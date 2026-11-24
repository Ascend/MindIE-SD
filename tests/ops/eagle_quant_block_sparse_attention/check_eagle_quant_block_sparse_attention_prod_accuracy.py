# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Resumable NPU accuracy and fixed-input stress; parent never initializes NPU."""

import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from tests.ops.eagle_quant_block_sparse_attention.eqbsa_cases import parse, print_preview, selections
from tests.ops.eagle_quant_block_sparse_attention.eqbsa_state import Batch, identity, read, save, select_device, tee_log
from tests.ops.eagle_quant_block_sparse_attention.eagle_quant_block_sparse_attention_accuracy_common import (
    bit_difference,
    compare,
    contract,
    defined_outputs,
    dut,
    fingerprint,
    golden,
    initialize,
    input_hashes,
    make_inputs,
    metadata,
    name,
    to_device,
)


def phases(plan):
    return ["accuracy", "stress"] if plan["mode"] == "all" else [plan["mode"]]


def key_for(phase, s):
    return phase + "/" + name(s)


def done(batch, phase, s):
    key = key_for(phase, s)
    if phase == "accuracy":
        return batch.get(key) is not None
    return all(batch.get(f"{key}/{i}") for i in range(1, batch.plan["parameters"]["stress_runs"] + 1))


def snapshot(path, tensors):
    import torch

    # Only the atomic record makes this independently written snapshot committed.
    torch.save(tensors, path)
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def worker(batch, phase, selected, device):
    for s in selected:
        key = key_for(phase, s)
        folder = batch.attempt(key)
        with tee_log(folder / "accuracy.log", stderr=True):
            worker_case(batch, phase, s, device, key, folder)


def worker_case(batch, phase, s, device, key, folder):
    active, stage_name = key, "initialize"
    started = time.monotonic()
    physical = {"device": device}
    try:
        print(f"START {key} device={device} stage={stage_name}", flush=True)
        physical = identity(device)
        torch, _ = initialize(device)
        save(folder / "execution.json", {"identity": physical, "phase": phase, "scenario": s})
        stage_name = "input preparation"
        print(f"START {key} {stage_name}", flush=True)
        stage = time.monotonic()
        cpu, parameters = make_inputs(s)
        hashes = input_hashes(cpu)
        inputs = to_device(cpu, device)
        del cpu
        torch.npu.synchronize()
        if input_hashes(inputs) != hashes:
            raise AssertionError("device copy changed encoded input bits")
        details = metadata(inputs, s, parameters)
        print(json.dumps(details), flush=True)
        save(folder / "inputs.json", {"metadata": details, "hashes": hashes})
        print(f"END {key} {stage_name} seconds={time.monotonic() - stage:.3f}", flush=True)
        artifacts = [folder / "execution.json", folder / "inputs.json"]
        if phase == "accuracy":
            stage_name = "DUT"
            print(f"START {key} {stage_name}", flush=True)
            stage = time.monotonic()
            actual = dut(inputs, s)
            torch.npu.synchronize()
            contract(actual, s)
            identity_metrics = None
            if s["softmax_lse_flag"]:
                off = dut(inputs, dict(s, softmax_lse_flag=0))
                torch.npu.synchronize()
                contract(off, dict(s, softmax_lse_flag=0))
                identity_metrics = dict(
                    passed=fingerprint(actual[0]) == fingerprint(off[0]), **bit_difference(actual[0], off[0])
                )
                del off
            print(f"END {key} DUT seconds={time.monotonic() - stage:.3f}", flush=True)
            stage_name = "NPU dequant + matmul/softmax/matmul/logsumexp golden"
            print(f"START {key} {stage_name}", flush=True)
            stage = time.monotonic()
            expected = golden(inputs, s)
            torch.npu.synchronize()
            print(f"END {key} golden seconds={time.monotonic() - stage:.3f}", flush=True)
            stage_name = "result validation"
            print(f"START {key} {stage_name}", flush=True)
            stage = time.monotonic()
            metrics = compare(actual, expected, s, large=True)
            metrics["O_bitwise_identity"] = identity_metrics
            metrics["input_unchanged"] = input_hashes(inputs) == hashes
            metrics["passed"] &= metrics["input_unchanged"] and (identity_metrics is None or identity_metrics["passed"])
            print(f"END {key} validation seconds={time.monotonic() - stage:.3f}", flush=True)
            if not metrics["passed"]:
                for label, outputs in (("actual", actual), ("golden", expected)):
                    path = folder / (label + ".pt")
                    snapshot(path, {k: v.cpu() for k, v in defined_outputs(outputs, s).items()})
                    artifacts.append(path)
            batch.commit(
                key,
                status="PASS" if metrics["passed"] else "FAIL",
                device=device,
                identity=physical,
                metrics=metrics,
                artifacts=artifacts,
                log=str((folder / "accuracy.log").relative_to(batch.path)),
            )
            if not metrics["passed"]:
                raise AssertionError("accuracy hard gate or O bitwise identity failed")
        else:
            total = batch.plan["parameters"]["stress_runs"]
            baseline_key = key + "/baseline"
            active = baseline_key
            record = batch.get(baseline_key)
            stage_name = "baseline restore" if record else "baseline creation"
            print(f"START {key} {stage_name}", flush=True)
            stage = time.monotonic()
            if record:
                if hashes != record["input_hashes"]:
                    raise AssertionError("inputs differ from the original stress baseline")
                baseline = torch.load(batch.path / record["baseline"], map_location="cpu", weights_only=True)
                if set(baseline) != ({"O", "LSE"} if s["softmax_lse_flag"] else {"O"}):
                    raise RuntimeError("baseline output set differs from the original contract")
                if input_hashes(baseline) != record["output_hashes"]:
                    raise RuntimeError("original baseline checksum mismatch")
            else:
                if any(batch.get(f"{key}/{i}") for i in range(1, total + 1)):
                    raise RuntimeError("stress rounds exist without original baseline")
                out = dut(inputs, s)
                torch.npu.synchronize()
                contract(out, s)
                baseline = {k: v.detach().cpu().clone() for k, v in defined_outputs(out, s).items()}
                del out
                if input_hashes(inputs) != hashes:
                    raise AssertionError("baseline call changed input bits")
                path = folder / "baseline.pt"
                snapshot(path, baseline)
                batch.commit(
                    baseline_key,
                    device=device,
                    identity=physical,
                    input_hashes=hashes,
                    output_hashes=input_hashes(baseline),
                    baseline=str(path.relative_to(batch.path)),
                    artifacts=[path, *artifacts],
                )
            print(f"END {key} {stage_name} seconds={time.monotonic() - stage:.3f}", flush=True)
            expected_hashes = input_hashes(baseline)
            history = sum(batch.get(f"{key}/{i}") is not None for i in range(1, total + 1))
            print(f"START {key} progress={history}/{total} historical={history} new=0", flush=True)
            last, executed = time.monotonic(), 0
            for i in range(1, total + 1):
                active = f"{key}/{i}"
                if batch.get(active):
                    continue
                stage_name = "stress DUT/synchronize"
                tick = time.monotonic()
                out = dut(inputs, s)
                torch.npu.synchronize()
                computed = time.monotonic()
                stage_name = "stress validation"
                contract(out, s)
                current = {k: v.detach().cpu() for k, v in defined_outputs(out, s).items()}
                del out
                current_hashes = input_hashes(current)
                unchanged = input_hashes(inputs) == hashes
                if current_hashes != expected_hashes or not unchanged:
                    metrics = {k: bit_difference(current[k], baseline[k]) for k in baseline}
                    metrics["input_unchanged"] = unchanged
                    path = folder / f"failure_round_{i}.pt"
                    snapshot(path, current)
                    batch.commit(
                        active,
                        status="FAIL",
                        device=device,
                        identity=physical,
                        metrics=metrics,
                        artifacts=[path, *artifacts],
                        baseline_key=baseline_key,
                    )
                    raise AssertionError(f"stress mismatch at original round {i}: {metrics}")
                del current
                checked = time.monotonic()
                stage_name = "stress record commit"
                batch.commit(
                    active,
                    device=device,
                    identity=physical,
                    input_hashes=hashes,
                    output_hashes=current_hashes,
                    baseline_key=baseline_key,
                    dut_sync_seconds=computed - tick,
                    validation_seconds=checked - computed,
                    log=str((folder / "accuracy.log").relative_to(batch.path)),
                    artifacts=artifacts,
                )
                now = time.monotonic()
                executed += 1
                if executed == 1 or i % 10 == 0 or i == total or now - last >= 30:
                    print(
                        f"PROGRESS {key} device={device} progress={history + executed}/{total} historical={history} "
                        f"new={executed} elapsed={now - started:.3f}s host_dut_sync={computed - tick:.3f}s "
                        f"host_validation={checked - computed:.3f}s host_commit={now - checked:.3f}s",
                        flush=True,
                    )
                    last = now
        print(f"PASS {key} device={device} elapsed={time.monotonic() - started:.3f}s", flush=True)
    except BaseException as error:
        print(f"FAILED {key} unit={active} device={device} stage={stage_name}: {error}", flush=True)
        if isinstance(error, AssertionError) and batch.get(active) is None:
            batch.commit(active, status="FAIL", device=device, identity=physical, error=traceback.format_exc())
        traceback.print_exc()
        raise


def summary(batch, reason):
    plan = batch.plan["parameters"]
    lines = [reason, "Type  Scenario  Status  Completed/Planned  Metrics", "Plan: " + json.dumps(plan)]
    any_results = False
    for phase in phases(plan):
        for s in selections(plan):
            key = key_for(phase, s)
            if phase == "accuracy":
                r = batch.get(key)
                any_results |= bool(r)
                lines.append(
                    f"{key} {r['status'] if r else 'PARTIAL'} {int(bool(r))}/1 "
                    + (json.dumps(r.get("metrics", r.get("error", {}))) if r else "")
                )
            else:
                rows = [batch.get(f"{key}/{i}") for i in range(1, plan["stress_runs"] + 1)]
                count = sum(bool(r and r["status"] == "PASS") for r in rows)
                baseline = batch.get(key + "/baseline")
                failed = any(r and r["status"] == "FAIL" for r in [baseline, *rows])
                any_results |= bool(count or failed)
                status = "FAIL" if failed else "PASS" if count == len(rows) else "PARTIAL"
                lines.append(f"{key} {status} {count}/{len(rows)} devices={sorted({r['device'] for r in rows if r})}")
                lines.extend(json.dumps(r) for r in [baseline, *rows] if r and r["status"] == "FAIL")
    if not any_results:
        lines.append("No completed results")
    lines.append("Stress across resumes is cumulative across process segments, not a continuous-process claim.")
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    (batch.path / "final_summary.txt").write_text(text, encoding="utf-8")


def run_batch(batch, args, plan):
    running = []
    reason, code = "PARTIAL", 1

    def stop(sig, frame):
        raise SystemExit(128 + sig)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        batch.ensure_no_failures()
        for phase in phases(plan):
            for s in selections(plan):
                print(f"SAVED {key_for(phase, s)} complete={done(batch, phase, s)}", flush=True)
        if all(done(batch, phase, s) for phase in phases(plan) for s in selections(plan)):
            reason, code = "COMPLETE (saved results; no device access)", 0
            return code
        cards = args.device_id
        if cards is None and args.resume:
            segments = sorted(batch.path.glob("segment_*.json"), key=lambda p: read(p)["started"])
            cards = read(segments[-1])["devices"] if segments else None
        cards = cards if cards is not None else [select_device()]
        batch.history(cards)
        assigned = {
            card: [s for i, s in enumerate(selections(plan)) if i % len(cards) == index]
            for index, card in enumerate(cards)
        }
        execution = batch.attempt("assignment")
        save(execution / "mapping.json", {str(k): [name(s) for s in v] for k, v in assigned.items()})
        # Parent has no torch import; fork preserves the lock in every live worker.
        context = multiprocessing.get_context("fork")
        for phase in phases(plan):
            for card, selected in assigned.items():
                pending = [s for s in selected if not done(batch, phase, s)]
                if pending:
                    process = context.Process(target=worker, args=(batch, phase, pending, card))
                    process.start()
                    running.append((process, card))
            while running:
                for process, card in list(running):
                    if process.is_alive():
                        continue
                    process.join()
                    running.remove((process, card))
                    if process.exitcode:
                        raise RuntimeError(f"worker card={card} exit={process.exitcode}")
                time.sleep(0.1)
        reason, code = "COMPLETE", 0
    except KeyboardInterrupt:
        reason, code = "PARTIAL SIGINT", 130
    except SystemExit as error:
        reason, code = "PARTIAL signal", int(error.code)
    except Exception:  # noqa: BLE001 - preserve every committed result on scheduler/tool errors.
        traceback.print_exc()
        reason, code = "PARTIAL error", 1
    finally:
        for process, _ in running:
            process.terminate()
        for process, _ in running:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        signal.signal(signal.SIGTERM, previous)
        summary(batch, reason)
    return code


def main(argv=None):
    args, plan = parse("accuracy", argv)
    if args.dry_run:
        print_preview(args, plan)
        return 0
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("unset ASCEND_RT_VISIBLE_DEVICES")
    batch = Batch("accuracy", plan, args.resume, args.output_dir)
    try:
        with tee_log(batch.path / "accuracy.log", stderr=True):
            return run_batch(batch, args, plan)
    finally:
        batch.close()


if __name__ == "__main__":
    sys.exit(main())
