"""Production accuracy and bitwise stability; parent never imports torch/NPU."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_common import (
    bit_difference,
    compare,
    contract,
    dut,
    fingerprint,
    initialize,
    input_hashes,
    make_inputs,
    metadata,
    npu_golden,
    preview,
    scenario,
    to_device,
)
from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_state import Batch, ROOT, digest, read, save, tee_log


def parser():
    p = argparse.ArgumentParser(
        description="910B INT8 ND QBMM + FP32 SwiGLU, BF16 output",
        epilog="Defaults: M=5672 K=5376 N=28672; x1=[M,K], x2=[K,N], scale=[N], token=[M], y=[M,N/2]. "
        "Positive M/K/N, N must be even. No transpose/bias. Both uniform and normal; seed=20261030; "
        "mode=all (all accuracy before stress); stress-runs=1000. Omitted device auto-selects one card; "
        "resume restores last devices. Output: repo/log/eagle_quant_batch_matmul_swiglu/accuracy_TIMESTAMP.",
    )
    p.add_argument("--device-id", type=int, nargs="+")
    p.add_argument("--mode", choices=("all", "accuracy", "stress"))
    p.add_argument("--stress-runs", type=int)
    for name in ("m", "k", "n", "seed"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--distribution", choices=("uniform", "normal"))
    p.add_argument("--resume")
    p.add_argument("--output-dir")
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve(args, saved=None):
    defaults = dict(m=5672, k=5376, n=28672, seed=20261030, mode="all", stress_runs=1000, distribution=None)
    values = dict(saved) if saved else defaults
    for key in defaults:
        value = getattr(args, key)
        if value is not None:
            if saved and value != values[key]:
                raise ValueError(f"resume conflicts with {key}")
            values[key] = value
    if min(values[k] for k in ("m", "k", "n", "stress_runs")) <= 0:
        raise ValueError("M, K, N and stress-runs must be positive")
    if values["n"] % 2:
        raise ValueError("N must be even")
    if args.device_id is not None and (min(args.device_id) < 0 or len(set(args.device_id)) != len(args.device_id)):
        raise ValueError("device-id must contain distinct nonnegative physical IDs")
    if args.resume and args.output_dir:
        raise ValueError("resume and output-dir are mutually exclusive")
    return values


def cases(plan):
    return [
        scenario(m=plan["m"], k=plan["k"], n=plan["n"], seed=plan["seed"], distribution=d)
        for d in ([plan["distribution"]] if plan["distribution"] else ["uniform", "normal"])
    ]


def units(plan):
    return [
        (phase, s)
        for phase in (["accuracy", "stress"] if plan["mode"] == "all" else [plan["mode"]])
        for s in cases(plan)
    ]


def done(batch, phase, s):
    key = phase + "/" + s["distribution"]
    if phase == "accuracy":
        return batch.get(key) is not None
    return all(batch.get(key + f"/{i}") for i in range(1, batch.plan["parameters"]["stress_runs"] + 1))


def worker(batch, phase, s, device):
    key = phase + "/" + s["distribution"]
    attempt = batch.attempt(key)
    with tee_log(attempt / "accuracy.log", stderr=True):
        worker_run(batch, phase, s, device, key, attempt)


def worker_run(batch, phase, s, device, key, attempt):
    active_key = key
    try:
        started = time.monotonic()
        from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_perf import identity

        physical = identity(device)
        save(attempt / "execution.json", dict(phase=phase, scenario=s, identity=physical))
        print(f"START phase={phase} distribution={s['distribution']} device={device}", flush=True)
        torch, _ = initialize(device)
        print("START CPU input generation", flush=True)
        cpu, params = make_inputs(s)
        original = input_hashes(cpu)
        inputs = to_device(cpu, device)
        del cpu
        torch.npu.synchronize()
        print(json.dumps(metadata(inputs, s, params)), flush=True)
        save(attempt / "inputs.json", dict(metadata=metadata(inputs, s, params), hashes=original))
        print(f"READY inputs elapsed={time.monotonic() - started:.3f}s", flush=True)
        if phase == "accuracy":
            print("START DUT", flush=True)
            stage = time.monotonic()
            actual = dut(inputs, s)
            torch.npu.synchronize()
            print(
                f"END DUT elapsed={time.monotonic() - stage:.3f}s; START INT32 matmul + FP32 SwiGLU golden", flush=True
            )
            stage = time.monotonic()
            golden = npu_golden(inputs, s)
            torch.npu.synchronize()
            print(f"END golden elapsed={time.monotonic() - stage:.3f}s; START accuracy metrics", flush=True)
            stage = time.monotonic()
            metrics = compare(actual, golden, s, large=True)
            print(f"END metrics elapsed={time.monotonic() - stage:.3f}s; START input integrity check", flush=True)
            if input_hashes(inputs) != original:
                raise AssertionError("DUT changed input tensors")
            batch.commit(
                key,
                status="PASS" if metrics["passed"] else "FAIL",
                device=device,
                metrics=metrics,
                identity=physical,
                artifacts=[attempt / "inputs.json", attempt / "execution.json"],
            )
            if not metrics["passed"]:
                raise AssertionError("production accuracy hard gate failed")
        else:
            baseline_key = key + "/baseline"
            record = batch.get(baseline_key)
            if record:
                print("START loading original baseline and checking regenerated inputs", flush=True)
                stage = time.monotonic()
                if record["input_hashes"] != original:
                    raise AssertionError("regenerated inputs differ from original baseline inputs")
                baseline = torch.load(batch.path / record["baseline"], map_location="cpu", weights_only=True)
                contract(baseline, s)
                if fingerprint(baseline) != record["output_hash"]:
                    raise AssertionError("baseline bit checksum mismatch")
                print(f"END baseline restore elapsed={time.monotonic() - stage:.3f}s", flush=True)
            else:
                if any(batch.get(key + f"/{i}") for i in range(1, batch.plan["parameters"]["stress_runs"] + 1)):
                    raise RuntimeError("committed rounds have no baseline")
                print("START baseline", flush=True)
                out = dut(inputs, s)
                torch.npu.synchronize()
                contract(out, s)
                baseline = out.detach().cpu().clone()
                del out
                if input_hashes(inputs) != original:
                    raise AssertionError("baseline call changed inputs")
                baseline_path = attempt / "baseline.pt"
                torch.save(baseline, baseline_path)
                with baseline_path.open("rb") as stream:
                    os.fsync(stream.fileno())
                batch.commit(
                    baseline_key,
                    device=device,
                    baseline=str(baseline_path.relative_to(batch.path)),
                    output_hash=fingerprint(baseline),
                    input_hashes=original,
                    identity=physical,
                    artifacts=[baseline_path, attempt / "inputs.json", attempt / "execution.json"],
                )
            reference_hash = fingerprint(baseline)
            total = batch.plan["parameters"]["stress_runs"]
            completed = sum(batch.get(key + f"/{i}") is not None for i in range(1, total + 1))
            print(f"START stress loop {key} device={device} completed={completed}/{total}", flush=True)
            last = time.monotonic()
            first_new = True
            for i in range(1, total + 1):
                active_key = key + f"/{i}"
                if batch.get(active_key):
                    continue
                round_start = time.monotonic()
                out = dut(inputs, s)
                torch.npu.synchronize()
                compute_end = time.monotonic()
                contract(out, s)
                current = out.detach().cpu()
                del out
                if fingerprint(current) != reference_hash:
                    torch.save(current, attempt / "current.pt")
                    torch.save(baseline, attempt / "baseline_failure.pt")
                    metrics = compare(current, baseline, s, large=True)
                    metrics.update(bit_difference(current, baseline))
                    metrics["input_unchanged"] = input_hashes(inputs) == original
                    batch.commit(
                        active_key,
                        status="FAIL",
                        device=device,
                        metrics=metrics,
                        artifacts=[attempt / "current.pt", attempt / "baseline_failure.pt"],
                    )
                    raise AssertionError("bitwise stability failed; tensors saved")
                del current
                # A completed round includes unchanged inputs, not just unchanged output.
                if input_hashes(inputs) != original:
                    raise AssertionError("stress call changed input tensors")
                check_end = time.monotonic()
                batch.commit(
                    active_key,
                    device=device,
                    input_hashes=original,
                    identity=physical,
                    baseline_key=baseline_key,
                    output_hash=reference_hash,
                    log=str((attempt / "accuracy.log").relative_to(batch.path)),
                    compute_seconds=compute_end - round_start,
                    check_seconds=check_end - compute_end,
                )
                now = time.monotonic()
                if first_new or i % 10 == 0 or i == total or now - last >= 30:
                    print(
                        f"PROGRESS {key} device={device} {i}/{total} elapsed={now - started:.3f}s "
                        f"dut_sync={compute_end - round_start:.3f}s check={check_end - compute_end:.3f}s "
                        f"commit={now - check_end:.3f}s",
                        flush=True,
                    )
                    last = now
                first_new = False
        print(f"PASS {key} device={device} elapsed={time.monotonic() - started:.3f}s", flush=True)
    except AssertionError:
        if batch.get(active_key) is None:
            batch.commit(active_key, status="FAIL", device=device, error=traceback.format_exc())
        traceback.print_exc()
        raise
    except Exception:
        # Initialization/driver/I/O failures do not establish a numerical verdict.
        # Preserve this attempt; the uncommitted unit remains resumable.
        traceback.print_exc()
        raise


def summary(batch, reason):
    lines = [
        reason,
        "Resolved plan: " + json.dumps(batch.plan["parameters"]),
        "x1 INT8 ND [M,K]; x2 INT8 ND [K,N]; scale FP32 [N]; token FP32 [M]; output BF16 [M,N]",
    ]
    for phase, s in units(batch.plan["parameters"]):
        key = phase + "/" + s["distribution"]
        if phase == "accuracy":
            record = batch.get(key)
            lines.append(
                f"{record['status'] if record else 'PARTIAL'} {key} device={record.get('device') if record else None} "
                + (json.dumps(record.get("metrics", {})) if record else "")
            )
        else:
            total = batch.plan["parameters"]["stress_runs"]
            records = [batch.get(key + f"/{i}") for i in range(1, total + 1)]
            count = sum(r is not None and r["status"] == "PASS" for r in records)
            status = (
                "FAIL" if any(r and r["status"] == "FAIL" for r in records) else "PASS" if count == total else "PARTIAL"
            )
            devices = sorted({r["device"] for r in records if r})
            lines.append(f"{status} {key} runs={count}/{total} devices={devices}")
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    (batch.path / "final_summary.txt").write_text(text, encoding="utf-8")


def main():
    args = parser().parse_args()
    flags = [arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")]
    if len(flags) != len(set(flags)):
        raise ValueError("duplicate option")
    saved = read(Path(args.resume) / "plan.json")["parameters"] if args.resume else None
    plan = resolve(args, saved)
    if args.dry_run:
        selected = cases(plan)
        mapping = [
            {
                "distribution": s["distribution"],
                "device": args.device_id[i % len(args.device_id)] if args.device_id else "auto/restored",
            }
            for i, s in enumerate(selected)
        ]
        print(json.dumps(dict(plan=plan, scenarios=[preview(s) for s in selected], mapping=mapping), indent=2))
        return 0
    batch = Batch("accuracy", plan, args.resume, args.output_dir)
    try:
        with tee_log(batch.path / "accuracy.log"):
            return run_batch(batch, plan, args)
    finally:
        batch.close()


def run_batch(batch, plan, args):
    running = []
    reason, code = "PARTIAL", 1
    try:
        batch.ensure_no_failures()
        print("RESUME progress" if args.resume else "NEW batch", batch.path, flush=True)
        for phase, s in units(plan):
            print(f"{phase}/{s['distribution']} complete={done(batch, phase, s)}", flush=True)
        if all(done(batch, phase, s) for phase, s in units(plan)):
            reason, code = "COMPLETE (committed results; no NPU access)", 0
            return code
        devices = args.device_id
        if devices is None and args.resume:
            history = sorted(batch.path.glob("segment_*.json"), key=lambda p: read(p)["started"])
            devices = read(history[-1])["devices"] if history else None
        if devices is None:
            devices = [
                int(
                    subprocess.check_output(
                        [sys.executable, str(ROOT / "tests/tools/select_npu_device.py"), "--format=id"], text=True
                    ).strip()
                )
            ]
        batch.history(devices)

        def stop(sig, frame):
            raise SystemExit(128 + sig)

        signal.signal(signal.SIGTERM, stop)
        ctx = multiprocessing.get_context("fork")
        for phase in ["accuracy", "stress"] if plan["mode"] == "all" else [plan["mode"]]:
            pending = [s for s in cases(plan) if not done(batch, phase, s)]
            while pending or running:
                used = {device for process, device in running}
                for device in devices:
                    if pending and device not in used:
                        process = ctx.Process(target=worker, args=(batch, phase, pending.pop(0), device))
                        process.start()
                        running.append((process, device))
                for process, device in list(running):
                    if not process.is_alive():
                        process.join()
                        running.remove((process, device))
                        if process.exitcode:
                            raise RuntimeError(f"worker device={device} exit={process.exitcode}")
                time.sleep(0.1)
        reason, code = "COMPLETE", 0
    except KeyboardInterrupt:
        reason, code = "PARTIAL SIGINT", 130
    except SystemExit as exc:
        reason, code = "PARTIAL signal", int(exc.code)
    except Exception:
        traceback.print_exc()
        reason, code = "PARTIAL error", 1
    finally:
        for process, _ in running:
            process.terminate()
        for process, _ in running:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        summary(batch, reason)
    return code


if __name__ == "__main__":
    sys.exit(main())
