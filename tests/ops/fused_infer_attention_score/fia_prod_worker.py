# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""One large FIA accuracy/stability scenario, with per-unit persistence."""

import argparse
import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace

from tests.ops.fused_infer_attention_score.profile_fia_dit_quant import (
    call_fia,
    describe_inputs,
    make_inputs,
    prepare_operator_environment,
)
from tests.ops.fused_infer_attention_score.fia_prod_state import (
    atomic_json,
    file_hash,
    load_record,
    plan_fingerprint,
    read_json,
    record_path,
)


def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8).numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def metrics_for(reference, output):
    import torch

    from tests.ops.fused_infer_attention_score.fia_accuracy_common import cosine_metrics

    metrics = cosine_metrics(reference, output)
    # Comparison only: the golden computation itself stays on NPU.
    expected = reference.detach().cpu().reshape(-1)
    actual = output.detach().cpu().reshape(-1)
    matched, maximum, golden_at_max = 0, -1.0, 0.0
    for start in range(0, actual.numel(), 1048576):
        a, g = actual[start : start + 1048576].float(), expected[start : start + 1048576].float()
        error = (a - g).abs()
        matched += int((error <= 0.125 + 0.125 * g.abs()).sum())
        block_max, index = error.max(dim=0)
        if float(block_max) > maximum:
            maximum, golden_at_max = float(block_max), float(g[index])
    low = torch.tensor(abs(golden_at_max), dtype=torch.bfloat16)
    ulp = float(torch.nextafter(low, torch.tensor(float("inf"), dtype=torch.bfloat16)).float() - low.float())
    metrics.update(
        matched_ratio=matched / actual.numel(),
        rtol=0.125,
        atol=0.125,
        tolerance_dtype="FLOAT8_E4M3",
        output_dtype="BF16",
        ulp=ulp,
        max_abs_limit=max(1.0, 32 * ulp),
        golden_at_max=golden_at_max,
    )
    metrics["soft_gate"] = metrics["matched_ratio"] >= 0.99 and maximum <= metrics["max_abs_limit"]
    return metrics


def run(args, batch, scenario, stage, count, fingerprint):
    prepare_operator_environment()
    import torch
    import torch_npu

    torch.npu.set_device(args.device_id)
    import mindiesd  # noqa: F401 pylint: disable=unused-import

    print("stage=inputs", flush=True)
    originals, encoded, scales_cpu = make_inputs(args)
    input_hashes = [tensor_hash(tensor) for tensor in (*encoded, *scales_cpu)]
    qkv = [tensor.to(f"npu:{args.device_id}") for tensor in encoded]
    scales = [tensor.to(f"npu:{args.device_id}") for tensor in scales_cpu]
    torch_npu.npu.synchronize()
    describe_inputs(qkv, scales)
    data_dir = Path(batch) / scenario
    data_dir.mkdir(exist_ok=True)
    baseline_file = data_dir / "baseline.pt"
    baseline_record = data_dir / "baseline.json"
    baseline = None
    artifacts = {}
    if stage == "stress":
        if baseline_record.exists():
            saved = read_json(baseline_record)
            if saved["inputs"] != input_hashes or file_hash(baseline_file) != saved["sha256"]:
                raise ValueError("Baseline or reproduced inputs changed; refusing resume")
            baseline = torch.load(baseline_file, map_location="cpu", weights_only=True)
        else:
            if any(record_path(batch, scenario, stage, index).exists() for index in range(1, count + 1)):
                raise ValueError("Completed stress records exist without baseline")
            print("stage=baseline", flush=True)
            baseline = call_fia(args, qkv, scales)[0].cpu()
            temporary = baseline_file.with_suffix(".pending")
            with temporary.open("wb") as stream:
                torch.save(baseline, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, baseline_file)
            atomic_json(baseline_record, {"inputs": input_hashes, "sha256": file_hash(baseline_file)})
        artifacts[str(baseline_file)] = file_hash(baseline_file)
        if [tensor_hash(tensor) for tensor in (*qkv, *scales)] != input_hashes:
            raise ValueError("Inputs changed during baseline creation")
    started, reported = time.monotonic(), time.monotonic()
    for iteration in range(1, count + 1):
        result_file = record_path(batch, scenario, stage, iteration)
        if result_file.exists():
            if load_record(result_file, fingerprint)["status"] == "FAIL":
                raise ValueError("Previously committed failure; use a new batch to retest")
            continue
        print(f"stage=DUT scenario={scenario} iteration={iteration}", flush=True)
        before = time.monotonic()
        output, lse = call_fia(args, qkv, scales)
        torch_npu.npu.synchronize()
        dut_seconds = time.monotonic() - before
        before = time.monotonic()
        valid = (
            tuple(output.shape) == tuple(originals[0].shape)
            and output.dtype == torch.bfloat16
            and output.is_contiguous()
            and (lse is None or lse.numel() == 0)
        )
        valid = valid and bool(torch.isfinite(output.float()).all().item())
        metrics = {}
        if stage == "accuracy":
            print("stage=NPU_golden", flush=True)
            inputs = [tensor.to(f"npu:{args.device_id}") for tensor in originals]
            reference = torch_npu.npu_fusion_attention(
                *inputs,
                head_num=args.num_query_heads,
                input_layout="BNSD",
                scale=args.head_dim**-0.5,
                pre_tockens=2147483647,
                next_tockens=2147483647,
            )[0]
            torch_npu.npu.synchronize()
            print("stage=compare golden=unquantized_BF16_npu_fusion_attention", flush=True)
            valid = valid and bool(torch.isfinite(reference.float()).all().item())
            if valid:
                try:
                    metrics = metrics_for(reference, output)
                    valid = metrics["cosine"] >= 0.99 and 0.9 <= metrics["norm_ratio"] <= 1.1
                except ValueError as exc:
                    metrics = {"failure": str(exc)}
                    valid = False
        else:
            current = output.cpu()
            valid = valid and torch.equal(current.view(torch.int16), baseline.view(torch.int16))
            if not valid:
                torch.save(current, data_dir / f"failed_{iteration}.pt")
            if iteration == count or not valid:
                valid = valid and [tensor_hash(tensor) for tensor in (*qkv, *scales)] == input_hashes
        check_seconds = time.monotonic() - before
        record = {
            "version": 1,
            "unit": result_file.stem,
            "plan_fingerprint": fingerprint,
            "status": "PASS" if valid else "FAIL",
            "device_id": args.device_id,
            "inputs": input_hashes,
            "data_dir": str(data_dir),
            "metrics": metrics,
            "artifacts": artifacts,
            "worker_log": args.worker_log,
            "dut_wall_seconds": dut_seconds,
            "check_wall_seconds": check_seconds,
        }
        atomic_json(result_file, record)
        now = time.monotonic()
        if iteration == 1 or iteration % 10 == 0 or iteration == count or now - reported >= 30 or not valid:
            print(
                f"scenario={scenario} stage={stage} progress={iteration}/{count} "
                f"elapsed={now - started:.3f}s dut={dut_seconds:.3f}s check={check_seconds:.3f}s "
                f"status={record['status']} metrics={metrics}",
                flush=True,
            )
            reported = now
        if not valid:
            raise ValueError(f"{scenario} {stage} iteration={iteration} FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--stage", choices=("accuracy", "stress"), required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--worker-log", required=True)
    cli = parser.parse_args()
    plan = read_json(Path(cli.batch) / "plan.json")
    scenario = next(item for item in plan["scenarios"] if item["id"] == cli.scenario)
    args = SimpleNamespace(
        **scenario["shape"],
        path=scenario["path"],
        distribution=scenario["distribution"],
        seed=plan["seed"],
        enhance_mode=2.0,
        device_id=cli.device_id,
        worker_log=cli.worker_log,
    )
    run(args, cli.batch, cli.scenario, cli.stage, plan["stages"][cli.stage], plan_fingerprint(plan))


if __name__ == "__main__":
    main()
