# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Report the BF16 kernel Task Duration and complete three-operator Event minima."""

import argparse
import json
import math
import re
from pathlib import Path

from tests.ops.sigmoid_mul_dynamic_mx_quant.batch import record_path, write_json


PATHS = {
    "kernel": ("SigmoidMulDynamicMxQuant(BF16, mode=1)", "Task Duration"),
    "native": ("Sigmoid(BF16)+Mul(BF16)+DynamicMxQuant", "Complete-call Event"),
}


def unique_artifact(folder, pattern):
    matches = list(folder.rglob(pattern))
    if len(matches) != 1 or matches[0].is_file() and not matches[0].stat().st_size:
        raise ValueError(f"expected one nonempty {pattern} in {folder}")
    return matches[0]


def read_run(folder, path, plan):
    if path == "kernel":
        log = (folder / "msprof.log").read_text(encoding="utf-8", errors="replace")
        modes = re.findall(
            r"Preparing BF16 CPU inputs \(8068, 1536\); precision_mode=1(?:, scale_alg=(\d+))?\s*$", log, re.MULTILINE
        )
        if len(modes) != 1 or int(modes[0] or 0) != plan.get("scale_alg", 0) or ("scale_alg" in plan and not modes[0]):
            raise ValueError(f"kernel scale algorithm does not match the plan in {folder}")
        durations = re.findall(r"Task\s+Duration\s*\(us\)\s*:\s*([\d.]+)", log)
        if len(durations) != 1 or "Completed kernel with outputs" not in log:
            raise ValueError(f"expected one completed fused kernel measurement in {folder}")
        value = float(durations[0])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid kernel duration in {folder}")
        unique_artifact(folder, "OpBasicInfo.csv")
        return {"duration_us": value, "data_dir": str(folder)}

    profiled = not plan["no_msprof"]
    timing = folder / ("profiled_timing.json" if profiled else "timing.json")
    event = json.loads(timing.read_text(encoding="utf-8"))
    expected = {
        "shape": [8068, 1536],
        "device": f"npu:{plan['device_id']}",
        "precision_mode": 1,
        "path": path,
        "profiled": profiled,
        "warmup": 5,
        "iterations": plan["iterations"],
        "samples": plan["iterations"],
    }
    # Before scale_alg was configurable, both the plan and timing always used algorithm 0.
    if "scale_alg" in plan or "scale_alg" in event:
        expected["scale_alg"] = plan.get("scale_alg", 0)
    for name, value in expected.items():
        if event[name] != value:
            raise ValueError(f"expected {name}={value} in {timing}, got {event[name]}")
    samples = event["event_samples_us"]
    if len(samples) != plan["iterations"] or not all(math.isfinite(v) and v > 0 for v in samples):
        raise ValueError(f"invalid Event samples in {timing}")
    if event["event_us"] != min(samples):
        raise ValueError(f"reported minimum differs from Event samples in {timing}")
    unique_artifact(folder, "event_samples.csv")
    profile = None
    if profiled:
        profile = unique_artifact(folder, "PROF_*")
        unique_artifact(profile, "op_summary*.csv")
        unique_artifact(profile, "msprof_*.json")
    return {
        "duration_us": min(samples),
        "sample": samples.index(min(samples)) + 1,
        "data_dir": str(profile or folder),
        "profile_dir": str(profile) if profile else None,
        "timing_file": str(timing),
        "event_samples_us": samples,
    }


def format_rows(rows):
    widths = [max(map(len, column)) for column in zip(*rows)]
    return ["\t".join(s.ljust(w) for s, w in zip(row[:-1], widths)) + "\t" + row[-1] for row in rows]


def summarize(root, partial_reason=None):
    root = root.resolve()
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))["parameters"]
    if plan.get("precision_mode") != 1:
        raise ValueError("expected a BF16 mode 1 performance plan")
    paths = [plan["path"]] if plan["path"] else PATHS
    rows = [("Scenario", "Test", "Metric", "Duration(us)", "DataDir")]
    minima = [("Scenario", "Test", "Metric", "Minimum(us)", "DataDir")]
    report = {}
    for path in paths:
        name, metric = PATHS[path]
        name = f"{name} [scale_alg={plan.get('scale_alg', 0)}]"
        runs = []
        for run in range(1, plan["runs"] + 1):
            key = f"mode1_{path}_{run}"
            record = record_path(root, key)
            if not record.exists() or json.loads(record.read_text(encoding="utf-8"))["status"] != "PASS":
                if partial_reason is not None:
                    continue
                raise ValueError(f"missing completed run {key} in {root}")
            result = read_run(root / key, path, plan)
            runs.append({"run": run, **result})
            rows.append((name, str(run), metric, f"{result['duration_us']:.6f}", result["data_dir"]))
        report[path] = {"scenario": name, "metric": metric, "runs": runs}
        if runs:
            best = min(runs, key=lambda item: item["duration_us"])
            report[path].update(min_us=best["duration_us"], best_run=best["run"], data_dir=best["data_dir"])
            minima.append((name, str(best["run"]), metric, f"{best['duration_us']:.6f}", best["data_dir"]))
    prefix = "partial_summary" if partial_reason is not None else "final_summary"
    lines = [*format_rows(rows), "", *format_rows(minima)]
    if "kernel" in report and len(report["kernel"]["runs"]) > 1:
        values = [r["duration_us"] for r in report["kernel"]["runs"]]
        spread = (max(values) - min(values)) / min(values) * 100
        lines.append(f"Kernel warmup={plan['warmup']}, max={max(values):.6f} us, spread={spread:.2f}%")
        if spread > 10:
            lines.append("Kernel spread exceeds 10%. Consider repeating the batch with more kernel warmup.")
    if partial_reason is not None:
        lines += [
            f"PARTIAL {partial_reason}",
            *(f"{p}: {len(report[p]['runs'])}/{plan['runs']} complete" for p in paths),
        ]
    else:
        lines += [
            "",
            "Note: Some hardware models may show increased latency across runs. The minimum across all "
            "groups for each scenario is reported as the best performance observed under the current test "
            "conditions; this does not mean the variation has been eliminated.",
        ]
    summary = "\n".join(lines)
    print(summary)
    (root / f"{prefix}.txt").write_text(summary + "\n", encoding="utf-8")
    write_json(root / ("partial_summary.json" if partial_reason is not None else "summary.json"), {"1": report})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
