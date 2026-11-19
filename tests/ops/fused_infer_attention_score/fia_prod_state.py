# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Atomic FIA test records and reports; no device initialization."""

import hashlib
import json
import math
import os
from pathlib import Path

_VERIFIED_ARTIFACTS = {}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".pending.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_path(batch, scenario, stage, iteration):
    return Path(batch) / "records" / f"{scenario}_{stage}_{iteration:04d}.json"


def plan_fingerprint(plan):
    contract = {key: plan[key] for key in ("version", "kind", "scenarios", "stages", "seed", "identity")}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def load_record(path, fingerprint=None):
    record = read_json(path)
    if record["unit"] != Path(path).stem or record["version"] != 1:
        raise ValueError(f"Unit identity mismatch: {path}")
    if fingerprint is not None and record["plan_fingerprint"] != fingerprint:
        raise ValueError(f"Plan identity mismatch: {path}")
    if record["status"] not in ("PASS", "FAIL"):
        raise ValueError(f"Invalid completed status: {path}")
    if not isinstance(record["device_id"], int) or not Path(record["data_dir"]).is_dir():
        raise ValueError(f"Invalid device or data directory: {path}")
    stage = record["unit"].split("_")[1]
    if stage in ("accuracy", "stress") and not Path(record["worker_log"]).is_file():
        raise ValueError(f"Missing worker log: {path}")
    if stage == "accuracy" and record["status"] == "PASS":
        metrics = record["metrics"]
        if not all(math.isfinite(metrics[key]) for key in ("cosine", "norm_ratio", "max_abs_error", "matched_ratio")):
            raise ValueError(f"Invalid accuracy metrics: {path}")
        if metrics["cosine"] < 0.99 or not 0.9 <= metrics["norm_ratio"] <= 1.1:
            raise ValueError(f"PASS contradicts accuracy gates: {path}")
    if stage == "stress" and (len(record["inputs"]) != 6 or not record["artifacts"]):
        raise ValueError(f"Missing stress input/baseline evidence: {path}")
    if stage == "performance" and (
        not math.isfinite(record["duration_us"]) or record["duration_us"] <= 0 or not record["artifacts"]
    ):
        raise ValueError(f"Invalid performance evidence: {path}")
    for artifact, digest in record.get("artifacts", {}).items():
        stat = Path(artifact).stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, digest)
        if _VERIFIED_ARTIFACTS.get(artifact) == signature:
            continue
        if file_hash(artifact) != digest:
            raise ValueError(f"Artifact changed: {artifact}")
        _VERIFIED_ARTIFACTS[artifact] = signature
    return record


def _table(header, rows):
    table = [header, *rows]
    widths = [max(map(len, column)) for column in zip(*table)]
    return ["\t".join(cell.ljust(width) for cell, width in zip(row[:-1], widths)) + "\t" + row[-1] for row in table]


def summarize(batch, plan, reason=""):
    rows = []
    totals = []
    minima = []
    notes = []
    complete = True
    failed = False
    for scenario in plan["scenarios"]:
        for stage, count in plan["stages"].items():
            durations = []
            done = 0
            for iteration in range(1, count + 1):
                path = record_path(batch, scenario["id"], stage, iteration)
                if not path.exists():
                    complete = False
                    continue
                result = load_record(path, plan_fingerprint(plan))
                done += 1
                failed |= result["status"] == "FAIL"
                duration = result.get("duration_us")
                if duration is not None and result["status"] == "PASS":
                    durations.append(duration)
                rows.append(
                    (
                        scenario["label"],
                        f"{stage}:{iteration}",
                        result["status"],
                        "Task Duration" if duration is not None else str(result.get("metrics", {})),
                        str(duration if duration is not None else "-"),
                        result["data_dir"],
                    )
                )
            totals.append((scenario["label"], stage, f"{done}/{count}", "completed", "-", str(batch)))
            if stage == "performance":
                minima.append(
                    (scenario["label"], str(min(durations)) if durations else "-", f"{len(durations)}/{count}")
                )
            if durations:
                variation = (max(durations) - min(durations)) / min(durations) * 100
                notes.append(
                    f"{scenario['label']}: warmup={scenario['warmup']} samples={len(durations)} "
                    f"max_us={max(durations)} variation={variation:.2f}%"
                    + ("; consider more warmup and a complete new batch" if variation > 10 else "")
                )
                if len(durations) < 2:
                    notes.append("Insufficient samples to assess variation.")
    status = "FAIL" if failed else "PASS" if complete else "PARTIAL"
    header = ("Scenario", "Test", "Status", "Metric", "Duration(us)", "DataDir")
    performance = plan["kind"] == "performance"
    lines = [f"{status} {reason}", *_table(header, rows if performance else rows + totals)]
    if not rows:
        lines.append("No completed results")
    if performance:
        lines.extend(
            ["", "Minimum Task Duration by scenario", *_table(("Scenario", "Min Task Duration(us)", "Samples"), minima)]
        )
        lines.extend(notes)
        lines.append(
            "Note: Some hardware models may show increased latency across runs. The minimum across all groups "
            "for each scenario is reported as the best performance observed under the current test conditions; "
            "this does not mean the variation has been eliminated."
        )
    report = "\n".join(lines) + "\n"
    print(report, end="", flush=True)
    (Path(batch) / "final_summary.txt").write_text(report, encoding="utf-8")
    return complete and not failed
