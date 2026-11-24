# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Crash-safe immutable plans and per-unit records for this operator's tests."""

import hashlib
from contextlib import contextmanager, redirect_stderr, redirect_stdout, ExitStack
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OP = HERE.name


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def json_safe(value):
    """Keep nonfinite diagnostic values explicit without changing pass/fail flags."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save(path, value):
    path = Path(path)
    value = json_safe(value)
    payload = {"value": value, "sha256": hashlib.sha256(canonical(value)).hexdigest()}
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temp.open("xb") as stream:
        stream.write(canonical(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def read(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if hashlib.sha256(canonical(payload["value"])).hexdigest() != payload["sha256"]:
        raise RuntimeError(f"checksum mismatch: {path}")
    return payload["value"]


def environment():
    paths = list(HERE.glob("*.py")) + list(HERE.glob("*.sh"))
    paths += list((ROOT / "csrc/ops" / OP).rglob("*"))
    paths += [ROOT / "csrc/plugin" / (OP + ext) for ext in (".cpp", ".h")]
    paths += [ROOT / "csrc/plugin/register_ops.cpp", ROOT / "mindiesd/layers/_custom_ops.py"]
    paths += [
        ROOT / "csrc/plugin/block_sparse_attention.cpp",
        ROOT / "csrc/plugin/block_sparse_attention.h",
        ROOT / "mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py",
        ROOT / "mindiesd/layers/quant/block_quant.py",
    ]
    paths += list((ROOT / "mindiesd").rglob("*.so"))
    paths += [p for p in (ROOT / "mindiesd/ops").rglob("*") if p.suffix in (".json", ".o", ".bin")]
    versions = {}
    for name in ("torch", "torch-npu", "mindiesd", "numpy", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    opp = Path(os.environ.get("ASCEND_OPP_PATH", "/nonexistent"))
    for path in (opp / "version.info", opp.parent / "version.cfg"):
        if path.is_file():
            versions[str(path)] = digest(path)
    return {
        "python": sys.version,
        "machine": platform.machine(),
        "versions": versions,
        "files": {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths)) if p.is_file()},
    }


class Batch:
    def __init__(self, kind, plan=None, resume=None, output=None):
        self.lock = None
        if resume:
            self.path = Path(resume).resolve()
            if not self.path.is_dir():
                raise ValueError("resume batch does not exist")
        else:
            self.path = (
                Path(output).resolve()
                if output
                else ROOT
                / "logs"
                / OP
                / (("perf" if kind == "perf" else "accuracy") + "_" + time.strftime("%Y%m%d_%H%M%S"))
            )
            if not output and self.path.exists():
                self.path = self.path.with_name(self.path.name + "_" + uuid.uuid4().hex[:8])
            self.path.mkdir(parents=True, exist_ok=False)
        try:
            self.lock = (self.path / "executor.lock").open("a+b")
            if os.name == "posix":
                import fcntl

                fcntl.lockf(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt

                self.lock.seek(0)
                if not self.lock.read(1):
                    self.lock.write(b"0")
                    self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.close()
            raise RuntimeError("batch already has an executor") from exc
        if resume:
            self.plan = read(self.path / "plan.json")
            if self.plan["kind"] != kind or self.plan["environment"] != environment():
                raise RuntimeError("resume kind/code/software/artifact fingerprint mismatch")
            if plan is not None and plan != self.plan["parameters"]:
                raise ValueError("resume parameters conflict with immutable plan")
        else:
            self.plan = {
                "kind": kind,
                "parameters": plan,
                "environment": environment(),
                "protocol": {
                    "cosine_min": 0.99,
                    "norm_ratio": [0.9, 1.1],
                    "matched_ratio_min": 0.99,
                    "outputs": {
                        "O": {"precision": "fp8_e4m3", "atol": 2**-3, "rtol": 2**-3, "fallback": 1.0},
                        "LSE": {"precision": "fp16", "atol": 2**-10, "rtol": 2**-10, "fallback": 0.1},
                        "analytic_LSE": {"precision": "fp32", "atol": 2**-13, "rtol": 2**-13, "fallback": 0.01},
                    },
                    "max_abs": "max(fallback,32*ULP(golden_at_max_error,actual_output_dtype))",
                    "small_mixed": "hard",
                    "production_mixed": "soft",
                    "stress": "bitwise",
                    "input": "Q/K INT8 per64; V FP8 E4M3 bits; FP32 scales; mask INT8",
                    "output": "O BF16/FP16; enabled LSE FP32; disabled LSE values undefined",
                },
            }
            save(self.path / "plan.json", self.plan)
            (self.path / "records").mkdir()
        self.validate()
        if os.name == "posix":
            for active in self.path.glob("active_*.json"):
                group = read(active)["pgid"]
                try:
                    os.killpg(group, 0)
                except ProcessLookupError:
                    continue
                raise RuntimeError(f"previous capture process group {group} is still alive")

    def validate(self):
        if not (self.path / "records").is_dir():
            raise RuntimeError("missing batch records directory")
        for path in (self.path / "records").glob("*.json"):
            record = read(path)
            if self.record_path(record["key"]) != path or record["status"] not in ("PASS", "FAIL", "SKIP"):
                raise RuntimeError(f"invalid committed record: {path}")
            if "baseline_key" in record:
                baseline = self.get(record["baseline_key"])
                if not baseline or baseline["status"] != "PASS":
                    raise RuntimeError("committed stress round has no valid original baseline")
            for rel, sha in record.get("artifacts", {}).items():
                artifact = self.path / rel
                if not artifact.is_file() or digest(artifact) != sha:
                    raise RuntimeError(f"missing/corrupt committed artifact: {artifact}")

    def ensure_no_failures(self):
        for path in (self.path / "records").glob("*.json"):
            record = read(path)
            if record["status"] == "FAIL":
                raise RuntimeError(f"committed hard failure {record['key']}: {record}")

    def record_path(self, key):
        return self.path / "records" / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def get(self, key):
        path = self.record_path(key)
        return read(path) if path.exists() else None

    def commit(self, key, status="PASS", artifacts=(), **values):
        path = self.record_path(key)
        if path.exists():
            raise RuntimeError(f"unit already committed: {key}")
        hashes = {str(Path(p).relative_to(self.path)): digest(p) for p in artifacts}
        save(path, dict(key=key, status=status, artifacts=hashes, time=time.time(), **values))

    def attempt(self, key):
        path = self.path / "attempts" / (key.replace("/", "_") + "_" + uuid.uuid4().hex)
        path.mkdir(parents=True)
        return path

    def history(self, devices):
        save(
            self.path / ("segment_" + uuid.uuid4().hex + ".json"),
            {"devices": devices, "started": time.time(), "pid": os.getpid(), "host": platform.node()},
        )

    def close(self):
        if self.lock:
            self.lock.close()
            self.lock = None


class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.previous = sys.stdout

    def write(self, value):
        self.previous.write(value)
        self.stream.write(value)

    def flush(self):
        self.previous.flush()
        self.stream.flush()


@contextmanager
def tee_log(path, *, stderr=False):
    with path.open("a", encoding="utf-8", buffering=1) as stream, ExitStack() as redirects:
        tee = Tee(stream)
        redirects.enter_context(redirect_stdout(tee))
        if stderr:
            redirects.enter_context(redirect_stderr(tee))
        yield


def duration(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8", errors="replace"))
    if text.count("Operator Basic Information") != 1:
        raise ValueError("expected exactly one Operator Basic Information section")
    section = text.split("Operator Basic Information", 1)[1]
    labels = re.findall(r"Task\s+Duration\s*\(us\)\s*:\s*([^\r\n]*)", section)
    if len(labels) != 1 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", labels[0].strip()):
        raise ValueError(f"expected one numeric Task Duration(us), found {labels}")
    value = float(labels[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Task Duration must be finite and positive")
    return value


def identity(device):
    text = subprocess.check_output(
        ["npu-smi", "info", "-t", "board", "-i", str(device)], text=True, stderr=subprocess.STDOUT
    )
    fields = {}
    for line in text.splitlines():
        match = re.match(r"\s*([^:]+):\s*(.*?)\s*$", line)
        if match and re.search(r"serial|\bSN\b|board id|chip id", match[1], re.IGNORECASE):
            fields[match[1].strip()] = match[2]
    if not fields or not any(re.search(r"serial|\bSN\b", key, re.IGNORECASE) for key in fields):
        raise RuntimeError("cannot establish stable physical identity from npu-smi board serial")
    return {"device": device, "board": fields, "host": platform.node()}


def small_device_identity(device):
    """Translate the pytest visible ordinal to an npu-smi board and chip for resume."""
    visibility = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    logic_id = device
    if visibility:
        visible_ids = []
        for entry in visibility.split(","):
            first, separator, last = entry.strip().partition("-")
            visible_ids.extend(range(int(first), int(last if separator else first) + 1))
        if device >= len(visible_ids):
            raise ValueError(f"Logical device {device} is outside ASCEND_RT_VISIBLE_DEVICES={visibility}")
        logic_id = visible_ids[device]

    # npu-smi board -i expects an NPU ID. A board may contain multiple chips.
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True).replace("|", " ")
    if re.search(r"NPU\s+ID\s+Slot\s+ID\s+Chip\s+ID", mapping):
        # Ascend 950 reports the unmasked runtime ID in the NPU ID column.
        row = rf"^[ \t]*({logic_id})[ \t]+\S+[ \t]+(\d+)\b"
    elif re.search(r"NPU\s+ID\s+Chip\s+ID\s+Chip\s+Logic\s+ID", mapping):
        row = rf"^[ \t]*(\d+)[ \t]+(\d+)[ \t]+{logic_id}\b"
    else:
        raise RuntimeError("cannot identify npu-smi device mapping columns")
    matches = re.findall(row, mapping, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"cannot map runtime ID {logic_id} to a unique npu-smi board and chip")
    card, chip = map(int, matches[0])
    return dict(identity(card), chip=chip)


def table(rows):
    widths = [((max(len(str(row[c])) for row in rows) // 8) + 1) * 8 for c in range(len(rows[0]))]
    return ["".join(str(v).ljust(w - 1) + "\t" for v, w in zip(row[:-1], widths[:-1])) + str(row[-1]) for row in rows]


def select_device():
    report = subprocess.check_output(
        [sys.executable, str(ROOT / "tests/tools/select_npu_device.py"), "--format=report"], text=True
    )
    print(report, flush=True)
    found = re.findall(r"Selected NPU:\s*(\d+)", report)
    if len(found) != 1:
        raise RuntimeError("cannot parse selected physical device")
    return int(found[0])
