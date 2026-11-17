"""Crash-safe immutable plans and per-unit records for this operator's tests."""

import hashlib
from contextlib import contextmanager, redirect_stderr, redirect_stdout, ExitStack
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
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
    payload = dict(value=value, sha256=hashlib.sha256(canonical(value)).hexdigest())
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
    paths += list((ROOT / "mindiesd").rglob("*.so"))
    paths += list((ROOT / "mindiesd/ops").rglob("*.json"))
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
    return dict(
        python=sys.version,
        machine=platform.machine(),
        versions=versions,
        files={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths)) if p.is_file()},
    )


class Batch:
    def __init__(self, kind, plan=None, resume=None, output=None):
        self.lock = None
        if resume:
            self.path = Path(resume).resolve()
            if not self.path.is_dir():
                raise ValueError("resume batch does not exist")
        else:
            self.path = (
                Path(output).resolve() if output else ROOT / "log" / OP / (kind + "_" + time.strftime("%Y%m%d_%H%M%S"))
            )
            self.path.mkdir(parents=True, exist_ok=False)
        try:
            import fcntl

            self.lock = (self.path / "executor.lock").open("a+b")
            fcntl.lockf(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # flock survives fork and inherited descriptors, protecting orphan workers.
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, OSError) as exc:
            self.close()
            raise RuntimeError("requires Linux and exclusive batch access") from exc
        if resume:
            self.plan = read(self.path / "plan.json")
            if self.plan["kind"] != kind or self.plan["environment"] != environment():
                raise RuntimeError("resume kind/code/software/artifact fingerprint mismatch")
            if plan is not None and plan != self.plan["parameters"]:
                raise ValueError("resume parameters conflict with immutable plan")
        else:
            self.plan = dict(
                kind=kind,
                parameters=plan,
                environment=environment(),
                protocol=dict(
                    cosine_min=0.99,
                    norm_ratio=[0.9, 1.1],
                    matched_ratio_min=0.99,
                    bf16_atol_rtol=2**-7,
                    fp16_atol_rtol=2**-10,
                    fallback=0.1,
                    max_abs="max(fallback,32*ULP(golden_at_max_error,actual_output_dtype))",
                    small_mixed="hard",
                    production_mixed="soft",
                    stress="bitwise",
                    input="INT8 ND; FP32 scale and pertoken_scale",
                    output="BF16 production",
                ),
            )
            save(self.path / "plan.json", self.plan)
            (self.path / "records").mkdir()
        self.validate()

    def validate(self):
        for path in (self.path / "records").glob("*.json"):
            record = read(path)
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
            dict(devices=devices, started=time.time(), pid=os.getpid(), host=platform.node()),
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
