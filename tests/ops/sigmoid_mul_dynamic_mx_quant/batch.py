# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Keep immutable test plans and per-unit results across interrupted runs."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    with path.open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(stream.read()).hexdigest()
        )


def write_json(path, data):
    temp = path.with_suffix(".pending")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def package_root():
    return Path(importlib.util.find_spec("mindiesd").origin).parent


def prepare_vendor():
    vendor = package_root() / "ops/vendors/aie_ascendc"
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    if str(vendor) not in current:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([str(vendor), *filter(None, current)])


def fingerprint():
    package = package_root()
    files = sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh"))
    files += sorted((package / "plugin").glob("*.so"))
    files += sorted((package / "ops/vendors/aie_ascendc").rglob("SigmoidMulDynamicMxQuant*.o"))
    versions = {k: importlib.metadata.version(k) for k in ("torch", "torch-npu")}
    return {"host": platform.node(), "versions": versions, "files": {str(p): digest(p) for p in files}}


def record_path(root, key):
    return root / "records" / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def save_record(root, key, status, artifacts=(), detail=""):
    hashes = {str(path.relative_to(root)): digest(path) for path in artifacts}
    write_json(record_path(root, key), {"key": key, "status": status, "artifacts": hashes, "detail": detail})


class Batch:
    def __init__(self, kind, plan, output=None, resume=None):
        import fcntl

        if output and resume:
            raise ValueError("select --output or --resume")
        self.path = (
            Path(resume or output)
            if resume or output
            else ROOT / "log" / HERE.name / time.strftime(kind + "_%Y%m%d_%H%M%S")
        ).resolve()
        if not resume:
            self.path.mkdir(parents=True, exist_ok=False)
        self.lock = (self.path / "batch.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.plan = {"kind": kind, "parameters": plan, "environment": fingerprint()}
            self.records = self.path / "records"
            if resume:
                saved = json.loads((self.path / "plan.json").read_text(encoding="utf-8"))
                if self.plan != saved:
                    raise ValueError("resume plan, hardware, software or source differs")
            else:
                self.records.mkdir()
                write_json(self.path / "plan.json", self.plan)
        except BaseException:
            self.lock.close()
            raise
        print(f"batch={self.path}", flush=True)

    def record(self, key):
        return record_path(self.path, key)

    def complete(self, key):
        path = self.record(key)
        if not path.exists():
            return False
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["key"] != key or result["status"] == "FAIL":
            raise RuntimeError(f"prior failure or invalid record for {key}: {result}")
        for name, expected in result["artifacts"].items():
            if digest(self.path / name) != expected:
                raise RuntimeError(f"artifact changed: {name}")
        return True

    def save(self, key, status, artifacts=(), detail=""):
        save_record(self.path, key, status, artifacts, detail)

    def close(self):
        self.lock.close()
