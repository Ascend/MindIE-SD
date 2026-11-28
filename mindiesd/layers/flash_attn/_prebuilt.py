"""Read-only spans executor selection for the packaged CATLASS 2.2 ABI."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from threading import RLock


class PrebuiltUnavailable(RuntimeError):
    """No packaged executor is compatible with this request."""


_ACTIVE = ContextVar("mindiesd_spans_store", default=None)
_LOCK = RLock()
_SEED = Path(__file__).resolve().parents[2] / "_block_causal_seed"


def _backend_prefix():
    import torch
    from ..register_ops import _get_torch_major_minor

    variant = "torch" + _get_torch_major_minor(torch.__version__).replace(".", "")
    return f"mindiesd.plugin.{variant}.flash_attn.flash_attn_npu_4.catlass_dsl"


def _cache_root():
    if os.environ.get("CATLASS_DSL_CACHE_DIR"):
        return Path(os.environ["CATLASS_DSL_CACHE_DIR"]).expanduser().resolve() / "bc"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "catlass" / "bc").expanduser().resolve()


def _stage_bc(compiler, seed, key, root):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".mindiesd-bc.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (root / key).exists():
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                stage = Path(tmp) / key
                shutil.copytree(seed / "bc" / key, stage)
                stage.rename(root / key)
        compiler.check_bc()


def _namespace(persistent, catlass, roots):
    # Reproduce the pinned FA4 v1 namespace with producer paths for relocated
    # installations. Hash the actual local sources, binaries and BC contents.
    from catlass import execution
    from catlass.bc_compile import get_bc_paths
    from catlass.compiler_bridge import resolve_bridge_extension_path
    import sys

    def producer_path(path):
        for current, original in roots:
            if path.is_relative_to(current):
                return str(original / path.relative_to(current))
        return str(path)

    sources = []
    for directory in (Path(persistent.__file__).parent, Path(catlass.__file__).parent):
        for path in sorted(directory.rglob("*.py")):
            sources.append((directory.name, path.relative_to(directory).as_posix(), persistent.file_digest(path)))
    ccec, includes = execution._ascendc_compiler_inputs()
    tools = [resolve_bridge_extension_path(), execution._find_hivmc_a5(), ccec]
    tools.extend(Path(catlass.__file__).parent.rglob("*.so"))
    for variant in ("normal", "print"):
        tools.extend(Path(p) for p in get_bc_paths("mix", variant=variant).split(","))
    binaries = [(producer_path(Path(p)), persistent.file_digest(p)) for p in set(tools) if p is not None]
    binaries.sort(key=lambda item: Path(item[0]))
    versions = {name: version(name) for name in ("ascend-catlass-dsl", "torch", "torch_npu")}
    return persistent._json_digest(
        (
            persistent.ABI_VERSION,
            sys.version_info[:3],
            versions,
            sources,
            binaries,
            tuple(producer_path(Path(p)) for p in includes),
            "--npu-arch 3510",
        )
    )


def executor_key(key):
    """Return the packaged mask or equal-head forward specialization."""
    family = key[0]
    if family == "mask":
        return key
    if (
        family == "fwd"
        and len(key) == 11
        and key[2:4] in (("torch.bfloat16", "3510"), ("torch.float16", "3510"))
        and 0 < key[4] == key[5]
    ):
        return (*key[:4], "dynamic-spans-heads-v1", *key[6:])
    return None


class _Store:
    def __init__(self, persistent, directory, dtypes):
        self.dtypes = dtypes
        self.persistent = persistent
        self.directory = directory
        self.lock = RLock()
        self.executors = {}

    @lru_cache(maxsize=512)
    def entry(self, key):
        candidate = executor_key(key)
        if candidate is None or not (digest := self.persistent.key_digest(candidate)):
            return None
        path = self.directory / key[0] / digest
        return path if (path / "manifest.json").is_file() else None

    def lookup(self, key):
        with self.lock:
            path = self.entry(key)
            if path is None:
                raise PrebuiltUnavailable(f"No packaged Flex spans {key[0]} executor matches this specialization.")
            # A digest is portable between NPUs. Loaded device handles are not.
            identity = (key[1], path)
            if identity not in self.executors:
                try:
                    self.executors[identity] = self.persistent.load_function(path)
                except (FileNotFoundError, ValueError) as exc:
                    raise PrebuiltUnavailable(f"Packaged Flex spans {key[0]} executor is unusable: {exc}") from exc
            return self.executors[identity]


def _bind_cache(cache):
    normal_lookup = cache.get_or_compile

    def get_or_compile(key, compile_fn):
        store = _ACTIVE.get()
        if store is None:
            return normal_lookup(key, compile_fn)
        return store.lookup(key)

    cache.get_or_compile = get_or_compile


@lru_cache(maxsize=1)
def _variants():
    index = _SEED / "variants.json"
    return json.loads(index.read_text()) if index.is_file() else ()


@lru_cache(maxsize=4)
def _dtype_packaged(dtype):
    return any(dtype in row["dtypes"] for row in _variants())


def _open_store():
    import catlass
    from catlass import bc_compile

    versions = [platform.python_version(), version("torch"), version("torch_npu"), version("ascend-catlass-dsl")]
    records = [row for row in _variants() if row["selector"][:4] == versions]
    if not records:
        raise PrebuiltUnavailable("No packaged Flex spans executor matches Python/Torch/torch_npu/CATLASS.")
    key = bc_compile.compute_cache_key(bc_compile._resolve_toolchain().ccec)
    record = next((row for row in records if row["selector"][4] == key), None)
    if record is None:
        raise PrebuiltUnavailable("No packaged Flex spans executor matches the installed CANN toolchain.")
    seed = _SEED / record["id"]
    root = _cache_root()
    _stage_bc(bc_compile, seed, key, root)
    origin = json.loads((seed / "origin.json").read_text())
    prefix = _backend_prefix()
    importlib.import_module(f"{prefix}.utils").get_tla()
    persistent = importlib.import_module(f"{prefix}.persistent")
    roots = [(root, Path(origin["bc"])), (Path(catlass.__file__).resolve().parent, Path(origin["catlass"]))]
    namespace = _namespace(persistent, catlass, roots)
    directory = seed / "executors" / namespace / "attention"
    if not directory.is_dir():
        raise PrebuiltUnavailable("Packaged Flex spans executor source or toolchain fingerprint differs.")
    runtime = importlib.import_module(f"{prefix}.runtime")
    _bind_cache(runtime.compile_cache)
    return _Store(persistent, directory, record["dtypes"])


@lru_cache(maxsize=1)
def _available_store():
    try:
        return _open_store(), None
    except (ImportError, OSError, RuntimeError) as exc:
        # Initialization checks the package and toolchain. Device execution is
        # outside this handler so numerical/runtime failures reach the caller.
        return None, f"Could not initialize packaged Flex spans executors: {exc}"


@contextmanager
def packaged_execution(dtype):
    if not _dtype_packaged(str(dtype)):
        raise PrebuiltUnavailable(f"This MindIE wheel has no packaged Flex spans executor for {dtype}.")
    with _LOCK:
        store, reason = _available_store()
    if store is None:
        raise PrebuiltUnavailable(reason)
    if str(dtype) not in store.dtypes:
        raise PrebuiltUnavailable(f"No packaged Flex spans executor for {dtype} in this environment.")
    token = _ACTIVE.set(store)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
