"""Check strict loading without NPU, CATLASS or compiler dependencies."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(name="loader")
def _loader():
    source = Path(__file__).resolve().parents[4] / "mindiesd/layers/flash_attn/_prebuilt.py"
    spec = importlib.util.spec_from_file_location("spans_prebuilt_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_scope_and_failure_restore_native_policy(loader, monkeypatch):
    compile_fn = Mock(return_value="compiled")
    native = SimpleNamespace(get_or_compile=lambda key, fn: fn())
    loader._bind_cache(native)
    store = SimpleNamespace(lookup=Mock(return_value="packaged"), dtypes=["bf16"])
    monkeypatch.setattr(loader, "_available_store", lambda: (store, None))
    monkeypatch.setattr(loader, "_dtype_packaged", lambda dtype: True)
    with loader.packaged_execution("bf16"):
        assert native.get_or_compile(("fwd",), compile_fn) == "packaged"
        compile_fn.assert_not_called()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(native.get_or_compile, ("fwd",), compile_fn).result() == "compiled"
        with pytest.raises(loader.PrebuiltUnavailable, match="fp16"):
            with loader.packaged_execution("fp16"):
                pytest.fail()
        assert native.get_or_compile(("fwd",), compile_fn) == "packaged"
    assert native.get_or_compile(("fwd",), compile_fn) == "compiled"
    assert compile_fn.call_count == 2


@pytest.mark.parametrize("dtype", ["torch.bfloat16", "torch.float16"])
def test_dynamic_head_keys_reuse_only_proven_families(loader, tmp_path, dtype):
    stored = tmp_path / "fwd" / dtype
    stored.mkdir(parents=True)
    (stored / "manifest.json").touch()
    persistent = SimpleNamespace(key_digest=lambda key: key[2], load_function=Mock(side_effect=lambda _: object()))
    store = loader._Store(persistent, tmp_path, [dtype])

    def key(qh, kvh, device="npu:0", precision=dtype):
        return ("fwd", device, precision, "3510", qh, kvh, (), (), (), False, False)

    first = store.lookup(key(4, 4))
    assert store.lookup(key(16, 16)) is first
    assert store.lookup(key(4, 4, "npu:1")) is not first
    assert persistent.load_function.call_count == 2
    other = "torch.float16" if dtype == "torch.bfloat16" else "torch.bfloat16"
    for unmatched in (key(4, 2), key(5, 2), key(4, 4, precision=other)):
        with pytest.raises(loader.PrebuiltUnavailable, match="fwd"):
            store.lookup(unmatched)


def test_kernel_load_error_preserves_native_diagnostic(loader, tmp_path):
    path = tmp_path / "mask" / "entry"
    path.mkdir(parents=True)
    (path / "manifest.json").touch()
    persistent = SimpleNamespace(key_digest=lambda _: "entry", load_function=Mock(side_effect=ValueError("corrupt")))
    store = loader._Store(persistent, tmp_path, ["torch.bfloat16"])
    with pytest.raises(loader.PrebuiltUnavailable, match="corrupt"):
        store.lookup(("mask", "npu:0"))


def test_missing_precision_skips_runtime_setup(loader, monkeypatch):
    monkeypatch.setattr(loader, "_variants", lambda: [{"dtypes": ["torch.bfloat16"]}])
    open_store = Mock(side_effect=AssertionError)
    monkeypatch.setattr(loader, "_available_store", open_store)
    with pytest.raises(loader.PrebuiltUnavailable, match="float16"):
        with loader.packaged_execution("torch.float16"):
            pytest.fail()
    open_store.assert_not_called()


@pytest.mark.parametrize(
    "python_version,cann,has_executors,error",
    [
        ("3.12", "cann", True, None),
        ("3.11", "cann", True, "Python/Torch"),
        ("3.12", "older-cann", True, "CANN toolchain"),
        ("3.12", "cann", False, "fingerprint"),
    ],
)
def test_open_packaged_release(loader, tmp_path, monkeypatch, python_version, cann, has_executors, error):
    seed = tmp_path / "release"
    seed.mkdir()
    dtype = "torch.bfloat16"
    record = {"id": seed.name, "selector": [python_version, "2.10", "2.10", "2.2", cann], "dtypes": [dtype]}
    (tmp_path / "variants.json").write_text(json.dumps([record]), encoding="utf8")
    (seed / "origin.json").write_text(json.dumps({"bc": "producer/bc", "catlass": "producer/catlass"}), encoding="utf8")
    directory = seed / "executors" / "signature" / "attention"
    if has_executors:
        directory.mkdir(parents=True)
    compiler = Mock()
    compiler.compute_cache_key.return_value = "cann"
    monkeypatch.setitem(sys.modules, "catlass", SimpleNamespace(bc_compile=compiler, __file__=str(tmp_path / "sdk.py")))
    monkeypatch.setattr(loader.platform, "python_version", lambda: "3.12")
    monkeypatch.setattr(
        loader, "version", {"torch": "2.10", "torch_npu": "2.10", "ascend-catlass-dsl": "2.2"}.__getitem__
    )
    monkeypatch.setattr(loader, "_SEED", tmp_path)
    monkeypatch.setenv("CATLASS_DSL_CACHE_DIR", str(tmp_path / "cache"))
    stage = Mock()
    monkeypatch.setattr(loader, "_stage_bc", stage)
    monkeypatch.setattr(loader, "_backend_prefix", lambda: "fa4")
    for module in ("utils", "persistent", "runtime"):
        monkeypatch.setitem(sys.modules, f"fa4.{module}", Mock())
    monkeypatch.setattr(loader, "_namespace", lambda *_: "signature")

    if error:
        with pytest.raises(loader.PrebuiltUnavailable, match=error), loader.packaged_execution(dtype):
            pytest.fail("An incompatible release was accepted")
        if error != "fingerprint":
            stage.assert_not_called()
    else:
        with loader.packaged_execution(dtype):
            assert loader._ACTIVE.get().directory == directory
        stage.assert_called_once_with(compiler, seed, "cann", tmp_path / "cache/bc")


def test_namespace_uses_producer_paths_and_local_file_contents(loader, tmp_path, monkeypatch):
    sdk = tmp_path / "installed"
    sdk.mkdir()
    source = sdk / "kernel.py"
    source.write_bytes(b"source")
    bridge = sdk / "bridge.so"
    bridge.write_bytes(b"bridge")
    ccec = tmp_path / "ccec"
    ccec.write_bytes(b"compiler")
    fa4 = tmp_path / "fa4/persistent.py"
    fa4.parent.mkdir()
    fa4.write_bytes(b"adapter")
    for variant in ("normal", "print"):
        (sdk / f"{variant}.bc").write_bytes(variant.encode())
    execution = SimpleNamespace(_ascendc_compiler_inputs=lambda: (ccec, [sdk]), _find_hivmc_a5=lambda: None)
    catlass = SimpleNamespace(__file__=str(source), execution=execution)
    monkeypatch.setitem(sys.modules, "catlass", catlass)
    monkeypatch.setitem(
        sys.modules,
        "catlass.bc_compile",
        SimpleNamespace(get_bc_paths=lambda family, variant: str(sdk / f"{variant}.bc")),
    )
    monkeypatch.setitem(
        sys.modules, "catlass.compiler_bridge", SimpleNamespace(resolve_bridge_extension_path=lambda: bridge)
    )
    monkeypatch.setattr(loader, "version", lambda name: name)
    persistent = SimpleNamespace(
        __file__=str(fa4), ABI_VERSION=1, file_digest=Path.read_bytes, _json_digest=lambda inputs: inputs
    )
    producer = tmp_path / "producer"
    fingerprint = loader._namespace(persistent, catlass, [(sdk, producer)])
    _, _, _, sources, binaries, includes, _ = fingerprint
    assert sources == [("fa4", fa4.name, b"adapter"), (sdk.name, source.name, b"source")]
    assert dict(binaries) == {
        str(producer / bridge.name): b"bridge",
        str(ccec): b"compiler",
        str(producer / "normal.bc"): b"normal",
        str(producer / "print.bc"): b"print",
    }
    assert includes == (str(producer),)
    bridge.write_bytes(b"changed")
    assert loader._namespace(persistent, catlass, [(sdk, producer)]) != fingerprint


def test_stage_bc_reuses_cached_toolchain(loader, monkeypatch, tmp_path):
    if sys.platform == "win32":
        monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=2, flock=Mock()))
    monkeypatch.delenv("CATLASS_DSL_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = loader._cache_root()
    assert root == tmp_path / "cache/catlass/bc"
    seed = tmp_path / "seed"
    binary = seed / "bc/toolchain/kernel.bc"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"packaged bitcode")
    compiler = Mock()
    loader._stage_bc(compiler, seed, "toolchain", root)
    binary.unlink()
    loader._stage_bc(compiler, seed, "toolchain", root)
    assert (root / "toolchain/kernel.bc").read_bytes() == b"packaged bitcode"
    assert compiler.check_bc.call_count == 2
