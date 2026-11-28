"""Check the private FA4 mask and lazy loading behind block_causal_attention."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mindiesd.layers.flash_attn import _prebuilt


def test_public_import_keeps_the_fa4_backend_lazy():
    code = """
import sys
sys.path.insert(0, "tests")
from utils.utils.torch_npu_mock import mock_torch_npu
mock_torch_npu()
from mindiesd import block_causal_attention
assert callable(block_causal_attention)
assert "mindiesd.layers.flash_attn._block_causal_fa4" not in sys.modules
assert not any(m.startswith(("catlass", "mindiesd.plugin.torch")) for m in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[4], check=True)


@pytest.mark.parametrize("version,variant", [("2.9.0+cpu", "torch29"), ("2.10.0", "torch210")])
def test_backend_uses_matching_torch_runtime(monkeypatch, version, variant):
    monkeypatch.setattr(torch, "__version__", version)
    assert _prebuilt._backend_prefix() == f"mindiesd.plugin.{variant}.flash_attn.flash_attn_npu_4.catlass_dsl"


@pytest.fixture(name="adapter")
def _adapter(monkeypatch):
    prefix = _prebuilt._backend_prefix()
    tla = SimpleNamespace(jit=lambda fn: fn, Int64=int, Bool=bool)
    backend = SimpleNamespace(compute_block_sparsity=Mock(return_value=object()), flash_attn_func=Mock())
    mods = SimpleNamespace(MaskMod=lambda callback, **options: SimpleNamespace(callback=callback, **options))
    monkeypatch.setitem(sys.modules, prefix, backend)
    monkeypatch.setitem(sys.modules, prefix + ".utils", SimpleNamespace(get_tla=lambda: tla))
    monkeypatch.setitem(sys.modules, prefix + ".mods", mods)
    spec = importlib.util.spec_from_file_location(
        "mindiesd.layers.flash_attn._fa4_test", Path(_prebuilt.__file__).with_name("_block_causal_fa4.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, backend


@pytest.mark.parametrize("spans", [(), ((0, 2), (2, 4)), ((1, 4), (6, 8))])
def test_mask_matches_dense_visibility_including_tile_tail(adapter, spans):
    module, backend = adapter
    length = 9
    device = torch.device("cpu")
    blocks, aux = module._prepare(spans, length, device, 0)
    assert blocks is backend.compute_block_sparsity.return_value
    args = backend.compute_block_sparsity.call_args.args
    assert args[:6] == (128, 128, None, None, length, length)
    mask = args[6]
    assert mask.broadcast_batch and mask.broadcast_head
    for q in range(length + 2):
        for k in range(length + 2):
            expected = q < length and k < length and (q >= k or any(a <= q < b and a <= k < b for a, b in spans))
            assert (
                bool(
                    mask.callback(
                        0, 0, torch.tensor(q, dtype=torch.int32), torch.tensor(k, dtype=torch.int32), None, aux, ()
                    )
                )
                == expected
            )


def test_reuses_geometry_only_for_the_same_spans_and_stream(adapter, monkeypatch):
    module, backend = adapter
    q = torch.randn(1, 9, 2, 128)
    spans = ((1, 4),)
    backend.flash_attn_func.return_value = (q, None)
    stream = Mock(return_value=0)
    monkeypatch.setattr(torch.npu, "current_stream", stream)
    assert module.run(q, q, q, spans, 0.125) is q
    assert module.run(q, q, q, spans, 0.125) is q
    backend.compute_block_sparsity.assert_called_once()
    kwargs = backend.flash_attn_func.call_args.kwargs
    assert (
        kwargs["softmax_scale"] == 0.125
        and kwargs["block_sparse_tensors"] is backend.compute_block_sparsity.return_value
    )
    stream.return_value = 1
    module.run(q, q, q, spans, None)
    module.run(q, q, q, ((2, 4),), None)
    assert backend.compute_block_sparsity.call_count == 3
