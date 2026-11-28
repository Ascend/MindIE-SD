import importlib
import inspect
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mindiesd import block_causal_attention
from mindiesd.layers.flash_attn import _block_causal as routing
from mindiesd.utils.exception import ParametersInvalid


@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("spans", [[], [(0, 11)], [(0, 2), (2, 5), (7, 9)], [(1, 2), (10, 11)]])
def test_math_matches_independent_mask(head_first, spans):
    generator = torch.Generator().manual_seed(837)
    query, key, value = (torch.randn(2, 3, 11, 8, generator=generator) for _ in range(3))
    keep = torch.ones(11, 11, dtype=torch.bool).tril()
    for start, end in spans:
        keep[start:end, start:end] = True
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=keep, scale=0.25)
    if not head_first:
        query, key, value, expected = (tensor.transpose(1, 2) for tensor in (query, key, value, expected))
    result = block_causal_attention(query, key, value, scale=0.25, fused=False, head_first=head_first, spans=spans)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("spans", [False, "", [1], [(0,)], [(True, 2)], [(0, 0)], [(3, 5)], [(1, 3), (2, 4)]])
def test_invalid_spans(spans):
    query = torch.ones(1, 4, 1, 8)
    with pytest.raises(ParametersInvalid, match="span"):
        block_causal_attention(query, query, query, spans=spans, fused=False)


def test_reject_cached_prefix():
    query = torch.ones(1, 4, 1, 8)
    with pytest.raises(ParametersInvalid, match="self-prefill"):
        block_causal_attention(query[:, -1:], query, query, spans=[])


@pytest.mark.parametrize("length", [27999, 28000, 28001, 70000])
@pytest.mark.parametrize("pieces", [7, 8, 9])
def test_geometry_thresholds(length, pieces):
    # Leading gap, separated spans, and an optional trailing gap.
    spans = [(2, 3), (4, 5), (6, 7), (8, length)]
    if pieces == 7:
        spans[0] = (0, 3)
    elif pieces == 9:
        spans[-1] = (8, 10)
    assert len(list(routing._segments(spans, length))) == pieces
    assert routing._candidate_geometry((2, length, 17, 128), spans) == (length >= 28000 and pieces >= 8)


@pytest.mark.parametrize(
    "mode,length,ready,result",
    [
        ("piecewise", 27999, True, "piecewise"),
        ("piecewise", 28000, True, "piecewise"),
        ("auto", 27999, True, "piecewise"),
        ("auto", 28000, True, "flex"),
        ("auto", 28000, False, "piecewise"),
        ("flex", 128, True, "flex"),
        ("flex", 128, False, "error"),
        ("piecewise", 70000, True, "piecewise"),
    ],
)
def test_backend_policy(mode, length, ready, result, monkeypatch, recwarn):
    query = torch.empty(1, length, 2, 128, dtype=torch.bfloat16, device="meta")
    spans = ((2, 3), (4, 5), (6, 7), (8, length))
    monkeypatch.setattr(routing, "_flex_unavailable_reason", lambda _: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda _: 0), raising=False)
    store = Mock(return_value=nullcontext()) if ready else Mock(side_effect=routing.PrebuiltUnavailable("missing"))
    monkeypatch.setattr(routing, "packaged_execution", store)
    flex = Mock(return_value="flex")
    monkeypatch.setattr(routing, "_run_flex", flex)
    piecewise = Mock(return_value="piecewise")
    monkeypatch.setattr(routing, "_piecewise", piecewise)
    if result == "error":
        with pytest.raises(routing.PrebuiltUnavailable, match="missing"):
            routing._dispatch_npu(query, query, query, spans, 1.0, mode)
        piecewise.assert_not_called()
    else:
        assert routing._dispatch_npu(query, query, query, spans, 1.0, mode) == result
    if mode == "auto" and not ready:
        assert "missing" in str(recwarn.pop(RuntimeWarning).message)
    assert not recwarn.list
    if mode == "piecewise" or (mode == "auto" and length < 28000):
        store.assert_not_called()
        flex.assert_not_called()


@pytest.mark.parametrize("mode", ["auto", "flex"])
def test_flex_import_failure(mode, monkeypatch):
    query = torch.empty(1, 28000, 2, 128, dtype=torch.bfloat16, device="meta")
    spans = ((2, 3), (4, 5), (6, 7), (8, 28000))
    monkeypatch.setattr(routing, "_flex_unavailable_reason", lambda _: None)
    monkeypatch.setattr(routing, "packaged_execution", lambda _: nullcontext())
    monkeypatch.setitem(sys.modules, f"{routing.__package__}._block_causal_fa4", None)
    piecewise = Mock(return_value="piecewise")
    monkeypatch.setattr(routing, "_piecewise", piecewise)
    if mode == "auto":
        with pytest.warns(RuntimeWarning, match="falling back to piecewise.*slower.*Could not load the FA4 backend"):
            assert routing._dispatch_npu(query, query, query, spans, 1.0, mode) == "piecewise"
    else:
        with pytest.raises(ImportError, match="Could not load the FA4 backend"):
            routing._dispatch_npu(query, query, query, spans, 1.0, mode)
        piecewise.assert_not_called()


@pytest.mark.parametrize("mode", ["auto", "flex"])
@pytest.mark.parametrize("error", [RuntimeError, ImportError])
def test_runtime_errors_surface(mode, error, monkeypatch):
    query = torch.empty(1, 28000, 2, 128, dtype=torch.bfloat16, device="meta")
    spans = ((2, 3), (4, 5), (6, 7), (8, 10))
    monkeypatch.setattr(routing, "_flex_unavailable_reason", lambda _: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda _: 0), raising=False)
    monkeypatch.setattr(routing, "packaged_execution", lambda _: nullcontext())
    monkeypatch.setattr(routing, "_run_flex", Mock(side_effect=error("execution error")))
    monkeypatch.setattr(routing, "_piecewise", Mock(side_effect=AssertionError))
    with pytest.raises(error, match="execution error"):
        routing._dispatch_npu(query, query, query, spans, 1.0, mode)


@pytest.mark.parametrize(
    "a5,dtype,dim,capture,grad,reason",
    [
        (False, torch.bfloat16, 128, False, False, "A5"),
        (True, torch.float32, 128, False, False, "BF16 or FP16"),
        (True, torch.bfloat16, 64, False, False, "head dimension"),
        (True, torch.bfloat16, 128, True, False, "graph capture"),
        (True, torch.bfloat16, 128, False, True, "inference"),
        (True, torch.float16, 128, False, False, None),
    ],
)
def test_flex_capabilities(a5, dtype, dim, capture, grad, reason, monkeypatch):
    monkeypatch.setattr(routing, "is_a5_device", lambda: a5)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: capture), raising=False)
    q = torch.empty(1, 8, 2, dim, device="meta", dtype=dtype)
    with torch.set_grad_enabled(grad):
        if reason:
            with pytest.raises(ParametersInvalid, match=reason):
                routing._dispatch_npu(q, q, q, (), 1.0, "flex")
        else:
            assert routing._flex_unavailable_reason(q) is None


def test_public_block_causal_signature():
    params = inspect.signature(block_causal_attention).parameters
    assert tuple(params) == ("query", "key", "value", "spans", "backend", "scale", "fused", "head_first")
    assert params["spans"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["spans"].default is inspect.Parameter.empty
    assert params["backend"].default == "piecewise"
    for path in ("mindiesd", "mindiesd.layers", "mindiesd.layers.flash_attn"):
        package = importlib.import_module(path)
        assert package.block_causal_attention is block_causal_attention
        assert "block_causal_attention" in package.__all__


@pytest.mark.parametrize("choice", [None, "", "unknown"])
def test_invalid_backend(choice):
    q = torch.empty(1, 8, 2, 128)
    with pytest.raises(ParametersInvalid, match="backend"):
        block_causal_attention(q, q, q, spans=[], backend=choice)


@pytest.mark.parametrize("choice", ["auto", "flex"])
def test_fused_backend_requires_fused(choice):
    q = torch.empty(1, 8, 2, 128)
    with pytest.raises(ParametersInvalid, match="requires fused=True"):
        block_causal_attention(q, q, q, spans=[], backend=choice, fused=False)


def test_piecewise_uses_prefix_kv_and_bottom_right_mask(monkeypatch):
    spans = ((2, 5), (6, 8))
    query, key, value = (torch.randn(1, 13, 2, 8) for _ in range(3))
    calls = []
    sentinel = object()
    mask = Mock(return_value=sentinel)
    monkeypatch.setattr(routing, "_compressed_mask", mask)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=lambda: False, current_stream=lambda _: 0),
        raising=False,
    )

    def emulate(query, key, value, *, head_num, input_layout, atten_mask, scale, keep_prob, sparse_mode):
        assert input_layout == "BSND" and head_num == 2 and keep_prob == 1.0
        assert atten_mask is (sentinel if sparse_mode == 3 else None)
        rows, cols = query.size(1), key.size(1)
        calls.append((rows, cols, sparse_mode))
        visible = torch.ones(rows, cols, dtype=torch.bool).tril(cols - rows) if sparse_mode else None
        result = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=visible, scale=scale
        )
        return (result.transpose(1, 2),)

    monkeypatch.setattr(routing.torch_npu, "npu_fusion_attention", emulate)
    actual = routing._piecewise(query, key, value, spans, 0.25, True)
    expected = block_causal_attention(query, key, value, spans=spans, scale=0.25, fused=False)
    torch.testing.assert_close(actual, expected)
    assert calls == [(2, 2, 3), (3, 5, 0), (1, 6, 3), (2, 8, 0), (5, 13, 3)]
    mask.assert_called_once()
