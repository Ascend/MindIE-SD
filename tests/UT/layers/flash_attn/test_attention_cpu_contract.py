"""Attention routing exercised through the actual registry to native API boundary."""

import importlib
from unittest.mock import patch

import pytest
import torch
import torch_npu

from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import NPUDevice

forward = importlib.import_module("mindiesd.layers.flash_attn.attention_forward")
functions = importlib.import_module("mindiesd.layers.flash_attn.attention_func")


@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("layout", ["BNSD", "BSND", "BSH"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("mode", ["manual", "static"])
def test_native_fas_dispatch_retains_layout(head_first, layout, masked, mode):
    x = torch.arange(48).reshape(1, 3, 2, 8).float()
    if head_first:
        x = x.transpose(1, 2)
    mask = torch.ones(3, 3, dtype=torch.bool) if masked else None
    with (
        patch.object(functions, "get_npu_device", return_value=NPUDevice.A5),
        patch.object(functions, "is_a5_device", return_value=True),
        patch.object(forward, "is_a5_device", return_value=True),
        patch.dict(functions.ATTN_DICT, {}, clear=True),
        patch.object(torch_npu, "npu_fusion_attention", side_effect=lambda q, *args, **kwargs: (q + 7,)) as native,
    ):
        result = forward.attention_forward(
            x,
            x,
            x,
            mask,
            head_first=head_first,
            opt_mode=mode,
            op_type="fused_attn_score",
            layout=layout,
            use_env_override=False,
        )
        torch.testing.assert_close(result, x + 7)
        assert native.call_args.kwargs["input_layout"] == (layout if mode == "manual" else "BNSD")
        assert native.call_args.kwargs["scale"] == 8**-0.5
        if masked:
            torch.testing.assert_close(native.call_args.kwargs["atten_mask"], ~mask)


@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("mask_rank", [None, 2, 4])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_attention_math_matches_sdpa(head_first, mask_rank, mask_dtype):
    torch.manual_seed(16)
    q, k, v = (torch.randn(1, 2, 3, 8) for _ in range(3))
    mask = None if mask_rank is None else torch.ones((3, 3) if mask_rank == 2 else (1, 2, 3, 3), dtype=torch.bool)
    if mask is not None:
        mask[..., 0] = False
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.3)
    # The frontend uses keep-mask semantics for floating 0/1 masks as well.
    if mask is not None:
        mask = mask.to(mask_dtype)
    inputs = (q, k, v) if head_first else tuple(t.transpose(1, 2) for t in (q, k, v))
    actual = forward.attention_forward(*inputs, attn_mask=mask, scale=0.3, fused=False, head_first=head_first)
    torch.testing.assert_close(actual, expected if head_first else expected.transpose(1, 2))


@pytest.mark.parametrize(
    "key,value",
    [
        ("query", None),
        ("key", None),
        ("value", None),
        ("query", torch.ones(3)),
        ("key", torch.ones(3)),
        ("value", torch.ones(3)),
        ("fused", 1),
        ("attn_mask", 1),
        ("scale", 1),
        ("opt_mode", "invalid"),
        ("op_type", "invalid"),
    ],
)
def test_attention_rejects_invalid_parameters(key, value):
    x = torch.ones(1, 3, 2, 8)
    inputs = dict(query=x, key=x, value=x, opt_mode="manual", use_env_override=False)
    inputs[key] = value
    with pytest.raises(ParametersInvalid):
        forward.attention_forward(**inputs)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (1, 2, 3, 4), (2, 2, 3, 3), (1, 3, 3, 3)])
def test_math_rejects_invalid_mask(shape):
    x = torch.ones(1, 2, 3, 8)
    with pytest.raises(ParametersInvalid):
        functions.attention_math(x, x, x, torch.ones(shape), 0.3, head_first=True)


@pytest.mark.parametrize(
    "platform,operator,layout,dtype",
    [
        (NPUDevice.UNDEFINED, "fused_attn_score", "BNSD", torch.float32),
        (NPUDevice.A2, "missing", "BNSD", torch.float32),
        (NPUDevice.A3, "fused_attn_score", "invalid", torch.float32),
        (NPUDevice.A5, "fused_attn_score", "BNSD", torch.int8),
        (NPUDevice.Duo, "missing", "BNSD", torch.float32),
    ],
)
def test_registry_rejects_platform_operator_layout_and_dtype(platform, operator, layout, dtype):
    param = functions.AttentionParam(1, 2, 8, 3, 3, dtype, True)
    with patch.object(functions, "get_npu_device", return_value=platform):
        with pytest.raises(ParametersInvalid):
            functions.get_attention_function(param, operator, layout)


def test_native_benchmark_records_failure_and_synchronizes_success():
    x = torch.ones(1, 2, 3, 8)
    param = functions.AttentionParam(1, 2, 8, 3, 3, torch.float32, True)
    selected = functions.get_all_func_forward_time
    calls = []

    def native_success(*args):
        calls.append(args)
        return x

    def native_failure(*args):
        raise RuntimeError("injected native launch failure")

    with patch.object(torch.npu, "synchronize") as sync, patch.object(functions.time, "time", side_effect=[1.0, 2.0]):
        timings = selected([["bad", "BNSD", native_failure], ["good", "BNSD", native_success]], x, x, x, param)
    assert len(timings) == 1 and timings[0][:2] == ["good", "BNSD"]
    assert timings[0][3] == 1 / functions.TEST_COUNT
    assert len(calls) == functions.WARM_UP_COUNT + functions.TEST_COUNT
    assert sync.call_count == 2
