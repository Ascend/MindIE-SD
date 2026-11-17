"""Norm native result validation using scoped device metadata on CPU storage."""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from mindiesd.layers import norm


@pytest.mark.parametrize("kind", ["add_layer_norm", "add_rms_norm", "group_norm_silu"])
@pytest.mark.parametrize("invalid", [False, True])
def test_native_norm_result_contract(kind, invalid):
    x = torch.ones(2, 8)
    residual = torch.ones_like(x)
    weight = torch.ones(8)
    bias = torch.zeros(8)
    output = torch.full_like(x, 3)
    result = (
        None
        if invalid
        else (
            (output, torch.ones(2, 1), torch.ones(2, 1), residual)
            if kind == "add_layer_norm"
            else (output, torch.ones(2, 1), residual)
        )
    )
    api = {
        "add_layer_norm": "npu_add_layer_norm",
        "add_rms_norm": "npu_add_rms_norm",
        "group_norm_silu": "npu_group_norm_silu",
    }[kind]
    with (
        patch.object(torch.Tensor, "device", new_callable=PropertyMock, return_value=SimpleNamespace(type="npu")),
        patch.object(norm.torch_npu, api, return_value=result) as native,
    ):

        def invoke():
            if kind == "add_layer_norm":
                return norm.add_layer_norm(x, residual, weight, bias)
            if kind == "add_rms_norm":
                return norm.add_rms_norm(x, residual, weight)
            return norm.group_norm_silu(x, weight, bias, 2)

        if invalid:
            with pytest.raises(RuntimeError, match="invalid result"):
                invoke()
        else:
            actual = invoke()
            assert (actual if kind == "group_norm_silu" else actual[0]) is output
        assert native.call_args.args[0] is x
    assert output.device.type == "cpu"


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("impl_mode", [0, 1, 2])
def test_fast_layernorm_native_options_and_reference(affine, impl_mode):
    x = torch.arange(16).reshape(2, 8).to(torch.float16)
    layer = torch.nn.LayerNorm(8, elementwise_affine=affine).half()
    expected = layer(x)
    with patch.object(torch.ops.mindiesd, "layernorm", return_value=(expected,), create=True) as native:
        torch.testing.assert_close(norm.fast_layernorm(layer, x, impl_mode=impl_mode), expected)
        assert native.call_args.kwargs["impl_mode"] == impl_mode
    torch.testing.assert_close(norm.fast_layernorm(layer, x, impl_mode=impl_mode, fused=False), expected)
