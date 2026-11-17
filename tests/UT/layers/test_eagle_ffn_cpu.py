"""CPU FFN reference and A5 native dispatch contracts."""

import importlib
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional

from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import NPUDevice

ffn = importlib.import_module("mindiesd.layers.eagle_ffn_linear")


@pytest.mark.parametrize("activation", ["GELU", "silu", "swiglu"])
@pytest.mark.parametrize("platform,fused", [(NPUDevice.A2, True), (NPUDevice.A5, False), (NPUDevice.A5, True)])
@pytest.mark.parametrize("bias", [False, True])
def test_ffn_chain_and_platform_dispatch(activation, platform, fused, bias):
    torch.manual_seed(11)
    x = torch.randn(2, 3, 4)
    w1 = torch.randn(12 if activation == "swiglu" else 6, 4)
    w2 = torch.randn(8, 6)
    b1 = torch.randn(w1.shape[0]) if bias else None
    b2 = torch.randn(8) if bias else None
    up = functional.linear(x, w1, b1)
    if activation == "swiglu":
        gate, value = up.chunk(2, dim=-1)
        middle = functional.silu(gate) * value
    else:
        middle = functional.gelu(up) if activation == "GELU" else functional.silu(up)
    expected = functional.linear(middle, w2, b2)
    with (
        patch.object(ffn, "get_npu_device", return_value=platform),
        patch.object(torch.ops.mindiesd, "eagle_ffn_linear", return_value=expected, create=True) as native,
    ):
        actual = ffn.eagle_ffn_linear(x, w1, w2, b1, b2, activation, fused)
        torch.testing.assert_close(actual, expected)
        if platform == NPUDevice.A5 and fused:
            assert native.call_args.args[-2:] == (activation.lower(), 0)
            assert native.call_args.args[0] is x
        else:
            native.assert_not_called()


@pytest.mark.parametrize("activation", [None, 1, "relu"])
def test_invalid_activation_rejected_before_platform_query(activation):
    with patch.object(ffn, "get_npu_device") as platform:
        with pytest.raises(ParametersInvalid):
            ffn.eagle_ffn_linear(torch.ones(2, 4), torch.ones(8, 4), torch.ones(4, 8), activation=activation)
        platform.assert_not_called()
