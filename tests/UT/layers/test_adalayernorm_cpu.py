"""AdaLayerNorm validation, platform dispatch and tokenwise layout contracts."""

import importlib
from unittest.mock import patch

import pytest
import torch

from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import NPUDevice

adaln = importlib.import_module("mindiesd.layers.adalayernorm")


@pytest.mark.parametrize("platform", [NPUDevice.A2, NPUDevice.A5])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize(
    "scale_shape,shift_shape",
    [((2, 8), (2, 8)), ((2, 1, 8), (2, 1, 8)), ((2, 3, 8), (2, 8)), ((2, 1, 8), (2, 3, 8)), ((2, 3, 8), (2, 3, 8))],
)
def test_adaln_platform_and_tokenwise_contract(platform, affine, scale_shape, shift_shape):
    torch.manual_seed(89)
    norm = torch.nn.LayerNorm(8, elementwise_affine=affine)
    x = torch.randn(2, 3, 8)
    scale, shift = torch.randn(scale_shape), torch.randn(shift_shape)
    expected = norm(x) * (1 + (scale[:, None] if scale.ndim == 2 else scale))
    expected += shift[:, None] if shift.ndim == 2 else shift

    def device_op(**kwargs):
        assert kwargs["x"].device.type == "cpu"
        local_scale, local_shift = kwargs["scale"], kwargs["shift"]
        if local_scale.ndim == 2:
            local_scale = local_scale[:, None]
        if local_shift.ndim == 2:
            local_shift = local_shift[:, None]
        result = torch.nn.functional.layer_norm(kwargs["x"], (8,), kwargs["weight"], kwargs["bias"], kwargs["epsilon"])
        result = result * (1 + local_scale) + local_shift
        return (result,) if platform == NPUDevice.A5 else result

    api = "adaln_v2" if platform == NPUDevice.A5 else "adaln"
    with patch.object(adaln, "npu_device", platform), patch.object(adaln.ops, api, side_effect=device_op) as launch:
        actual = adaln.layernorm_scale_shift(norm, x, scale, shift)
        torch.testing.assert_close(actual, expected)
        tokenwise = scale_shape == (2, 3, 8) or shift_shape == (2, 3, 8)
        assert launch.call_args.kwargs["x"].shape == ((6, 1, 8) if tokenwise else x.shape)
        assert (launch.call_args.kwargs["weight"] is not None) == affine
    torch.testing.assert_close(adaln.layernorm_scale_shift(norm, x, scale, shift, fused=False), expected)


@pytest.mark.parametrize(
    "key,value",
    [
        ("layernorm", None),
        ("x", None),
        ("scale", None),
        ("shift", None),
        ("fused", 1),
        ("x", torch.zeros(2, 8)),
        ("scale", torch.zeros(8)),
        ("shift", torch.zeros(8)),
        ("scale", torch.zeros(1, 8)),
        ("shift", torch.zeros(1, 8)),
        ("scale", torch.zeros(2, 2, 8)),
        ("shift", torch.zeros(2, 2, 8)),
        ("scale", torch.zeros(2, 9)),
        ("shift", torch.zeros(2, 9)),
    ],
)
def test_adaln_rejects_invalid_contract_before_launch(key, value):
    inputs = dict(
        layernorm=torch.nn.LayerNorm(8),
        x=torch.ones(2, 3, 8),
        scale=torch.ones(2, 8),
        shift=torch.ones(2, 8),
        fused=True,
    )
    inputs[key] = value
    with patch.object(adaln.ops, "adaln") as a2, patch.object(adaln.ops, "adaln_v2") as a5:
        with pytest.raises(ParametersInvalid):
            adaln.layernorm_scale_shift(**inputs)
        a2.assert_not_called()
        a5.assert_not_called()


def test_single_token_modulation_is_broadcast_path():
    x = torch.ones(2, 1, 8)
    assert not adaln._is_tokenwise_modulation(x, x)
    actual, scale, shift, original = adaln._preprocess_tokenwise_modulation(x, x, x)
    assert actual is x and scale is x and shift is x and original is None
