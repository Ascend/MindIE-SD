"""Real norm/RoPE input validation with a scoped device-property boundary."""

import importlib
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from mindiesd.utils import ParametersInvalid

module = importlib.import_module("mindiesd.layers.norm_rope_concat")


@pytest.mark.parametrize("input_layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("output_layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("encoder", [False, True])
def test_valid_layout_conversion_and_native_forwarding(input_layout, output_layout, encoder):
    q = torch.arange(48).reshape(1, 3, 2, 8).float()
    if input_layout == "BNSD":
        q = q.transpose(1, 2)
    outputs = tuple(torch.ones(1, 2, 3, 8) for _ in range(3)) + tuple(torch.ones(1) for _ in range(8))
    kwargs = dict(input_layout=input_layout, output_layout=output_layout)
    if encoder:
        kwargs.update(encoder_query=q, encoder_key=q, encoder_value=q)
    device_boundary = SimpleNamespace(type="npu")
    with (
        patch.object(torch.Tensor, "device", new_callable=PropertyMock, return_value=device_boundary),
        patch.object(torch.ops.mindiesd, "norm_rope_concat", return_value=outputs, create=True) as native,
    ):
        actual = module.norm_rope_concat(q, q, q, **kwargs)
        assert native.call_args.args[0].shape == (1, 3, 2, 8)
        assert native.call_args.args[0].is_contiguous()
        assert (native.call_args.kwargs["encoder_query"] is not None) == encoder
        assert actual[0].shape == ((1, 3, 2, 8) if output_layout == "BSND" else (1, 2, 3, 8))
        assert actual[3] is outputs[3]
    assert actual[0].device.type == "cpu"


@pytest.mark.parametrize(
    "name,value",
    [
        ("input_layout", "bad"),
        ("output_layout", None),
        ("query", None),
        ("query", torch.ones(2, 8)),
        ("query", torch.ones(1, 2, 3, 7)),
        ("key", None),
        ("key", []),
        ("key", torch.ones(1, 3, 2, 8, dtype=torch.float64)),
        ("value", torch.ones(1, 4, 2, 8)),
        ("encoder_key", torch.ones(1, 3, 2, 8)),
        ("encoder_value", torch.ones(1, 3, 2, 8)),
        ("key", torch.ones(2, 3, 2, 8)),
        ("norm_type", True),
        ("norm_added_type", 5),
        ("rope_type", -1),
        ("concat_order", 3),
        ("eps", 0),
        ("eps", float("inf")),
        ("is_training", 1),
        ("norm_type", 2),
        ("norm_type", 4),
        ("norm_query_weight", torch.ones(7)),
        ("rope_type", 1),
        ("rope_sin", torch.ones(3, 8)),
    ],
)
def test_invalid_inputs_fail_before_native(name, value):
    q = torch.ones(1, 3, 2, 8)
    inputs = dict(query=q, key=q, value=q)
    inputs[name] = value
    with patch.object(torch.ops.mindiesd, "norm_rope_concat", create=True) as native:
        with pytest.raises(ParametersInvalid):
            module.norm_rope_concat(**inputs)
        native.assert_not_called()


@pytest.mark.parametrize("norm_type", [0, 1, 2, 3, 4])
def test_norm_and_rope_parameter_validation_reaches_device_check(norm_type):
    q = torch.ones(1, 3, 2, 8)
    kwargs = dict(
        norm_type=norm_type,
        norm_added_type=norm_type,
        encoder_query=q,
        encoder_key=q,
        encoder_value=q,
        rope_type=1,
        rope_sin=torch.ones(3, 8),
        rope_cos=torch.ones(3, 8),
    )
    for name in (
        "norm_query_weight",
        "norm_query_bias",
        "norm_key_weight",
        "norm_key_bias",
        "norm_added_query_weight",
        "norm_added_query_bias",
        "norm_added_key_weight",
        "norm_added_key_bias",
    ):
        kwargs[name] = torch.ones(8)
    with pytest.raises(ParametersInvalid, match="NPU"):
        module.norm_rope_concat(q, q, q, **kwargs)
    kwargs["rope_cos"] = torch.ones(4, 8)
    with pytest.raises(ParametersInvalid, match="sin/cos shapes"):
        module.norm_rope_concat(q, q, q, **kwargs)
