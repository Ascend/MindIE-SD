"""FIA validation and dtype normalization without a device runtime."""

import importlib
from unittest.mock import patch

import pytest
import torch

from mindiesd.utils import ParametersInvalid
from mindiesd.layers.flash_attn.attention_operate import AttentionOperateBase, AttnOpRegistry

fia = importlib.import_module("mindiesd.layers.flash_attn.fused_infer_attention_score")


@pytest.mark.parametrize(
    "name,value",
    [
        ("input_layout", "bad"),
        ("num_query_heads", True),
        ("num_query_heads", 0),
        ("num_query_heads", "2"),
        ("num_key_value_heads", -1),
    ],
)
def test_invalid_attributes_fail_before_native(name, value):
    x = torch.ones(1, 3, 16)
    with patch.object(torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True) as native:
        with pytest.raises(ParametersInvalid):
            fia.fused_infer_attention_score_v2(x, x, x, **{name: value})
        native.assert_not_called()


@pytest.mark.parametrize("layout,shape", [("BSH", (1, 3, 15)), ("BNSD", (1, 3, 4, 8)), ("BSND", (1, 4, 3, 8))])
def test_head_mismatch(layout, shape):
    x = torch.ones(shape)
    with pytest.raises(ParametersInvalid):
        fia.fused_infer_attention_score_v2(x, x, x, input_layout=layout, num_query_heads=2)


@pytest.mark.parametrize("layout,shape", [("BSH", (1, 3, 16)), ("BNSD", (1, 2, 4, 8)), ("BSND", (1, 4, 2, 8))])
def test_valid_heads_and_dtype_options(layout, shape):
    x = torch.ones(shape)
    with patch.object(torch.ops.mindiesd, "fused_infer_attention_score_v2", return_value=(x,), create=True) as native:
        assert (
            fia.fused_infer_attention_score_v2(
                x,
                x,
                x,
                input_layout=layout,
                num_query_heads=2,
                query_dtype=torch.float16,
                key_dtype=36,
                quant_scale_p=1,
            )[0]
            is x
        )
        assert native.call_args.kwargs["query_dtype"] is None
        assert native.call_args.kwargs["key_dtype"] == 36
        assert "quant_scale_p" not in native.call_args.kwargs


@pytest.mark.parametrize("method", ["is_supported_layout", "is_supported_dtype"])
def test_base_requires_declared_capabilities(method):
    with pytest.raises(ParametersInvalid):
        getattr(AttentionOperateBase, method)(None)


@pytest.mark.parametrize("method", ["forward_attn_bnsd", "forward_attn_bsnd", "forward_attn_bsh"])
def test_abstract_forward_raises(method):
    with pytest.raises(NotImplementedError):
        getattr(AttentionOperateBase, method)(None, None, None, None)


def test_registry_contract_and_missing_operator():
    registry = AttnOpRegistry()
    with pytest.raises(ParametersInvalid):
        registry.register(1, lambda: None)
    with pytest.raises(ParametersInvalid):
        registry.register("bad", None)
    with pytest.raises(ParametersInvalid):
        registry.get("absent")
    registry.register("valid", torch.ones)
    assert registry.get("valid") is torch.ones
    assert registry.get_all() == {"valid": torch.ones}
