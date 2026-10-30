"""Additional CPU expert MLP scale and routing contracts."""

import importlib
from unittest.mock import patch

import pytest
import torch

from mindiesd.layers.moe.moe_dataclass import MoEMlpComputeInput, MoEWeights, MoEStaticCombineMetadata

mlp = importlib.import_module("mindiesd.layers.moe.moe_mlp")


def test_weight_format_and_scale_layout_validation():
    weight = torch.ones(2, 4, 8)
    with (
        patch("torch_npu.get_npu_format", return_value=0),
        patch("torch_npu.npu_format_cast", return_value=weight) as cast,
    ):
        assert mlp._ensure_nz_weight(weight) is weight
        cast.assert_called_once_with(weight, 29)
    assert mlp._normalize_mxfp_scale_layout(None) is None
    scale = torch.ones(3, 1, 2)
    assert mlp._normalize_mxfp_scale_layout(scale) is scale
    with pytest.raises(ValueError, match="scale shape"):
        mlp._normalize_mxfp_scale_layout(torch.ones(3, 3))


@pytest.mark.parametrize("quantized", [False, True])
def test_mxfp8_static_finalize_uses_expanded_indices(quantized):
    x = torch.ones(4, 8)
    weights = MoEWeights(
        w13_weight=torch.ones(2, 8, 16),
        w2_weight=torch.ones(2, 8, 8),
        w13_weight_scale=torch.ones(2, 16),
        w2_weight_scale=torch.ones(2, 8),
    )
    inputs = MoEMlpComputeInput(
        hidden_states=x,
        group_list=torch.tensor([2, 4]),
        group_list_type=0,
        weights=weights,
        mlp_output_dtype=torch.float16,
        dynamic_scale=torch.ones(4, 2) if quantized else None,
    )
    metadata = MoEStaticCombineMetadata(
        expanded_row_idx=torch.tensor([2, 0, 3, 1]),
        topk_weights=torch.tensor([[0.1, 0.9], [0.4, 0.6]]),
        restore_shape=(2, 8),
    )
    with (
        patch("torch_npu.npu_dynamic_mx_quant", return_value=(x, torch.ones(4, 1, 2))) as quant,
        patch("torch_npu.npu_grouped_matmul_swiglu_quant_v2", return_value=(x, torch.ones(4, 1, 2))),
        patch("torch_npu.npu_grouped_matmul_finalize_routing", return_value=torch.ones(2, 8)) as finalize,
    ):
        actual = mlp.w8a8_mxfp8_apply_mlp(inputs, metadata)
        assert actual.shape == (2, 8) and actual.dtype == torch.float16
        torch.testing.assert_close(finalize.call_args.kwargs["row_index"], torch.tensor([1, 0, 1, 0]))
        torch.testing.assert_close(finalize.call_args.kwargs["logit"], torch.tensor([0.4, 0.1, 0.6, 0.9]))
        assert quant.call_count == (0 if quantized else 1)
