"""CPU sparse module configuration and native AdaBSA argument contracts."""

import importlib
from unittest.mock import patch

import pytest
import torch

from mindiesd.utils import ParametersInvalid

linear = importlib.import_module("mindiesd.layers.flash_attn.sparse_linear_attn")
ada = importlib.import_module("mindiesd.layers.flash_attn.sparse_flash_attn_ada_bsa")


@pytest.mark.parametrize("feature", ["elu", "softmax", "relu"])
@pytest.mark.parametrize("tie", [False, True])
def test_feature_map_math_and_initialized_projection(feature, tie):
    with patch.object(linear, "is_a5_device", return_value=True):
        module = linear.SparseLinearAttention(head_dim=64, topk=0.5, feature_map=feature, tie_feature_map_qk=tie)
    x = torch.tensor([-2.0, 0.0, 1.0])
    expected = {"elu": torch.nn.functional.elu(x) + 1, "softmax": x.softmax(-1), "relu": x.relu()}[feature]
    torch.testing.assert_close(module.feature_map_q(x), expected)
    torch.testing.assert_close(module.feature_map_k(x), expected)
    assert torch.count_nonzero(module.proj_l.weight) == torch.count_nonzero(module.proj_l.bias) == 0
    if tie:
        assert module.feature_map_q is module.feature_map_k
    with pytest.raises(ParametersInvalid, match="head dimension"):
        module(torch.ones(1, 2, 3, 32), torch.ones(1, 2, 3, 32), torch.ones(1, 2, 3, 32))
    with pytest.raises(ParametersInvalid, match="on NPU"):
        module(torch.ones(1, 2, 3, 64), torch.ones(1, 2, 3, 64), torch.ones(1, 2, 3, 64))


def test_sparse_constructor_rejects_invalid_options():
    with pytest.raises(ParametersInvalid, match="positive"):
        linear.SparseLinearAttention(head_dim=64, topk=0.5, BLKQ=0)
    with patch.object(linear, "is_a5_device", return_value=True):
        with pytest.raises(NotImplementedError):
            linear.SparseLinearAttention(head_dim=64, topk=0.5, feature_map="invalid")


def test_adabsa_native_options_and_estimate_scale():
    q = torch.ones(1, 2, 3, 8)
    mask = torch.ones(1)
    count = torch.ones(1, dtype=torch.int32)
    with (
        patch.object(ada, "is_a5_device", return_value=False),
        patch.object(torch.ops.mindiesd, "sparse_block_estimate", return_value=(mask, count), create=True) as estimate,
        patch.object(torch.ops.mindiesd, "ada_block_sparse_attention", return_value=q, create=True) as native,
    ):
        actual_mask, actual_count = ada.get_estimate_mask(q, q, q, scale=0.5, stride=4, head_num=2, sparsity=0.3)
        assert actual_mask is mask and actual_count is count
        assert estimate.call_args.kwargs["scale_value"] == 0.125
        assert estimate.call_args.kwargs["row_sparse"] == 0.7
        assert ada.ada_block_sparse_attention(q, q, q, mask, count, scale=0.5, head_num=2) is q
        assert native.call_args.kwargs["sparse_mask"] is mask
        assert native.call_args.kwargs["sparse_count_table"] is count
