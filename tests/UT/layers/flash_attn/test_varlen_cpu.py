"""Variable-length attention parameter validation and native launch contracts."""

from unittest.mock import patch

import pytest
import torch

from mindiesd.layers.flash_attn.attention_forward_varlen import attention_forward_varlen
from mindiesd.utils import ParametersInvalid


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale", [None, 0.3])
def test_varlen_native_arguments(causal, scale):
    q = torch.ones(5, 2, 8)
    k = torch.ones(7, 2, 8)
    sentinel = torch.full_like(q, 7)
    with patch("torch_npu.npu_fusion_attention", return_value=(sentinel,)) as native:
        assert attention_forward_varlen(q, k, k, [0, 2, 5], [0, 3, 7], causal=causal, softmax_scale=scale) is sentinel
    args = native.call_args.kwargs
    assert args["actual_seq_qlen"] == [2, 5] and args["actual_seq_kvlen"] == [3, 7]
    assert args["scale"] == (8**-0.5 if scale is None else scale)
    assert args["sparse_mode"] == (3 if causal else 0)
    if causal:
        mask = args["atten_mask"]
        assert mask.device.type == "cpu" and mask[0, 1] and not mask[1, 0]
    else:
        assert args["atten_mask"] is None


@pytest.mark.parametrize(
    "name",
    [
        "max_seqlen_q",
        "max_seqlen_k",
        "window_size",
        "softcap",
        "alibi_slopes",
        "deterministic",
        "return_attn_probs",
        "block_table",
        "dropout_p",
    ],
)
def test_varlen_rejects_unsupported_options(name):
    x = torch.ones(5, 2, 8)
    with pytest.raises(ParametersInvalid):
        attention_forward_varlen(x, x, x, [0, 5], [0, 5], **{name: 1})


@pytest.mark.parametrize("index", [0, 1, 2])
def test_varlen_rejects_invalid_tensor_rank(index):
    inputs = [torch.ones(5, 2, 8) for _ in range(3)]
    inputs[index] = torch.ones(5, 8)
    with pytest.raises(ParametersInvalid):
        attention_forward_varlen(*inputs, [0, 5], [0, 5])
