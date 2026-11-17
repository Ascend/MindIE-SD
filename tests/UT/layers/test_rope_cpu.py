"""CPU numerical RoPE reference and native boundary contracts."""

from unittest.mock import patch

import pytest
import torch

from mindiesd.layers.rope import rotary_position_embedding


@pytest.mark.parametrize("mode", ["rotated_half", "rotated_interleaved"])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("cache_rank", [2, 4])
def test_rotary_math_and_fused_arguments(mode, head_first, cache_rank):
    x = torch.arange(96).reshape(1, 3, 2, 16).to(torch.float16)
    if head_first:
        x = x.transpose(1, 2)
    cos, sin = torch.randn(3, 16), torch.randn(3, 16)
    shape = (1, 1, 3, 16) if head_first else (1, 3, 1, 16)
    c, s = cos.reshape(shape), sin.reshape(shape)
    real = x.float()
    if mode == "rotated_half":
        rotated = torch.cat((-real[..., 8:], real[..., :8]), -1)
    else:
        rotated = torch.stack((-real[..., 1::2], real[..., ::2]), -1).flatten(-2)
    expected = (real * c + rotated * s).to(x.dtype)
    cache = (cos, sin) if cache_rank == 2 else (c, s)
    torch.testing.assert_close(rotary_position_embedding(x, *cache, mode, head_first, False), expected)
    with patch("torch_npu.npu_rotary_mul", return_value=expected.float()) as native:
        torch.testing.assert_close(rotary_position_embedding(x, *cache, mode, head_first, True), expected)
        assert native.call_args.args[-1] == ("half" if mode == "rotated_half" else "interleave")
        assert native.call_args.args[0].dtype == cos.dtype
