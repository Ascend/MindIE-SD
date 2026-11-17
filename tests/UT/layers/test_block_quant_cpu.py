"""CPU preprocessing and Hadamard mathematical contracts."""

from unittest.mock import patch

import pytest
import torch

from mindiesd.layers.quant import block_quant as quant
from mindiesd.utils import ParametersInvalid


@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_block_quant_layout_and_options(layout, explicit_dtype):
    x = torch.arange(2 * 3 * 8).reshape(1, 2, 3, 8).float()
    native_input = x.transpose(1, 2).squeeze(0) if layout == "BSND" else x.squeeze(0)
    with patch.object(quant.torch_npu, "npu_dynamic_block_quant", return_value=(native_input, torch.ones(2))) as native:
        kwargs = {"dst_type": torch.int8} if explicit_dtype else {}
        output, scales = quant.fa_block_quant_preprocess(x, block_size=256, col_block_size=64, layout=layout, **kwargs)
        torch.testing.assert_close(native.call_args.args[0], native_input)
        torch.testing.assert_close(output, native_input.unsqueeze(0))
        assert scales.shape == (1, 2)
        assert native.call_args.kwargs["row_block_size"] == 256
        assert native.call_args.kwargs["col_block_size"] == 64
    with pytest.raises(ParametersInvalid):
        quant.fa_block_quant_preprocess(torch.ones(2, 3))


def test_missing_fp8_capability_fails_before_launch():
    with patch.object(quant.torch_npu, "float8_e4m3fn", None):
        with pytest.raises(RuntimeError, match="float8"):
            quant.fa_block_quant_preprocess(torch.ones(1, 2, 3, 8))


@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_hadamard_seed_orthogonality_and_block_rotation(left, transpose):
    state = torch.random.get_rng_state().clone()
    matrix = quant.generate_hadamard_matrix(4, transpose=transpose, seed=51)
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(4) * 4)
    weight = torch.arange(2 * 8 * 8).reshape(2, 8, 8).float()
    actual = quant.apply_hadamard_weight(weight, matrix, left=left)
    expected = torch.empty_like(weight)
    if left:
        for start in (0, 4):
            expected[:, start : start + 4] = matrix @ weight[:, start : start + 4] / 4
    else:
        for start in (0, 4):
            expected[..., start : start + 4] = weight[..., start : start + 4] @ matrix / 4
    torch.testing.assert_close(actual, expected)
    with pytest.raises(AssertionError, match="not divisible"):
        quant.apply_hadamard_weight(torch.ones(3, 3), matrix, left=left)
