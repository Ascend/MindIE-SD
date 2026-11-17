"""CPU shape inference and native dispatch contracts for custom operators."""

from unittest.mock import patch

import pytest
import torch

from mindiesd.layers import _custom_ops as ops
from mindiesd.utils import ParametersInvalid


@pytest.mark.parametrize("layout,shape", [("TND", (5, 2, 8)), ("BNSD", (2, 3, 5, 8))])
@pytest.mark.parametrize(
    "kind", ["block_sparse_attention", "eagle_block_sparse_attention", "eagle_quant_block_sparse_attention"]
)
@pytest.mark.parametrize("optional", [False, True])
def test_sparse_attention_optional_native_arguments(layout, shape, kind, optional):
    q = torch.ones(shape)
    kwargs = dict(q_input_layout=layout)
    if optional:
        kwargs.update(block_shape=[16, 32], actual_seq_lengths=[5], actual_seq_lengths_kv=[7])
        if kind == "block_sparse_attention":
            kwargs.update(
                q_dequant_scale=q,
                k_dequant_scale=q,
                v_dequant_scale=q,
                quant_mode=2,
                dst_type_max=4.0,
                q_dtype=40,
                k_dtype=40,
                v_dtype=40,
                q_scale_dtype=1,
                k_scale_dtype=1,
                v_scale_dtype=1,
            )
        elif kind == "eagle_quant_block_sparse_attention":
            kwargs.update(
                query_scale=q,
                key_scale=q,
                value_scale=q,
                query_dtype=torch.uint8,
                key_dtype=torch.uint8,
                value_dtype=torch.uint8,
                output_dtype=torch.bfloat16,
            )
    sentinel = (torch.ones(shape), torch.zeros(1))
    with patch.object(torch.ops.mindiesd, kind, return_value=sentinel, create=True) as native:
        assert getattr(ops, kind)(q, q, q, **kwargs) is sentinel
        actual = native.call_args.kwargs
        assert actual["query"] is q
        assert actual["block_shape"] == ([16, 32] if optional else [128, 128])
        assert ("actual_seq_lengths" in actual) == optional
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                assert actual[name] is value
            else:
                assert actual[name] == value
    output, lse = getattr(ops, kind + "_fake")(q, q, q, **kwargs)
    packed = kind == "block_sparse_attention" and optional
    assert output.shape == (*shape[:-1], shape[-1] * (2 if packed else 1))
    assert lse.shape == (*shape[:-1], 1)
    assert lse.dtype == torch.float32
    expected_dtype = torch.bfloat16 if optional and kind != "eagle_block_sparse_attention" else q.dtype
    assert output.dtype == expected_dtype


@pytest.mark.parametrize("shape", [(2, 3, 8), (5, 3, 8)])
def test_adaln_and_layernorm_fake_shapes(shape):
    x = torch.zeros(shape)
    assert ops.adaln_fake(x, x, x).shape == x.shape
    out, mean, rstd = ops.adaln_v2_fake(x, x, x)
    assert out.shape == x.shape and mean.shape == (*shape[:2], 1) and rstd.shape == mean.shape
    for normalized in ([8], [3, 8]):
        output, mean, rstd = ops.layernorm_fake(x, normalized)
        assert output.shape == x.shape
        assert mean.shape == (*shape[: -len(normalized)], *([1] * len(normalized)))
        assert rstd.shape == mean.shape
    with pytest.raises(ParametersInvalid):
        ops.layernorm_fake(x, [2, 2, 2, 2])
    assert ops.layernorm_fake(torch.empty(0, 8), [8])[1].shape == (0,)


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("output_bnsd", [False, True])
@pytest.mark.parametrize("cache_rank", [2, 3, 4])
@pytest.mark.parametrize("affine", [False, True])
def test_split_norm_rope_shape_contract(rank, output_bnsd, cache_rank, affine):
    shape = (2, 3, 8, 128) if rank == 4 else (3, 8, 128)
    q = torch.zeros(shape, dtype=torch.bfloat16)
    cache = torch.zeros({2: (3, 64), 3: (1, 3, 64), 4: (1, 3, 1, 64)}[cache_rank], dtype=q.dtype)
    weight = torch.ones(128, dtype=q.dtype) if affine else None
    results = ops.eagle_split_norm_rope_concat_fake(
        q, weight, weight, cache, cache, [4, 2, 2], 3 if affine else 2, 0, output_bnsd=output_bnsd
    )
    for result, heads in zip(results, [4, 2, 2]):
        expected = (
            ((2 if rank == 4 else 1), heads, 3, 128)
            if output_bnsd
            else ((2, 3, heads, 128) if rank == 4 else (3, heads, 128))
        )
        assert result.shape == expected and result.dtype == q.dtype


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf"), 1e100, 1e-100])
def test_indexed_norm_rejects_nonpositive_fp32_epsilon(eps):
    x = torch.ones(3, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="eps"):
        ops.eagle_indexed_gate_norm_modulate_fake(
            x, x, x, x, x, torch.zeros(3, dtype=torch.int64), weight=x[0], eps=eps
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 3])
def test_indexed_norm_output_shapes(dtype, rows):
    x = torch.ones(rows, 8, dtype=dtype)
    modulation = torch.ones(2, 8, dtype=dtype)
    output = ops.eagle_indexed_gate_norm_modulate_fake(
        x,
        modulation,
        x,
        modulation,
        modulation,
        torch.zeros(rows, dtype=torch.int32),
        weight=torch.ones(8, dtype=dtype),
    )
    assert all(tensor.shape == x.shape and tensor.dtype == dtype for tensor in output)


@pytest.mark.parametrize("shape", [(2, 64), (2, 3, 96)])
@pytest.mark.parametrize("axis", [-1, 0])
def test_four_over_six_packed_shape(shape, axis):
    x = torch.ones(shape, dtype=torch.bfloat16)
    packed, scales = ops.quant_four_over_six_a5_fake(x, axis=axis)
    assert packed.shape == (*shape[:-1], shape[-1] // 2)
    expected = list(shape)
    expected[axis] = ((shape[axis] + 31) // 32 + 1) // 2
    assert scales.shape == (*expected, 2)
    assert packed.dtype == scales.dtype == torch.uint8
    with pytest.raises(ParametersInvalid):
        ops.quant_four_over_six_a5_fake(x, dst_type=39)


@pytest.mark.parametrize("shape", [(3, 2, 8), (2, 3, 2, 8)])
def test_laser_preprocess_padding_contract(shape):
    q = torch.ones(shape)
    outputs = ops.attention_preprocess_fake(q, q, q, 4)
    expected = (4, 2, 8) if len(shape) == 3 else (2, 4, 2, 8)
    assert all(tensor.shape == expected for tensor in outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_fake_output_dtype(dtype):
    q = torch.zeros(2, 3, 4, 8, dtype=dtype)
    lse, output = ops.attention_fake(q, q, q)
    assert lse.shape == (2, 3, 4)
    assert output.dtype == (torch.bfloat16 if dtype == torch.bfloat16 else torch.float32)
