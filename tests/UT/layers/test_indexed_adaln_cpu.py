"""CPU coverage for indexed AdaLayerNorm validation and cache behavior."""

import importlib

import pytest
import torch

from mindiesd.utils import ParametersInvalid

indexed_adaln = importlib.import_module("mindiesd.layers.indexed_adaln")


@pytest.fixture(autouse=True)
def clear_index_validation_cache():
    indexed_adaln._INDEX_VALIDATION_CACHE.clear()
    yield
    indexed_adaln._INDEX_VALIDATION_CACHE.clear()


def _valid_inputs():
    hidden = 16
    return {
        "layernorm": torch.nn.LayerNorm(hidden, elementwise_affine=False),
        "x": torch.randn(1, 2, hidden),
        "scale_table": torch.randn(1, 2, hidden),
        "shift_table": torch.randn(1, 2, hidden),
        "mod_index": torch.tensor([0, 1], dtype=torch.int64),
    }


def _call(**overrides):
    inputs = _valid_inputs()
    inputs.update(overrides)
    return indexed_adaln.layernorm_scale_shift_indexed(**inputs, fused=False)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"layernorm": None}, "input layernorm must be torch.nn.LayerNorm"),
        ({"x": None}, "input x must be torch.Tensor"),
        ({"scale_table": None}, "input scale_table must be torch.Tensor"),
        ({"shift_table": None}, "input shift_table must be torch.Tensor"),
        ({"mod_index": None}, "input mod_index must be torch.Tensor"),
        ({"x": torch.zeros(1, 16)}, "x must be 3D"),
        ({"scale_table": torch.zeros(1, 16)}, "scale_table/shift_table must be 3D"),
        ({"shift_table": torch.zeros(1, 16)}, "scale_table/shift_table must be 3D"),
        ({"mod_index": torch.zeros(1, 2, dtype=torch.int64)}, "mod_index must be 1D"),
        ({"scale_table": torch.zeros(2, 2, 16)}, "Batch size"),
        ({"shift_table": torch.zeros(2, 2, 16)}, "Batch size"),
        ({"scale_table": torch.zeros(1, 2, 32)}, "Hidden size"),
        ({"shift_table": torch.zeros(1, 2, 32)}, "Hidden size"),
        ({"scale_table": torch.zeros(1, 3, 16)}, "same K"),
        (
            {"scale_table": torch.zeros(1, 0, 16), "shift_table": torch.zeros(1, 0, 16)},
            "K must be positive",
        ),
        ({"mod_index": torch.zeros(1, dtype=torch.int64)}, "mod_index length must equal T"),
        ({"layernorm": torch.nn.LayerNorm((4, 4), elementwise_affine=False)}, "normalized_shape must be"),
        ({"layernorm": torch.nn.LayerNorm(8, elementwise_affine=False)}, "normalized_shape must be"),
        ({"x": torch.zeros(1, 2, 16, dtype=torch.int32)}, "x must be BF16, FP16 or FP32"),
        ({"scale_table": torch.zeros(1, 2, 16, dtype=torch.float16)}, "dtype must match x"),
        ({"shift_table": torch.zeros(1, 2, 16, dtype=torch.float16)}, "dtype must match x"),
        ({"mod_index": torch.zeros(2, dtype=torch.int32)}, "mod_index must be INT64"),
        (
            {
                "layernorm": torch.nn.LayerNorm(15, elementwise_affine=False),
                "x": torch.zeros(1, 2, 15),
                "scale_table": torch.zeros(1, 2, 15),
                "shift_table": torch.zeros(1, 2, 15),
            },
            "multiple of 32 bytes",
        ),
        ({"layernorm": torch.nn.LayerNorm(16, elementwise_affine=True)}, "elementwise_affine=False only"),
        ({"mod_index": torch.tensor([-1, 0], dtype=torch.int64)}, "mod_index values must be in"),
        ({"mod_index": torch.tensor([0, 2], dtype=torch.int64)}, "mod_index values must be in"),
    ],
)
def test_rejects_invalid_inputs_before_fallback(overrides, message):
    with pytest.raises(ParametersInvalid, match=message):
        _call(**overrides)


def test_cpu_fallback_matches_reference_for_both_fused_flags():
    inputs = _valid_inputs()
    layernorm = inputs["layernorm"]
    x = inputs["x"]
    scale = torch.index_select(inputs["scale_table"], 1, inputs["mod_index"])
    shift = torch.index_select(inputs["shift_table"], 1, inputs["mod_index"])
    expected = layernorm(x) * (1 + scale) + shift

    for fused in (False, True):
        actual = indexed_adaln.layernorm_scale_shift_indexed(**inputs, fused=fused)
        torch.testing.assert_close(actual, expected)


def test_validation_can_skip_index_range_check():
    inputs = _valid_inputs()
    inputs["mod_index"] = torch.tensor([0, 2], dtype=torch.int64)
    indexed_adaln._check_inputs(**inputs, validate_mod_index=False)


def test_index_range_cache_hits_for_same_tensor_and_key():
    indices = torch.tensor([0, 1], dtype=torch.int64)
    indexed_adaln._validate_mod_index_range(indices, 2)
    entry = indexed_adaln._INDEX_VALIDATION_CACHE[id(indices)]

    indexed_adaln._validate_mod_index_range(indices, 2)

    assert indexed_adaln._INDEX_VALIDATION_CACHE[id(indices)] is entry


def test_index_range_cache_revalidates_after_tensor_mutation():
    indices = torch.tensor([0, 1], dtype=torch.int64)
    indexed_adaln._validate_mod_index_range(indices, 2)
    indices[0] = 2

    with pytest.raises(ParametersInvalid, match="mod_index values must be in"):
        indexed_adaln._validate_mod_index_range(indices, 2)


def test_index_range_cache_revalidates_for_a_different_table_size():
    indices = torch.tensor([1], dtype=torch.int64)
    indexed_adaln._validate_mod_index_range(indices, 2)

    with pytest.raises(ParametersInvalid, match="mod_index values must be in"):
        indexed_adaln._validate_mod_index_range(indices, 1)


def test_index_range_cache_clears_after_more_than_256_entries():
    tensors = [torch.tensor([0], dtype=torch.int64) for _ in range(258)]

    for indices in tensors:
        indexed_adaln._validate_mod_index_range(indices, 1)

    assert len(indexed_adaln._INDEX_VALIDATION_CACHE) == 1


def test_index_range_cache_handles_non_weakrefable_empty_input():
    class NonWeakrefableEmpty:
        __slots__ = ()

        @staticmethod
        def numel():
            return 0

    indexed_adaln._validate_mod_index_range(NonWeakrefableEmpty(), 1)
    assert not indexed_adaln._INDEX_VALIDATION_CACHE


def test_empty_index_tensor_skips_min_max_validation():
    indexed_adaln._validate_mod_index_range(torch.empty(0, dtype=torch.int64), 1)
