# SPDX-License-Identifier: Apache-2.0
"""Rectangular RF indices and independent Q/K pools retain stock values."""

import pytest
import torch

from cpu_support import RF_SOURCE, load_functions, native_rf_functions


@pytest.mark.parametrize("q_blocks,kv_blocks", [(2, 5), (5, 2)])
def test_rectangular_mask_indices(q_blocks, kv_blocks):
    get_index = load_functions(RF_SOURCE, {"get_mask_index"})["get_mask_index"]
    mask = torch.arange(2 * q_blocks * kv_blocks).reshape(1, 2, q_blocks, kv_blocks) % 3 == 0
    indices = get_index(mask)
    for head in range(2):
        for row in range(q_blocks):
            selected = torch.where(mask[0, head, row])[0]
            wanted = torch.cat((selected, torch.full((kv_blocks - selected.numel(),), -1)))
            assert torch.equal(indices[0, head, row], wanted)


@pytest.mark.parametrize("batch,heads,length", [(1, 1, 129), (2, 3, 257)])
def test_separate_pool_matches_combined_reference(batch, heads, length):
    functions = load_functions(RF_SOURCE, {"avgpool", "pool_qk_separately"})
    generator = torch.Generator().manual_seed(42)
    q, k = [torch.randn(batch, length, heads, 128, generator=generator).bfloat16() for _ in range(2)]
    reference = functions["avgpool"](torch.cat((q, k)), 128, "BSND")
    actual = functions["pool_qk_separately"](q, k, 128, "BSND")
    assert torch.equal(actual, reference)


def test_retained_blocks_accept_sequences():
    function = native_rf_functions()["get_multi_span_blockwise_mask"]
    pool = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4).bfloat16() / 16
    options = {"return_binary": True, "pool_parts": 2}
    wanted = function(pool, 0.8, 0.5, {0}, {1}, "BSND", **options)
    actual = function(pool, 0.8, 0.5, (0,), [1], "BSND", **options)
    assert torch.equal(actual, wanted)
