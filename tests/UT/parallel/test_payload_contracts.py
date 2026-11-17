# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU contracts for distributed quantization and V send layouts."""

# pylint: disable=redefined-outer-name  # pytest fixtures are requested by parameter name.

import pytest

torch = pytest.importorskip("torch")
from cpu_support import load_parallel_module, new_package


@pytest.fixture
def quant(monkeypatch):
    package = new_package("_parallel_quant_contracts", monkeypatch)
    load_parallel_module(package, "config", monkeypatch)
    return load_parallel_module(package, "distributed_quant", monkeypatch)


def make_boundary_owner(local_peak, rank):
    """Build one half-owner of a 64-row block with a rank-specific local amax."""
    rows = 32
    torch.manual_seed(rank)
    x = (torch.rand(1, rows, 2, 4, dtype=torch.float32) * local_peak).to(torch.bfloat16)
    local_rank_mapping = {
        "rank": rank,
        "tok_lo": 0,
        "tok_hi": rows,
        "n_tok": rows,
        "blocks": torch.tensor([0], dtype=torch.int64),
        "owned_per_block": torch.tensor([rows], dtype=torch.int64),
        "full": torch.tensor([False]),
        "src_local": torch.arange(rows, dtype=torch.int64),
        "blk_idx": torch.zeros(rows, dtype=torch.int64),
        "row_in_block": torch.arange(rows, dtype=torch.int64),
        "amax_entry": torch.tensor([3], dtype=torch.int64),
        "cells_unique": torch.tensor([0], dtype=torch.int64),
    }
    return x, local_rank_mapping


def test_a_boundary_block_is_repaired_to_one_global_scale(quant):
    """Both owners must quantize against the globally repaired scale."""
    xa, part_a = make_boundary_owner(1.0, rank=0)
    xb, part_b = make_boundary_owner(8.0, rank=1)
    st_a = quant.make_quant_state(part_a, 2, 4, 1, 32, torch.device("cpu"), 4, head_row=None)
    st_b = quant.make_quant_state(part_b, 2, 4, 1, 32, torch.device("cpu"), 4, head_row=None)
    boundary_amax_a = quant.quantize_amax_for_allreduce(xa, st_a)
    boundary_amax_b = quant.quantize_amax_for_allreduce(xb, st_b)
    assert float(boundary_amax_a[:, 3].abs().max()) > 0
    assert float(boundary_amax_a[:, :3].abs().max()) == 0
    assert float(boundary_amax_b[:, 3].abs().max()) > 0

    global_amax = torch.maximum(boundary_amax_a, boundary_amax_b)
    quant.quantize_and_pool_for_amax(xa, st_a, emit_amax=False, gather=False)
    quant.quantize_and_pool_for_amax(xb, st_b, emit_amax=False, gather=False)
    quant.repair_boundary_and_scatter(st_a, global_amax)
    quant.repair_boundary_and_scatter(st_b, global_amax)
    assert torch.equal(st_a.scales[:, 0], st_b.scales[:, 0])
    want = global_amax[:, 3].to(torch.bfloat16).to(torch.float32) * quant._C_INV127
    assert torch.equal(st_a.scales[:, 0], want)
    assert float(st_a.scales[:, 0].max() / quant._C_INV127) > 7.0
    for state, value in ((st_a, xa), (st_b, xb)):
        for head in range(2):
            rows = value[0, :, head, :].float() / state.scales[head, 0]
            assert torch.equal(state.payload[head, :32], torch.round(rows).to(torch.int8))


def test_check_sendable_owns_the_v_send_contract(quant):
    assert quant.check_sendable((1, 4, 4, 2), 2, 2) is None
    assert quant.check_sendable((1, 4, 4, 2), 1, 4) is None
    with pytest.raises(ValueError):
        quant.check_sendable((1, 4, 8, 2), 2, 2)
    with pytest.raises(ValueError):
        quant.check_sendable((2, 4, 4, 2), 2, 2)
    with pytest.raises(ValueError):
        quant.check_sendable((1, 4, 9, 2), 2, 2)


def test_v_send_layout_round_trips_into_the_operator_layout(quant):
    world, chunks, s_local, hidden = 2, 3, 2, 2
    heads = chunks * world
    base = 1 + torch.arange(s_local).reshape(s_local, 1) * heads + torch.arange(heads).reshape(1, heads)
    value = torch.stack((base, base * 2), dim=-1).reshape(1, s_local, heads, hidden).to(torch.bfloat16)
    scales = torch.ones(1, heads, hidden, dtype=torch.float32)
    sent = quant.to_send(value, scales, chunks, world)
    assert tuple(sent.shape) == (chunks, world, s_local, 1, 1, hidden)
    for chunk in range(chunks):
        got = quant.from_send(sent[chunk]).view(torch.float8_e4m3fn).float().reshape(world * s_local, hidden)
        want = torch.stack([value[0, :, rank * chunks + chunk, :] for rank in range(world)]).reshape(-1, hidden)
        assert torch.equal(got, want.float())
    assert len({tuple(sent[chunk].reshape(-1).tolist()) for chunk in range(chunks)}) == chunks


def test_a_subnormal_v_input_flushes_to_positive_zero(quant):
    zero = torch.zeros(1, 1, 1, 2, dtype=torch.bfloat16)
    got = quant.to_send(zero, torch.zeros(1, 1, 2), 1, 1).view(torch.uint8)
    assert got.reshape(-1).tolist() == [0x00, 0x00]

    value = torch.tensor([[[[1e-39, 1.0]], [[-1e-39, -1.0]]]], dtype=torch.bfloat16)
    one = torch.ones(1, 1, 2, dtype=torch.float32)
    got = quant.to_send(value, one, 1, 1).view(torch.uint8).reshape(-1)
    assert got.tolist() == [0x00, 0x38, 0x00, 0xB8]

    small = 2.0**-127
    subnormal = torch.tensor([[[[small, small]], [[-small, -small]]]], dtype=torch.float32)
    divisor = torch.full((1, 1, 2), small, dtype=torch.float32)
    got = quant.to_send(subnormal, divisor, 1, 1).view(torch.uint8).reshape(-1)
    assert got.tolist() == [0x00, 0x00, 0x00, 0x00]
