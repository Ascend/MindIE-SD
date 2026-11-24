#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Single-NPU CATLASS FlexAttention compared to dense FP32 attention."""

import pytest
import torch

from mindiesd import FlexAttentionMask, flex_attention, make_block_causal_mask
from mindiesd.utils.get_platform import is_a5_device


@pytest.mark.parametrize(
    "dtype, layout, position_dtype",
    [(torch.bfloat16, "BSND", torch.int64), (torch.float16, "BNSD", torch.int32)],
)
@torch.inference_mode()
def test_block_causal_mask_with_padding_and_rectangular_shards(dtype, layout, position_dtype):
    if not is_a5_device():
        pytest.skip("CATLASS DSL FlexAttention requires Ascend 950")
    device = torch.device("npu", torch.npu.current_device())
    length, width = 263, 128
    q, k, v = (torch.randn((1, size, 2, width), dtype=dtype, device=device) * 0.2 for size in (131, length, length))
    q_pos = torch.cat(
        (
            torch.arange(3, 10, device=device, dtype=position_dtype),
            torch.arange(130, 254, device=device, dtype=position_dtype),
        )
    )
    k_pos = torch.arange(length, device=device, dtype=position_dtype)
    valid = k_pos != 6
    spans = [(9, 139), (140, 262)]
    mask = make_block_causal_mask(spans, length, device=device, valid_tokens=valid)
    inputs = (q, k, v) if layout == "BSND" else tuple(t.transpose(1, 2) for t in (q, k, v))
    output, lse = flex_attention(
        *inputs, mask=mask, layout=layout, query_positions=q_pos, key_positions=k_pos, return_lse=True
    )
    ids = torch.full((length,), -1, device=device)
    for index, (start, end) in enumerate(spans):
        ids[start:end] = index
    q_ids = ids[q_pos, None]
    keep = ((q_pos[:, None] >= k_pos) | ((q_ids >= 0) & (q_ids == ids))) & valid[q_pos, None] & valid
    qb, kb, vb = (t.transpose(1, 2).float() for t in (q, k, v))
    scores = (qb @ kb.transpose(-2, -1) * width**-0.5).masked_fill(~keep, -torch.inf)
    expected = torch.nan_to_num(scores.softmax(-1)) @ vb
    if layout == "BSND":
        expected = expected.transpose(1, 2)
    torch.testing.assert_close(output.float(), expected, atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=3e-3, rtol=2e-3)


@torch.inference_mode()
def test_custom_dsl_score_and_mask():
    if not is_a5_device():
        pytest.skip("CATLASS DSL FlexAttention requires Ascend 950")
    from catlass import tla

    @tla.jit
    def causal(b, h, q, k, info, aux, scalars):
        return q >= k

    @tla.jit
    def bias(score, b, h, q, k, info, aux, scalars):
        return score + aux[0][h] * (k - q).to(tla.Float32) * scalars[0]

    device = torch.device("npu", torch.npu.current_device())
    q, k, v = (torch.randn(1, 35, 2, 128, dtype=torch.bfloat16, device=device) * 0.2 for _ in range(3))
    weights = torch.arange(4, device=device, dtype=torch.float32)
    heads = torch.tensor([1, 3], device=device)
    mask = FlexAttentionMask(
        causal, bias, aux_tensors=(weights,), aux_scalars=(0.01,), batch_shared=True, head_shared=True
    )
    positions = torch.arange(35, device=device) + (2**31 - 16)
    actual = flex_attention(q, k, v, mask=mask, head_indices=heads, query_positions=positions, key_positions=positions)
    qb, kb, vb = (t.transpose(1, 2).float() for t in (q, k, v))
    distance = positions - positions[:, None]
    scores = qb @ kb.transpose(-2, -1) * 128**-0.5
    scores = (scores + weights[heads][None, :, None, None] * distance * 0.01).masked_fill(distance > 0, -torch.inf)
    expected = (scores.softmax(-1) @ vb).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=3e-3, rtol=2e-2)
