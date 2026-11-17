#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Public FIA fake metadata contracts without a native schema or NPU fixture."""

import pytest
import torch

from mindiesd.layers import _custom_ops as ops


def meta(shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def key_for_layout(query, value, layout):
    shape = list(value.shape)
    # Data construction only: K has the query head dimension and two KV heads.
    shape[-1] = query.shape[-1] // 2 if layout.startswith("BSH") else query.shape[-1]
    return meta(shape, query.dtype)


def assert_outputs(outputs, shape, lse_shape, dtype=torch.bfloat16):
    output, lse = outputs
    assert tuple(output.shape) == shape
    assert output.dtype == dtype and output.device.type == "meta"
    assert tuple(lse.shape) == lse_shape
    assert lse.dtype == torch.float32 and lse.device.type == "meta"


@pytest.mark.parametrize(
    "layout,query_shape,value_shape,output_shape,lse_shape",
    [
        ("BSH", (2, 3, 32), (2, 7, 12), (2, 3, 24), (2, 4, 3, 1)),
        ("BSND", (2, 3, 4, 8), (2, 7, 2, 6), (2, 3, 4, 6), (2, 4, 3, 1)),
        ("BNSD", (2, 4, 3, 8), (2, 2, 7, 6), (2, 4, 3, 6), (2, 4, 3, 1)),
        ("TND", (9, 4, 8), (11, 2, 6), (9, 4, 6), (9, 4, 1)),
        ("NTD", (4, 9, 8), (2, 11, 6), (4, 9, 6), (9, 4, 1)),
        ("BNSD_BSND", (2, 4, 3, 8), (2, 2, 7, 6), (2, 3, 4, 6), (2, 4, 3, 1)),
        ("BSH_BNSD", (2, 3, 32), (2, 7, 12), (2, 4, 3, 6), (2, 4, 3, 1)),
        ("BSND_BNSD", (2, 3, 4, 8), (2, 7, 2, 6), (2, 4, 3, 6), (2, 4, 3, 1)),
        ("NTD_TND", (4, 9, 8), (2, 11, 6), (9, 4, 6), (9, 4, 1)),
        ("BSH_NBSD", (2, 3, 32), (2, 7, 12), (4, 2, 3, 6), (2, 4, 3, 1)),
        ("BSND_NBSD", (2, 3, 4, 8), (2, 7, 2, 6), (4, 2, 3, 6), (2, 4, 3, 1)),
        ("BNSD_NBSD", (2, 4, 3, 8), (2, 2, 7, 6), (4, 2, 3, 6), (2, 4, 3, 1)),
        ("TND_NTD", (9, 4, 8), (11, 2, 6), (4, 9, 6), (9, 4, 1)),
        ("NSD", (4, 3, 8), (2, 7, 6), (4, 3, 6), (1, 4, 3, 1)),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_fia_layout_conversion_uses_value_head_dimension(
    layout, query_shape, value_shape, output_shape, lse_shape, return_lse
):
    query, value = meta(query_shape), meta(value_shape)
    outputs = ops.fused_infer_attention_score_v2_fake(
        query,
        key_for_layout(query, value, layout),
        value,
        input_layout=layout,
        num_query_heads=4,
        num_key_value_heads=2,
        return_softmax_lse=return_lse,
    )
    assert_outputs(outputs, output_shape, lse_shape if return_lse else (0,))


def test_fia_default_layout_and_head_count():
    query, value = meta((2, 3, 8)), meta((2, 7, 6))
    outputs = ops.fused_infer_attention_score_v2_fake(query, meta((2, 7, 8)), value, return_softmax_lse=True)
    assert_outputs(outputs, (2, 3, 6), (2, 1, 3, 1))


@pytest.mark.parametrize("kv_heads,value_shape", [(None, (2, 7, 24)), (0, (2, 7, 24)), (2, (2, 7, 12))])
def test_fia_grouped_query_heads_and_zero_kv_head_default(kv_heads, value_shape):
    query, value = meta((2, 3, 32)), meta(value_shape)
    kwargs = {"num_query_heads": 4}
    if kv_heads is not None:
        kwargs["num_key_value_heads"] = kv_heads
    key = meta((2, 7, 16 if kv_heads == 2 else 32))
    outputs = ops.fused_infer_attention_score_v2_fake(query, key, value, **kwargs)
    assert_outputs(outputs, (2, 3, 24), (0,))


@pytest.mark.parametrize(
    "value_shape,key_shape",
    [((6, 16, 12), (6, 16, 16)), ((6, 2, 16, 6), (6, 2, 16, 8)), ((6, 2, 3, 16, 2), (6, 2, 4, 16, 2))],
)
def test_fia_paged_attention_value_storage_layouts(value_shape, key_shape):
    query, value = meta((2, 4, 3, 8)), meta(value_shape)
    outputs = ops.fused_infer_attention_score_v2_fake(
        query,
        meta(key_shape),
        value,
        input_layout="BNSD",
        num_query_heads=4,
        num_key_value_heads=2,
        block_table=meta((2, 3), torch.int32),
        return_softmax_lse=True,
    )
    assert_outputs(outputs, (2, 4, 3, 6), (2, 4, 3, 1))


@pytest.mark.parametrize(
    "query_dtype,rope_dtype,quantized_output,explicit_dtype,expected_dtype",
    [
        (torch.float16, None, False, None, torch.float16),
        (torch.bfloat16, None, False, None, torch.bfloat16),
        (torch.int8, None, False, None, torch.float16),
        (torch.int8, torch.bfloat16, False, None, torch.bfloat16),
        (torch.float16, torch.bfloat16, False, None, torch.float16),
        (torch.bfloat16, None, True, None, torch.int8),
        (torch.int8, torch.bfloat16, True, None, torch.int8),
        (torch.int8, torch.bfloat16, True, torch.float16, torch.float16),
        (torch.float16, None, False, torch.bfloat16, torch.bfloat16),
    ],
)
def test_fia_output_dtype_priority(query_dtype, rope_dtype, quantized_output, explicit_dtype, expected_dtype):
    query, value = meta((2, 4, 3, 8), query_dtype), meta((2, 2, 7, 6), query_dtype)
    outputs = ops.fused_infer_attention_score_v2_fake(
        query,
        meta((2, 2, 7, 8), query_dtype),
        value,
        input_layout="BNSD",
        num_query_heads=4,
        num_key_value_heads=2,
        query_rope=meta((2, 4, 3, 2), rope_dtype) if rope_dtype is not None else None,
        quant_scale_out=meta((1,), torch.float32) if quantized_output else None,
        out_dtype=explicit_dtype,
    )
    assert_outputs(outputs, (2, 4, 3, 6), (0,), expected_dtype)


@pytest.mark.parametrize(
    "layout,query_shape,value_shape,output_shape,lse_shape",
    [
        ("BSH", (2, 0, 32), (2, 7, 12), (2, 0, 32), (2, 4, 0, 1)),
        ("BSH", (2, 3, 32), (2, 7, 0), (2, 3, 32), (2, 4, 3, 1)),
        ("BSND", (2, 3, 4, 8), (2, 7, 2, 0), (2, 3, 4, 8), (2, 4, 3, 1)),
        ("BSND", (2, 3, 4, 0), (2, 7, 2, 6), (2, 3, 4, 0), (2, 4, 3, 1)),
        ("BNSD", (2, 4, 3, 8), (2, 2, 7, 0), (2, 4, 3, 8), (2, 4, 3, 1)),
        ("BNSD", (2, 4, 3, 0), (2, 2, 7, 6), (2, 4, 3, 0), (2, 4, 3, 1)),
        ("BNSD_NBSD", (2, 4, 3, 8), (2, 2, 7, 0), (4, 2, 3, 8), (2, 4, 3, 1)),
        ("BNSD_NBSD", (2, 4, 3, 0), (2, 2, 7, 6), (4, 2, 3, 0), (2, 4, 3, 1)),
        ("TND", (9, 4, 8), (11, 2, 0), (9, 4, 8), (9, 4, 1)),
        ("TND", (9, 4, 0), (11, 2, 6), (9, 4, 0), (9, 4, 1)),
        ("NTD", (4, 9, 8), (2, 11, 0), (4, 9, 8), (9, 4, 1)),
        ("NTD", (4, 9, 0), (2, 11, 6), (4, 9, 0), (9, 4, 1)),
        ("NSD", (4, 3, 8), (2, 7, 0), (4, 3, 8), (1, 4, 3, 1)),
        ("NSD", (4, 3, 0), (2, 7, 6), (4, 3, 0), (1, 4, 3, 1)),
    ],
)
def test_fia_empty_dimensions_preserve_native_metadata_rules(layout, query_shape, value_shape, output_shape, lse_shape):
    query, value = meta(query_shape), meta(value_shape)
    outputs = ops.fused_infer_attention_score_v2_fake(
        query,
        key_for_layout(query, value, layout),
        value,
        input_layout=layout,
        num_query_heads=4,
        num_key_value_heads=2,
        return_softmax_lse=True,
    )
    assert_outputs(outputs, output_shape, lse_shape)


@pytest.mark.parametrize("layout", ["NBSD", "invalid", ""])
def test_fia_rejects_unsupported_input_layout(layout):
    query = meta((2, 4, 3, 8))
    with pytest.raises(RuntimeError, match="unsupported.*input_layout"):
        ops.fused_infer_attention_score_v2_fake(query, query, query, input_layout=layout)


@pytest.mark.parametrize("layout", ["BSH", "BNSD", "BSND", "TND", "NTD", "NSD", "BNSD_BSND", "NTD_TND"])
def test_fia_rejects_query_rank_mismatch(layout):
    query = meta((2, 8))
    with pytest.raises(RuntimeError, match="query dim does not match input_layout"):
        ops.fused_infer_attention_score_v2_fake(query, query, query, input_layout=layout)


@pytest.mark.parametrize("heads", [0, -1])
def test_fia_rejects_nonpositive_query_heads(heads):
    query = meta((2, 4, 3, 8))
    with pytest.raises(RuntimeError, match="num_query_heads should be greater than 0"):
        ops.fused_infer_attention_score_v2_fake(query, query, query, input_layout="BNSD", num_query_heads=heads)


def test_fia_rejects_nonpaged_value_rank_mismatch():
    query, value = meta((2, 4, 3, 8)), meta((2, 7, 12))
    with pytest.raises(RuntimeError, match="value dim should equal query dim"):
        ops.fused_infer_attention_score_v2_fake(query, value, value, input_layout="BNSD")


@pytest.mark.parametrize("shape", [(6, 12), (1, 2, 3, 4, 5, 6)])
def test_fia_rejects_unsupported_paged_value_rank(shape):
    query, value = meta((2, 4, 3, 8)), meta(shape)
    with pytest.raises(RuntimeError, match="value dim should be 3, 4, or 5"):
        ops.fused_infer_attention_score_v2_fake(
            query, value, value, input_layout="BNSD", block_table=meta((2, 3), torch.int32)
        )
