#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import importlib.util
import sys
import types

import pytest
import torch

from mindiesd.layers.flash_attn.fused_infer_attention_score import (
    _normalize_dtype_arg,
    fused_infer_attention_score_v2,
)
from mindiesd.utils.exception import ParametersInvalid


MINIMUM_BNSD_SD_CASES = (
    {
        "name": "B1_N1_S128_D128",
        "batch_size": 1,
        "seq_len": 128,
        "num_heads": 1,
        "head_dim": 128,
    },
    {
        "name": "B2_N1_S256_D128",
        "batch_size": 2,
        "seq_len": 256,
        "num_heads": 1,
        "head_dim": 128,
    },
)


def test_fused_infer_attention_score_v2_routes_to_mindiesd_op(monkeypatch):
    calls = []

    def fake_fia(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return query, torch.empty(0, dtype=torch.float32)

    monkeypatch.setattr(
        torch.ops.mindiesd,
        "fused_infer_attention_score_v2",
        fake_fia,
        raising=False,
    )

    query = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    key = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    value = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    out, lse = fused_infer_attention_score_v2(
        query,
        key,
        value,
        input_layout="BNSD",
        num_query_heads=1,
        quant_scale_p=1.0,
        query_dtype=torch.float16,
        key_dtype=torch.float16,
        dequant_scale_query_dtype=torch.float32,
        out_dtype=torch.float16,
    )

    assert out is query
    assert lse.numel() == 0
    assert len(calls) == 1
    assert "quant_scale_p" not in calls[0][3]
    assert calls[0][3]["query_dtype"] is None
    assert calls[0][3]["key_dtype"] is None
    assert calls[0][3]["dequant_scale_query_dtype"] is None
    assert calls[0][3]["out_dtype"] is torch.float16


def test_fused_infer_attention_score_v2_rejects_non_tensor_query():
    key = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    value = torch.empty(1, 1, 4, 8, dtype=torch.float16)

    with pytest.raises(ParametersInvalid, match="input query must be torch.Tensor"):
        fused_infer_attention_score_v2(
            None,
            key,
            value,
            input_layout="BNSD",
            num_query_heads=1,
        )


def test_fused_infer_attention_score_v2_rejects_unsupported_layout():
    query = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    key = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    value = torch.empty(1, 1, 4, 8, dtype=torch.float16)

    with pytest.raises(ParametersInvalid, match="input_layout"):
        fused_infer_attention_score_v2(
            query,
            key,
            value,
            input_layout="NTD_TND",
            num_query_heads=1,
        )


def test_fused_infer_attention_score_v2_rejects_bnsd_head_mismatch():
    query = torch.empty(1, 2, 4, 8, dtype=torch.float16)
    key = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    value = torch.empty(1, 1, 4, 8, dtype=torch.float16)

    with pytest.raises(ParametersInvalid, match="head num of input query"):
        fused_infer_attention_score_v2(
            query,
            key,
            value,
            input_layout="BNSD",
            num_query_heads=1,
        )


def test_fused_infer_attention_score_v2_rejects_bool_head_num():
    query = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    key = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    value = torch.empty(1, 1, 4, 8, dtype=torch.float16)

    with pytest.raises(ParametersInvalid, match="num_query_heads must be int"):
        fused_infer_attention_score_v2(
            query,
            key,
            value,
            input_layout="BNSD",
            num_query_heads=True,
        )


def test_fused_infer_attention_score_v2_rejects_key_value_seq_mismatch():
    query = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    key = torch.empty(1, 1, 4, 8, dtype=torch.float16)
    value = torch.empty(1, 1, 5, 8, dtype=torch.float16)

    with pytest.raises(ParametersInvalid, match="sequence length of key/value"):
        fused_infer_attention_score_v2(
            query,
            key,
            value,
            input_layout="BNSD",
            num_query_heads=1,
        )


def test_normalize_dtype_arg_maps_torch_npu_pseudo_dtypes(monkeypatch):
    fake_torch_npu = types.SimpleNamespace(
        hifloat8=object(),
        float8_e8m0fnu=object(),
        float4_e2m1fn_x2=object(),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    assert _normalize_dtype_arg(fake_torch_npu.hifloat8) == 290
    assert _normalize_dtype_arg(fake_torch_npu.float8_e8m0fnu) == 293
    assert _normalize_dtype_arg(fake_torch_npu.float4_e2m1fn_x2) == 296


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None,
    reason="torch_npu is required for NPU fused_infer_attention_score_v2 validation.",
)
@pytest.mark.parametrize(
    "case",
    MINIMUM_BNSD_SD_CASES,
    ids=[case["name"] for case in MINIMUM_BNSD_SD_CASES],
)
def test_fused_infer_attention_score_v2_minimum_bnsd_sd_cases(case):
    import torch_npu

    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available.")

    torch_npu.npu.set_device(0)
    batch_size = case["batch_size"]
    seq_len = case["seq_len"]
    num_heads = case["num_heads"]
    head_dim = case["head_dim"]

    query = torch.zeros(
        (batch_size, seq_len, num_heads, head_dim),
        dtype=torch.float16,
        device="npu:0",
    )
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)

    attention_out, softmax_lse = fused_infer_attention_score_v2(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        num_query_heads=num_heads,
        softmax_scale=1.0 / (head_dim**0.5),
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="BNSD",
        out_dtype=torch.float16,
    )

    assert attention_out.shape == (batch_size, num_heads, seq_len, head_dim)
    assert attention_out.dtype is torch.float16
    assert softmax_lse.numel() == 0
