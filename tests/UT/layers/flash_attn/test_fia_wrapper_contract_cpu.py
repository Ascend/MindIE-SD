# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
import sys
import types
import pytest
import torch
from mindiesd.utils.exception import ParametersInvalid
from mindiesd.layers.flash_attn.fused_infer_attention_score import _normalize_dtype_arg, fused_infer_attention_score_v2


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
