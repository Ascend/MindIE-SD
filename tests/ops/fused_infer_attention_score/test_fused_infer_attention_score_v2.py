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
import os
import sys
import types

import pytest
import torch

from mindiesd.layers.flash_attn.fused_infer_attention_score import (
    _normalize_dtype_arg,
    fused_infer_attention_score_v2,
)
from mindiesd.utils.exception import ParametersInvalid

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from fia_accuracy_common import (  # noqa: E402
    DEFAULT_ENHANCE_MODE,
    K_BLOCK,
    Q_BLOCK,
    QUANT_MODE,
    V_BLOCK,
    check_mixed_tolerance,
    cosine_metrics,
    cpu_c8v16_fp8_fia_golden,
    synthesize_bf16,
)


MINIMUM_FP8_PERBLOCK_BNSD_CASES = (
    {
        "name": "B1_N1_S128_D128_FP16_OUT",
        "batch_size": 1,
        "seq_len": 128,
        "num_heads": 1,
        "head_dim": 128,
        "out_dtype": torch.float16,
    },
    {
        "name": "B1_N1_S256_D128_BF16_OUT",
        "batch_size": 1,
        "seq_len": 256,
        "num_heads": 1,
        "head_dim": 128,
        "out_dtype": torch.bfloat16,
    },
    {
        "name": "B1_N1_S128_D64_FP16_OUT",
        "batch_size": 1,
        "seq_len": 128,
        "num_heads": 1,
        "head_dim": 64,
        "out_dtype": torch.float16,
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
    MINIMUM_FP8_PERBLOCK_BNSD_CASES,
    ids=[case["name"] for case in MINIMUM_FP8_PERBLOCK_BNSD_CASES],
)
def test_fused_infer_attention_score_v2_fp8_perblock_bnsd_cases(case):
    import torch_npu
    from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available.")

    torch_npu.npu.set_device(0)
    batch_size = case["batch_size"]
    seq_len = case["seq_len"]
    num_heads = case["num_heads"]
    head_dim = case["head_dim"]

    query = torch.ones(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=torch.float16,
        device="npu:0",
    )
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    out_dtype = case["out_dtype"]
    q, q_scale = fa_block_quant_preprocess(query, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout="BNSD")
    k, k_scale = fa_block_quant_preprocess(key, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BNSD")
    v, v_scale = fa_block_quant_preprocess(value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BNSD")

    attention_out, softmax_lse = fused_infer_attention_score_v2(
        q,
        k,
        v,
        num_query_heads=num_heads,
        softmax_scale=1.0 / (head_dim**0.5),
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="BNSD",
        query_quant_mode=7,
        key_quant_mode=7,
        value_quant_mode=7,
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=out_dtype,
    )

    assert attention_out.shape == (batch_size, num_heads, seq_len, head_dim)
    assert attention_out.dtype == out_dtype
    assert softmax_lse.numel() == 0


def _run_fp8_small_vs_cpu_four_stage(inner_precise=None):
    import torch_npu
    from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available.")

    torch_npu.npu.set_device(0)
    device = "npu:0"
    batch_size = 1
    num_query_heads = 8
    num_kv_heads = 2
    query_seq_len = 128
    kv_seq_len = 256
    head_dim = 128
    generator = torch.Generator(device="cpu").manual_seed(20260811)

    query = synthesize_bf16(
        "query", (batch_size, num_query_heads, query_seq_len, head_dim), generator
    )
    key = synthesize_bf16(
        "key", (batch_size, num_kv_heads, kv_seq_len, head_dim), generator
    )
    value = synthesize_bf16(
        "value", (batch_size, num_kv_heads, kv_seq_len, head_dim), generator
    )
    softmax_scale = 1.0 / (head_dim**0.5)
    query_npu = query.to(device)
    key_npu = key.to(device)
    value_npu = value.to(device)
    q, q_scale = fa_block_quant_preprocess(
        query_npu, block_size=Q_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
    )
    k, k_scale = fa_block_quant_preprocess(
        key_npu, block_size=K_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
    )
    v, v_scale = fa_block_quant_preprocess(
        value_npu, block_size=V_BLOCK, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
    )
    cpu_out = cpu_c8v16_fp8_fia_golden(
        q.float().cpu(),
        k.float().cpu(),
        v.float().cpu(),
        q_scale.cpu(),
        k_scale.cpu(),
        v_scale.cpu(),
        softmax_scale,
        out_dtype=torch.bfloat16,
    )
    fia_kwargs = {
        "num_query_heads": num_query_heads,
        "num_key_value_heads": num_kv_heads,
        "softmax_scale": softmax_scale,
        "pre_tokens": 2147483647,
        "next_tokens": 2147483647,
        "input_layout": "BNSD",
        "query_quant_mode": QUANT_MODE,
        "key_quant_mode": QUANT_MODE,
        "value_quant_mode": QUANT_MODE,
        "dequant_scale_query": q_scale,
        "dequant_scale_key": k_scale,
        "dequant_scale_value": v_scale,
        "out_dtype": torch.bfloat16,
    }
    if inner_precise is not None:
        fia_kwargs["inner_precise"] = inner_precise
    fia_out, softmax_lse = fused_infer_attention_score_v2(q, k, v, **fia_kwargs)

    assert fia_out.shape == cpu_out.shape
    assert cpu_out.dtype == torch.bfloat16
    assert fia_out.dtype == torch.bfloat16
    assert softmax_lse.numel() == 0
    assert torch.isfinite(fia_out.float()).all()

    metrics = cosine_metrics(cpu_out, fia_out)
    gate = check_mixed_tolerance(fia_out, cpu_out, dtype=torch.float8_e4m3fn)
    precise_label = "0 (default)" if inner_precise is None else str(inner_precise)
    print(
        f"golden=cpu_c8v16_fp8_fia_golden (C8V16 FP8 FullQuant)  "
        f"dut=fused_infer_attention_score_v2 (FP8 7/7/7 inner_precise={precise_label})"
    )
    print(
        f"cosine={metrics['cosine']:.8f}  "
        f"max_abs_error={metrics['max_abs_error']:.6f}  "
        f"norm_ratio={metrics['norm_ratio']:.6f}  "
        f"enhance={DEFAULT_ENHANCE_MODE}"
    )
    print(
        f"mixed_tolerance={gate['result']}  dtype={gate['dtype_key']}  "
        f"matched_ratio={gate['matched_ratio'] * 100.0:.4f}%  "
        f"max_abs={gate['max_abs_error']:.6f}  "
        f"rtol={gate['rtol']} atol={gate['atol']}  "
        f"required_ratio={gate['required_matched_ratio']}  "
        f"max_abs_limit={gate['max_abs_error_limit']}"
    )
    assert gate["result"] == "Pass", (
        f"opbase mixed tolerance {gate['result']}: "
        f"dtype={gate['dtype_key']} matched_ratio={gate['matched_ratio'] * 100.0:.4f}% "
        f"max_abs={gate['max_abs_error']:.6f} limit={gate['max_abs_error_limit']} "
        f"rtol={gate['rtol']} atol={gate['atol']}"
    )
    assert metrics["cosine"] >= 0.99, (
        f"cosine {metrics['cosine']:.8f} < 0.99  "
        f"max_abs={metrics['max_abs_error']:.6f}  "
        f"norm_ratio={metrics['norm_ratio']:.6f}"
    )
    assert 0.9 <= metrics["norm_ratio"] <= 1.1, (
        f"norm_ratio {metrics['norm_ratio']:.6f} is outside [0.9, 1.1]  "
        f"cosine={metrics['cosine']:.8f}  "
        f"max_abs={metrics['max_abs_error']:.6f}"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None,
    reason="torch_npu is required to run MindIE-SD quantized FIA.",
)
def test_fused_infer_attention_score_v2_fp8_small_vs_cpu_four_stage():
    _run_fp8_small_vs_cpu_four_stage()


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None,
    reason="torch_npu is required to run MindIE-SD quantized FIA.",
)
def test_fused_infer_attention_score_v2_fp8_small_vs_cpu_four_stage_inner_precise4():
    _run_fp8_small_vs_cpu_four_stage(inner_precise=4)
