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

import pytest
import torch

from tests.ops.device_support import init_test_device, require_support
from tests.ops.fused_infer_attention_score.fia_accuracy_common import (
    DEFAULT_ENHANCE_MODE,
    check_mixed_tolerance,
    cosine_metrics,
    cpu_c8v16_fp8_fia_golden,
    synthesize_bf16,
)
from tests.ops.fused_infer_attention_score.fia_quant_common import (
    D128_CHANNEL_BLOCK,
    FIA_QUANT_CASE_BY_NAME,
    K_TOKEN_BLOCK,
    Q_TOKEN_BLOCK,
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


@pytest.fixture(scope="module", name="fia_device")
def fixture_fia_device():
    device = init_test_device()
    require_support("fused_infer_attention_score")
    return device


@pytest.mark.parametrize(
    "case",
    MINIMUM_FP8_PERBLOCK_BNSD_CASES,
    ids=[case["name"] for case in MINIMUM_FP8_PERBLOCK_BNSD_CASES],
)
def test_fused_infer_attention_score_v2_fp8_perblock_bnsd_cases(case, fia_device):
    import torch_npu

    from mindiesd.layers.flash_attn.fused_infer_attention_score import fused_infer_attention_score_v2
    from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

    batch_size = case["batch_size"]
    seq_len = case["seq_len"]
    num_heads = case["num_heads"]
    head_dim = case["head_dim"]

    query = torch.ones(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=torch.float16,
        device=fia_device,
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
    assert torch.count_nonzero(attention_out).item() == 0


@pytest.mark.parametrize(
    "path,num_query_heads,num_kv_heads,query_seq_len,kv_seq_len",
    [
        pytest.param("original", 2, 2, 127, 255, id="original-mha-q127-kv255"),
        pytest.param("c8v16", 4, 2, 128, 256, id="c8v16-gqa-q128-kv256"),
        pytest.param("c8v16", 2, 2, 129, 257, id="c8v16-mha-q129-kv257"),
        pytest.param("v512", 2, 2, 127, 511, id="v512-mha-q127-kv511"),
        pytest.param("v512_d64", 4, 2, 129, 512, id="v512-d64-gqa-q129-kv512"),
        *[pytest.param(path, 8, 2, 128, 513, id=f"{path}-gqa-q128-kv513") for path in FIA_QUANT_CASE_BY_NAME],
    ],
)
def test_fused_infer_attention_score_v2_fp8_small_vs_cpu_golden(
    path, num_query_heads, num_kv_heads, query_seq_len, kv_seq_len, fia_device
):
    import torch_npu

    from mindiesd.layers.flash_attn.fused_infer_attention_score import fused_infer_attention_score_v2
    from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

    case = FIA_QUANT_CASE_BY_NAME[path]
    batch_size = 1
    head_dim = 128
    generator = torch.Generator(device="cpu").manual_seed(20260811)

    query = synthesize_bf16("query", (batch_size, num_query_heads, query_seq_len, head_dim), generator)
    key = synthesize_bf16("key", (batch_size, num_kv_heads, kv_seq_len, head_dim), generator)
    value = synthesize_bf16("value", (batch_size, num_kv_heads, kv_seq_len, head_dim), generator)
    softmax_scale = 1.0 / (head_dim**0.5)
    query_npu = query.to(fia_device)
    key_npu = key.to(fia_device)
    value_npu = value.to(fia_device)
    q, q_scale = fa_block_quant_preprocess(
        query_npu,
        block_size=Q_TOKEN_BLOCK,
        col_block_size=D128_CHANNEL_BLOCK,
        dst_type=torch_npu.float8_e4m3fn,
        layout="BNSD",
    )
    k, k_scale = fa_block_quant_preprocess(
        key_npu,
        block_size=K_TOKEN_BLOCK,
        col_block_size=D128_CHANNEL_BLOCK,
        dst_type=torch_npu.float8_e4m3fn,
        layout="BNSD",
    )
    v, v_scale = fa_block_quant_preprocess(
        value_npu,
        block_size=case.value_token_block,
        col_block_size=case.value_channel_block,
        dst_type=torch_npu.float8_e4m3fn,
        layout="BNSD",
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
        s2_tile=case.value_token_block,
        q_row_block=Q_TOKEN_BLOCK,
        k_row_block=K_TOKEN_BLOCK,
        v_row_block=case.value_token_block,
        qk_col_block=D128_CHANNEL_BLOCK,
        v_col_block=case.value_channel_block,
    )
    fia_kwargs = {
        "num_query_heads": num_query_heads,
        "num_key_value_heads": num_kv_heads,
        "softmax_scale": softmax_scale,
        "pre_tokens": 2147483647,
        "next_tokens": 2147483647,
        "input_layout": "BNSD",
        "query_quant_mode": case.quant_modes[0],
        "key_quant_mode": case.quant_modes[1],
        "value_quant_mode": case.quant_modes[2],
        "inner_precise": case.inner_precise,
        "dequant_scale_query": q_scale,
        "dequant_scale_key": k_scale,
        "dequant_scale_value": v_scale,
        "out_dtype": torch.bfloat16,
    }
    fia_out, softmax_lse = fused_infer_attention_score_v2(q, k, v, **fia_kwargs)

    assert fia_out.shape == cpu_out.shape
    assert cpu_out.dtype == torch.bfloat16
    assert fia_out.dtype == torch.bfloat16
    assert softmax_lse.numel() == 0
    assert torch.isfinite(fia_out.float()).all()

    metrics = cosine_metrics(cpu_out, fia_out)
    gate = check_mixed_tolerance(fia_out, cpu_out, dtype=torch.float8_e4m3fn)
    print(
        f"golden=cpu_c8v16_fp8_fia_golden (C8V16 FP8 FullQuant)  "
        f"dut=fused_infer_attention_score_v2 ({case.description}, "
        f"qkv={case.quant_modes}, inner_precise={case.inner_precise})"
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
