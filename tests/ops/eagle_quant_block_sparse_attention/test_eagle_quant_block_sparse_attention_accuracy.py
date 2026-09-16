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

"""Small-shape EQBSA mix accuracy vs CPU dequant compose golden. Device is fixed to npu:0."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from eagle_quant_block_sparse_attention_accuracy_common import (  # noqa: E402
    BLOCK_SHAPE,
    INNER_PRECISE,
    OUTPUT_DTYPE,
    SEED,
    SMALL_ALIGNED_SEQ,
    SMALL_BATCH,
    SMALL_HEAD_DIM,
    SMALL_NUM_KV_HEADS,
    SMALL_NUM_Q_HEADS,
    SMALL_TAIL_SEQ,
    build_eqbsa_prequant_inputs,
    call_eagle_quant_block_sparse_attention,
    check_mixed_tolerance,
    collect_gate_failures,
    cosine_metrics,
    cpu_eqbsa_golden,
    print_accuracy_report,
    quantize_eqbsa_qkv,
    softmax_scale,
)

SMALL_CASES = (
    {"name": "aligned_s256_mix", "seq_len": SMALL_ALIGNED_SEQ},
    {"name": "tail_s257_mix", "seq_len": SMALL_TAIL_SEQ},
)


def _require_950():
    from mindiesd.utils.get_platform import NPUDevice, get_npu_device

    npu = get_npu_device()
    if npu != NPUDevice.A5:
        pytest.skip(f"eagle_quant_block_sparse_attention is an Ascend 950 operator, got {npu}")


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None,
    reason="torch_npu is required for NPU eagle_quant_block_sparse_attention.",
)
@pytest.mark.parametrize("case", SMALL_CASES, ids=[case["name"] for case in SMALL_CASES])
def test_eagle_quant_block_sparse_attention_small_vs_cpu_golden(case):
    import torch_npu

    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available.")
    _require_950()

    torch_npu.npu.set_device(0)
    device = "npu:0"
    seq_len = case["seq_len"]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    scale = softmax_scale(SMALL_HEAD_DIM)
    query, key, value, mask = build_eqbsa_prequant_inputs(
        SMALL_BATCH,
        SMALL_NUM_Q_HEADS,
        SMALL_NUM_KV_HEADS,
        seq_len,
        seq_len,
        SMALL_HEAD_DIM,
        generator,
        BLOCK_SHAPE,
    )
    query_npu = query.to(device)
    key_npu = key.to(device)
    value_npu = value.to(device)
    mask_npu = mask.to(device)
    query_q, key_q, value_q, query_scale, key_scale, value_scale = quantize_eqbsa_qkv(
        query_npu, key_npu, value_npu
    )
    cpu_out = cpu_eqbsa_golden(
        query_q, key_q, value_q, query_scale, key_scale, value_scale, mask, BLOCK_SHAPE, scale
    )
    dut_out, softmax_lse = call_eagle_quant_block_sparse_attention(
        query_q, key_q, value_q, mask_npu, query_scale, key_scale, value_scale, BLOCK_SHAPE, scale
    )
    torch_npu.npu.synchronize()

    assert dut_out.shape == cpu_out.shape
    assert dut_out.dtype == OUTPUT_DTYPE
    assert cpu_out.dtype == OUTPUT_DTYPE
    assert torch.isfinite(dut_out.float()).all()
    if softmax_lse is not None and softmax_lse.numel() != 0:
        assert torch.isfinite(softmax_lse.float()).all()

    metrics = cosine_metrics(cpu_out, dut_out)
    gate = check_mixed_tolerance(dut_out, cpu_out, dtype=torch.float8_e4m3fn)
    print_accuracy_report(
        golden_name="cpu_eqbsa_golden (dequant mix Q/K/V, FP32 compose, last-cast BF16)",
        dut_name=(
            f"eagle_quant_block_sparse_attention ({case['name']}, "
            f"block_shape={list(BLOCK_SHAPE)}, inner_precise={INNER_PRECISE})"
        ),
        metrics=metrics,
        gate=gate,
    )
    failures = collect_gate_failures(metrics, gate)
    assert not failures, "; ".join(failures)
