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

"""Small-shape Laser Attention accuracy vs CPU kernel-stage golden. Device is fixed to npu:0."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from laser_attention_accuracy_common import (  # noqa: E402
    ACCURACY_DISTRIBUTIONS,
    COMPUTE_DTYPE,
    MAX_TOKEN,
    PROD_DTYPE,
    SEED,
    SMALL_ALIGNED_KV_SEQ,
    SMALL_ALIGNED_Q_SEQ,
    SMALL_BATCH,
    SMALL_HEAD_DIM,
    SMALL_NUM_HEADS,
    SMALL_MASKED_KV_SEQ,
    SMALL_MASKED_Q_SEQ,
    SMALL_MASKED_PRE_TOKENS,
    build_laser_attention_inputs,
    call_laser_attention,
    check_mixed_tolerance,
    collect_gate_failures,
    cosine_metrics,
    cpu_laser_attention_golden,
    kernel_output_dtype,
    print_operator_inputs,
    print_accuracy_report,
    softmax_scale,
)

SMALL_CASES = (
    {
        "name": "aligned_q256_kv512_fp16",
        "q_seq": SMALL_ALIGNED_Q_SEQ,
        "kv_seq": SMALL_ALIGNED_KV_SEQ,
        "dtype": COMPUTE_DTYPE,
        "pre_tokens": MAX_TOKEN,
    },
    {
        "name": "masked_q512_kv768_fp16",
        "q_seq": SMALL_MASKED_Q_SEQ,
        "kv_seq": SMALL_MASKED_KV_SEQ,
        "dtype": COMPUTE_DTYPE,
        "pre_tokens": SMALL_MASKED_PRE_TOKENS,
    },
    {
        "name": "aligned_q256_kv512_native_bf16",
        "q_seq": SMALL_ALIGNED_Q_SEQ,
        "kv_seq": SMALL_ALIGNED_KV_SEQ,
        "dtype": PROD_DTYPE,
        "pre_tokens": MAX_TOKEN,
    },
    {
        "name": "masked_q512_kv768_native_bf16",
        "q_seq": SMALL_MASKED_Q_SEQ,
        "kv_seq": SMALL_MASKED_KV_SEQ,
        "dtype": PROD_DTYPE,
        "pre_tokens": SMALL_MASKED_PRE_TOKENS,
    },
)


def _skip_a5():
    from mindiesd.utils.get_platform import NPUDevice, get_npu_device

    npu = get_npu_device()
    if npu == NPUDevice.A5:
        pytest.skip(f"ascend_laser_attention is not supported on A5, got {npu}")


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None,
    reason="torch_npu is required for NPU laser_attention.",
)
@pytest.mark.parametrize("case", SMALL_CASES, ids=[case["name"] for case in SMALL_CASES])
@pytest.mark.parametrize("distribution", ACCURACY_DISTRIBUTIONS)
def test_laser_attention_small_vs_cpu_golden(case, distribution):
    import torch_npu

    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available.")
    _skip_a5()

    torch_npu.npu.set_device(0)
    device = "npu:0"
    dtype = case["dtype"]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    scale = softmax_scale(SMALL_HEAD_DIM)
    print(f"scenario={case['name']} distribution={distribution} seed={SEED}")
    query, key, value = build_laser_attention_inputs(
        SMALL_BATCH,
        SMALL_NUM_HEADS,
        case["q_seq"],
        case["kv_seq"],
        SMALL_HEAD_DIM,
        dtype,
        generator,
        distribution=distribution,
        log_distribution=True,
    )
    cpu_out = cpu_laser_attention_golden(query, key, value, scale, pre_tokens=case["pre_tokens"])
    query_npu = query.to(device)
    key_npu = key.to(device)
    value_npu = value.to(device)
    print_operator_inputs(query_npu, key_npu, value_npu, scale, case["pre_tokens"])
    with torch.inference_mode():
        dut_out = call_laser_attention(query_npu, key_npu, value_npu, scale, case["pre_tokens"])
    gate_dtype = dtype
    golden_name = "cpu_laser_attention_golden (same BNSD inputs, FP16 kernel stages, FP32 PV/normalize)"
    dut_name = f"torch.ops.mindiesd.la ({case['name']}, {distribution}, BNSD)"
    torch_npu.npu.synchronize()

    assert dut_out.shape == cpu_out.shape
    assert dut_out.dtype == kernel_output_dtype(dtype)
    assert cpu_out.dtype == kernel_output_dtype(dtype)
    assert torch.isfinite(dut_out.float()).all()

    metrics = cosine_metrics(cpu_out, dut_out)
    gate = check_mixed_tolerance(dut_out, cpu_out, dtype=gate_dtype)
    print_accuracy_report(
        golden_name=golden_name,
        dut_name=dut_name,
        metrics=metrics,
        gate=gate,
    )
    failures = collect_gate_failures(metrics, gate)
    assert not failures, "; ".join(failures)
