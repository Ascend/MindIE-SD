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

"""Small-shape BSA accuracy vs CPU compose golden. Device is fixed to npu:0."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from eagle_block_sparse_attention_accuracy_common import (  # noqa: E402
    BLOCK_SHAPE,
    BLOCK_SHAPE_K64,
    SEED,
    SMALL_ALIGNED_SEQ,
    SMALL_BATCH,
    SMALL_HEAD_DIM,
    SMALL_NUM_KV_HEADS,
    SMALL_NUM_Q_HEADS,
    SMALL_TAIL_SEQ,
    build_bsa_inputs,
    call_eagle_block_sparse_attention,
    check_mixed_tolerance,
    collect_gate_failures,
    cosine_metrics,
    cpu_block_sparse_attention_golden,
    print_accuracy_report,
    softmax_scale,
)

SMALL_CASES = (
    {
        "name": "aligned_s256_fp16",
        "seq_len": SMALL_ALIGNED_SEQ,
        "dtype": torch.float16,
        "block_shape": BLOCK_SHAPE,
    },
    {
        "name": "aligned_s256_bf16",
        "seq_len": SMALL_ALIGNED_SEQ,
        "dtype": torch.bfloat16,
        "block_shape": BLOCK_SHAPE,
    },
    {
        "name": "tail_s257_fp16",
        "seq_len": SMALL_TAIL_SEQ,
        "dtype": torch.float16,
        "block_shape": BLOCK_SHAPE,
    },
    {
        "name": "tail_s257_bf16",
        "seq_len": SMALL_TAIL_SEQ,
        "dtype": torch.bfloat16,
        "block_shape": BLOCK_SHAPE,
    },
    {
        "name": "aligned_s256_fp16_blockk64",
        "seq_len": SMALL_ALIGNED_SEQ,
        "dtype": torch.float16,
        "block_shape": BLOCK_SHAPE_K64,
    },
    {
        "name": "aligned_s256_bf16_blockk64",
        "seq_len": SMALL_ALIGNED_SEQ,
        "dtype": torch.bfloat16,
        "block_shape": BLOCK_SHAPE_K64,
    },
    {
        "name": "tail_s257_fp16_blockk64",
        "seq_len": SMALL_TAIL_SEQ,
        "dtype": torch.float16,
        "block_shape": BLOCK_SHAPE_K64,
    },
    {
        "name": "tail_s257_bf16_blockk64",
        "seq_len": SMALL_TAIL_SEQ,
        "dtype": torch.bfloat16,
        "block_shape": BLOCK_SHAPE_K64,
    },
)


def _require_910b():
    from mindiesd.utils.get_platform import NPUDevice, get_npu_device

    npu = get_npu_device()
    if npu not in (NPUDevice.A2, NPUDevice.A3):
        pytest.skip(f"eagle_block_sparse_attention is a 910B/910_93 operator, got {npu}")


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None,
    reason="torch_npu is required for NPU eagle_block_sparse_attention.",
)
@pytest.mark.parametrize("case", SMALL_CASES, ids=[case["name"] for case in SMALL_CASES])
def test_eagle_block_sparse_attention_small_vs_cpu_golden(case):
    import torch_npu

    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available.")
    _require_910b()

    torch_npu.npu.set_device(0)
    device = "npu:0"
    dtype = case["dtype"]
    seq_len = case["seq_len"]
    block_shape = case["block_shape"]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    scale = softmax_scale(SMALL_HEAD_DIM)
    query, key, value, mask = build_bsa_inputs(
        SMALL_BATCH,
        SMALL_NUM_Q_HEADS,
        SMALL_NUM_KV_HEADS,
        seq_len,
        seq_len,
        SMALL_HEAD_DIM,
        dtype,
        generator,
        block_shape,
    )
    cpu_out = cpu_block_sparse_attention_golden(query, key, value, mask, block_shape, scale)
    query_npu = query.to(device)
    key_npu = key.to(device)
    value_npu = value.to(device)
    mask_npu = mask.to(device)
    dut_out, softmax_lse = call_eagle_block_sparse_attention(
        query_npu, key_npu, value_npu, mask_npu, block_shape, scale
    )
    torch_npu.npu.synchronize()

    assert dut_out.shape == cpu_out.shape
    assert dut_out.dtype == dtype
    assert cpu_out.dtype == dtype
    assert torch.isfinite(dut_out.float()).all()
    if softmax_lse is not None and softmax_lse.numel() != 0:
        assert torch.isfinite(softmax_lse.float()).all()

    metrics = cosine_metrics(cpu_out, dut_out)
    gate = check_mixed_tolerance(dut_out, cpu_out, dtype=dtype)
    print_accuracy_report(
        golden_name="cpu_block_sparse_attention_golden (FP32 compose, last-cast compute dtype)",
        dut_name=f"eagle_block_sparse_attention ({case['name']}, block_shape={list(block_shape)}, inner_precise=0)",
        metrics=metrics,
        gate=gate,
    )
    failures = collect_gate_failures(metrics, gate)
    assert not failures, "; ".join(failures)
