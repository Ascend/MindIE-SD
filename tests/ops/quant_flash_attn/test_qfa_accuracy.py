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

# pylint: disable=redefined-outer-name

"""Accuracy and regression tests for direct and public dense MXFP4 QFA."""

import pytest

from tests.ops.device_support import init_test_device, require_support

from tests.ops.quant_flash_attn.qfa_accuracy_common import (
    Case,
    bind,
    check_outputs,
    compare,
    cpu_golden,
    make_inputs,
)


@pytest.fixture(scope="module")
def qfa_device():
    require_support("quant_flash_attn")
    device = init_test_device()
    import torch
    import torch_npu
    import mindiesd  # noqa: F401  # Register native operators after selecting a visible device.

    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    return torch, torch_npu, str(device)


@pytest.mark.parametrize("layout", ("BNSD", "BSND"))
@pytest.mark.parametrize(
    "q_seq,kv_seq,q_valid,kv_valid",
    (
        pytest.param(128, 256, 128, 256, id="native-q128-kv256"),
        pytest.param(129, 257, 129, 257, id="native-q129-kv257"),
        pytest.param(129, 513, 129, 513, id="native-q129-kv513"),
        pytest.param(512, 512, 128, 256, id="pad512-q128-kv256"),
        pytest.param(512, 512, 129, 257, id="pad512-q129-kv257"),
        pytest.param(512, 512, 511, 511, id="pad512-q511-kv511"),
        pytest.param(512, 512, 512, 512, id="pad512-q512-kv512"),
        pytest.param(1024, 1024, 513, 513, id="pad512-q513-kv513"),
    ),
)
@pytest.mark.parametrize("length_mode", ("used",))
@pytest.mark.parametrize("distribution", ("uniform", "normal"))
def test_qfa_small_accuracy(qfa_device, layout, q_seq, kv_seq, q_valid, kv_valid, length_mode, distribution):
    from tests.ops.quant_flash_attn.qfa_tolerance import BUDGET

    # Mixed per-batch lengths exercise the shared scheduler's cost accounting.
    q_used = (q_valid, 65)
    kv_used = (kv_valid, 129)
    case = Case(
        batch=2,
        q_heads=4,
        kv_heads=2,
        q_seq=q_seq,
        kv_seq=kv_seq,
        layout=layout,
        length_mode=length_mode,
        q_used=q_used,
        kv_used=kv_used,
    )
    torch, torch_npu, device = qfa_device
    with torch.inference_mode():
        native, encodings = make_inputs(case, distribution, 21, torch, device)
        invoke, _ = bind(case, native, torch, torch_npu, device)
        reference = cpu_golden(case, encodings, torch)
        output = invoke()
        torch.npu.synchronize()
        check_outputs(output, case, torch, device)
        # Check valid rows separately so cleared padding cannot dilute the metrics.
        for batch, valid in enumerate(case.q_used):
            actual = output[0][batch, :, :valid, :] if layout == "BNSD" else output[0][batch, :valid, :, :]
            golden = reference[batch, :, :valid, :] if layout == "BNSD" else reference[batch, :valid, :, :]
            padding = output[0][batch, :, valid:, :] if layout == "BNSD" else output[0][batch, valid:, :, :]
            print(f"valid rows: batch={batch}, q_used={valid}, kv_used={case.kv_used[batch]}", flush=True)
            compare(actual, golden, BUDGET, small=True)
            assert torch.count_nonzero(padding).item() == 0, f"Nonzero query padding in batch {batch}"
        compare(output[0], reference, BUDGET, small=True)


def test_qfa_prescales_qk_before_fp16_cast(qfa_device):
    from mindiesd import quant_attention

    torch, _, device = qfa_device
    with torch.inference_mode():
        query = torch.full((1, 130, 14, 128), 24, dtype=torch.bfloat16, device=device)
        key = query.new_full((1, 64, 14, 128), 24)
        key[:, 32:, :, 0] = 16
        value = torch.ones_like(key)
        value[:, 32:] = -1

        # The two groups have raw logits 73728 and 73536, scaled to 288 and 287.25.
        actual = quant_attention(query, key, value, precision="mxfp4", layout="BSND", scale=1 / 256)
        control = quant_attention(query / 2, key, value, precision="mxfp4", layout="BSND", scale=1 / 128)

        # Both groups round to the same FP4 probability with E8M0 scales in a 2:1 ratio.
        expected = torch.full_like(actual, 1 / 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, control, rtol=0, atol=0)


def test_qfa_regenerates_legacy_padded_metadata(qfa_device):
    from mindiesd import prepare_mxfp4_attention_metadata, quant_attention

    torch, _, device = qfa_device
    padded_query = torch.ones((1, 1, 512, 128), dtype=torch.bfloat16, device=device)
    padded_key = padded_query.neg()
    with torch.inference_mode():
        old_metadata = prepare_mxfp4_attention_metadata(padded_query, padded_key).tensor
        query = padded_query[:, :, :17]
        key = padded_key[:, :, :23]
        value = torch.ones_like(key)
        actual = quant_attention(query, key, value, precision="mxfp4", metadata=old_metadata)
        torch.testing.assert_close(actual, torch.ones_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize("layout", ("BNSD", "BSND"))
def test_qfa_partial_group_with_reused_metadata(qfa_device, layout):
    from mindiesd import prepare_mxfp4_attention_metadata, quant_attention

    torch, _, device = qfa_device
    query = torch.ones((1, 130, 2, 128), dtype=torch.bfloat16, device=device)
    query[:, 1::2] *= -1
    query[:, :, 1] *= -1
    key = query.new_ones((1, 33, 1, 128))
    key[:, -1] = -1
    if layout == "BNSD":
        query, key = (tensor.transpose(1, 2).contiguous() for tensor in (query, key))
    with torch.inference_mode():
        cached = prepare_mxfp4_attention_metadata(query, key, layout=layout)
        kwargs = {"precision": "mxfp4", "layout": layout, "scale": 1 / 128}
        automatic = quant_attention(query, key, key, **kwargs)
        reused = quant_attention(query, key, key, metadata=cached, **kwargs)
        # The logits are +1/-1. Both groups quantize to 6 and differ in E8M0 scale by 8x.
        expected = torch.where(query > 0, query.new_tensor(255 / 257), query.new_tensor(3 / 5))
        torch.testing.assert_close(automatic, expected, atol=0, rtol=0)
        torch.testing.assert_close(reused, automatic, atol=0, rtol=0)


def test_qfa_rescales_history_after_4096_keys(qfa_device):
    from mindiesd import quant_attention

    torch, _, device = qfa_device
    query = torch.full((1, 130, 14, 128), 16, dtype=torch.bfloat16, device=device)
    query[..., 0] = 8
    query[..., 1] = 24
    key = query.new_full((1, 4352, 14, 128), 16)
    key[:, :4096, :, 0] = 12
    value = torch.ones_like(key)
    value[:, 4096:] = -1
    with torch.inference_mode():
        actual = quant_attention(query, key, value, precision="mxfp4", layout="BSND", scale=1 / 16)
        # Scaled scores 2046/2048 quantize to equal FP4 mantissas. A -3 exponent
        # delta gives old:new weight 2:1, while the old -2 result gives 4:1.
        torch.testing.assert_close(actual, torch.full_like(actual, 1 / 3), atol=0, rtol=0)
