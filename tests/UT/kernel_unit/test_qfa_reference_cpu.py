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

"""CPU regressions for the dense MXFP4 attention references."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.ops.quant_flash_attn.qfa_reference import attention_rows


def test_qfa_reference_keeps_e8m0_floor_within_one_rescale_group():
    case = SimpleNamespace(
        batch=1, q_heads=1, kv_heads=1, physical_q=1, physical_kv=512, q_used=(1,), kv_used=(512,), dim=128, scale=1
    )
    query = np.ones((1, 1, 1, 128), dtype=np.float32)
    key = np.zeros((1, 1, 512, 128), dtype=np.float32)
    key[:, :, 256:, :100] = 1
    value = np.zeros_like(key)
    value[:, :, :256] = 2**100
    actual = attention_rows(case, {"q": query, "k": key, "v": value}, [(0, 0, 0)])
    np.testing.assert_array_equal(actual, np.full((1, 128), 2.0**-25))


def test_qfa_debug_golden_prescales_dumped_logits(tmp_path, monkeypatch):
    import torch

    debug_dir = Path(__file__).resolve().parents[2] / "ops" / "quant_flash_attn"
    monkeypatch.syspath_prepend(str(debug_dir))
    from tests.ops.quant_flash_attn import flash_attention_cpu_golden as golden

    monkeypatch.chdir(tmp_path)
    (tmp_path / "dump_data").mkdir()
    query = torch.full((1, 128, 128), 24.0)
    key = torch.full((1, 256, 128), 24.0)
    key[:, 128:, 0] = 16.0

    golden._flash_attn_single_batch_kernel(
        query,
        key,
        torch.ones_like(key),
        1 / 256,
        False,
        0,
        128,
        256,
        True,
        32,
        "baseline",
        s_layout="DN",
        quantize_p_mode="blockwise_snap_local",
        s_dtype="fp16",
    )
    logits = np.fromfile(tmp_path / "dump_data/mm1Res.bin", dtype=np.float16).reshape(256, 128)
    np.testing.assert_array_equal(logits[:128], np.full((128, 128), 288, dtype=np.float16))
    np.testing.assert_array_equal(logits[128:], np.full((128, 128), 287.25, dtype=np.float16))
