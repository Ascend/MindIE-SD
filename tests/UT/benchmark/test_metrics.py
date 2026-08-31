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

"""Unit tests for common.metrics.util_metrics."""

from common.metrics import util_metrics


def test_normal_case():
    # 280/560 = 0.5, 637.5/1275 = 0.5
    assert util_metrics(280.0, 637.5, 560.0, 1275.0) == (0.5, 0.5)


def test_rounding_to_4_decimals():
    mfu, mbu = util_metrics(1.0, 1.0, 3.0, 3.0)
    assert mfu == round(1 / 3, 4)
    assert mbu == round(1 / 3, 4)


def test_missing_flops_yields_none_mfu():
    assert util_metrics(None, 637.5, 560.0, 1275.0) == (None, 0.5)


def test_missing_peak_yields_none():
    assert util_metrics(280.0, 637.5, None, 1275.0) == (None, 0.5)
    assert util_metrics(280.0, 637.5, 560.0, None) == (0.5, None)


def test_zero_peak_yields_none():
    # peak 0 is treated as missing (falsy), not a division-by-zero.
    assert util_metrics(280.0, 637.5, 0.0, 0.0) == (None, None)


def test_zero_bw_yields_zero_mbu():
    # mem_bw=0.0 is a real zero (io_bytes=0), not a missing value: the formula
    # yields MBU 0.0. A missing mem_bw (None) yields None instead.
    assert util_metrics(280.0, 0.0, 560.0, 1275.0) == (0.5, 0.0)


def test_consistent_with_readme_example():
    # README: MFU = calc_flops_power(tflops) / peak_flops, MBU = mem_bw / peak_bw.
    # Ascend910_9382 env: peak_flops=560, peak_bw=1275.
    assert util_metrics(560.0, 1275.0, 560.0, 1275.0) == (1.0, 1.0)


def test_clamped_to_one_when_peak_accounting_misses():
    # flops above peak (e.g. missing per-dtype peak for mxfp4) must not
    # produce MFU > 1 in reports.
    assert util_metrics(700.0, 2000.0, 560.0, 1275.0) == (1.0, 1.0)
    assert util_metrics(1120.0, 637.5, 560.0, 1275.0) == (1.0, 0.5)


def test_clamp_applies_after_rounding():
    mfu, mbu = util_metrics(561.0, 1276.0, 560.0, 1275.0)
    assert mfu == 1.0
    assert mbu == 1.0
