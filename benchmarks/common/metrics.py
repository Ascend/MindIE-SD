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

"""Metric accounting shared by the runtime summary and the offline report.

MFU/MBU use one formula (`util_metrics`) so the runtime Mixin
(op_defs/_common.py) and the offline recompute (benchmark_report.py) cannot
drift apart when the accounting convention changes.

The CUBE peak flops / peak bandwidth are NOT hard-coded here: the user
provides them per run via --config ({"peak_flops": ..., "peak_bw": ...}),
each case carries them, and MFU/MBU are computed from the case's own peaks.
"""


def util_metrics(flops_power, mem_bw, peak_flops, peak_bw):
    """Return (MFU, MBU) rounded to 4 decimals, None when a peak is missing.

    Args:
        flops_power: calc_flops_power(tflops) or None.
        mem_bw: mem_bw(GB/s) or None.
        peak_flops / peak_bw: device peaks (tflops / GB/s) or None
            (peak_flops = CUBE peak flops).

    MFU/MBU are clamped to <= 1.0: a value above 1 means the measured path
    exceeds the accounted peak (e.g. quantized tiers whose real CUBE
    throughput is higher than the bf16 peak); clamping keeps reports readable.
    """
    mfu = round(flops_power / peak_flops, 4) if peak_flops and flops_power is not None else None
    mbu = round(mem_bw / peak_bw, 4) if peak_bw and mem_bw is not None else None
    if mfu is not None:
        mfu = min(mfu, 1.0)
    if mbu is not None:
        mbu = min(mbu, 1.0)
    return mfu, mbu
