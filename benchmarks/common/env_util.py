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

"""Shared env.json parsing for peak_flops / peak_bw.

Single source of truth for MFU/MBU peak resolution, used by both BackendNPU
(runtime accounting) and benchmark_report.py (offline recompute / compare), so
the two can never disagree about which device entry is active.

Resolution: start from the "common" entry, then overlay the entry matching
`device_name` if provided and present; otherwise fall back to the first
non-common entry. Missing file or empty peak values yield None peaks.
"""

import json


def load_peaks(env_file, device_name=None):
    """Return (peak_flops, peak_bw) floats from env_file, or (None, None).

    Args:
        env_file: path to env.json ({common: {...}, <device>: {...}}).
        device_name: optional exact device key to prefer.
    """
    if not env_file:
        return None, None
    try:
        with open(env_file, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None

    merged = dict(data.get("common", {}) or {})
    device_entry = data.get(device_name) if device_name else None
    if device_entry is not None:
        merged.update(device_entry)
    else:
        for key, val in data.items():
            if key == "common":
                continue
            merged.update(val)
            break
    return _to_float(merged.get("peak_flops")), _to_float(merged.get("peak_bw"))


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
