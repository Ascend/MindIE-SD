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

"""Legacy env.json peak parsing (kept for compatibility, NO production callers).

Peak_flops / peak_bw are now provided per run via --config (each case carries
them in its arguments); this module's load_peaks / _merged_env helpers are
retained as an unused compatibility layer (covered by tests) and document the
old env.json format for reference:

    --env vendor_ops/NPU/env.json
    --env '{"peak_flops": 560, "peak_bw": 1275}'

Resolution (file form): start from the "common" entry, then overlay the entry
matching `device_name` if provided and present; otherwise fall back to the
first non-common entry. Missing file / unparsable input / empty peak values
yield None peaks.
"""

import json
import os


def _merged_env(env_spec, device_name=None):
    """Resolve the active env entry into a merged dict (common + device).

    With a ``device_name`` the matching device entry overlays ``common``.
    Without one, ``common`` acts as the default-device entry when it carries
    peaks; otherwise the first non-common device entry is used (legacy shape).
    """
    data = _load_env_data(env_spec)
    if not isinstance(data, dict):
        return {}
    merged = dict(data.get("common", {}) or {})
    device_entry = data.get(device_name) if device_name else None
    if device_entry is not None:
        merged.update(device_entry)
    elif "peak_flops" in data or "peak_bw" in data:
        # Flat inline JSON: {"peak_flops": 560, "peak_bw": 1275}.
        merged.update(data)
    elif "peak_flops" not in merged and "peak_bw" not in merged:
        # No device match and common carries no peaks: legacy fallback to the
        # first non-common device entry.
        for key, val in data.items():
            if key == "common":
                continue
            if isinstance(val, dict):
                merged.update(val)
                break
    return merged


def load_peaks(env_spec, device_name=None):
    """Return (peak_flops, peak_bw) floats from env_spec, or (None, None).

    Args:
        env_spec: env.json path or inline JSON string
            ({"peak_flops": <float>, "peak_bw": <float>}).
        device_name: optional exact device key to prefer (file form only).

    ``peak_flops`` is the CUBE (AI Core matrix) peak flops of the device.
    """
    merged = _merged_env(env_spec, device_name)
    if not merged:
        return None, None
    return _to_float(merged.get("peak_flops")), _to_float(merged.get("peak_bw"))


def _load_env_data(env_spec):
    """Parse env_spec into a dict: file path first, then inline JSON."""
    if not env_spec:
        return None
    if os.path.isfile(env_spec):
        try:
            with open(env_spec, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None
    try:
        return json.loads(env_spec)
    except (TypeError, ValueError):
        return None


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
