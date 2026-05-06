#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any

import torch

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    pass

_NUM_AICORE = -1
_NUM_VECTORCORE = -1
_extension_module = None

if _HAS_TRITON:
    try:
        import triton.language.extra.cann.extension as _extension_module  # type: ignore[no-redef]
    except ImportError:
        _extension_module = None


def _check_triton_ascend():
    if not _HAS_TRITON:
        return False
    if not torch.npu.is_available():
        return False
    try:
        triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        return True
    except Exception:
        return False


_TRITON_ON_ASCEND = _check_triton_ascend()


def init_device_properties_triton():
    global _NUM_AICORE, _NUM_VECTORCORE
    if _NUM_AICORE == -1 and _HAS_TRITON:
        try:
            device_properties: dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(
                torch.npu.current_device()
            )
            _NUM_AICORE = device_properties.get("num_aicore", -1)
            _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        except Exception:
            _NUM_AICORE = -1
            _NUM_VECTORCORE = -1


def get_aicore_num():
    global _NUM_AICORE
    if _NUM_AICORE <= 0:
        if _HAS_TRITON:
            init_device_properties_triton()
        if _NUM_AICORE <= 0:
            return 1
    return _NUM_AICORE


def get_vectorcore_num():
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE <= 0:
        if _HAS_TRITON:
            init_device_properties_triton()
        if _NUM_VECTORCORE <= 0:
            return 1
    return _NUM_VECTORCORE
