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

"""Detection of quant kernels unavailable on the current platform.

torch_npu raises RuntimeError with the missing kernel's name in the message
(e.g. DynamicMxQuant, npu_quant_matmul). All fallback decisions route through
`is_quant_unsupported` so an upstream message change only needs one edit here
instead of a silent behavior change across vendor impls.
"""

_QUANT_UNSUPPORTED_SIGNALS = ("DynamicMxQuant", "npu_quant_matmul")


def is_quant_unsupported(exc):
    """True when `exc` indicates a quantization kernel missing on the platform."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return any(signal in msg for signal in _QUANT_UNSUPPORTED_SIGNALS)
