#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import warnings
from .attention_forward import attention_forward
from .attention_forward_varlen import attention_forward_varlen
from .sparse_flash_attn import sparse_attention
from ..triton_utils import _HAS_TRITON, _extension_module

if _HAS_TRITON and _extension_module is not None:
    from .sparse_linear_attn import SparseLinearAttention
else:

    def SparseLinearAttention(*args, **kwargs):
        raise RuntimeError("SparseLinearAttention requires Triton-Ascend >= 3.2.1 but it is not available.")

    warnings.warn(
        "Triton-Ascend is not available or is below 3.2.1."
        "SparseLinearAttention is disabled. Install required dependencies to use it.",
        UserWarning,
    )

__all__ = [
    "attention_forward",
    "attention_forward_varlen",
    "sparse_attention",
    "SparseLinearAttention",
]
