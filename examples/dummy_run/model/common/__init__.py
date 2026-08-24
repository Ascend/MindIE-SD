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

"""dummy run 通用能力（跨模型共享，按职责分类）。

- ``precision``: 模型级 bf16/fp32 精度机制（``--quant bf16`` / ``fp32``）
- ``compile_patches``: compile 图层性能问题的模型层补丁（纯性能）
- ``quantization``: W8A8-MXFP8 在线量化（``--quant mxfp8``）
"""

from .compile_patches import (
    replace_pos_embed_with_buffers,
    replace_zero_dropout,
)
from .precision import (
    apply_compute_precision,
    compute_dtype_from_precision,
    verify_compute_precision_graph,
)
from .quantization import (
    apply_mxfp8_quant,
    apply_w8a8_quant,
    report_quant_layers,
)

__all__ = [
    "apply_compute_precision",
    "apply_mxfp8_quant",
    "apply_w8a8_quant",
    "compute_dtype_from_precision",
    "replace_pos_embed_with_buffers",
    "replace_zero_dropout",
    "report_quant_layers",
    "verify_compute_precision_graph",
]
