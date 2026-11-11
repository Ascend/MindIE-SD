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

"""QuantBatchMatmulV3 + SwiGLU + DynamicMxQuant fused operator.

The device entry point is :func:`run`.  It requires the CANN CATLASS Python DSL
(``catlass.tla``) at call time, so importing :mod:`mindiesd` does not depend on
the DSL being installed.

The CPU reference implementation lives with the tests in
``tests/ops/qbmm_swiglu_mxquant/reference.py``; the operator contract is
documented in ``docs/zh/features/qbmm_swiglu_mxquant.md``.
"""

__all__ = ["run"]


def run(a, scale_a, b, scale_b, in_fp4, out_fp4):
    """Run the fused operator on NPU."""
    from .kernel import run as _run

    return _run(a, scale_a, b, scale_b, in_fp4, out_fp4)
