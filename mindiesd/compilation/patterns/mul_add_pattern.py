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

import torch
from ..passes.register_pattern_to_pass import PatternBase
from ...layers import muls_add


def create(dtype, scale=1.0):
    class MulAddPattern(PatternBase):
        @staticmethod
        def name():
            return f"MulAddPattern-{dtype}-{scale}"

        @staticmethod
        def inputs():
            x = torch.empty(2, 2, dtype=dtype, device="meta")
            y = torch.empty(2, 2, dtype=dtype, device="meta")
            return [x, y]

        @staticmethod
        def pattern(x, y):
            def func(x, y):
                return x * scale + y

            return func(x, y)

        @staticmethod
        def replacement(x, y):
            def func(x, y):
                return muls_add(x, y, scale)

            return func(x, y)

    return MulAddPattern


MulAddPatternGroup = [create(torch.bfloat16)]
