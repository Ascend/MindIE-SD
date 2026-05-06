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

import unittest

from mindiesd.compilation.compiliation_config import CompilationConfig


class TestFusionConfigIntegration(unittest.TestCase):
    def test_fusion_config_has_new_fields(self):
        config = CompilationConfig.fusion_patterns
        self.assertTrue(hasattr(config, "enable_mul_add"))

    def test_fusion_config_fields_default_true(self):
        config = CompilationConfig.fusion_patterns
        self.assertTrue(config.enable_mul_add)

    def test_fusion_config_can_be_disabled(self):
        saved = CompilationConfig.fusion_patterns.enable_mul_add
        CompilationConfig.fusion_patterns.enable_mul_add = False
        self.assertFalse(CompilationConfig.fusion_patterns.enable_mul_add)
        CompilationConfig.fusion_patterns.enable_mul_add = saved


if __name__ == "__main__":
    unittest.main()
