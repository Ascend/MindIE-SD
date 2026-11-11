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
from unittest.mock import MagicMock, patch

from mindiesd.compilation import passes

from mindiesd.compilation.compiliation_config import CompilationConfig


class TestFusionConfigIntegration(unittest.TestCase):
    def test_add_layer_norm_registers_before_shorter_patterns(self):
        groups = MagicMock()
        with (
            patch.object(passes.once_flag, "is_set", return_value=False),
            patch.object(passes.once_flag, "set"),
            patch("mindiesd.compilation.passes.register_pattern_to_pass.register_pattern_to_pass") as register,
            patch("importlib.import_module", return_value=groups),
        ):
            passes.activate_pattern_once()
        self.assertEqual(register.call_args_list[0].args, (groups.AddLayerNormPatternGroup,))
        self.assertEqual(
            sum(call.args == (groups.AddLayerNormPatternGroup,) for call in register.call_args_list),
            1,
        )

    def test_add_layer_norm_can_be_disabled(self):
        groups = MagicMock()
        with (
            patch.object(CompilationConfig.fusion_patterns, "enable_add_layer_norm", False),
            patch.object(passes.once_flag, "is_set", return_value=False),
            patch.object(passes.once_flag, "set"),
            patch("mindiesd.compilation.passes.register_pattern_to_pass.register_pattern_to_pass") as register,
            patch("importlib.import_module", return_value=groups),
        ):
            passes.activate_pattern_once()
        self.assertFalse(any(call.args == (groups.AddLayerNormPatternGroup,) for call in register.call_args_list))

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

    def test_qwen_residual_gate_config_can_be_disabled(self):
        config = CompilationConfig.fusion_patterns
        saved = config.enable_qwen_residual_gate
        try:
            self.assertTrue(saved)
            config.enable_qwen_residual_gate = False
            self.assertFalse(config.enable_qwen_residual_gate)
        finally:
            config.enable_qwen_residual_gate = saved


if __name__ == "__main__":
    unittest.main()
