#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from enum import IntFlag

from mindiesd.quantization.mode import QuantAlgorithm, QuantMode, QuantFlag
from mindiesd.quantization.mode import QuantModeDescriptor
from mindiesd.utils import ParametersInvalid


class TestQuantization(unittest.TestCase):

    def test_quant_algo(self):
        self.assertEqual(QuantAlgorithm.W8A8, 'W8A8')
        self.assertEqual(QuantAlgorithm.W8A16, 'W8A16')
        self.assertEqual(QuantAlgorithm.W4A16, 'W4A16')
        self.assertEqual(QuantAlgorithm.W4A16_AWQ, 'W4A16_AWQ')
        self.assertEqual(QuantAlgorithm.W4A8_AWQ, 'W4A8_AWQ')
        self.assertEqual(QuantAlgorithm.W4A16_GPTQ, 'W4A16_GPTQ')
        self.assertEqual(QuantAlgorithm.W8A16_GPTQ, 'W8A16_GPTQ')
        self.assertEqual(QuantAlgorithm.W8A8_PER_CHANNEL, 'W8A8_PER_CHANNEL')
        self.assertEqual(QuantAlgorithm.W8A8_PER_TENSOR, 'W8A8_PER_TENSOR')
        self.assertEqual(QuantAlgorithm.W8A8_PER_CHANNEL_PER_TENSOR, 'W8A8_PER_CHANNEL_PER_TENSOR')
        self.assertEqual(QuantAlgorithm.W8A8, 'W8A8')
        self.assertEqual(QuantAlgorithm.W8A8_PER_CHANNEL_PER_TOKEN, 'W8A8_PER_CHANNEL_PER_TOKEN')
        self.assertEqual(QuantAlgorithm.W8A8_PER_TENSOR_PER_TOKEN, 'W8A8_PER_TENSOR_PER_TOKEN')

        self.assertEqual(QuantAlgorithm.INT8, 'INT8')
        self.assertEqual(QuantAlgorithm.MIXED_PERCISION, 'MIXED_PERCISION')
        self.assertEqual(QuantAlgorithm.NO_QUANT, 'NO_QUANT')
        self.assertEqual(QuantAlgorithm.W8A8_MXFP8, 'W8A8_MXFP8')

    def test_quant_mode_descriptor(self):
        desc = QuantModeDescriptor()
        self.assertFalse(desc.quantize_weights)
        self.assertFalse(desc.quantize_activations)
        self.assertFalse(desc.per_token)
        self.assertFalse(desc.per_channel)
        self.assertFalse(desc.per_group)
        self.assertFalse(desc.use_int4_weights)


    def test_quant_mode(self):
        self.assertIsInstance(QuantFlag.INT4_WEIGHTS, IntFlag)
        self.assertIsInstance(QuantFlag.INT8_WEIGHTS, IntFlag)
        self.assertIsInstance(QuantFlag.ACTIVATION, IntFlag)
        self.assertIsInstance(QuantFlag.PER_CHANNEL, IntFlag)
        self.assertIsInstance(QuantFlag.PER_TENSOR, IntFlag)
        self.assertIsInstance(QuantFlag.PER_TOKEN, IntFlag)
        self.assertIsInstance(QuantFlag.PER_GROUP, IntFlag)

        self.assertIsInstance(QuantFlag.COUNT, IntFlag)
        self.assertIsInstance(QuantFlag.WEIGHTS_AND_ACTIVATION, IntFlag)
        self.assertIsInstance(QuantFlag.VALID_FLAG, IntFlag)

    def test_quant_mode_from_descriptor(self):
        desc = QuantModeDescriptor(quantize_weights=True, quantize_activations=True, per_token=True, per_channel=True)
        mode = QuantMode.from_descriptor(desc)
        self.assertTrue(mode.contains_activation_and_weight_quant())
        self.assertTrue(mode.contains_per_channel_scale())
        self.assertTrue(mode.contains_weight_quantization())

    def test_quant_mode_use_smooth_quant(self):
        mode = QuantMode.use_smooth_quant(per_token=True, per_channel=True)
        self.assertTrue(mode.contains_activation_and_weight_quant())
        self.assertTrue(mode.contains_per_channel_scale())

    def test_quant_mode_use_weight_only(self):
        mode = QuantMode.use_weight_only(use_int4_weights=True, per_group=True)
        self.assertTrue(mode.check_weight_int4_only_with_group())
        self.assertFalse(mode.contains_activation_and_weight_quant())

    def test_quant_mode_from_quant_algo(self):
        mode = QuantMode.from_quant_algo(QuantAlgorithm.W8A16)
        self.assertTrue(mode.check_weight_int8_only())
        self.assertFalse(mode.contains_activation_and_weight_quant())

    def test_quant_mode_is_int8_weight_only(self):
        mode = QuantMode(QuantFlag.INT8_WEIGHTS)
        self.assertTrue(mode.check_weight_int8_only())

    def test_quant_mode_is_int4_weight_only(self):
        mode = QuantMode(QuantFlag.INT4_WEIGHTS)
        self.assertTrue(mode.check_weight_int4_only())

    def test_quant_mode_is_weights_only(self):
        mode = QuantMode(QuantFlag.INT8_WEIGHTS)
        self.assertTrue(mode.check_weight_only_mode())

    def test_quant_mode_is_int8_weight_only_per_group(self):
        mode = QuantMode(QuantFlag.INT8_WEIGHTS | QuantFlag.PER_GROUP)
        self.assertTrue(mode.check_weight_int8_only_with_group())

    def test_quant_mode_is_int4_weight_only_per_group(self):
        mode = QuantMode(QuantFlag.INT4_WEIGHTS | QuantFlag.PER_GROUP)
        self.assertTrue(mode.check_weight_int4_only_with_group())

    def test_quant_mode_has_act_or_weight_quant(self):
        mode = QuantMode(QuantFlag.INT8_WEIGHTS)
        self.assertTrue(mode.contains_activation_or_weight_quant())

    def test_quant_mode_has_act_and_weight_quant(self):
        mode = QuantMode(QuantFlag.INT8_WEIGHTS | QuantFlag.ACTIVATION)
        self.assertTrue(mode.contains_activation_and_weight_quant())

    def test_quant_mode_has_per_channel_scaling(self):
        mode = QuantMode(QuantFlag.PER_CHANNEL)
        self.assertTrue(mode.contains_per_channel_scale())

    def test_quant_mode_has_per_group_scaling(self):
        mode = QuantMode(QuantFlag.PER_GROUP)
        self.assertTrue(mode.contains_per_group_scale())



    def test_quant_mode_has_weight_quant(self):
        mode = QuantMode(QuantFlag.INT4_WEIGHTS | QuantFlag.INT8_WEIGHTS)
        self.assertTrue(mode.contains_weight_quantization())

    def test_quant_mode_has_fa_quant(self):
        mode = QuantMode(QuantFlag.FA_QUANT)
        self.assertTrue(mode.contains_fa_quantization())



    def test_quant_mode_to_dict(self):
        mode = QuantMode(QuantFlag.INT8_WEIGHTS | QuantFlag.ACTIVATION | QuantFlag.PER_CHANNEL | QuantFlag.PER_TOKEN)
        mode_dict = mode.to_dict()
        self.assertTrue(mode_dict["use_smooth_quant"])
        self.assertFalse(mode_dict["use_weight_only"])
        self.assertEqual(mode_dict["weight_only_precision"], 'int4')


class TestFromDescriptor(unittest.TestCase):
    def test_quantize_weights_only(self):
        desc = QuantModeDescriptor(quantize_weights=True)
        self.assertEqual(QuantMode.from_descriptor(desc).flag, QuantFlag.INT8_WEIGHTS)

    def test_quantize_activations_only(self):
        desc = QuantModeDescriptor(quantize_activations=True)
        with self.assertRaises(ParametersInvalid):
            QuantMode.from_descriptor(desc)

    def test_quantize_both(self):
        desc = QuantModeDescriptor(quantize_weights=True, quantize_activations=True)
        self.assertEqual(QuantMode.from_descriptor(desc).flag, QuantFlag.INT8_WEIGHTS | QuantFlag.ACTIVATION)

    def test_fa_quant(self):
        desc = QuantModeDescriptor(use_fa_quant=True)
        self.assertEqual(QuantMode.from_descriptor(desc).flag, QuantFlag.FA_QUANT)

    def test_per_channel_without_activation(self):
        desc = QuantModeDescriptor(per_channel=True)
        with self.assertRaises(ParametersInvalid):
            QuantMode.from_descriptor(desc)

    def test_per_token_without_activation(self):
        desc = QuantModeDescriptor(per_token=True)
        with self.assertRaises(ParametersInvalid):
            QuantMode.from_descriptor(desc)

    def test_per_channel_and_per_token(self):
        desc = QuantModeDescriptor(quantize_weights=True, quantize_activations=True, per_channel=True, per_token=True)
        self.assertEqual(QuantMode.from_descriptor(desc).flag,
                         QuantFlag.INT8_WEIGHTS | QuantFlag.ACTIVATION | QuantFlag.PER_CHANNEL | QuantFlag.PER_TOKEN)

    def test_int4_weights(self):
        desc = QuantModeDescriptor(quantize_weights=True, use_int4_weights=True)
        self.assertEqual(QuantMode.from_descriptor(desc).flag, QuantFlag.INT4_WEIGHTS)



    def test_all_options(self):
        desc = QuantModeDescriptor(quantize_weights=True, quantize_activations=True, use_int4_weights=True,
                                   per_token=True, per_channel=True, per_group=True)
        expected_mode = QuantFlag.INT4_WEIGHTS | QuantFlag.ACTIVATION | QuantFlag.PER_CHANNEL | QuantFlag.PER_TOKEN | \
                        QuantFlag.PER_GROUP
        self.assertEqual(QuantMode.from_descriptor(desc).flag, expected_mode)


if __name__ == '__main__':
    unittest.main()
