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
import os
import unittest
from unittest.mock import Mock, patch

import torch
import torch_npu

import mindiesd
from mindiesd import apply_rotary_pos_emb as top_level_apply_rotary_pos_emb
from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from mindiesd.layers import apply_rotary_pos_emb as public_apply_rotary_pos_emb
from mindiesd.layers._custom_ops import (
    FREQUENCY_REGULATOR_MAX_FREQ,
    apply_rotary_pos_emb,
    frequency_regulator,
    frequency_regulator_fake,
    laser_attention,
    laser_attention_preprocess,
)
from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import is_a5_device


class TestApplyRotaryPosEmbWrapper(unittest.TestCase):
    def setUp(self):
        self.query = torch.empty((2, 8, 6, 64), device="meta", dtype=torch.float16)
        self.key = torch.empty((2, 8, 2, 64), device="meta", dtype=torch.float16)
        self.cos = torch.empty((2, 8, 1, 64), device="meta", dtype=torch.float16)
        self.sin = torch.empty_like(self.cos)

    def test_public_api_is_exported(self):
        self.assertIs(public_apply_rotary_pos_emb, apply_rotary_pos_emb)

    def test_top_level_api_is_exported(self):
        self.assertIs(top_level_apply_rotary_pos_emb, apply_rotary_pos_emb)
        self.assertIn("apply_rotary_pos_emb", mindiesd.__all__)
        self.assertIn("apply_rotary_pos_emb", mindiesd.layers.__all__)

    def test_wrapper_forwards_default_arguments(self):
        expected = (self.query, self.key)
        mock_op = Mock(return_value=expected)

        with patch.object(torch_npu, "npu_apply_rotary_pos_emb", mock_op, create=True):
            result = apply_rotary_pos_emb(self.query, self.key, self.cos, self.sin)

        self.assertEqual(result, expected)
        mock_op.assert_called_once_with(self.query, self.key, self.cos, self.sin, layout="BSND", rotary_mode="half")

    def test_wrapper_forwards_explicit_arguments(self):
        expected = (self.query, self.key)
        mock_op = Mock(return_value=expected)

        with patch.object(torch_npu, "npu_apply_rotary_pos_emb", mock_op, create=True):
            result = apply_rotary_pos_emb(
                self.query, self.key, self.cos, self.sin, layout="TND", rotary_mode="interleave"
            )

        self.assertEqual(result, expected)
        mock_op.assert_called_once_with(
            self.query, self.key, self.cos, self.sin, layout="TND", rotary_mode="interleave"
        )

    def test_unsupported_layout_is_rejected(self):
        with patch.object(torch_npu, "npu_apply_rotary_pos_emb", Mock(), create=True):
            with self.assertRaisesRegex(ParametersInvalid, "layout must be"):
                apply_rotary_pos_emb(self.query, self.key, self.cos, self.sin, layout="INVALID")

    def test_unsupported_rotary_mode_is_rejected(self):
        with patch.object(torch_npu, "npu_apply_rotary_pos_emb", Mock(), create=True):
            with self.assertRaisesRegex(ParametersInvalid, "rotary_mode must be"):
                apply_rotary_pos_emb(self.query, self.key, self.cos, self.sin, rotary_mode="INVALID")


class TestFrequencyRegulatorWrapper(unittest.TestCase):
    def test_frequency_regulator_forwards_valid_freq(self):
        expected = torch.empty((1,), device="meta", dtype=torch.int64)
        mock_op = Mock(return_value=expected)

        with patch.object(torch.ops.mindiesd, "frequency_regulator", mock_op, create=True):
            result = frequency_regulator(1650)

        self.assertIs(result, expected)
        mock_op.assert_called_once_with(1650)

    def test_frequency_regulator_accepts_uint32_max_freq(self):
        expected = torch.empty((1,), device="meta", dtype=torch.int64)
        mock_op = Mock(return_value=expected)

        with patch.object(torch.ops.mindiesd, "frequency_regulator", mock_op, create=True):
            result = frequency_regulator(FREQUENCY_REGULATOR_MAX_FREQ)

        self.assertIs(result, expected)
        mock_op.assert_called_once_with(FREQUENCY_REGULATOR_MAX_FREQ)

    def test_frequency_regulator_fake_returns_int64_status(self):
        result = frequency_regulator_fake(1650)

        self.assertEqual(result.dtype, torch.int64)

    def test_frequency_regulator_rejects_bool_freq(self):
        with self.assertRaises(ParametersInvalid):
            frequency_regulator(True)

    def test_frequency_regulator_rejects_non_int_freq(self):
        with self.assertRaises(ParametersInvalid):
            frequency_regulator(1650.0)

    def test_frequency_regulator_rejects_out_of_range_freq(self):
        with self.assertRaises(ParametersInvalid):
            frequency_regulator(-1)
        with self.assertRaises(ParametersInvalid):
            frequency_regulator(FREQUENCY_REGULATOR_MAX_FREQ + 1)


class TestCustomOps(unittest.TestCase):
    def test_laser_attention_fake_shape(self):
        from mindiesd.layers._custom_ops import attention_fake

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            query = torch.empty(2, 8, 256, 128, dtype=dtype, device="meta")
            stats, output = attention_fake(query, query, query, head_num=8)
            self.assertEqual(stats.shape, (2, 8, 256))
            self.assertEqual(stats.dtype, torch.float32)
            self.assertEqual(output.shape, query.shape)
            self.assertEqual(output.dtype, torch.bfloat16 if dtype == torch.bfloat16 else torch.float32)
            self.assertEqual(output.device.type, "meta")

    def test_laser_attention_preprocess_fake_shape(self):
        from mindiesd.layers._custom_ops import attention_preprocess_fake

        for shape in ((2, 65, 8, 16), (65, 8, 16)):
            query = torch.empty(shape, dtype=torch.float16, device="meta")
            outputs = attention_preprocess_fake(query, query, query, 32)
            expected = (2, 96, 8, 16) if len(shape) == 4 else (96, 8, 16)
            self.assertEqual(len(outputs), 3)
            for output in outputs:
                self.assertEqual(tuple(output.shape), expected)
                self.assertEqual(output.dtype, query.dtype)
                self.assertEqual(output.device.type, "meta")


if __name__ == '__main__':
    unittest.main()
