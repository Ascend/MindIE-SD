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

from mindiesd.compilation import MindieSDBackend  # pylint: disable=no-name-in-module
from mindiesd.layers._custom_ops import (
    FREQUENCY_REGULATOR_MAX_FREQ,
    frequency_regulator,
    frequency_regulator_fake,
    laser_attention,
    laser_attention_preprocess,
)
from mindiesd.utils import ParametersInvalid
from mindiesd.utils.get_platform import is_a5_device


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


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
@unittest.skipIf(
    is_a5_device(), "laser_attention / laser_attention_preprocess are unsupported on A5; routed elsewhere."
)
class TestCustomOps(unittest.TestCase):
    def test_laser_attention_fake_shape(self):
        class LaserAttentionModel(torch.nn.Module):
            def forward(
                self,
                query,
                key,
                value,
                atten_mask,
                alibi_mask,
                drop_mask,
                scale_value,
                head_num,
                input_layout,
                keep_prob,
                pre_tokens,
                next_tokens,
                is_high_precision,
            ):
                return laser_attention(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=atten_mask,
                    alibi_mask=alibi_mask,
                    drop_mask=drop_mask,
                    scale_value=scale_value,
                    head_num=head_num,
                    input_layout=input_layout,
                    keep_prob=keep_prob,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    is_high_precision=is_high_precision,
                )[0]

        batch_size = 2
        seq_len = 256
        head_num = 8
        head_dim = 128

        query = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="npu")
        key = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="npu")
        value = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="npu")

        layout = "BNSD"
        pre_tokens = 0

        scale_value = 1.0
        keep_prob = 1.0
        input_layout = layout
        is_high_precision = True
        next_tokens = 1

        atten_mask = None
        alibi_mask = None
        drop_mask = None
        model = LaserAttentionModel()
        compiled_model = torch.compile(model, backend=MindieSDBackend())

        output_original = model(
            query,
            key,
            value,
            atten_mask,
            alibi_mask,
            drop_mask,
            scale_value,
            head_num,
            input_layout,
            keep_prob,
            pre_tokens,
            next_tokens,
            is_high_precision,
        )
        output_compiled = compiled_model(
            query,
            key,
            value,
            atten_mask,
            alibi_mask,
            drop_mask,
            scale_value,
            head_num,
            input_layout,
            keep_prob,
            pre_tokens,
            next_tokens,
            is_high_precision,
        )

        self.assertEqual(output_original.shape, output_compiled.shape)

    def test_laser_attention_preprocess_fake_shape(self):
        class LaserAttentionPreprocessModel(torch.nn.Module):
            def forward(self, query, key, value, align_len):
                return laser_attention_preprocess(query, key, value, align_len)

        batch_size = 2
        seq_len = 64
        head_num = 8
        head_dim = 16
        align_len = 32

        query = torch.randn(batch_size, seq_len, head_num, head_dim, dtype=torch.float16, device="npu")
        key = torch.randn(batch_size, seq_len, head_num, head_dim, dtype=torch.float16, device="npu")
        value = torch.randn(batch_size, seq_len, head_num, head_dim, dtype=torch.float16, device="npu")

        model = LaserAttentionPreprocessModel()
        compiled_model = torch.compile(model, backend=MindieSDBackend())

        output_original = model(query, key, value, align_len)
        output_compiled = compiled_model(query, key, value, align_len)

        self.assertEqual(len(output_original), len(output_compiled))
        for orig, comp in zip(output_original, output_compiled):
            self.assertEqual(orig.shape, comp.shape)


if __name__ == '__main__':
    unittest.main()
