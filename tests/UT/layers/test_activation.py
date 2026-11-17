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

import unittest
from unittest.mock import patch
import torch
from torch import nn
import torch_npu

from mindiesd.layers.activation import get_activation_layer, GELU
from mindiesd.utils import ParametersInvalid


class TestActivation(unittest.TestCase):
    def test_get_activation_layer_valid(self):
        func = get_activation_layer('gelu')
        self.assertIsInstance(func, nn.Module)

    def test_get_activation_layer_invalid(self):
        with self.assertRaises(ParametersInvalid):
            get_activation_layer('test')

    def test_gelu(self):
        tensor = torch.randn(size=(1, 2, 3))
        gelu = GELU(approximate="test")
        with self.assertRaises(ParametersInvalid):
            gelu(tensor)

    def test_reference_modes_preserve_values_and_dtype(self):
        tensor = torch.linspace(-3, 3, 24).reshape(2, 3, 4)
        for approximate in ("none", "tanh"):
            with self.subTest(approximate=approximate):
                actual = GELU(approximate)(tensor)
                expected = torch.nn.functional.gelu(tensor, approximate=approximate)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.dtype, tensor.dtype)

    def test_fast_mode_delegates_exact_input(self):
        tensor = torch.randn(2, 4)
        expected = tensor + 1
        with patch.object(torch_npu, "npu_fast_gelu", return_value=expected, create=True) as backend:
            self.assertIs(get_activation_layer("GELU-FAST")(tensor), expected)
        backend.assert_called_once_with(tensor)


if __name__ == '__main__':
    torch.manual_seed(1234)
    unittest.main()
