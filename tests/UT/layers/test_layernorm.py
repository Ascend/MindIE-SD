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

import torch
from torch import nn

from mindiesd import fast_layernorm
from mindiesd.utils import ParametersInvalid


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn([2, 1024, 128], dtype=torch.float32)

    def test_normalized_shape_too_large(self):
        """Test that normalized_shape with more dims than input raises ParametersInvalid."""
        # input is 3-D [2, 1024, 128], normalized_shape is 4-D
        bad_layernorm = nn.LayerNorm(normalized_shape=(2, 1024, 128, 64), elementwise_affine=False)
        with self.assertRaises(ParametersInvalid) as ctx:
            fast_layernorm(bad_layernorm, self.x, 0)
        self.assertIn("normalized_shape must fit within input dimensions", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
