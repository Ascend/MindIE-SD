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

# pylint: disable=duplicate-code

# pylint: disable=no-name-in-module
import unittest
import os
import torch
import torch_npu

from device import DEVICE_ID
from mindiesd.layers.flash_attn.sparse_flash_attn import sparse_attention
from mindiesd.utils.get_platform import is_a5_device
from utils.utils.precision_compare import data_compare


if __name__ == '__main__':
    torch_npu.npu.set_device(DEVICE_ID)
    unittest.main()
