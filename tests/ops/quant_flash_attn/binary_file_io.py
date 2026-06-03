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

import torch
import os
import ctypes


def safe_save_bin_file(tensor, filepath):
    """
    纯 PyTorch 保存原始数据，格式与 numpy.tofile 完全一致
    """
    data = tensor.contiguous().cpu()
    temp_path = filepath + '.tmp'

    # 从 storage 获取原始字节的正确方式
    storage = data.storage()
    num_bytes = storage.nbytes()
    ptr = storage.data_ptr()

    # 用 ctypes 从内存地址读取字节
    raw_bytes = ctypes.string_at(ptr, num_bytes)

    with open(temp_path, 'wb') as f:
        f.write(raw_bytes)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, filepath)


def safe_load_bin_file(filepath, dtype, shape=None):
    """
    纯 PyTorch 读取原始二进制，与 numpy.fromfile 行为一致
    """
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()

    tensor = torch.frombuffer(bytearray(raw_bytes), dtype=dtype)

    if shape is not None:
        tensor = tensor.reshape(shape)

    return tensor
