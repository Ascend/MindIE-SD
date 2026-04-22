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

import math
import torch_npu
import torch.nn.functional as F
from ...utils import ParametersInvalid


def fa_block_quant_preprocess(input_tensor, block_size=128, dst_type=torch_npu.float8_e4m3fn, **kwargs):
    """
    Preprocess for FA quant. Input layout must be 'BNSD' or 'BSND'.
    Args:
        input_tensor (torch.Tensor): Input tensor to be quantized.
        block_size (int, optional): Block size for quantization. Support 128/256/512. Default: 128.
        dst_type (torch.dtype, optional): Target quantization data type. Default: torch_npu.float8_e4m3fn.
        **kwargs: 
            layout (str): Tensor layout format, supports 'BNSD' (Batch, Num_heads, Seq_len, Dim) 
                         or 'BSND' (Batch, Seq_len, Num_heads, Dim).

    Returns:
        torch.Tensor: Preprocessed tensor ready for FA block quantization.
    """

    if len(input_tensor.shape) != 4:
        raise ParametersInvalid(f"fa block quant preprocess only support qkv quant, dim = 4, \
                                but got {len(input_tensor.shape)}.")

    layout = kwargs.get("layout", "BNSD")
    if layout == "BSND":
        input_tensor = input_tensor.transpose(1, 2)

    # block_quant only support BNSD
    input_quant, input_scale = torch_npu.npu_dynamic_block_quant(input_tensor.squeeze(0),
                                                                 dst_type=dst_type,
                                                                 row_block_size=block_size,
                                                                 col_block_size=128)

    return input_quant.unsqueeze(0), input_scale.unsqueeze(0)
