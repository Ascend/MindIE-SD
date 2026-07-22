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


import torch
from .attention_operate import AttentionOperateBase, register_op_800
from ...utils.exception import ParametersInvalid
from ...utils.get_platform import is_a5_device
from .. import _custom_ops as ops


_A5_LA_PREPROCESS_UNSUPPORTED_MSG = (
    "ascend_laser_preprocess is not supported on A5 devices. The LA pre-processing kernel is no longer "
    "needed because the CANN-native FA operator handles padding internally. "
    "Please drop this call and use 'mindiesd.layers.flash_attn.attention_forward' (or "
    "'attention_forward_varlen') directly."
)


@register_op_800("ascend_laser_preprocess")
class AscendLaserPreprocess(AttentionOperateBase):
    supported_layout = ["BSND"]
    supported_dtype = [torch.float16, torch.bfloat16]

    @classmethod
    def forward_preprocess(
        cls, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, align_len: int = 256
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if is_a5_device():
            raise ParametersInvalid(_A5_LA_PREPROCESS_UNSUPPORTED_MSG)

        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ParametersInvalid("LA_preprocess输入必须是4D张量")
        if query.shape[2:] != key.shape[2:]:
            raise ParametersInvalid(
                f"key head dimensions mismatch: query{list(query.shape[2:])} vs key{list(key.shape[2:])}"
            )
        if query.shape[2:] != value.shape[2:]:
            raise ParametersInvalid(
                f"value head dimensions mismatch: query{list(query.shape[2:])} vs value{list(value.shape[2:])}"
            )
        batch_size, seq_len, head_num, head_dim = query.shape

        out_query, out_key, out_value = ops.laser_attention_preprocess(query, key, value, align_len)
        return out_query.contiguous(), out_key.contiguous(), out_value.contiguous()


def la_preprocess(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, align_len: int = 256):
    return AscendLaserPreprocess.forward_preprocess(query, key, value, align_len)
