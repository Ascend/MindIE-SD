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
"""FFN 融合层：y = act(x @ W1^T + b1) @ W2^T + b2（ascend950/arch35 单 kernel）。

 fused=True 且 A5 平台走融合算子，否则回退 PyTorch 原生小算子链。
"""

import torch
import torch.nn.functional as F

from ..utils import ParametersInvalid
from ..utils.get_platform import get_npu_device, NPUDevice
from . import _custom_ops as ops

_SUPPORTED_ACTIVATIONS = ("gelu", "silu", "swiglu")


def _chain_fallback(x, weight1, weight2, bias1, bias2, activation):
    up = F.linear(x, weight1, bias1)
    if activation == "swiglu":
        gate, up_half = up.chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up_half, weight2, bias2)
    if activation == "silu":
        return F.linear(F.silu(up), weight2, bias2)
    return F.linear(F.gelu(up), weight2, bias2)


def eagle_ffn_linear(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    activation: str = "gelu",
    fused: bool = True,
) -> torch.Tensor:
    """FFN 前向：y = act(x @ W1^T + b1) @ W2^T + b2。

    Args:
        x (torch.Tensor): 输入，[..., M, K]，bf16/fp16。
        weight1 (torch.Tensor): 上投权重。linear 布局 [H, K]（PyTorch Linear 惯例）；
            swiglu 为单 matmul 形式 [2H, K]。canonical [K, H] 亦可（自动检测）。
        weight2 (torch.Tensor): 下投权重，[N, H]（linear）或 [H, N]（canonical）。
        bias1/bias2 (torch.Tensor | None): 偏置，需同时给/同时不给、同 dtype（与 x 同
            或 fp32）；swiglu 时 bias1 长度为 2H。
        activation (str): "gelu" / "silu" / "swiglu"（大小写不敏感）。
        fused (bool): True 且 A5 平台走融合算子；否则回退原生小算子链。

    Returns:
        torch.Tensor: [..., M, N]，dtype 与 x 相同。
    """
    if not isinstance(activation, str) or activation.lower() not in _SUPPORTED_ACTIVATIONS:
        raise ParametersInvalid(
            f"activation must be one of {_SUPPORTED_ACTIVATIONS}, got {activation!r}"
        )
    act = activation.lower()

    npu_device = get_npu_device()
    if fused and npu_device == NPUDevice.A5:
        return ops.eagle_ffn_linear(x, weight1, weight2, bias1, bias2, act, 0)
    return _chain_fallback(x, weight1, weight2, bias1, bias2, act)
