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

import torch
from ..utils import ParametersInvalid
from ..utils.get_platform import get_npu_device, NPUDevice
from ..utils.logs.logging import logger
from . import _custom_ops as ops

npu_device = get_npu_device()


def check_input_params(layernorm, x, scale, shift, fused):
    if not isinstance(layernorm, torch.nn.LayerNorm):
        raise ParametersInvalid(f"The type of input layernorm must be torch.nn.LayerNorm, but got {type(layernorm)}.")
    if not isinstance(x, torch.Tensor):
        raise ParametersInvalid(f"The data type of input x must be torch.Tensor, but got {type(x)}.")
    if not isinstance(scale, torch.Tensor):
        raise ParametersInvalid(f"The data type of input scale must be torch.Tensor, but got {type(scale)}.")
    if not isinstance(shift, torch.Tensor):
        raise ParametersInvalid(f"The data type of input shift must be torch.Tensor, but got {type(shift)}.")
    if not isinstance(fused, bool):
        raise ParametersInvalid(f"The data type of input fused must be bool, but got {type(fused)}.")

    if x.dim() != 3:  # 3: BSH输入dim
        raise ParametersInvalid(f"The dimensional of input x must be a 3, but got {x.dim()}.")
    if scale.dim() not in [2, 3]:  # 2: BH输入dim; 3: B1H/BSH输入dim
        raise ParametersInvalid(f"The dimensional of input scale must be a 2 or 3, but got {scale.dim()}.")
    if shift.dim() not in [2, 3]:  # 2: BH输入dim; 3: B1H/BSH输入dim
        raise ParametersInvalid(f"The dimensional of input shift must be a 2 or 3, but got {shift.dim()}.")
    if scale.size()[0] != x.size()[0]:
        raise ParametersInvalid(
            f"The first dimensions of input x and input scale must be equal, but {x.size()[0]} != {scale.size()[0]}."
        )
    if shift.size()[0] != x.size()[0]:
        raise ParametersInvalid(
            f"The first dimensions of input x and input shift must be equal, but {x.size()[0]} != {shift.size()[0]}."
        )
    if scale.dim() == 3 and scale.size()[1] not in [1, x.size()[1]]:
        raise ParametersInvalid(
            f"If scale is a 3D tensor, the second dimension must be 1 or match x.size(1), but got {scale.size()[1]}."
        )
    if shift.dim() == 3 and shift.size()[1] not in [1, x.size()[1]]:
        raise ParametersInvalid(
            f"If shift is a 3D tensor, the second dimension must be 1 or match x.size(1), but got {shift.size()[1]}."
        )

    last_dim_x = x.size()[-1]
    last_dim_scale = scale.size()[-1]
    last_dim_shift = shift.size()[-1]
    if last_dim_x != last_dim_scale:
        raise ParametersInvalid(
            f"The last dimensions of input x and input scale must be equal,  but {last_dim_x} != {last_dim_scale}."
        )
    if last_dim_scale != last_dim_shift:
        raise ParametersInvalid(
            f"The last dimensions of input scale and input shift must be equal,  "
            f"but {last_dim_scale} != {last_dim_shift}."
        )


def _is_tokenwise_modulation(x: torch.Tensor, tensor: torch.Tensor) -> bool:
    # [B,1,H] 仍然走原有广播路径, 只有真正的 [B,S,H] (S>1) 才视为 token-wise modulation
    return tensor.dim() == 3 and tensor.size(1) == x.size(1) and x.size(1) != 1


def _expand_modulation_to_tokenwise(x: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor[:, None, :].expand(-1, x.size(1), -1)
    if tensor.size(1) == 1:
        return tensor.expand(-1, x.size(1), -1)
    return tensor


def _preprocess_tokenwise_modulation(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> tuple:
    if not (_is_tokenwise_modulation(x, scale) or _is_tokenwise_modulation(x, shift)):
        return x, scale, shift, None

    batch_size, sequence_length, hidden_size = x.shape
    x_bsh = x
    x = x_bsh.reshape(batch_size * sequence_length, 1, hidden_size)
    scale = _expand_modulation_to_tokenwise(x_bsh, scale).reshape(batch_size * sequence_length, hidden_size)
    shift = _expand_modulation_to_tokenwise(x_bsh, shift).reshape(batch_size * sequence_length, hidden_size)
    return x, scale, shift, (batch_size, sequence_length, hidden_size)


def _postprocess_tokenwise_output(out: torch.Tensor, tokenwise_shape: tuple) -> torch.Tensor:
    if tokenwise_shape is None:
        return out
    batch_size, sequence_length, hidden_size = tokenwise_shape
    return out.reshape(batch_size, sequence_length, hidden_size)


_modulation_dtype_align_warned = False


def _modulation_dtype_align_enabled() -> bool:
    """adaLN 调制量 dtype 对齐开关（默认开启）。

    aclnnAdaLayerNorm/V2 要求所有输入 dtype 一致；混合精度图（例如 bf16 主激活 +
    fp32 调制量）不对齐会在捕获期失败。设置 ``MINDIE_SD_ADALN_DTYPE_ALIGN=0``
    可关闭对齐（此时需要调用方自行保证 dtype 一致）。
    """
    return os.environ.get("MINDIE_SD_ADALN_DTYPE_ALIGN", "1") != "0"


def _align_modulation_dtype(x: torch.Tensor, *tensors: torch.Tensor | None) -> tuple:
    # 混合精度编译图(如 Wan2.2 模型级 bf16 改写未覆盖的 .float() 岛)命中 adaLN 融合
    # pattern 后, 融合算子(aclnnAdaLayerNorm/V2)要求所有输入 dtype 一致, 不能像 eager
    # 一样靠类型提升; 此处统一对齐到 x.dtype。dtype 比较在 trace 期静态求值, 一致时
    # 不产生冗余 cast 节点。
    global _modulation_dtype_align_warned
    if not _modulation_dtype_align_enabled():
        return tensors
    aligned = []
    changed = []
    for tensor in tensors:
        if tensor is not None and tensor.dtype != x.dtype:
            changed.append((str(tensor.dtype), str(x.dtype)))
            tensor = tensor.to(x.dtype)
        aligned.append(tensor)
    if changed and not _modulation_dtype_align_warned:
        _modulation_dtype_align_warned = True
        logger.warning(
            "[MindIE-SD/compilation] adaLN modulation dtype aligned to x.dtype (%s -> %s, %d tensor(s)). "
            "aclnnAdaLayerNorm requires uniform input dtype; this differs from eager type promotion. "
            "set MINDIE_SD_ADALN_DTYPE_ALIGN=0 to opt out.",
            changed[0][0],
            changed[0][1],
            len(changed),
        )
    return tuple(aligned)


def layernorm_scale_shift(
    layernorm: torch.nn.LayerNorm, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, fused: bool = True
) -> torch.Tensor:
    """
    Apply AdaLayerNorm to input tensors:
        out = layernorm(x) * (1 + scale) + shift

    Args:
        layernorm (torch.nn.LayerNorm):
            The LayerNorm module.
        x (torch.Tensor):
            Tensor to apply AdaLayerNorm. x must be 3-dimensional.
            The supported layout: [B,S,H].
        scale (torch.Tensor):
            Adaptive Scaling Parameters. scale must be 2 or 3-dimensional.
            The supported layout: [B, H], [B, 1, H], [B, S, H].
        shift (torch.Tensor):
            Adaptive offset parameter. shift must be 2 or 3-dimensional.
            The supported layout: [B, H], [B, 1, H], [B, S, H].
        fused (bool):
            If fused is True, using high-performance AdaLayerNorm operator.

    Returns:
        (torch.Tensor): modified tensor with AdaLayerNorm.
    """
    check_input_params(layernorm, x, scale, shift, fused)

    if fused:
        if layernorm.elementwise_affine:
            weight = layernorm.weight
            bias = layernorm.bias
        else:
            weight = None
            bias = None
        scale, shift, weight, bias = _align_modulation_dtype(x, scale, shift, weight, bias)
        x, scale, shift, tokenwise_shape = _preprocess_tokenwise_modulation(x, scale, shift)
        if npu_device == NPUDevice.A5:
            out = ops.adaln_v2(x=x, scale=scale, shift=shift, weight=weight, bias=bias, epsilon=layernorm.eps)[0]
        else:
            out = ops.adaln(x=x, scale=scale, shift=shift, weight=weight, bias=bias, epsilon=layernorm.eps)
        out = _postprocess_tokenwise_output(out, tokenwise_shape)
    else:
        if scale.dim() == 2:
            scale = scale[:, None]
        if shift.dim() == 2:
            shift = shift[:, None]
        out = layernorm(x) * (1 + scale) + shift

    return out
