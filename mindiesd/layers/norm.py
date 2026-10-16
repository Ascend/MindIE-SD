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
from numbers import Real

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn

from ..utils.exception import ParametersInvalid
from . import _custom_ops as ops


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, if_fused=True):
        if hidden_states.dim() < 2 or hidden_states.dim() > 8:
            raise ParametersInvalid("The input dimension should be greater than 1 and less than 9.")
        if if_fused:
            # pylint: disable=no-member
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


_SUPPORTED_ADD_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_add_norm_inputs(x, residual, weight, bias, eps, fused):
    for name, value in (("x", x), ("residual", residual), ("weight", weight)):
        if not isinstance(value, torch.Tensor):
            raise ParametersInvalid(f"The data type of input {name} must be torch.Tensor, but got {type(value)}.")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise ParametersInvalid(f"The data type of input bias must be torch.Tensor, but got {type(bias)}.")
    if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ParametersInvalid(f"The input eps must be a finite positive number, but got {eps}.")
    if not isinstance(fused, bool):
        raise ParametersInvalid(f"The data type of input fused must be bool, but got {type(fused)}.")
    if x.dim() < 2 or x.dim() > 8:
        raise ParametersInvalid("The input dimension must be between 2 and 8.")
    if x.dtype not in _SUPPORTED_ADD_NORM_DTYPES:
        raise ParametersInvalid(f"The input dtype must be one of {_SUPPORTED_ADD_NORM_DTYPES}, but got {x.dtype}.")
    if x.shape != residual.shape:
        raise ParametersInvalid(
            f"The shape of x must be equal to the shape of residual, but got {tuple(x.shape)} and "
            f"{tuple(residual.shape)}."
        )
    if x.device != residual.device or x.dtype != residual.dtype:
        raise ParametersInvalid("The device and dtype of x and residual must be the same.")
    if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
        raise ParametersInvalid(
            f"The shape of weight must match the last dimension of x, but got {tuple(weight.shape)} and {x.shape[-1]}."
        )
    if bias is not None and bias.shape != weight.shape:
        raise ParametersInvalid(
            "The shape of bias must be equal to the shape of weight, but got "
            f"{tuple(bias.shape)} and {tuple(weight.shape)}."
        )
    parameters = (("weight", weight),) if bias is None else (("weight", weight), ("bias", bias))
    for name, parameter in parameters:
        if parameter.device != x.device or parameter.dtype != x.dtype:
            raise ParametersInvalid(f"The device and dtype of {name} must match x.")
    if fused and x.device.type != "npu":
        raise ParametersInvalid("The fused implementation only supports NPU tensors.")


def add_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    fused: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a residual tensor and apply LayerNorm.

    Args:
        x: Input tensor with 2 to 8 dimensions.
        residual: Residual tensor with the same shape, dtype and device as ``x``.
        weight: One-dimensional LayerNorm weight matching the last dimension of ``x``.
        bias: One-dimensional LayerNorm bias with the same shape as ``weight``.
        eps: Finite positive value added to the variance for numerical stability.
        fused: Use ``torch_npu.npu_add_layer_norm`` when ``True``. This mode requires NPU tensors.
            When ``False``, use the native PyTorch reference implementation.

    Returns:
        A tuple of ``(normalized_output, residual_output)``, where ``residual_output`` is
        ``x + residual``. Both outputs have the same shape, dtype and device as ``x``.

    Raises:
        ParametersInvalid: If any input violates the documented shape, dtype or device constraints.
        RuntimeError: If the fused backend returns an invalid output structure.
    """
    _check_add_norm_inputs(x, residual, weight, bias, eps, fused)
    if fused:
        result = torch_npu.npu_add_layer_norm(  # pylint: disable=no-member
            x, residual, weight, bias, float(eps), True
        )
        if not isinstance(result, (tuple, list)) or len(result) < 4:
            raise RuntimeError("npu_add_layer_norm returned an invalid result")
        return result[0], result[3]

    residual_output = x + residual
    normalized_output = F.layer_norm(
        residual_output.float(),
        tuple(weight.shape),
        weight.float(),
        bias.float(),
        float(eps),
    ).to(x.dtype)
    return normalized_output, residual_output


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    fused: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a residual tensor and apply RMSNorm.

    Args:
        x: Input tensor with 2 to 8 dimensions.
        residual: Residual tensor with the same shape, dtype and device as ``x``.
        weight: One-dimensional RMSNorm weight matching the last dimension of ``x``.
        eps: Finite positive value added to the mean square for numerical stability.
        fused: Use ``torch_npu.npu_add_rms_norm`` when ``True``. This mode requires NPU tensors.
            When ``False``, use the native PyTorch reference implementation.

    Returns:
        A tuple of ``(normalized_output, residual_output)``, where ``residual_output`` is
        ``x + residual``. Both outputs have the same shape, dtype and device as ``x``.

    Raises:
        ParametersInvalid: If any input violates the documented shape, dtype or device constraints.
        RuntimeError: If the fused backend returns an invalid output structure.
    """
    _check_add_norm_inputs(x, residual, weight, None, eps, fused)
    if fused:
        result = torch_npu.npu_add_rms_norm(  # pylint: disable=no-member
            x, residual, weight, float(eps)
        )
        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise RuntimeError("npu_add_rms_norm returned an invalid result")
        return result[0], result[2]

    residual_output = x + residual
    residual_fp32 = residual_output.float()
    variance = residual_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized_output = residual_fp32 * torch.rsqrt(variance + float(eps))
    normalized_output = (normalized_output * weight.float()).to(x.dtype)
    return normalized_output, residual_output


def check_input_params(layernorm, x, impl_mode, fused):
    if not isinstance(layernorm, torch.nn.LayerNorm):
        raise ParametersInvalid(f"The type of input layernorm must be torch.nn.LayerNorm, but got {type(layernorm)}.")
    if not isinstance(fused, bool):
        raise ParametersInvalid(f"The data type of input fused must be bool, but got {type(fused)}.")
    if impl_mode not in [0, 1, 2]:
        raise ParametersInvalid(f"Expected impl_mode to be in [0, 1, 2], but now got [{impl_mode}]")
    if len(layernorm.normalized_shape) > x.dim():
        raise ParametersInvalid(
            f"normalized_shape must fit within input dimensions, but got "
            f"normalized_shape={list(layernorm.normalized_shape)} (ndim={len(layernorm.normalized_shape)}) "
            f"and input.dim()={x.dim()}"
        )
    if impl_mode == 2:
        if not (
            x.dtype == torch.float16
            and (layernorm.weight is None or layernorm.weight.dtype == torch.float16)
            and (layernorm.bias is None or layernorm.bias.dtype == torch.float16)
        ):
            raise ParametersInvalid("only support all input dtype float16!")


def fast_layernorm(norm: torch.nn.LayerNorm, x: torch.Tensor, impl_mode: int = 0, fused: bool = True) -> torch.Tensor:
    """
    Args:
        norm (torch.nn.LayerNorm):
            The LayerNorm module.
        x (torch.Tensor):
            Tensor to apply LayerNorm. x must be 3-dimensional.
            The supported layout: [B,S,H].
        impl_mode (int):
            Specifies the compute mode for the kernel. The value must be in [0, 1, 2]. The default value is 0.
            0 indicates the high-precision mode, 1 indicates the high-performance mode, and 2 indicates the
            float16 mode. The float16 mode is supported only when all inputs are float16.
        fused (bool):
            If fused is True, can enable different layernorm mode by specifying 'impl_mode'.
    """
    check_input_params(norm, x, impl_mode, fused)
    if fused:
        out = ops.layernorm(
            x=x,
            normalized_shape=list(norm.normalized_shape),
            weight=norm.weight,
            bias=norm.bias,
            eps=norm.eps,
            impl_mode=impl_mode,
        )[0]
    else:
        out = norm(x)
    return out
