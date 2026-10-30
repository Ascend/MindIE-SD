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


"""Explicit opaque CPU doubles for hardware dispatch boundaries in FX tests.

These operators stay opaque to make_fx so replacement graphs cannot silently
collapse back into their unfused reference. They are never installed globally.
"""
# pylint: disable=function-redefined

import torch
import torch.nn.functional as F


@torch.library.custom_op("mindiesd_test_cpu::fast_gelu", mutates_args=())
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


@fast_gelu.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("mindiesd_test_cpu::rms", mutates_args=())
def rms(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
    return (x.float() * inverse * weight.float()).to(x.dtype), inverse


@rms.register_fake
def _(x, weight, epsilon):
    return torch.empty_like(x), torch.empty_like(x[..., :1], dtype=torch.float32)


@torch.library.custom_op("mindiesd_test_cpu::rope", mutates_args=())
def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "interleave":
        real, imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        rotated = torch.stack((-imag, real), -1).flatten(-2)
    else:
        real, imag = x.chunk(2, -1)
        rotated = torch.cat((-imag, real), -1)
    return x * cos + rotated * sin


@rope.register_fake
def _(x, cos, sin, mode):
    return torch.empty_like(x)


@torch.library.custom_op("mindiesd_test_cpu::adaln", mutates_args=())
def adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    if scale.ndim == x.ndim - 1:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
    return F.layer_norm(x, (x.shape[-1],), eps=eps) * (scale + 1) + shift


@adaln.register_fake
def _(x, scale, shift, eps):
    return torch.empty_like(x)


@torch.library.custom_op("mindiesd_test_cpu::gate", mutates_args=())
def gate(x: torch.Tensor, value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x + value * scale


@gate.register_fake
def _(x, value, scale):
    return torch.empty_like(x)


@torch.library.custom_op("mindiesd_test_cpu::mul_add", mutates_args=())
def mul_add(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale + y


@mul_add.register_fake
def _(x, y, scale):
    return torch.empty_like(x)


@torch.library.custom_op("mindiesd_test_cpu::group_norm_silu", mutates_args=())
def group_norm_silu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized, mean, rstd = torch.native_group_norm(
        x, weight, bias, x.shape[0], x.shape[1], x.numel() // (x.shape[0] * x.shape[1]), groups, eps
    )
    return F.silu(normalized), mean, rstd


@group_norm_silu.register_fake
def _(x, weight, bias, groups, eps):
    stats = torch.empty((x.shape[0], groups), dtype=x.dtype, device=x.device)
    return torch.empty_like(x), stats, torch.empty_like(stats)
