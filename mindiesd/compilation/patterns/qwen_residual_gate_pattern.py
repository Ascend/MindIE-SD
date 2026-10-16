#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Qwen-Image-only residual-gate fusion.

This intentionally does not reuse Wan's generic ``x + y * gate`` pattern.
Qwen's gate is ``[B,1,D]`` and must broadcast only over the token dimension of
``residual`` and ``branch`` (both ``[B,S,D]``).
"""

import torch

from ..passes.register_pattern_to_pass import PatternBase

npu_available = (
    hasattr(torch, "npu")
    and hasattr(torch.npu, "is_available")
    and torch.npu.is_available()
)
if npu_available:
    import torch_npu  # noqa: F401

import mindiesd


def _is_qwen_residual_gate_match(match) -> bool:
    try:
        residual = match.kwargs["residual"].meta["val"]
        branch = match.kwargs["branch"].meta["val"]
        gate = match.kwargs["gate"].meta["val"]
    except (AttributeError, KeyError):
        return False
    if not all(isinstance(value, torch.Tensor) for value in (residual, branch, gate)):
        return False
    if residual.dim() != 3 or branch.dim() != 3 or gate.dim() != 3:
        return False
    if residual.shape != branch.shape:
        return False
    batch, _, dim = residual.shape
    return (
        gate.shape == (batch, 1, dim)
        and residual.dtype == branch.dtype == gate.dtype
        and residual.dtype in (torch.bfloat16, torch.float32)
    )


def create(dtype):
    class QwenResidualGatePattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            residual = torch.empty(1, 16, 5120, dtype=dtype, device="meta")
            branch = torch.empty(1, 16, 5120, dtype=dtype, device="meta")
            gate = torch.empty(1, 1, 5120, dtype=dtype, device="meta")
            return [residual, branch, gate]

        @staticmethod
        def pattern(residual, branch, gate):
            def func(residual, branch, gate):
                return residual + gate * branch

            return func(residual, branch, gate)

        @staticmethod
        def replacement(residual, branch, gate):
            def func(residual, branch, gate):
                return mindiesd.layers.qwen_residual_gate_add(residual, branch, gate)

            return func(residual, branch, gate)

        extra_check = staticmethod(_is_qwen_residual_gate_match)

    return QwenResidualGatePattern


QwenResidualGatePatternGroup = [create(torch.bfloat16), create(torch.float32)]
