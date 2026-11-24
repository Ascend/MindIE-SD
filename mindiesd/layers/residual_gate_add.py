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

"""Fused residual + gate: out = x + y * gate, with gate broadcast over S."""

import torch


def _normalize_residual_gate_inputs(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate inputs and normalize either multiplication operand order."""
    if x.dim() != 3 or y.dim() != 3 or gate.dim() != 3:
        raise ValueError(
            f"x, y, and gate must all be [B,S,D]; got x={tuple(x.shape)} y={tuple(y.shape)} gate={tuple(gate.shape)}"
        )

    # FX may preserve either source order: x + y * gate or x + gate * y.
    if y.shape != x.shape and gate.shape == x.shape:
        y, gate = gate, y

    if y.shape != x.shape:
        raise ValueError(f"x and y must have the same shape; got x={tuple(x.shape)} y={tuple(y.shape)}")
    batch, _, hidden_size = x.shape
    expected_gate_shape = (batch, 1, hidden_size)
    if gate.shape != expected_gate_shape:
        raise ValueError(f"gate must have shape {expected_gate_shape}; got gate={tuple(gate.shape)}")
    if x.numel() == 0:
        raise ValueError("x must not be empty")
    if x.device != y.device or x.device != gate.device:
        raise ValueError(f"x, y, and gate must be on the same device; got x={x.device} y={y.device} gate={gate.device}")
    if x.dtype != y.dtype or x.dtype != gate.dtype:
        raise ValueError(f"x, y, and gate must have the same dtype; got x={x.dtype} y={y.dtype} gate={gate.dtype}")
    return x, y, gate


def _validate_qwen_residual_gate_inputs(residual: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> None:
    if residual.dim() != 3 or branch.dim() != 3:
        raise ValueError(
            "Qwen residual and branch must both be [B,S,D]; "
            f"got residual={tuple(residual.shape)} branch={tuple(branch.shape)}"
        )
    if residual.shape != branch.shape:
        raise ValueError(
            "Qwen residual and branch must have identical [B,S,D] shapes; "
            f"got residual={tuple(residual.shape)} branch={tuple(branch.shape)}"
        )
    batch, _, dim = residual.shape
    if gate.dim() != 3 or gate.shape != (batch, 1, dim):
        raise ValueError(
            "Qwen gate must be exactly [B,1,D] for token-dimension broadcast; "
            f"got residual={tuple(residual.shape)} gate={tuple(gate.shape)}"
        )
    if residual.dtype != branch.dtype or residual.dtype != gate.dtype:
        raise ValueError(
            "Qwen residual, branch, and gate must share a dtype; "
            f"got residual={residual.dtype} branch={branch.dtype} gate={gate.dtype}"
        )


@torch.library.custom_op("mindiesd::qwen_residual_gate_add", mutates_args=())
def qwen_residual_gate_add(residual: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Qwen-Image residual + gated branch using native PyTorch ``addcmul``."""
    _validate_qwen_residual_gate_inputs(residual, branch, gate)
    return torch.addcmul(residual, branch, gate).contiguous()


@torch.library.custom_op("mindiesd::residual_gate_add", mutates_args=())
def residual_gate_add(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Residual + gate using native PyTorch ``addcmul``."""
    x, y, gate = _normalize_residual_gate_inputs(x, y, gate)
    return torch.addcmul(x, y, gate).contiguous()


@residual_gate_add.register_fake
def _residual_gate_add_fake(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x, _, _ = _normalize_residual_gate_inputs(x, y, gate)
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@qwen_residual_gate_add.register_fake
def _qwen_residual_gate_add_fake(residual: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    _validate_qwen_residual_gate_inputs(residual, branch, gate)
    return torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
