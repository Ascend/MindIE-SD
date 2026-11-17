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
import torch
import torch_npu
import scipy.linalg
from ...utils import ParametersInvalid


def fa_block_quant_preprocess(input_tensor, block_size=128, col_block_size=128, dst_type=None, **kwargs):
    """
    Preprocess for FA quant. Input layout must be 'BNSD' or 'BSND'.
    Args:
        input_tensor (torch.Tensor): Input tensor to be quantized.
        block_size (int, optional): Block size for quantization. Support 128/256/512. Default: 128.
        col_block_size (int, optional): D-axis block size for quantization. Default: 128.
        dst_type (torch.dtype, optional): Target quantization data type. Default: torch_npu.float8_e4m3fn.
        **kwargs:
            layout (str): Tensor layout format, supports 'BNSD' (Batch, Num_heads, Seq_len, Dim)
                         or 'BSND' (Batch, Seq_len, Num_heads, Dim).

    Returns:
        torch.Tensor: Preprocessed tensor ready for FA block quantization.
    """

    # Resolve the optional dtype at execution time, not during package import.
    if dst_type is None:
        dst_type = getattr(torch_npu, "float8_e4m3fn", None)
        if dst_type is None:
            raise RuntimeError("Block-FP8 quantization requires torch_npu.float8_e4m3fn.")

    if len(input_tensor.shape) != 4:
        raise ParametersInvalid(
            f"fa block quant preprocess only support qkv quant, dim = 4, \
                                but got {len(input_tensor.shape)}."
        )

    return _fa_block_quant_preprocess(
        input_tensor,
        block_size=block_size,
        col_block_size=col_block_size,
        dst_type=dst_type,
        layout=kwargs.get("layout", "BNSD"),
    )


def _fa_block_quant_preprocess(input_tensor, *, block_size, dst_type, layout, col_block_size=128):
    """Quantize a validated four-dimensional input and restore the batch axis."""
    if layout == "BSND":
        input_tensor = input_tensor.transpose(1, 2)

    # block_quant only support BNSD
    input_quant, input_scale = torch_npu.npu_dynamic_block_quant(
        input_tensor.squeeze(0), dst_type=dst_type, row_block_size=block_size, col_block_size=col_block_size
    )

    return input_quant.unsqueeze(0), input_scale.unsqueeze(0)


def generate_hadamard_matrix(block_size=32, device='cpu', transpose=True, seed=42):
    """
    Generate a block_size-order Hadamard matrix with random signs.

    Each column of the standard Hadamard matrix is multiplied by a random
    ±1 sign, which spreads out outliers in weights/activations and reduces
    the error of MXFP4 group-wise quantization.

    Args:
        block_size: Matrix order, default 32 to match the MXFP4 group size.
        device: Device on which the returned tensor lives.
        transpose: If True, return the transpose so it can be used directly
            for left multiplication.
        seed: Seed for the random signs, default 42. Only work in local func.

    Returns:
        torch.Tensor: A (block_size, block_size) float32 tensor with ±1 entries
        (no 1/sqrt(n) normalization applied).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(low=0, high=2, size=(block_size,), generator=generator, device="cpu").to(torch.float32)
    q = q * 2 - 1
    h = torch.tensor(scipy.linalg.hadamard(block_size), dtype=torch.float32, device="cpu") * q
    h = h.to(device)
    if transpose:
        h = h.T
    return h


def apply_hadamard_weight(weight, hadamard, left=True):
    """
    Apply hadamard to weight block-wise: left-multiply the K dim or right-multiply the last dim.

    Args:
        weight: (..., K, N) weight tensor to rotate.
        hadamard: (n, n), un-normalized with entries ±1 (H @ H.T = n * I).
        left=True:  K (second-to-last dim) of weight must be a multiple of n;
            each K-dim block computes H @ W / n.
        left=False: N (last dim) of weight must be a multiple of n; each N-dim
            block computes W @ H / n.

    Returns:
        torch.Tensor: The rotated weight tensor, same shape as the input and
        cast to float32. Each block along the rotated dim equals H @ W / n
        (left=True) or W @ H / n (left=False).
    """
    weight = weight.float()
    hadamard = hadamard.float()
    out_shape = weight.shape

    n = hadamard.shape[0]
    if left:
        k = weight.shape[-2]
        assert k % n == 0, f"K={k} is not divisible by the hadamard order {n}"
        lead = weight.shape[:-2]

        weight = weight.reshape(*lead, k // n, n, -1)
        weight = hadamard @ weight / n
    else:
        k = weight.shape[-1]
        assert k % n == 0, f"N={k} is not divisible by the hadamard order {n}"
        lead = weight.shape[:-1]

        weight = weight.reshape(*lead, k // n, n)
        weight = weight @ hadamard / n
    return weight.reshape(out_shape)
