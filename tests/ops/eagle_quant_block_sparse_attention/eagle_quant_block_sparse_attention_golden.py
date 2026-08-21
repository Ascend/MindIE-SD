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

# Single-operator precision test for eagle_quant_block_sparse_attention.
# Ported from the standalone qbsa test.py; keeps wiki/source public kwargs
# (query_scale / *_dtype / output_dtype) on torch.ops.mindiesd.eagle_quant_block_sparse_attention.
#
# NOTE: this operator is an Ascend 950 (A5/arch35) operator. It must be run on a
# 950 device. On other devices the op kernel is not available.
#
# Usage:
#   python eagle_quant_block_sparse_attention_golden.py

import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu

# Load the MindIE-SD custom op library (registers torch.ops.mindiesd.*).
from mindiesd.layers.register_ops import _load_mindie_ops_library

_load_mindie_ops_library()

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))
device = "npu:" + str(DEVICE_ID)


def check_nan_inf(x, name, max_print=100):
    """Check tensor / scalar for NaN or Inf; raise if found."""
    if isinstance(x, float):
        if math.isnan(x):
            print(f"\n[NaN FOUND] {name} is NaN", flush=True)
            raise RuntimeError(f"NaN detected in {name}")
        if math.isinf(x):
            print(f"\n[Inf FOUND] {name} is Inf: {x}", flush=True)
            raise RuntimeError(f"Inf detected in {name}")
        return x

    if not torch.is_tensor(x):
        return x

    if not (x.is_floating_point() or x.is_complex()):
        return x

    x_cpu = x.detach().cpu()
    nan_mask = torch.isnan(x_cpu)
    inf_mask = torch.isinf(x_cpu)
    has_nan = nan_mask.any().item()
    has_inf = inf_mask.any().item()

    if has_nan or has_inf:
        print(f"\n[INVALID VALUE FOUND] tensor name: {name}", flush=True)
        print(f"shape  = {tuple(x.shape)}", flush=True)
        print(f"dtype  = {x.dtype}", flush=True)
        print(f"device = {x.device}", flush=True)
        if has_nan:
            nan_idx = nan_mask.nonzero(as_tuple=False)
            nan_count = nan_idx.shape[0]
            print(f"\nNaN count = {nan_count}", flush=True)
            print(f"NaN positions, first {min(max_print, nan_count)}:", flush=True)
            print(nan_idx[:max_print].tolist(), flush=True)
        if has_inf:
            inf_idx = inf_mask.nonzero(as_tuple=False)
            inf_count = inf_idx.shape[0]
            print(f"\nInf count = {inf_count}", flush=True)
            print(f"Inf positions, first {min(max_print, inf_count)}:", flush=True)
            print(inf_idx[:max_print].tolist(), flush=True)
            print(f"Inf values, first {min(max_print, inf_count)}:", flush=True)
            print(x_cpu[inf_mask][:max_print].tolist(), flush=True)
        raise RuntimeError(f"NaN or Inf detected in {name}")

    return x


def block_sparse_attention_cpu(query, key, value, smask, causal=False, blocksize=128):
    """CPU float reference (non-quant) for block sparse attention."""
    bs, nq, seq, dim = query.shape
    nkv = key.shape[1]
    gqa = nq // nkv

    output = torch.zeros(bs, nq, seq, dim, dtype=torch.float)
    query = query.float().cpu().numpy()
    key = key.float().cpu().numpy()
    value = value.float().cpu().numpy()
    smask = smask.cpu().numpy()

    for bi in range(bs):
        for ni in range(nq):
            num_blocks = (seq + blocksize - 1) // blocksize
            for s1 in range(num_blocks):
                mask_block = smask[bi, ni, s1, :num_blocks]
                mask_seq = np.repeat(mask_block, blocksize)[:seq].astype(bool)
                start = s1 * blocksize
                end = min((s1 + 1) * blocksize, seq)
                q = query[bi, ni, start:end]

                k_head = ni // gqa
                k = key[bi, k_head][mask_seq]
                kt = k.T

                p = q @ kt
                p = p / np.sqrt(dim)
                if causal:
                    t = end - start
                    cm = np.triu(np.ones((t, t)), k=1) * (-10000.0)
                    p[:, -t:] += cm

                p = p - p.max(axis=-1, keepdims=True)
                exp_p = np.exp(p)
                exp_sum = exp_p.sum(axis=-1, keepdims=True)
                attn = exp_p / (exp_sum + 1e-12)
                v = value[bi, k_head][mask_seq]
                out = attn @ v
                output[bi, ni, start:end] = torch.from_numpy(out)
    return output


def mask_to_indices_4d_for_loop(mask: torch.Tensor) -> torch.Tensor:
    """Convert a 4D bool mask into a left-packed int32 index tensor (pad -1)."""
    _, _, _, W = mask.shape
    result = torch.full_like(mask, -1, dtype=torch.long)
    mask_flat = mask.view(-1, W)
    result_flat = result.view(-1, W)
    for i in range(mask_flat.size(0)):
        current_row_mask = mask_flat[i]
        valid_indices = current_row_mask.nonzero(as_tuple=True)[0]
        num_valid = valid_indices.numel()
        if num_valid > 0:
            result_flat[i, :num_valid] = valid_indices
    return result.to(torch.int32)


def ref_compare1(golden: torch.Tensor, actual: torch.Tensor, err=None, print_flag=False):
    """Single-baseline float compare: |actual - expected| <= err * max(1, |expected|)."""
    if err is None:
        if actual.dtype == torch.float16:
            err = 2 ** (-10)
        else:
            err = 2 ** (-7)
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= err * golden_nmax).all()
    EB = torch.mean(abs_error / golden_nmax)
    if print_flag:
        print(f"----> EB: {EB.item():.3e} | max err: {abs_error.max().item():.3e}")
    return result.item(), EB.item(), abs_error.max().item()


@torch.no_grad()
def perblock_quant(input_tensor, block_size=128, dst_type=torch_npu.float8_e4m3fn, smooth=False, **kwargs):
    """Per-block quant preprocess for Q/K. Input layout 'BNSD' or 'BSND'."""
    assert len(input_tensor.shape) == 4, (
        f"fa block quant preprocess only support qkv quant, dim = 4, but got {len(input_tensor.shape)}."
    )

    layout = kwargs.get("layout", "BNSD")
    if layout == "BNSD":
        b, n, s, d = input_tensor.shape
    elif layout == "BSND":
        input_tensor = input_tensor.transpose(1, 2)
        b, n, s, d = input_tensor.shape
    else:
        raise ValueError("unsupport layout")

    if smooth:
        input_tensor = input_tensor - input_tensor.mean(dim=2, keepdim=True)

    if not s % block_size == 0:
        padding_length = (block_size - (s % block_size)) % block_size
        input_tensor = F.pad(input_tensor, (0, 0, 0, padding_length))

    input_tensor = input_tensor.reshape(b, n, math.ceil(s / block_size), -1)
    input_quant, input_scale = torch_npu.npu_dynamic_quant(input_tensor, dst_type=dst_type)

    if layout == "BNSD":
        input_quant = input_quant.reshape(b, n, -1, d)[:, :, :s, :]
    elif layout == "BSND":
        input_quant = input_quant.transpose(1, 2).reshape(b, -1, n, d)[:, :s, :, :]

    return input_quant, input_scale


def test_quant_eagle_block_sparse_attention(
    b=1, n1=1, s1=1024, d=128, n2=None, s2=None, sparsity=0.5, dtype=torch.bfloat16
):
    if not n2:
        n2 = n1
    if not s2:
        s2 = s1
    causal = False
    sparse_size = 128
    sn1 = (s1 + sparse_size - 1) // sparse_size
    query = torch.randn(b, n1, s1, d, dtype=dtype).npu()
    key = torch.randn(b, n2, s2, d, dtype=dtype).npu()
    value = torch.randn(b, n2, s2, d, dtype=dtype).npu()

    smask = torch.rand(b, n1, sn1, sn1) > sparsity
    smask[:, :, :, 0] = True
    smask[:, :, :, sn1:] = False
    smask[:, :, sn1 - 1 : sn1, :] = True
    smask[:, :, :, sn1 - 1 : sn1] = True
    smask = smask.npu()

    sn1 = smask.shape[2]

    q_block = 64
    q_q, q_scales = perblock_quant(query, block_size=q_block, dst_type=torch.int8, smooth=False)
    k_q, k_scales = perblock_quant(key, block_size=q_block, dst_type=torch.int8, smooth=False)
    v_q, v_scales = torch_npu.npu_dynamic_quant(value.transpose(-1, -2), dst_type=torch_npu.float8_e4m3fn)
    v_q = v_q.transpose(-1, -2)


    bsa_cpu = block_sparse_attention_cpu(query.cpu(), key.cpu(), value.cpu(), smask.cpu(), causal=causal, blocksize=128)
    check_nan_inf(bsa_cpu, "CPU-no-quant")

    # ---- run with mask (int8 block sparse mask) ----
    out, _ = torch.ops.mindiesd.eagle_quant_block_sparse_attention(
        query=q_q,
        key=k_q,
        value=v_q.view(torch.int8),
        block_sparse_mask=smask.view(torch.int8),
        block_shape=[128, 128],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=n2,
        scale_value=128 ** -0.5,
        inner_precise=4,
        softmax_lse_flag=0,
        actual_seq_lengths=[s1] * b,
        actual_seq_lengths_kv=[s1] * b,
        query_scale=q_scales,
        key_scale=k_scales,
        value_scale=v_scales,
        query_dtype=torch.int8,
        key_dtype=torch.int8,
        value_dtype=torch_npu.float8_e4m3fn,
        output_dtype=torch.bfloat16,
    )
    check_nan_inf(out, "npu quant eagle bsa (mask)")
    print("Compare with CPU-no-quant (mask mode):")
    _, eb_mask, err_mask = ref_compare1(bsa_cpu.ravel().cpu().float(), out.ravel().cpu().float(), print_flag=True)
    # ---- run with index (int32 indices derived from mask) ----
    sindex = mask_to_indices_4d_for_loop(smask)
    out, _ = torch.ops.mindiesd.eagle_quant_block_sparse_attention(
        query=q_q,
        key=k_q,
        value=v_q.view(torch.int8),
        block_sparse_mask=sindex,
        block_shape=[128, 128],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=n2,
        scale_value=128 ** -0.5,
        inner_precise=4,
        softmax_lse_flag=0,
        actual_seq_lengths=[s1] * b,
        actual_seq_lengths_kv=[s1] * b,
        query_scale=q_scales,
        key_scale=k_scales,
        value_scale=v_scales,
        query_dtype=torch.int8,
        key_dtype=torch.int8,
        value_dtype=torch_npu.float8_e4m3fn,
        output_dtype=torch.bfloat16,
    )
    check_nan_inf(out, "npu quant eagle bsa (index)")
    print("Compare with CPU-no-quant (index mode):")
    _, eb_index, err_index = ref_compare1(bsa_cpu.ravel().cpu().float(), out.ravel().cpu().float(), print_flag=True)

    # Pass criterion: mean relative error (EB) within tolerance for both modes
    # (NaN/Inf already raise above). bf16 + INT8/FP8 quant vs float golden:
    # EB_TOL = 1e-2 is a comfortable bound for this op.
    eb_tol = 1e-2
    passed = (eb_mask <= eb_tol) and (eb_index <= eb_tol)
    print("\n" + "=" * 60)
    print(f"[mask  mode] EB={eb_mask:.3e} max_err={err_mask:.3e}  (EB_TOL={eb_tol:.0e})")
    print(f"[index mode] EB={eb_index:.3e} max_err={err_index:.3e}  (EB_TOL={eb_tol:.0e})")
    print("=" * 60)
    if passed:
        print(">>> eagle_quant_block_sparse_attention TEST PASSED")
    else:
        print(">>> eagle_quant_block_sparse_attention TEST FAILED (EB exceeds tolerance)")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    ok = test_quant_eagle_block_sparse_attention(dtype=torch.bfloat16)
    sys.exit(0 if ok else 1)
