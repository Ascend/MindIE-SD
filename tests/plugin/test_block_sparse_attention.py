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

# pylint: disable=duplicate-code

import os
import sys
import math
import unittest

import numpy as np
import torch
import torch_npu

# Add project root to sys.path for mindiesd import.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mindiesd.utils.get_platform import is_a5_device  # noqa: E402

# 加载自定义库
if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


# CPU reference implementation


def block_sparse_attention_cpu(query, key, value, block_sparse_mask, blocksize=128):
    """CPU reference: block_sparse_mask (int8 [B,N,q_blocks,kv_blocks]); 1=attend, 0=skip."""
    bs, nq, seq, dim = query.shape
    nkv = key.shape[1]
    gqa = nq // nkv
    output = torch.zeros(bs, nq, seq, dim, dtype=torch.float32)

    query_f = query.float().cpu().numpy()
    key_f = key.float().cpu().numpy()
    value_f = value.float().cpu().numpy()
    mask_np = block_sparse_mask.cpu().numpy()

    for bi in range(bs):
        for ni in range(nq):
            num_blocks = math.ceil(seq / blocksize)
            for s1 in range(num_blocks):
                mask_block = mask_np[bi, ni, s1, :num_blocks]  # [kv_blocks]
                mask_seq = np.repeat(mask_block, blocksize)[:seq].astype(bool)
                start = s1 * blocksize
                end = min((s1 + 1) * blocksize, seq)
                q = query_f[bi, ni, start:end]  # [q_len, dim]
                k_idx = ni // gqa
                k = key_f[bi, k_idx][mask_seq]
                v = value_f[bi, k_idx][mask_seq]
                if k.shape[0] == 0:
                    out = np.zeros((end - start, dim), dtype=np.float32)
                else:
                    p = q @ k.T / np.sqrt(dim)
                    p = p - p.max(axis=-1, keepdims=True)
                    exp_p = np.exp(p)
                    attn = exp_p / (exp_p.sum(axis=-1, keepdims=True) + 1e-12)
                    out = attn @ v
                output[bi, ni, start:end] = torch.from_numpy(out)
    return output


def ref_compare(golden, actual, err):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    EB = torch.mean(abs_error / golden_nmax)
    result = (abs_error <= err * golden_nmax).all() and EB <= err / 2
    return EB.item(), result.item(), abs_error.max().item()


def make_block_sparse_mask(batch, head_num, seq_len, sparse_size, sparsity=0.5, seed=42):
    """Generate random int8 block_sparse_mask [B, N, q_blocks, kv_blocks]."""
    rng = np.random.default_rng(seed)
    q_blocks = math.ceil(seq_len / sparse_size)
    kv_blocks = math.ceil(seq_len / sparse_size)
    mask = (rng.random((batch, head_num, q_blocks, kv_blocks)) > sparsity).astype(np.int8)
    # Ensure at least one block per row is 1.
    for b in range(batch):
        for n in range(head_num):
            for q in range(q_blocks):
                if mask[b, n, q].sum() == 0:
                    mask[b, n, q, 0] = 1
    return torch.from_numpy(mask)


# NPU tests.
# Fake-op tests (Meta device) are in test_block_sparse_attention_fake_op.py.
# Importing _custom_ops registers a fake op into PyTorch's abstract impl table
# (process-level global state), which causes aclnn shape inference to return
# invalid values (0xFFFFFFFF) in non-Meta paths. The two test files must run
# in separate processes.


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
@unittest.skipIf(not is_a5_device(), "Block Sparse Attention requires A5 (950) NPU.")
class TestNpuBlockSparseAttentionNPU(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        self.batch = 1
        self.head_num = 1
        self.head_dim = 128
        self.seq_len = 75392  # minimum viable sequence length
        self.sparse_size = 128
        self.scale = self.head_dim**-0.5
        # 950 series: inner_precise=4 (op vendor requirement); others: 1 (fp16 fast)
        dev_name = torch.npu.get_device_properties(self.device).name
        self.inner_precise = 4 if "950" in dev_name else 1

    def _full_mask(self):
        """All-ones block_sparse_mask [B, N, q_blocks, kv_blocks]."""
        q_blocks = math.ceil(self.seq_len / self.sparse_size)
        kv_blocks = math.ceil(self.seq_len / self.sparse_size)
        return torch.ones(self.batch, self.head_num, q_blocks, kv_blocks, dtype=torch.int8)

    def _call_op(
        self,
        q,
        k,
        v,
        mask,
        layout="BNSD",
        actual_seq_lengths=None,
        actual_seq_lengths_kv=None,
        softmax_lse_flag=0,
        block_shape=None,
        q_dequant_scale=None,
        k_dequant_scale=None,
        v_dequant_scale=None,
    ):
        # Default actual_seq_lengths if not provided.
        if actual_seq_lengths is None:
            actual_seq_lengths = [self.seq_len] * self.batch
        if actual_seq_lengths_kv is None:
            actual_seq_lengths_kv = [self.seq_len] * self.batch
        if block_shape is None:
            block_shape = [self.sparse_size, self.sparse_size]
        kwargs = dict(
            query=q.to(self.device),
            key=k.to(self.device),
            value=v.to(self.device),
            block_sparse_mask=mask.to(self.device),
            block_shape=block_shape,
            q_input_layout=layout,
            kv_input_layout=layout,
            num_key_value_heads=self.head_num,
            scale_value=self.scale,
            inner_precise=self.inner_precise,
            softmax_lse_flag=softmax_lse_flag,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        if q_dequant_scale is not None:
            kwargs.update(
                q_dequant_scale=q_dequant_scale.to(self.device),
                k_dequant_scale=k_dequant_scale.to(self.device),
                v_dequant_scale=v_dequant_scale.to(self.device),
            )
        return torch.ops.mindiesd.block_sparse_attention(**kwargs)

    # smoke test 1: BNSD full mask (BF16)

    def test_smoke_bnsd(self):
        """BNSD smoke test: output shape matches query."""
        B, N, S, D = self.batch, self.head_num, self.seq_len, self.head_dim
        q = torch.randn(B, N, S, D, dtype=torch.float16)
        k = torch.randn(B, N, S, D, dtype=torch.float16)
        v = torch.randn(B, N, S, D, dtype=torch.float16)
        mask = self._full_mask()
        attn_out, lse = self._call_op(q, k, v, mask, layout="BNSD")
        self.assertEqual(tuple(attn_out.shape), (B, N, S, D))
        self.assertEqual(attn_out.dtype, torch.float16)

    # smoke test 2: TND full mask (BF16)

    def test_smoke_tnd(self):
        """TND smoke test: output shape is [T, N, D]."""
        B, N, S, D = self.batch, self.head_num, self.seq_len, self.head_dim
        T = B * S
        q = torch.randn(T, N, D, dtype=torch.float16)
        k = torch.randn(T, N, D, dtype=torch.float16)
        v = torch.randn(T, N, D, dtype=torch.float16)
        mask = self._full_mask()
        seq_lens = [S] * B
        attn_out, lse = self._call_op(
            q,
            k,
            v,
            mask,
            layout="TND",
            actual_seq_lengths=seq_lens,
            actual_seq_lengths_kv=seq_lens,
        )
        self.assertEqual(tuple(attn_out.shape), (T, N, D))
        self.assertEqual(attn_out.dtype, torch.float16)

    # smoke test 3: BNSD FP8 with dequant scales

    def test_smoke_bnsd_fp8(self):
        """FP8 BNSD smoke test: FP8 QKV + dequant scales → BF16 output."""
        B, N, S, D = self.batch, self.head_num, self.seq_len, self.head_dim
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

        q_bf16 = torch.randn(B, N, S, D, dtype=torch.bfloat16).npu()
        k_bf16 = torch.randn(B, N, S, D, dtype=torch.bfloat16).npu()
        v_bf16 = torch.randn(B, N, S, D, dtype=torch.bfloat16).npu()

        # Block-quantize to FP8 (BNSD layout, function handles squeeze/unsqueeze internally)
        q_block, kv_block = 128, 256
        q_fp8, q_scale = fa_block_quant_preprocess(
            q_bf16, block_size=q_block, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
        )
        k_fp8, k_scale = fa_block_quant_preprocess(
            k_bf16, block_size=kv_block, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
        )
        v_fp8, v_scale = fa_block_quant_preprocess(
            v_bf16, block_size=kv_block, dst_type=torch_npu.float8_e4m3fn, layout="BNSD"
        )

        # FP8 uses [q_block, kv_block] = [128, 256]; mask must match this granularity
        q_blocks = math.ceil(self.seq_len / q_block)
        kv_blocks = math.ceil(self.seq_len / kv_block)
        mask = torch.ones(self.batch, self.head_num, q_blocks, kv_blocks, dtype=torch.int8)
        attn_out, lse = self._call_op(
            q_fp8,
            k_fp8,
            v_fp8,
            mask,
            layout="BNSD",
            block_shape=[q_block, kv_block],
            q_dequant_scale=q_scale,
            k_dequant_scale=k_scale,
            v_dequant_scale=v_scale,
        )
        self.assertEqual(tuple(attn_out.shape), (B, N, S, D))
        self.assertEqual(attn_out.dtype, torch.bfloat16)

    # BF16 backward compatibility: passing no scales should work

    def test_bnsd_bf16_no_scales(self):
        """BF16 BNSD without dequant scales: backward compatible with V1 behavior."""
        B, N, S, D = self.batch, self.head_num, self.seq_len, self.head_dim
        q = torch.randn(B, N, S, D, dtype=torch.bfloat16)
        k = torch.randn(B, N, S, D, dtype=torch.bfloat16)
        v = torch.randn(B, N, S, D, dtype=torch.bfloat16)
        mask = self._full_mask()
        attn_out, lse = self._call_op(q, k, v, mask, layout="BNSD")
        self.assertEqual(tuple(attn_out.shape), (B, N, S, D))
        self.assertEqual(attn_out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
