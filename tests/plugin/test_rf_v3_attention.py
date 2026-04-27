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
import sys
import math
import unittest
import torch
import torch_npu

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


def ref_compare(golden, actual, err):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    EB = torch.mean(abs_error / golden_nmax)
    result = (abs_error <= err * golden_nmax).all() and EB <= err / 2
    return EB.item(), result.item(), abs_error.max().item()


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
class TestRfV3Attention(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        self.batch       = 1
        self.head_num    = 24
        self.head_dim    = 128
        self.pool_size   = 128
        self.dtype       = torch.bfloat16

        # h, w must be divisible by 8 for the rearrange logic.
        self.t, self.h, self.w = 4, 16, 16
        self.latent_shape = (self.t, self.h, self.w)
        self.seq_len = self.t * self.h * self.w  # 1024，pool_size 的整数倍

        self.scale = self.head_dim ** -0.5

        # 950 series requires inner_precise=4.
        dev_name = torch.npu.get_device_properties(self.device).name
        self.inner_precise = 4 if "950" in dev_name else 1

    def _make_qkv_bsnd(self, t=None, h=None, w=None):
        """Create BSND q/k/v tensors, defaulting to setUp dimensions."""
        t = t or self.t
        h = h or self.h
        w = w or self.w
        seq_len = t * h * w
        shape = (self.batch, seq_len, self.head_num, self.head_dim)
        q = torch.randn(shape, dtype=self.dtype, device=self.device)
        k = torch.randn(shape, dtype=self.dtype, device=self.device)
        v = torch.randn(shape, dtype=self.dtype, device=self.device)
        return q, k, v

    # mask shape and dtype tests

    def test_block_sparse_mask_shape_bsnd(self):
        """get_blockwise_mask with return_binary=True returns correct int8 mask shape (BSND)."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
            do_tensor_rearrange_pooling, get_blockwise_mask,
        )

        q, k, v = self._make_qkv_bsnd()
        _, _, _, qkv_pool = do_tensor_rearrange_pooling(
            q, k, v, 0, self.pool_size,
            self.latent_shape, self.latent_shape, "BSND"
        )
        mask = get_blockwise_mask(
            qkv_pool, 0, 0.5, self.scale, self.pool_size,
            self.latent_shape, self.latent_shape, "BSND",
            return_binary=True,
        )
        q_blocks  = math.ceil(self.seq_len / self.pool_size)
        kv_blocks = math.ceil(self.seq_len / self.pool_size)
        self.assertEqual(tuple(mask.shape), (self.batch, self.head_num, q_blocks, kv_blocks))
        self.assertEqual(mask.dtype, torch.int8)

    # first-frame protection tests

    def test_firstframe_protection_in_mask(self):
        """First-frame blocks must all be 1 regardless of sparsity."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
            do_tensor_rearrange_pooling, get_blockwise_mask,
        )

        q, k, v = self._make_qkv_bsnd()
        _, _, _, qkv_pool = do_tensor_rearrange_pooling(
            q, k, v, 0, self.pool_size,
            self.latent_shape, self.latent_shape, "BSND"
        )
        mask = get_blockwise_mask(
            qkv_pool, 0, 0.9, self.scale, self.pool_size,
            self.latent_shape, self.latent_shape, "BSND",
            return_binary=True,
        )
        first_frame_len = self.h * self.w
        firstframe_block_num = math.ceil(first_frame_len / self.pool_size)
        self.assertTrue(mask[:, :, :firstframe_block_num, :].eq(1).all().item(),
                        "first-frame row blocks are not all 1")
        self.assertTrue(mask[:, :, :, :firstframe_block_num].eq(1).all().item(),
                        "first-frame column blocks are not all 1")

    # bsa_sparse_attention_v3 output shape/dtype tests

    def test_bsa_sparse_attention_v3_output_shape(self):
        """bsa_sparse_attention_v3 output shape and dtype match input."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        out, mask = bsa_sparse_attention_v3(
            q, k, v,
            latent_shape_q=self.latent_shape,
            pool_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
        )
        self.assertEqual(out.shape, q.shape,
                         f"output shape {out.shape} != input {q.shape}")
        self.assertEqual(out.dtype, self.dtype)

    # unaligned S tests

    def test_bsa_sparse_attention_v3_unaligned_seq_len(self):
        """bsa_sparse_attention_v3 returns original shape when S is not a multiple of pool_size."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        # h=20, w=20 -> S = t*400; 400 % 128 = 16 != 0
        t, h, w = 3, 20, 20
        latent_shape = (t, h, w)
        q, k, v = self._make_qkv_bsnd(t=t, h=h, w=w)

        out, _ = bsa_sparse_attention_v3(
            q, k, v,
            latent_shape_q=latent_shape,
            pool_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
        )
        self.assertEqual(out.shape, q.shape,
                         f"unaligned: output shape {out.shape} != input {q.shape}")

    # accuracy tests: sparsity=0 vs dense

    def test_bsa_sparse_attention_v3_vs_dense(self):
        """With sparsity=0, bsa_sparse_attention_v3 should be statistically close
        to npu_fusion_attention (token order differs due to rearrange)."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        # Use float16 for easier comparison (bfloat16 has lower precision).
        dtype = torch.float16
        t, h, w = 2, 16, 16   # 较小尺寸，加快测试
        latent_shape = (t, h, w)
        seq_len = t * h * w
        shape_bsnd = (self.batch, seq_len, self.head_num, self.head_dim)

        q = torch.randn(shape_bsnd, dtype=dtype, device=self.device)
        k = torch.randn(shape_bsnd, dtype=dtype, device=self.device)
        v = torch.randn(shape_bsnd, dtype=dtype, device=self.device)

        # dense attention via npu_fusion_attention (no rearrange)
        q_bnsd = q.permute(0, 2, 1, 3)
        k_bnsd = k.permute(0, 2, 1, 3)
        v_bnsd = v.permute(0, 2, 1, 3)
        out_dense = torch_npu.npu_fusion_attention(
            q_bnsd, k_bnsd, v_bnsd,
            head_num=self.head_num,
            input_layout="BNSD",
            scale=self.scale,
            pre_tockens=2147483647,
            next_tockens=2147483647,
        )[0].permute(0, 2, 1, 3)  # → BSND

        # bsa_sparse_attention_v3 with sparsity=0 (all blocks retained, ~dense)
        out_v3, _ = bsa_sparse_attention_v3(
            q.clone(), k.clone(), v.clone(),
            latent_shape_q=latent_shape,
            pool_size=self.pool_size,
            sparsity=0.0,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
        )

        # Token order differs (v3 applies spatial rearrange), so compare statistics.
        dense_mean = out_dense.to(torch.float32).mean()
        v3_mean    = out_v3.to(torch.float32).mean()
        dense_std  = out_dense.to(torch.float32).std()
        v3_std     = out_v3.to(torch.float32).std()

        mean_rel_err = abs(dense_mean.item() - v3_mean.item()) / max(abs(dense_mean.item()), 1e-6)
        std_rel_err  = abs(dense_std.item()  - v3_std.item())  / max(abs(dense_std.item()),  1e-6)

        self.assertLess(mean_rel_err, 0.1,
                        f"mean rel err too large: dense={dense_mean:.4f}, v3={v3_mean:.4f}")
        self.assertLess(std_rel_err, 0.1,
                        f"std rel err too large: dense={dense_std:.4f}, v3={v3_std:.4f}")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
