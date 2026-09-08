#!/usr/bin/env python
# pylint: disable=duplicate-code
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
from unittest import mock

import torch
import torch_npu

from mindiesd.utils.get_platform import is_a5_device  # noqa: E402

# 加载自定义库
if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    from mindiesd.layers.register_ops import _load_mindie_ops_library

    _load_mindie_ops_library()


def _make_rotation_matrices(head_dim, device, dtype=torch.float32):
    """Create orthogonal rotation matrices (same as WanSelfAttention init)."""
    rand_mat = torch.randn(head_dim, head_dim, dtype=dtype, device=device)
    rot, _ = torch.linalg.qr(rand_mat)  # pylint: disable=not-callable
    return rot, rot


def _is_bsa_v2_available():
    """Detect whether aclnnBlockSparseAttentionV2 exists in libopapi.so.

    Older CANN only ships V1 (aclnnBlockSparseAttention); its FP8 path is provided by
    the V2 kernel. When V2 is absent the plugin falls back to V1 (BF16/FP16 only), so
    FP8 test scenarios cannot run and must be skipped.
    """
    import ctypes

    try:
        lib = ctypes.CDLL("libopapi.so")
    except OSError:
        return False
    return hasattr(lib, "aclnnBlockSparseAttentionV2")


# FP8 BSA relies on aclnnBlockSparseAttentionV2; skip FP8 cases on CANN without V2.
_SKIP_NO_BSA_V2 = unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU" and not _is_bsa_v2_available(),
    "FP8 BSA path requires aclnnBlockSparseAttentionV2 (newer CANN); skipped on CANN without V2.",
)


def _is_bsa_v3_available():
    """Detect whether aclnnBlockSparseAttentionV3 exists in libopapi.so.

    MXFP4 BSA is only provided by the V3 kernel. When V3 is absent the plugin
    rejects quant_mode >= 2 at the TORCH_CHECK, so MXFP4 test scenarios must be
    skipped instead of failing.
    """
    import ctypes

    try:
        lib = ctypes.CDLL("libopapi.so")
    except OSError:
        return False
    return hasattr(lib, "aclnnBlockSparseAttentionV3")


# MXFP4 BSA relies on aclnnBlockSparseAttentionV3; skip MXFP4 cases on CANN without V3.
_SKIP_NO_BSA_V3 = unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU" and not _is_bsa_v3_available(),
    "MXFP4 BSA path requires aclnnBlockSparseAttentionV3 (newest CANN); skipped on CANN without V3.",
)


def _has_mxfp4_dtypes():
    """Whether torch_npu exposes the FP4 (E2M1) and E8M0 scale dtypes.

    The MXFP4 path threads these dtype codes into the kernel call and evaluates
    them as dst_type arguments inside _mxfp4_quant_qkv, so even fully mocked
    tests need the attributes to exist. Skip instead of failing on torch_npu
    builds without them.
    """
    return hasattr(torch_npu, "float4_e2m1fn_x2") and hasattr(torch_npu, "float8_e8m0fnu")


# MXFP4 cases reference the FP4/E8M0 dtype codes; skip on torch_npu without them.
_SKIP_NO_MXFP4_DTYPES = unittest.skipIf(
    not _has_mxfp4_dtypes(),
    "MXFP4 tests require torch_npu.float4_e2m1fn_x2 / float8_e8m0fnu (newer torch_npu).",
)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
@unittest.skipIf(not is_a5_device(), "Block Sparse Attention requires A5 (950) NPU.")
class TestRfV3Attention(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        self.batch = 1
        self.head_num = 24
        self.head_dim = 128
        self.pool_size = 128
        self.dtype = torch.bfloat16

        # h, w must be divisible by 8 for the rearrange logic.
        self.t, self.h, self.w = 4, 16, 16
        self.latent_shape = (self.t, self.h, self.w)
        self.seq_len = self.t * self.h * self.w  # 1024，pool_size 的整数倍

        self.scale = self.head_dim**-0.5

        # 950 series requires inner_precise=4.
        dev_name = torch.npu.get_device_properties(self.device).name
        self.inner_precise = 4 if "950" in dev_name else 0

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
            do_tensor_rearrange_pooling,
            get_blockwise_mask,
        )

        q, k, v = self._make_qkv_bsnd()
        _, _, _, qkv_pool = do_tensor_rearrange_pooling(
            q, k, v, 0, self.pool_size, self.latent_shape, self.latent_shape, "BSND"
        )
        mask = get_blockwise_mask(
            qkv_pool,
            0,
            0.5,
            self.scale,
            self.pool_size,
            self.latent_shape,
            self.latent_shape,
            "BSND",
            return_binary=True,
        )
        q_blocks = math.ceil(self.seq_len / self.pool_size)
        kv_blocks = math.ceil(self.seq_len / self.pool_size)
        self.assertEqual(tuple(mask.shape), (self.batch, self.head_num, q_blocks, kv_blocks))
        self.assertEqual(mask.dtype, torch.int8)

    # first-frame protection tests

    def test_firstframe_protection_in_mask(self):
        """First-frame blocks must all be 1 regardless of sparsity."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
            do_tensor_rearrange_pooling,
            get_blockwise_mask,
        )

        q, k, v = self._make_qkv_bsnd()
        _, _, _, qkv_pool = do_tensor_rearrange_pooling(
            q, k, v, 0, self.pool_size, self.latent_shape, self.latent_shape, "BSND"
        )
        mask = get_blockwise_mask(
            qkv_pool,
            0,
            0.9,
            self.scale,
            self.pool_size,
            self.latent_shape,
            self.latent_shape,
            "BSND",
            return_binary=True,
        )
        first_frame_len = self.h * self.w
        firstframe_block_num = math.ceil(first_frame_len / self.pool_size)
        self.assertTrue(
            mask[:, :, :firstframe_block_num, :].eq(1).all().item(),
            "first-frame row blocks are not all 1",
        )
        self.assertTrue(
            mask[:, :, :, :firstframe_block_num].eq(1).all().item(),
            "first-frame column blocks are not all 1",
        )

    # bsa_sparse_attention_v3 BF16 output shape/dtype tests

    def test_bsa_sparse_attention_v3_output_shape(self):
        """bsa_sparse_attention_v3 BF16 output shape and dtype match input."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        out, mask = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            precision="bf16",
        )
        self.assertEqual(out.shape, q.shape, f"output shape {out.shape} != input {q.shape}")
        self.assertEqual(out.dtype, self.dtype)

    def test_multi_video_sparse_attention_uses_rf_v3(self):
        """The multi-video sparse_attention path dispatches to rf_v3 on A5."""
        from mindiesd.layers.flash_attn.sparse_flash_attn import sparse_attention

        q, k, v = self._make_qkv_bsnd()
        spans = [
            {"start": 64, "latent_shape": [2, 8, 16]},
            {"start": 512, "latent_shape": [2, 8, 16]},
        ]
        out = sparse_attention(
            q,
            k,
            v,
            video_spans=spans,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            sparse_type="rf_v2",
        )

        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, self.dtype)

    # bsa_sparse_attention_v3 FP8 output shape/dtype tests

    @_SKIP_NO_BSA_V2
    def test_bsa_sparse_attention_v3_fp8_output_shape(self):
        """bsa_sparse_attention_v3 FP8 path: BF16 output with q_rot/k_rot provided."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)
        out, mask = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            q_rot=q_rot,
            k_rot=k_rot,
            precision="fp8",
        )
        self.assertEqual(out.shape, q.shape, f"FP8 output shape {out.shape} != input {q.shape}")
        self.assertEqual(out.dtype, torch.bfloat16)

    # mix (EagleQBSA) path tests

    def test_bsa_sparse_attention_v3_mix_output_shape(self):
        """bsa_sparse_attention_v3 mix (EagleQBSA) path: shape and dtype match input."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        out, _ = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            precision="mix",
        )
        self.assertEqual(out.shape, q.shape, f"mix output shape {out.shape} != input {q.shape}")
        self.assertEqual(out.dtype, self.dtype)

    def test_bsa_sparse_attention_v3_mix_unaligned_seq_len(self):
        """mix path: unaligned S still produces correct output shape (per-block quant pads)."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        t, h, w = 3, 20, 20
        latent_shape = (t, h, w)
        q, k, v = self._make_qkv_bsnd(t=t, h=h, w=w)

        out, _ = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            precision="mix",
        )
        self.assertEqual(out.shape, q.shape, f"mix unaligned: output shape {out.shape} != input {q.shape}")

    def test_mix_with_rotation_matrices_mask_stays_128(self):
        """mix + caller-provided q_rot/k_rot: mask KV granularity must stay 128.

        Regression for fp8_mode keying off q_rot presence (yjy_ac review 1):
        the mask block size must follow precision, not the rotation matrices.
        """
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)
        _, new_mask = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            precision="mix",
            q_rot=q_rot,
            k_rot=k_rot,
        )
        q_blocks = math.ceil(self.seq_len / self.pool_size)
        kv_blocks = math.ceil(self.seq_len / self.pool_size)
        self.assertEqual(new_mask.shape[2], q_blocks)
        self.assertEqual(new_mask.shape[3], kv_blocks)

    def test_invalid_precision_raises(self):
        """Unsupported precision values must raise ValueError at the entry check."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        with self.assertRaises(ValueError):
            bsa_sparse_attention_v3(
                q,
                k,
                v,
                latent_shape_q=self.latent_shape,
                block_size=self.pool_size,
                sparsity=0.5,
                input_layout="BSND",
                head_num=self.head_num,
                inner_precise=self.inner_precise,
                precision="bogus",
            )

    # unaligned S tests

    def test_bsa_sparse_attention_v3_unaligned_seq_len(self):
        """bsa_sparse_attention_v3 returns original shape when S is not a multiple of pool_size."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        # h=20, w=20 -> S = t*400; 400 % 128 = 16 != 0
        t, h, w = 3, 20, 20
        latent_shape = (t, h, w)
        q, k, v = self._make_qkv_bsnd(t=t, h=h, w=w)

        out, _ = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            precision="bf16",
        )
        self.assertEqual(out.shape, q.shape, f"unaligned: output shape {out.shape} != input {q.shape}")

    @_SKIP_NO_BSA_V2
    def test_bsa_sparse_attention_v3_fp8_unaligned_seq_len(self):
        """bsa_sparse_attention_v3 FP8 path: unaligned S still produces correct shape."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        t, h, w = 3, 20, 20
        latent_shape = (t, h, w)
        q, k, v = self._make_qkv_bsnd(t=t, h=h, w=w)
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)

        out, _ = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            q_rot=q_rot,
            k_rot=k_rot,
            precision="fp8",
        )
        self.assertEqual(out.shape, q.shape, f"FP8 unaligned: output shape {out.shape} != input {q.shape}")
        self.assertEqual(out.dtype, torch.bfloat16)

    # cached mask reuse tests

    @_SKIP_NO_BSA_V2
    def test_bsa_sparse_attention_v3_cached_mask_fp8(self):
        """FP8 path with cached_mask: reuse mask from BF16 step, output shape unchanged."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)

        # First call: generate mask
        out1, new_mask = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            precision="fp8",
        )
        # Second call: reuse mask with FP8
        out2, _ = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            cached_mask=new_mask,
            q_rot=q_rot,
            k_rot=k_rot,
            precision="fp8",
        )
        self.assertEqual(out2.shape, q.shape)
        self.assertEqual(out2.dtype, torch.bfloat16)

    # FP8 cached mask with explicit block_size_kv (regression for double-merge bug)

    @_SKIP_NO_BSA_V2
    def test_bsa_sparse_attention_v3_fp8_cached_mask_block_size_kv_256(self):
        """FP8 both steps with block_size_kv=256: mask at [128,256] is reused correctly."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)

        # Step 1: FP8, generate mask at [128, 256] granularity.
        out1, new_mask = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            block_size_kv=256,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            q_rot=q_rot,
            k_rot=k_rot,
            precision="fp8",
        )
        self.assertEqual(out1.shape, q.shape)
        self.assertEqual(out1.dtype, torch.bfloat16)

        # Verify mask shape: q_blocks=ceil(S/128), kv_blocks=ceil(S/256).
        q_blocks = math.ceil(self.seq_len / self.pool_size)
        kv_blocks = math.ceil(self.seq_len / 256)
        self.assertEqual(new_mask.shape[2], q_blocks)
        self.assertEqual(new_mask.shape[3], kv_blocks)

        # Step 2: FP8, reuse cached mask — must NOT double-merge KV blocks.
        out2, _ = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            block_size_kv=256,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            cached_mask=new_mask,
            q_rot=q_rot,
            k_rot=k_rot,
            precision="fp8",
        )
        self.assertEqual(out2.shape, q.shape)
        self.assertEqual(out2.dtype, torch.bfloat16)

    # bsa_sparse_attention_v3 MXFP4 output shape/dtype tests

    @_SKIP_NO_BSA_V3
    @_SKIP_NO_MXFP4_DTYPES
    def test_bsa_sparse_attention_v3_mxfp4_output_shape(self):
        """MXFP4 path: BF16 output for both CX (dst_type_max>0) and OCP (dst_type_max=0)."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)
        for dst_type_max in (0.0, 7.25):
            with self.subTest(dst_type_max=dst_type_max):
                out, _ = bsa_sparse_attention_v3(
                    q,
                    k,
                    v,
                    latent_shape_q=self.latent_shape,
                    block_size=self.pool_size,
                    sparsity=0.5,
                    input_layout="BSND",
                    head_num=self.head_num,
                    inner_precise=self.inner_precise,
                    q_rot=q_rot,
                    k_rot=k_rot,
                    precision="mxfp4",
                    mxfp4_dst_type_max=dst_type_max,
                )
                self.assertEqual(out.shape, q.shape, f"MXFP4 output shape {out.shape} != input {q.shape}")
                self.assertEqual(out.dtype, torch.bfloat16)

    @_SKIP_NO_BSA_V3
    @_SKIP_NO_MXFP4_DTYPES
    def test_bsa_sparse_attention_v3_mxfp4_unaligned_seq_len(self):
        """MXFP4 path: S is padded to a 64 base before quant, output cropped back to S."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        # h=20, w=20 -> S = t*400; 400 % 64 = 16, so quant pads and the kernel output is cropped.
        t, h, w = 3, 20, 20
        latent_shape = (t, h, w)
        q, k, v = self._make_qkv_bsnd(t=t, h=h, w=w)
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)

        out, _ = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            q_rot=q_rot,
            k_rot=k_rot,
            precision="mxfp4",
            mxfp4_dst_type_max=7.25,
        )
        self.assertEqual(out.shape, q.shape, f"MXFP4 unaligned: output shape {out.shape} != input {q.shape}")
        self.assertEqual(out.dtype, torch.bfloat16)

    @_SKIP_NO_BSA_V3
    @_SKIP_NO_MXFP4_DTYPES
    def test_bsa_sparse_attention_v3_mxfp4_cached_mask(self):
        """MXFP4 path with cached_mask: mask generated on the first step is reused unchanged."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)

        out1, new_mask = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            q_rot=q_rot,
            k_rot=k_rot,
            precision="mxfp4",
            mxfp4_dst_type_max=7.25,
        )
        # Verify mask granularity: q_blocks=ceil(S/128), kv_blocks=ceil(S/256).
        q_blocks = math.ceil(self.seq_len / self.pool_size)
        kv_blocks = math.ceil(self.seq_len / 256)
        self.assertEqual(new_mask.shape[2], q_blocks)
        self.assertEqual(new_mask.shape[3], kv_blocks)

        out2, _ = bsa_sparse_attention_v3(
            q,
            k,
            v,
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.5,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            cached_mask=new_mask,
            q_rot=q_rot,
            k_rot=k_rot,
            precision="mxfp4",
            mxfp4_dst_type_max=7.25,
        )
        self.assertEqual(out2.shape, q.shape)
        self.assertEqual(out2.dtype, torch.bfloat16)

    @_SKIP_NO_BSA_V3
    @_SKIP_NO_MXFP4_DTYPES
    def test_bsa_sparse_attention_v3_mxfp4_vs_bf16(self):
        """With sparsity=0 and a shared mask, MXFP4 output stays close to the BF16 path."""
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        q, k, v = self._make_qkv_bsnd()
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)
        common = dict(
            latent_shape_q=self.latent_shape,
            block_size=self.pool_size,
            sparsity=0.0,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            q_rot=q_rot,
            k_rot=k_rot,
        )

        out_bf16, _ = bsa_sparse_attention_v3(q.clone(), k.clone(), v.clone(), precision="bf16", **common)
        out_fp4, _ = bsa_sparse_attention_v3(
            q.clone(), k.clone(), v.clone(), precision="mxfp4", mxfp4_dst_type_max=7.25, **common
        )

        # Relative quantization noise: mean abs diff normalized by the BF16 output std.
        diff = (out_fp4.to(torch.float32) - out_bf16.to(torch.float32)).abs().mean()
        ref_std = out_bf16.to(torch.float32).std()
        self.assertLess((diff / ref_std).item(), 0.2, f"MXFP4 vs BF16 noise too large: {diff / ref_std:.4f}")

    @_SKIP_NO_MXFP4_DTYPES
    def test_mxfp4_quant_mode_and_dtype_threading(self):
        """MXFP4: quant_mode/dst_type_max/dtype codes must reach the kernel call correctly.

        Mocks the quantizer and the kernel wrapper so the test only asserts the
        parameter threading inside bsa_sparse_attention_v3 (no V3 kernel needed).
        """
        from mindiesd.layers.flash_attn import sparse_flash_attn_rf_v3 as rf_v3_mod

        q, k, v = self._make_qkv_bsnd()
        q_rot, k_rot = _make_rotation_matrices(self.head_dim, self.device)

        def fake_rain_fusion(q_, k_, v_, **kwargs):
            captured.update(kwargs)
            b, s, n, d = q.shape
            return torch.zeros((b, n, s, d), dtype=self.dtype, device=self.device)

        def fake_quant(q_, k_, v_, q_rot_, k_rot_, **kwargs):
            return q_, k_, v_, None, None, None

        for dst_type_max, expected_quant_mode in ((7.25, 3), (0.0, 2)):
            with self.subTest(dst_type_max=dst_type_max):
                captured = {}

                with (
                    mock.patch.object(rf_v3_mod, "rain_fusion_attention_v3", side_effect=fake_rain_fusion),
                    mock.patch.object(rf_v3_mod, "_mxfp4_quant_qkv", side_effect=fake_quant) as quant_mock,
                ):
                    out, _ = rf_v3_mod.bsa_sparse_attention_v3(
                        q,
                        k,
                        v,
                        latent_shape_q=self.latent_shape,
                        block_size=self.pool_size,
                        sparsity=0.5,
                        input_layout="BSND",
                        head_num=self.head_num,
                        inner_precise=self.inner_precise,
                        q_rot=q_rot,
                        k_rot=k_rot,
                        precision="mxfp4",
                        mxfp4_dst_type_max=dst_type_max,
                        mxfp4_scale_alg=2,
                    )

                self.assertEqual(out.shape, q.shape)
                # V3 quant mode: 2 = OCP (dst_type_max<=0), 3 = CX (dst_type_max>0).
                # dst_type_max threads through unchanged at this (caller) boundary;
                # rain_fusion_attention_v3 drops it from the op kwargs when it is 0.0.
                self.assertEqual(captured["quant_mode"], expected_quant_mode)
                self.assertEqual(captured["dst_type_max"], dst_type_max)
                # Packed FP4 data / E8M0 scales are UINT8 storage: CANN dtype codes required.
                self.assertEqual(captured["q_dtype"], torch_npu.float4_e2m1fn_x2)
                self.assertEqual(captured["k_dtype"], torch_npu.float4_e2m1fn_x2)
                self.assertEqual(captured["v_dtype"], torch_npu.float4_e2m1fn_x2)
                self.assertEqual(captured["q_scale_dtype"], torch_npu.float8_e8m0fnu)
                self.assertEqual(captured["k_scale_dtype"], torch_npu.float8_e8m0fnu)
                self.assertEqual(captured["v_scale_dtype"], torch_npu.float8_e8m0fnu)
                # Quantized path feeds the kernel BNSD tensors with the original lengths.
                self.assertEqual(captured["input_layout"], "BNSD")
                self.assertEqual(captured["actual_seq_lengths"], [self.seq_len])
                self.assertEqual(captured["block_size_kv"], 256)
                # mxfp4_scale_alg is forwarded to the quantizer.
                self.assertEqual(quant_mock.call_args.kwargs["scale_alg"], 2)
                self.assertEqual(quant_mock.call_args.kwargs["dst_type_max"], dst_type_max)

    # accuracy tests: sparsity=0 vs dense

    def test_bsa_sparse_attention_v3_vs_dense(self):
        """With sparsity=0, bsa_sparse_attention_v3 should be statistically close
        to npu_fusion_attention (token order differs due to rearrange).
        """
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

        # Use float16 for easier comparison (bfloat16 has lower precision).
        dtype = torch.float16
        t, h, w = 2, 16, 16  # 较小尺寸，加快测试
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
        # pylint: disable=no-member
        out_dense = torch_npu.npu_fusion_attention(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            head_num=self.head_num,
            input_layout="BNSD",
            scale=self.scale,
            pre_tockens=2147483647,
            next_tockens=2147483647,
        )[0].permute(0, 2, 1, 3)  # → BSND

        # bsa_sparse_attention_v3 with sparsity=0 (all blocks retained, ~dense)
        out_v3, _ = bsa_sparse_attention_v3(
            q.clone(),
            k.clone(),
            v.clone(),
            latent_shape_q=latent_shape,
            block_size=self.pool_size,
            sparsity=0.0,
            input_layout="BSND",
            head_num=self.head_num,
            inner_precise=self.inner_precise,
            precision="bf16",
        )

        # Token order differs (v3 applies spatial rearrange), so compare statistics.
        dense_mean = out_dense.to(torch.float32).mean()
        v3_mean = out_v3.to(torch.float32).mean()
        dense_std = out_dense.to(torch.float32).std()
        v3_std = out_v3.to(torch.float32).std()

        mean_rel_err = abs(dense_mean.item() - v3_mean.item()) / max(abs(dense_mean.item()), 1e-6)
        std_rel_err = abs(dense_std.item() - v3_std.item()) / max(abs(dense_std.item()), 1e-6)

        self.assertLess(mean_rel_err, 0.1, f"mean rel err too large: dense={dense_mean:.4f}, v3={v3_mean:.4f}")
        self.assertLess(std_rel_err, 0.1, f"std rel err too large: dense={dense_std:.4f}, v3={v3_std:.4f}")


class TestResolveSparseTypeForA5(unittest.TestCase):
    """Pure-Python unit tests for _resolve_sparse_type_for_a5 (no NPU required).

    Guards the A5 inner_precise policy added for yjy_ac review 3: explicit
    rf_v3 requires inner_precise=4 on A5, while rf_v2 keeps its remap.
    """

    @mock.patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=True)
    def test_rf_v3_rejects_wrong_inner_precise_on_a5(self, _mock_is_a5):
        """Explicit rf_v3 with inner_precise != 4 must raise ParametersInvalid on A5."""
        from mindiesd.layers.flash_attn.sparse_flash_attn import _resolve_sparse_type_for_a5
        from mindiesd.utils.exception import ParametersInvalid

        with self.assertRaises(ParametersInvalid):
            _resolve_sparse_type_for_a5("rf_v3", 0)
        # inner_precise=4 passes through untouched
        self.assertEqual(_resolve_sparse_type_for_a5("rf_v3", 4), ("rf_v3", 4))

    @mock.patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=True)
    def test_rf_v2_still_remaps_to_4_on_a5(self, _mock_is_a5):
        """rf_v2 remap must still force inner_precise=4 (vllm-omni depends on it)."""
        from mindiesd.layers.flash_attn.sparse_flash_attn import _resolve_sparse_type_for_a5

        self.assertEqual(_resolve_sparse_type_for_a5("rf_v2", 0), ("rf_v3", 4))

    @mock.patch("mindiesd.layers.flash_attn.sparse_flash_attn.is_a5_device", return_value=False)
    def test_non_a5_passthrough(self, _mock_is_a5):
        """Non-A5 devices pass sparse_type/inner_precise through unchanged."""
        from mindiesd.layers.flash_attn.sparse_flash_attn import _resolve_sparse_type_for_a5

        self.assertEqual(_resolve_sparse_type_for_a5("rf_v3", 0), ("rf_v3", 0))
        self.assertEqual(_resolve_sparse_type_for_a5("rf_v2", 0), ("rf_v2", 0))


@_SKIP_NO_MXFP4_DTYPES
class TestMxfp4ScaleReshapeGuards(unittest.TestCase):
    """CPU-runnable tests for the MXFP4 scale reshapes in _mxfp4_quant_qkv.

    Regression for the EZ1001 6D-scale bug: newer torch_npu returns the
    byte-grouped 5D scale directly from npu_dynamic_mx_quant, so the q/k scale
    reshape must be idempotent (dim==4 guard) and the V scale must go through
    the shared helper. npu_dynamic_mx_quant is mocked, so no NPU op runs.
    """

    B, S, N, D = 1, 128, 2, 64

    def _fake_dynamic_mx_quant(self, scale_dim5):
        """Return (data, scale) mimicking npu_dynamic_mx_quant for axis -1 / 2."""

        def _quant(tensor, dst_type, **kwargs):
            axis = kwargs.get("axis", -1)
            b, n, s, d = tensor.shape
            data = torch.zeros(b, n, s, d // 2, dtype=torch.uint8)
            if axis == -1:
                scale = torch.ones(b, n, s, d // 32, dtype=torch.uint8)
                if scale_dim5:
                    scale = scale.reshape(b, n, s, d // 64, 2)
            else:
                scale = torch.ones(b, n, s // 32, d, dtype=torch.uint8)
                if scale_dim5:
                    scale = scale.reshape(b, n, s // 64, d, 2)
            return data, scale

        return _quant

    def _run_quant_qkv(self, scale_dim5, **quant_kwargs):
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import _mxfp4_quant_qkv

        shape_bsnd = (self.B, self.S, self.N, self.D)
        q = torch.randn(shape_bsnd, dtype=torch.float32)
        k = torch.randn(shape_bsnd, dtype=torch.float32)
        v = torch.randn(shape_bsnd, dtype=torch.float32)
        q_rot = torch.eye(self.D, dtype=torch.float32)
        k_rot = torch.eye(self.D, dtype=torch.float32)

        with mock.patch(
            "mindiesd.quantization.layer._dynamic_mx_quant", side_effect=self._fake_dynamic_mx_quant(scale_dim5)
        ) as quant_mock:
            result = _mxfp4_quant_qkv(q, k, v, q_rot, k_rot, layout="BSND", **quant_kwargs)
        return result, quant_mock

    def test_4d_scales_are_regrouped_to_5d(self):
        """Older torch_npu returns 4D scales: they must be reshaped into the V3 5D layout."""
        (q_fp4, k_fp4, v_fp4, q_scale, k_scale, v_scale), _ = self._run_quant_qkv(scale_dim5=False)

        # q/k scale: [B,N,S,D/32] -> [B,N,S,D/64,2]; v scale: [B,N,S/32,D] -> [B,N,S/64,D,2].
        self.assertEqual(tuple(q_scale.shape), (self.B, self.N, self.S, self.D // 64, 2))
        self.assertEqual(tuple(k_scale.shape), (self.B, self.N, self.S, self.D // 64, 2))
        self.assertEqual(tuple(v_scale.shape), (self.B, self.N, self.S // 64, self.D, 2))

    def test_5d_scales_pass_through_unchanged(self):
        """Newer torch_npu returns byte-grouped 5D scales: reshapes must be idempotent."""
        (q_fp4, k_fp4, v_fp4, q_scale, k_scale, v_scale), _ = self._run_quant_qkv(scale_dim5=True)

        self.assertEqual(tuple(q_scale.shape), (self.B, self.N, self.S, self.D // 64, 2))
        self.assertEqual(tuple(k_scale.shape), (self.B, self.N, self.S, self.D // 64, 2))
        self.assertEqual(tuple(v_scale.shape), (self.B, self.N, self.S // 64, self.D, 2))

    def test_quant_kwargs_threading(self):
        """dst_type_max>0 and scale_alg must be forwarded to the quantizer; absent when unset."""
        _, quant_mock = self._run_quant_qkv(scale_dim5=False, scale_alg=2, dst_type_max=7.25)
        forwarded = quant_mock.call_args.kwargs
        self.assertEqual(forwarded["scale_alg"], 2)
        self.assertEqual(forwarded["dst_type_max"], 7.25)
        # Q/K quantize rowwise (axis -1), V columnwise along the sequence dim (axis 2).
        axes = [call.kwargs["axis"] for call in quant_mock.call_args_list]
        self.assertEqual(axes, [-1, -1, 2])

        _, quant_mock = self._run_quant_qkv(scale_dim5=False)
        forwarded = quant_mock.call_args.kwargs
        self.assertNotIn("scale_alg", forwarded)
        self.assertNotIn("dst_type_max", forwarded)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
