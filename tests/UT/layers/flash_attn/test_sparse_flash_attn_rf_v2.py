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

import math
import unittest
import torch

from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
    avgpool,
    get_mask_index,
    get_blockwise_mask,
    rearrange_with_remaining,
    inv_rearrange_with_remaining,
    do_tensor_rearrange_pooling,
    do_tensor_inv_rearrange,
)
from mindiesd.utils.get_platform import is_a5_device

_SKIP_A5_RF_V2 = "sparse_flash_attn_rf_v2 is unsupported on A5; routed to rf_v3."
_SKIP_IF_A5 = unittest.skipIf(is_a5_device(), _SKIP_A5_RF_V2)


# ---------------------------------------------------------------------------
# 参数配置
# ---------------------------------------------------------------------------
# hw 能整除 8 的场景: h=40, w=80
HW_DIV = {"t": 3, "h": 40, "w": 80}
# hw 不能整除 8 的场景 (h, w 均不整除): h=30, w=52
HW_NODIV_BOTH = {"t": 3, "h": 30, "w": 52}
# 仅 h 不整除: h=21, w=32
HW_NODIV_H = {"t": 2, "h": 21, "w": 32}
# 仅 w 不整除: h=24, w=45
HW_NODIV_W = {"t": 2, "h": 24, "w": 45}

BATCH = 1
HEAD = 2
HEADDIM = 64
POOL_SIZE = 128
DTYPE = torch.float32


def _make_latent_shape(cfg):
    return (cfg["t"], cfg["h"], cfg["w"])


def _seqlen(cfg):
    return cfg["t"] * cfg["h"] * cfg["w"]


def _make_qkv(cfg, layout, batch=BATCH, head=HEAD, headdim=HEADDIM, dtype=DTYPE):
    s = _seqlen(cfg)
    if layout == "BSND":
        shape = (batch, s, head, headdim)
    else:
        shape = (batch, head, s, headdim)
    q = torch.randn(shape, dtype=dtype)
    k = torch.randn(shape, dtype=dtype)
    v = torch.randn(shape, dtype=dtype)
    return q, k, v


# ============================================================================
# 1. TestAvgpool
# ============================================================================
@_SKIP_IF_A5
class TestAvgpool(unittest.TestCase):
    """avgpool 在 BNSD / BSND 布局以及 seqlen 整除 / 不整除 pool_size 场景下的正确性。"""

    def _check_avgpool(self, layout, seqlen, pool_size=POOL_SIZE):
        if layout == "BSND":
            tensor = torch.randn(BATCH, seqlen, HEAD, HEADDIM)
        else:
            tensor = torch.randn(BATCH, HEAD, seqlen, HEADDIM)

        out = avgpool(tensor, pool_size=pool_size, input_layout=layout)

        num_full = seqlen // pool_size
        tail = seqlen % pool_size
        expected_blocks = num_full + (1 if tail > 0 else 0)

        if layout == "BSND":
            self.assertEqual(out.shape, (BATCH, expected_blocks, HEAD, HEADDIM))
        else:
            self.assertEqual(out.shape, (BATCH, HEAD, expected_blocks, HEADDIM))

    # --- BSND ---
    def test_bsnd_seqlen_divisible(self):
        self._check_avgpool("BSND", seqlen=768, pool_size=128)

    def test_bsnd_seqlen_not_divisible(self):
        self._check_avgpool("BSND", seqlen=800, pool_size=128)

    # --- BNSD ---
    def test_bnsd_seqlen_divisible(self):
        self._check_avgpool("BNSD", seqlen=768, pool_size=128)

    def test_bnsd_seqlen_not_divisible(self):
        self._check_avgpool("BNSD", seqlen=800, pool_size=128)

    def test_bsnd_mean_value(self):
        """验证池化结果确实是分块均值。"""
        tensor = torch.arange(256, dtype=torch.float32).reshape(1, 256, 1, 1).expand(1, 256, 1, 1).clone()
        out = avgpool(tensor, pool_size=128, input_layout="BSND")
        expected_first = torch.arange(128, dtype=torch.float32).mean()
        expected_second = torch.arange(128, 256, dtype=torch.float32).mean()
        self.assertAlmostEqual(out[0, 0, 0, 0].item(), expected_first.item(), places=4)
        self.assertAlmostEqual(out[0, 1, 0, 0].item(), expected_second.item(), places=4)

    def test_bnsd_mean_value(self):
        tensor = torch.arange(256, dtype=torch.float32).reshape(1, 1, 256, 1).expand(1, 1, 256, 1).clone()
        out = avgpool(tensor, pool_size=128, input_layout="BNSD")
        expected_first = torch.arange(128, dtype=torch.float32).mean()
        expected_second = torch.arange(128, 256, dtype=torch.float32).mean()
        self.assertAlmostEqual(out[0, 0, 0, 0].item(), expected_first.item(), places=4)
        self.assertAlmostEqual(out[0, 0, 1, 0].item(), expected_second.item(), places=4)

    def test_bsnd_tail_only(self):
        """seqlen < pool_size，只有 tail block。"""
        self._check_avgpool("BSND", seqlen=64, pool_size=128)

    def test_bnsd_tail_only(self):
        self._check_avgpool("BNSD", seqlen=64, pool_size=128)


# ============================================================================
# 2. TestRearrangeWithRemaining — 覆盖 hw 整除 / 不整除 8
# ============================================================================
@_SKIP_IF_A5
class TestRearrangeWithRemaining(unittest.TestCase):
    """rearrange_with_remaining + inv_rearrange_with_remaining 的 roundtrip 测试。
    覆盖:
      - BNSD / BSND 布局
      - hw 整除 8 / h 不整除 / w 不整除 / 均不整除
    """

    def _roundtrip(self, cfg, layout):
        latent_shape = _make_latent_shape(cfg)
        q, _, _ = _make_qkv(cfg, layout)

        rearranged = rearrange_with_remaining(q, latent_shape, latent_shape, layout)
        self.assertEqual(rearranged.shape, q.shape, "rearrange 后 shape 应不变")

        recovered = inv_rearrange_with_remaining(rearranged, latent_shape, latent_shape, layout)
        self.assertEqual(recovered.shape, q.shape, "inv_rearrange 后 shape 应不变")
        self.assertTrue(
            torch.allclose(recovered, q, atol=1e-6),
            f"roundtrip 精度不满足, max diff={torch.max(torch.abs(recovered - q)).item()}",
        )

    # BSND, hw 整除 8
    def test_bsnd_hw_divisible(self):
        self._roundtrip(HW_DIV, "BSND")

    # BSND, hw 均不整除 8
    def test_bsnd_hw_both_not_divisible(self):
        self._roundtrip(HW_NODIV_BOTH, "BSND")

    # BSND, 仅 h 不整除 8
    def test_bsnd_h_not_divisible(self):
        self._roundtrip(HW_NODIV_H, "BSND")

    # BSND, 仅 w 不整除 8
    def test_bsnd_w_not_divisible(self):
        self._roundtrip(HW_NODIV_W, "BSND")

    # BNSD, hw 整除 8
    def test_bnsd_hw_divisible(self):
        self._roundtrip(HW_DIV, "BNSD")

    # BNSD, hw 均不整除 8
    def test_bnsd_hw_both_not_divisible(self):
        self._roundtrip(HW_NODIV_BOTH, "BNSD")

    # BNSD, 仅 h 不整除 8
    def test_bnsd_h_not_divisible(self):
        self._roundtrip(HW_NODIV_H, "BNSD")

    # BNSD, 仅 w 不整除 8
    def test_bnsd_w_not_divisible(self):
        self._roundtrip(HW_NODIV_W, "BNSD")


# ============================================================================
# 3. TestGetMaskIndex
# ============================================================================
@_SKIP_IF_A5
class TestGetMaskIndex(unittest.TestCase):
    def test_shape_and_dtype(self):
        s = 8
        mask = torch.ones(1, 2, s, s, dtype=torch.bool)
        pos = get_mask_index(mask)
        self.assertEqual(pos.shape, (1, 2, s, s))
        self.assertEqual(pos.dtype, torch.int64)

    def test_all_true_mask(self):
        """全 True mask，每行的 valid indices 应该是 0..s-1。"""
        s = 4
        mask = torch.ones(1, 1, s, s, dtype=torch.bool)
        pos = get_mask_index(mask)
        for row in range(s):
            expected = torch.arange(s, dtype=torch.int64)
            self.assertTrue(torch.equal(pos[0, 0, row, :s], expected))

    def test_partial_mask(self):
        """部分 False 的 mask，验证有效索引排在前面，无效位为 -1。"""
        s = 4
        mask = torch.zeros(1, 1, s, s, dtype=torch.bool)
        mask[0, 0, 0, 0] = True
        mask[0, 0, 0, 2] = True
        pos = get_mask_index(mask)
        self.assertEqual(pos[0, 0, 0, 0].item(), 0)
        self.assertEqual(pos[0, 0, 0, 1].item(), 2)
        self.assertEqual(pos[0, 0, 0, 2].item(), -1)


# ============================================================================
# 4. TestGetBlockwiseMask — BNSD / BSND
# ============================================================================
@_SKIP_IF_A5
class TestGetBlockwiseMask(unittest.TestCase):
    def _run_blockwise_mask(self, layout, cfg, txt_len=0, pool_size=POOL_SIZE, sparsity=0.0):
        latent_shape = _make_latent_shape(cfg)
        q, k, v = _make_qkv(cfg, layout)
        total_seq = _seqlen(cfg) + txt_len

        if txt_len > 0:
            if layout == "BSND":
                txt_q = torch.randn(BATCH, txt_len, HEAD, HEADDIM)
                q = torch.cat([txt_q, q], dim=1)
                k = torch.cat([txt_q, k], dim=1)
                v = torch.cat([txt_q, v], dim=1)
            else:
                txt_q = torch.randn(BATCH, HEAD, txt_len, HEADDIM)
                q = torch.cat([txt_q, q], dim=2)
                k = torch.cat([txt_q, k], dim=2)
                v = torch.cat([txt_q, v], dim=2)

        qkv_pool = avgpool(torch.cat([q, k, v], dim=0), pool_size=pool_size, input_layout=layout)
        scale = HEADDIM**-0.5
        select_idx, select_num_idx = get_blockwise_mask(
            qkv_pool, txt_len, sparsity, scale, pool_size, latent_shape, latent_shape, layout
        )

        num_pool_blocks = math.ceil(total_seq / pool_size)
        self.assertEqual(select_idx.shape[0], num_pool_blocks)
        self.assertEqual(select_idx.shape[1], HEAD)
        self.assertEqual(select_num_idx.shape[0], num_pool_blocks)
        self.assertEqual(select_num_idx.shape[1], HEAD)

    def test_bnsd_hw_divisible(self):
        self._run_blockwise_mask("BNSD", HW_DIV)

    def test_bsnd_hw_divisible(self):
        self._run_blockwise_mask("BSND", HW_DIV)

    def test_bnsd_hw_not_divisible(self):
        self._run_blockwise_mask("BNSD", HW_NODIV_BOTH)

    def test_bsnd_hw_not_divisible(self):
        self._run_blockwise_mask("BSND", HW_NODIV_BOTH)

    def test_bnsd_with_text(self):
        self._run_blockwise_mask("BNSD", HW_DIV, txt_len=256)

    def test_bsnd_with_text(self):
        self._run_blockwise_mask("BSND", HW_DIV, txt_len=256)

    def test_sparsity_nonzero(self):
        self._run_blockwise_mask("BNSD", HW_DIV, sparsity=0.5)


# ============================================================================
# 5. TestDoTensorRearrangePooling — BNSD / BSND × text / no text × hw 整除/不整除
# ============================================================================
@_SKIP_IF_A5
class TestDoTensorRearrangePooling(unittest.TestCase):
    """do_tensor_rearrange_pooling 输出 shape 和 roundtrip 一致性。"""

    def _run(self, layout, cfg, txt_len=0, pool_size=POOL_SIZE):
        latent_shape = _make_latent_shape(cfg)
        q, k, v = _make_qkv(cfg, layout)
        total_seq = _seqlen(cfg) + txt_len

        if txt_len > 0:
            if layout == "BSND":
                txt_t = torch.randn(BATCH, txt_len, HEAD, HEADDIM)
                q = torch.cat([txt_t, q], dim=1)
                k = torch.cat([txt_t.clone(), k], dim=1)
                v = torch.cat([txt_t.clone(), v], dim=1)
            else:
                txt_t = torch.randn(BATCH, HEAD, txt_len, HEADDIM)
                q = torch.cat([txt_t, q], dim=2)
                k = torch.cat([txt_t.clone(), k], dim=2)
                v = torch.cat([txt_t.clone(), v], dim=2)

        q_, k_, v_, pool = do_tensor_rearrange_pooling(q, k, v, txt_len, pool_size, latent_shape, latent_shape, layout)

        if layout == "BSND":
            self.assertEqual(q_.shape[1], total_seq)
            self.assertEqual(k_.shape[1], total_seq)
        else:
            self.assertEqual(q_.shape[2], total_seq)
            self.assertEqual(k_.shape[2], total_seq)

        expected_pool_blocks = math.ceil(total_seq / pool_size)
        if layout == "BSND":
            self.assertEqual(pool.shape[1], expected_pool_blocks)
        else:
            self.assertEqual(pool.shape[2], expected_pool_blocks)

    # BSND, 无 text
    def test_bsnd_hw_div_no_text(self):
        self._run("BSND", HW_DIV)

    def test_bsnd_hw_nodiv_both_no_text(self):
        self._run("BSND", HW_NODIV_BOTH)

    def test_bsnd_hw_nodiv_h_no_text(self):
        self._run("BSND", HW_NODIV_H)

    def test_bsnd_hw_nodiv_w_no_text(self):
        self._run("BSND", HW_NODIV_W)

    # BSND, 有 text
    def test_bsnd_hw_div_with_text(self):
        self._run("BSND", HW_DIV, txt_len=256)

    def test_bsnd_hw_nodiv_with_text(self):
        self._run("BSND", HW_NODIV_BOTH, txt_len=256)

    # BNSD, 无 text
    def test_bnsd_hw_div_no_text(self):
        self._run("BNSD", HW_DIV)

    def test_bnsd_hw_nodiv_both_no_text(self):
        self._run("BNSD", HW_NODIV_BOTH)

    def test_bnsd_hw_nodiv_h_no_text(self):
        self._run("BNSD", HW_NODIV_H)

    def test_bnsd_hw_nodiv_w_no_text(self):
        self._run("BNSD", HW_NODIV_W)

    # BNSD, 有 text
    def test_bnsd_hw_div_with_text(self):
        self._run("BNSD", HW_DIV, txt_len=256)

    def test_bnsd_hw_nodiv_with_text(self):
        self._run("BNSD", HW_NODIV_BOTH, txt_len=256)


# ============================================================================
# 6. TestDoTensorInvRearrange — roundtrip 验证
# ============================================================================
@_SKIP_IF_A5
class TestDoTensorInvRearrange(unittest.TestCase):
    """do_tensor_rearrange_pooling 之后 do_tensor_inv_rearrange 能恢复原始顺序。"""

    def _roundtrip(self, layout, cfg, txt_len=0):
        latent_shape = _make_latent_shape(cfg)
        q, k, v = _make_qkv(cfg, layout)

        if txt_len > 0:
            if layout == "BSND":
                txt_t = torch.randn(BATCH, txt_len, HEAD, HEADDIM)
                q_full = torch.cat([txt_t, q], dim=1)
            else:
                txt_t = torch.randn(BATCH, HEAD, txt_len, HEADDIM)
                q_full = torch.cat([txt_t, q], dim=2)
        else:
            q_full = q.clone()

        q_, k_, v_, pool = do_tensor_rearrange_pooling(
            q_full, q_full.clone(), q_full.clone(), txt_len, POOL_SIZE, latent_shape, latent_shape, layout
        )
        recovered = do_tensor_inv_rearrange(q_, txt_len, latent_shape, latent_shape, layout)

        if txt_len > 0:
            if layout == "BSND":
                recovered_text = recovered[:, :txt_len, :, :]
            else:
                recovered_text = recovered[:, :, :txt_len, :]
            self.assertTrue(torch.allclose(recovered_text, txt_t, atol=1e-6), "text 部分 roundtrip 失败")

        self.assertEqual(recovered.shape, q_full.shape, "roundtrip 后 shape 不一致")

    # BSND
    def test_bsnd_hw_div_no_text(self):
        self._roundtrip("BSND", HW_DIV)

    def test_bsnd_hw_nodiv_no_text(self):
        self._roundtrip("BSND", HW_NODIV_BOTH)

    def test_bsnd_hw_div_with_text(self):
        self._roundtrip("BSND", HW_DIV, txt_len=256)

    def test_bsnd_hw_nodiv_with_text(self):
        self._roundtrip("BSND", HW_NODIV_BOTH, txt_len=256)

    def test_bsnd_h_nodiv_no_text(self):
        self._roundtrip("BSND", HW_NODIV_H)

    def test_bsnd_w_nodiv_no_text(self):
        self._roundtrip("BSND", HW_NODIV_W)

    # BNSD
    def test_bnsd_hw_div_no_text(self):
        self._roundtrip("BNSD", HW_DIV)

    def test_bnsd_hw_nodiv_no_text(self):
        self._roundtrip("BNSD", HW_NODIV_BOTH)

    def test_bnsd_hw_div_with_text(self):
        self._roundtrip("BNSD", HW_DIV, txt_len=256)

    def test_bnsd_hw_nodiv_with_text(self):
        self._roundtrip("BNSD", HW_NODIV_BOTH, txt_len=256)

    def test_bnsd_h_nodiv_no_text(self):
        self._roundtrip("BNSD", HW_NODIV_H)

    def test_bnsd_w_nodiv_no_text(self):
        self._roundtrip("BNSD", HW_NODIV_W)


if __name__ == '__main__':
    unittest.main()
