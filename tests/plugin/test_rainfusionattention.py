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

import unittest
import torch
import torch_npu
import os
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.utils.precision_compare import data_compare

if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestRainFusionAttention(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        self.batch_size = 1
        self.head = 3
        self.q_seqlen = 4096
        self.kv_seqlen = 4096
        self.headdim = 128
        self.scale = self.headdim ** -0.5

        q_shape = (self.batch_size, self.q_seqlen, self.head, self.headdim)
        kv_shape = (self.batch_size, self.kv_seqlen, self.head, self.headdim)
        self.q = torch.randn(q_shape, dtype=torch.float16, device=self.device)
        self.k = torch.randn(kv_shape, dtype=torch.float16, device=self.device)
        self.v = torch.randn(kv_shape, dtype=torch.float16, device=self.device)
        self.q_tnd = self.q.reshape(-1, self.head, self.headdim)
        self.k_tnd = self.k.reshape(-1, self.head, self.headdim)
        self.v_tnd = self.v.reshape(-1, self.head, self.headdim)

        q_blocknum = math.ceil(self.q_seqlen / 128)
        kv_blocknum = math.ceil(self.kv_seqlen / 128)
        self.block_shape = [128, 128]
        self.actual_seq_lengths = [self.q_seqlen for _ in range(self.batch_size)]
        self.actual_seq_lengths_kv = [self.kv_seqlen for _ in range(self.batch_size)]
        self.select_idx, self.select_num_idx = self._generate_sparse_mask(
                                                q_blocknum, self.head, kv_blocknum, ratio=1.0)

    def _generate_sparse_mask(self, q_blocknum, head, kv_blocknum, device='npu', ratio=1.0):
        select_idx = torch.full(
            (q_blocknum, head, kv_blocknum),
            -1,
            dtype=torch.int64,
            device=device
        )

        select_num_idx = torch.tensor(
            kv_blocknum,
            dtype=torch.int64,
            device=device
        ).repeat(q_blocknum, head)

        base_indices = torch.arange(kv_blocknum, dtype=torch.int64, device=device)
        select_idx[...] = base_indices.repeat(q_blocknum, head, 1)

        for q in range(q_blocknum):
            for h in range(head):
                selected_kvs = base_indices[:int(kv_blocknum*ratio)]
                select_idx[q, h, :len(selected_kvs)] = selected_kvs
                select_num_idx[q, h] = len(selected_kvs)

        return select_idx, select_num_idx

    def test_rainfusionattention_vs_fusionattention(self):
        ra, _ = torch.ops.mindiesd.rainfusionattention(
            self.q_tnd, self.k_tnd, self.v_tnd,
            self.select_idx, self.select_num_idx,
            self.block_shape,
            attn_mask=None,
            actual_seq_qlen=self.actual_seq_lengths,
            actual_seq_kvlen=self.actual_seq_lengths_kv,
            block_table=None,
            q_input_layout="TND",
            kv_input_layout="TND",
            head_num=self.head,
            mask_type=0, scale=self.scale,
            inner_precise=0, block_size=0)
        fascore = torch_npu.npu_fusion_attention(
                    self.q, self.k, self.v,
                    input_layout="BSND",
                    scale=self.headdim ** -0.5,
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                    head_num=self.head)[0]
        result, _, max_err = data_compare(ra.cpu(), fascore.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    def test_rainfusionattention_bnsd(self):
        q = self.q.transpose(1, 2)
        k = self.k.transpose(1, 2)
        v = self.v.transpose(1, 2)
        ra, _ = torch.ops.mindiesd.rainfusionattention(
            q, k, v,
            self.select_idx, self.select_num_idx,
            self.block_shape,
            attn_mask=None,
            actual_seq_qlen=self.actual_seq_lengths,
            actual_seq_kvlen=self.actual_seq_lengths_kv,
            block_table=None,
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            head_num=self.head,
            mask_type=0, scale=self.scale,
            inner_precise=0, block_size=0)
        fascore = torch_npu.npu_fusion_attention(
                    q, k, v,
                    input_layout="BNSD",
                    scale=self.headdim ** -0.5,
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                    head_num=self.head)[0]
        result, _, max_err = data_compare(ra.cpu(), fascore.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    def test_ra_output_shape(self):
            expected_shape = (self.batch_size * self.q_seqlen, self.head, self.headdim)
            ra, _ = torch.ops.mindiesd.rainfusionattention(
                self.q_tnd, self.k_tnd, self.v_tnd,
                self.select_idx, self.select_num_idx,
                self.block_shape,
                attn_mask=None,
                actual_seq_qlen=self.actual_seq_lengths,
                actual_seq_kvlen=self.actual_seq_lengths_kv,
                block_table=None,
                q_input_layout="TND",
                kv_input_layout="TND",
                head_num=self.head,
                mask_type=0, scale=self.scale,
                inner_precise=0, block_size=0)
            self.assertEqual(ra.shape, expected_shape, "Output shape does not match expected shape.")

    def test_ra_invalid_inputlayout(self):
        with self.assertRaises(RuntimeError):
            ra, _ = torch.ops.mindiesd.rainfusionattention(
                self.q, self.k, self.v,
                self.select_idx, self.select_num_idx,
                self.block_shape,
                attn_mask=None,
                actual_seq_qlen=self.actual_seq_lengths,
                actual_seq_kvlen=self.actual_seq_lengths_kv,
                block_table=None,
                q_input_layout="BSND",
                kv_input_layout="BSND",
                head_num=self.head,
                mask_type=0, scale=self.scale,
                inner_precise=0, block_size=0)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
