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
import unittest
import torch
import torch.nn as nn
import torch_npu
import numpy as np
from math import sqrt

# 加载自定义库
if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


def ref_compare(golden:torch.Tensor, actual:torch.Tensor, err):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    EB = torch.mean(abs_error / golden_nmax)
    result = (abs_error <= err * golden_nmax).all() and EB <= err/2
    return EB.item(),result.item(),abs_error.max().item()

 
def block_sparse_attention_cpu(query, key, value, smask, causal=False, blocksize=128):
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
            num_blocks = (seq + blocksize - 1) // blocksize  # 向上取整

            for s1 in range(num_blocks):  # 当前 query 所在的 block 索引

                mask_block = smask[bi, ni, s1, :num_blocks]  # bool array

                # 展开为序列级掩码：每个 block 重复 blocksize 次
                mask_seq = np.repeat(mask_block, blocksize)[:seq].astype(bool)  # [seq], bool
                # 提取当前 query 块
                start = s1 * blocksize
                end = min((s1 + 1) * blocksize, seq)
                q = query[bi, ni, start:end]  # [q_len, dim]

                k_head = ni // gqa
                k = key[bi, k_head][mask_seq]  # [k_eff, dim]
                if k.shape[0] == 0:
                    out = np.zeros((end - start, dim), dtype=np.float32)
                else:
                    kt = k.T  # [dim, k_eff]
                    p = q @ kt  # [q_len, k_eff]
                    p = p / np.sqrt(dim)
                    if causal : 
                        t = end - start
                        cm = np.triu(np.ones((t, t)), k=1) * (-10000.0)
                        p[:, -t:] += cm

                    p =  p -p.max(axis=-1, keepdims=True)
                    exp_p = np.exp(p)
                    exp_sum = exp_p.sum(axis=-1, keepdims=True)
                    attn = exp_p / (exp_sum + 1e-12)  # softmax
                    # 提取对应的 value
                    v = value[bi, k_head][mask_seq]  # [v_eff, dim]

                    # 输出: attn @ V
                    out = attn @ v  # [q_len, dim]

                out_tensor = torch.from_numpy(out)
                output[bi, ni, start:end] = out_tensor

    return output


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestBsaMindieSd(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        self.batch = 1
        self.head_num = 1
        self.head_num_key = 16
        self.qseqlen = 8192
        self.head_dim = 128
        self.dtype = torch.bfloat16
        self.input_layout = "BNSD"
        self.stride = 8
        self.sparse_size = 128
        self.threshold = 0.85
        self.row_sparse = 1.0
        self.causal = False
        if self.causal:
            self.row_sparse = 1.0
        self.keep_sink = True
        self.keep_recent = True
        self.scale_value = 1.0 / (sqrt(self.head_dim))

        self.query_shape = (self.batch, self.head_num, self.qseqlen, self.head_dim)
        self.key_value_shape = (self.batch, self.head_num, self.qseqlen, self.head_dim)
        self.query = torch.randn(self.query_shape, dtype=self.dtype)
        self.key = torch.randn(self.key_value_shape, dtype=self.dtype)
        self.value = torch.randn(self.key_value_shape, dtype=self.dtype)

        s1 = (self.qseqlen + self.sparse_size - 1) // self.sparse_size
        realS2 = s1 
        s2 = (realS2 + 31) // 32 * 32
        self.smask_shape = (self.batch, self.head_num, s1, s2)
        self.sct_shape = (self.batch, self.head_num, s1)

    def bsa_preprocess_input(self):
        query = self.query.clone()
        key = self.key.clone()
        value = self.value.clone()
        return query, key, value


    def test_bsa_mindie_sd_vs_block_sparse_attention_cpu(self):
        """对比 block_sparse_attention 与 cpu 实现的结果"""
        query, key, value = self.bsa_preprocess_input()

        sn1 = (self.qseqlen + self.sparse_size - 1) // self.sparse_size
        realsn2 = (self.qseqlen + self.sparse_size - 1) // self.sparse_size
        sn2 = (realsn2 + 31) // 32 * 32 
        sparsity = 0.5
        smask = torch.rand(self.batch, self.head_num, sn1, sn2) > sparsity
        smask[:,:,:,0] = True
        smask[:,:,1,:] = False
        smask[:,:,sn1-2,:] = False
        smask[:,:,sn1-1,:] = False

        smask[:, :, :, realsn2:] = False
        if self.causal:
            for j in range(sn1):
                smask[:, :, j, j] = True
                smask[:, :, j, j+1:] = False
        smask = smask.to(torch.int8)
        sparse_count_table = smask.sum(dim=-1, dtype=torch.int32)

        bsa_npu = torch.ops.mindiesd.block_sparse_attention(
            query=query.to(self.device),
            key=key.to(self.device),
            value=value.to(self.device),
            sparse_mask=smask.to(self.device),
            sparse_count_table=sparse_count_table.to(self.device),
            input_layout="BNSD",
            sparse_size=self.sparse_size,
            num_heads=self.head_num,
            num_key_value_heads=self.head_num,
            scale_value = self.scale_value,
            causal=self.causal
        )
        
        bsa_cpu = block_sparse_attention_cpu(query, key, value, smask, causal=self.causal, blocksize=self.sparse_size)

        # compare result
        err_threshold = 2**(-6)
        EB, result, max_err = ref_compare(bsa_cpu.ravel(), bsa_npu.ravel().cpu().float(),err_threshold)
        assert result, f'eb should < {err_threshold}, but got {EB}. max_err:{max_err}'

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)