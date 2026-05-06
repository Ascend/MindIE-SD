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
import math
import numpy as np

if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


def softmax_flash(src, inmax=None, insum=None, update=False):
    if not update:
        x_max = np.max(src, axis=-1, keepdims=True)
        x_sub = src - x_max
        dst = np.exp(x_sub)
        x_sum = np.sum(dst, axis=-1, keepdims=True)
        exp_max = None
    else:
        x_max = np.max(np.concatenate((inmax, src), axis=-1), axis=-1, keepdims=True)
        dst = np.exp(src - x_max)
        exp_max = np.exp(inmax - x_max)
        x_sum = np.sum(dst, axis=-1, keepdims=True)
        x_sum = exp_max * insum +  x_sum
    return dst, x_max, x_sum, exp_max

def align_s_dim(q,block_size = 128):
    B,N,S,D = q.shape
    target_length = (S + block_size-1)//block_size * block_size
    if(S==target_length):
        return q
    else:
        aligned_q = torch.zeros(B,N,target_length,D,dtype = q.dtype,device = q.device)
        aligned_q[:,:,:S,:] = q
    return aligned_q

def sparse_estimate_cpu(query, key, causal, blocksize=128, stride=8, threshold=0.5, force_sparse=1.0):
    reduce_size = blocksize // stride
    bs, nq, seq, dim = query.shape
    gqa = nq // key.shape[1]
    block_num = (seq + blocksize - 1) // blocksize
    mask = np.zeros([bs, nq, block_num, block_num], dtype=np.bool_)
    M, N = 128, 1024
    FLASH = True

    for bi in range(bs):
        for ni in range(nq):
            qtimes = (seq + M*stride -1) // (M*stride)
            for outeridx in range(qtimes):
                m = M * stride if outeridx < qtimes - 1 else seq - outeridx* (M*stride)
                q = query[bi, ni, outeridx* (M*stride):  outeridx* (M*stride)+m, :]
                if m % stride > 0:
                    z = np.zeros((stride-m % stride, dim), dtype=np.float32)
                    q = np.concatenate([q, z], axis=-2)
                q = q.reshape(-1, stride, dim)[:, ::-1, :]
                q = q.reshape(-1, stride * dim)

                kseq = seq if not causal else (outeridx * M * stride + m) // stride * stride
                ktimes = (kseq + N*stride -1) // (N*stride)
                first_reduce_gm = []
                x_max_loop_ub = []
                x_max, x_sum = None, 0

                for innerIdx in range(ktimes):
                    n = N * stride if innerIdx < ktimes - 1 else kseq - innerIdx * (N * stride)
                    if causal: n = n // stride * stride # 尾块是diag 舍弃
                    k = key[bi, ni // gqa, innerIdx * (N * stride): innerIdx * (N * stride) + n, :]  # (n, dim)

                    # 补零到 stride 的整数倍
                    if n % stride > 0:
                        pad_size = stride - (n % stride)
                        z = np.zeros((pad_size, dim), dtype=np.float32)
                        k = np.concatenate([k, z], axis=-2)  # (padded_n, dim)

                    k = k.reshape(-1, stride * dim)

                    p = np.dot(q / (np.sqrt(dim) * stride), k.T)
                    if FLASH:
                        if innerIdx == 0:
                            p, x_max, x_sum, exp_max = softmax_flash(p)
                        else:
                            p, x_max[:], x_sum[:], exp_max = softmax_flash(p, x_max, x_sum, True)
                    else:
                        p = np.exp(p-20.0)
                        x_sum = x_sum + p.sum(axis=-1, keepdims=True)

                    # first reduce
                    n = p.shape[-1]
                    if n % reduce_size > 0:
                        pad_size = reduce_size - (n % reduce_size)
                        z = np.zeros((p.shape[0], pad_size), dtype=np.float32)
                        p = np.concatenate([p, z], axis=-1)

                    p = p.reshape(-1, p.shape[-1] // reduce_size, reduce_size).sum(axis=-1)
                    first_reduce_gm.append(p.copy())
                    x_max_loop_ub.append(x_max.copy() if x_max is not None else None)
                # second reduce
                x_max_global = x_max_loop_ub[-1]

                for i in range(len(first_reduce_gm)):
                    reduce_ub, x_max = first_reduce_gm[i], x_max_loop_ub[i]
                    upd = np.exp(x_max - x_max_global) / x_sum if FLASH else 1.0 / x_sum
                    reduce_ub *= upd
                    n, m = reduce_ub.shape
                    if n % reduce_size > 0:
                        z = np.zeros((reduce_size - n % reduce_size, m), dtype=np.float32)
                        reduce_ub = np.concatenate([reduce_ub, z], axis=0)
                    reduce_ub = reduce_ub.reshape(reduce_ub.shape[0] // reduce_size, reduce_size, reduce_ub.shape[1])
                    reduce_ub = reduce_ub.sum(axis=1)
                    first_reduce_gm[i] = reduce_ub.copy()

                # score compute
                reduce_ub = np.concatenate(first_reduce_gm, axis=-1)

                offset = outeridx * M // reduce_size
                for i in range(reduce_ub.shape[0]):
                    if causal and offset + i <= 1:
                        mask[bi, ni, offset + i, :offset + i+1] = True
                        continue
                    to_sort = reduce_ub[i, :offset+i+1].copy() if causal else reduce_ub[i] # include diag
                    to_sort[-1] = 0

                    score = -np.sort(-to_sort, axis=-1).astype(np.float32)
                    cnt = (score.cumsum(axis=-1) < threshold * score.sum(axis=-1)).astype(np.int32).sum(axis=-1) + 1
                    if cnt > 0:
                        if cnt > force_sparse * score.shape[-1] and force_sparse > 0 :
                            cnt = int(force_sparse * score.shape[-1])
                        guard = score[cnt-1]
                        to_sort[0] = guard +1
                        if not causal: to_sort[-1] = guard + 1
                        mask[bi, ni, offset + i, :to_sort.shape[-1]] = (to_sort >= guard).astype(np.bool_)

    return mask


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestSparseBlockEstimate(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        self.batch = 1
        self.head_num = 1
        self.head_num_key = 16
        self.qseqlen = 8192
        self.head_dim = 128
        self.dtype = torch.bfloat16
        self.input_layout="BNSD"
        self.stride = 8
        self.sparse_size = 128
        self.threshold = 0.85
        self.row_sparse = 1.0
        self.causal = False
        self.keep_sink = True
        self.keep_recent = True
        self.scale_value = self.head_dim ** -0.5 / self.stride

        self.query_shape = (self.batch, self.head_num, self.qseqlen, self.head_dim)
        self.key_value_shape = (self.batch, self.head_num, self.qseqlen, self.head_dim)
        self.query = torch.randn(self.query_shape, dtype=self.dtype)
        self.key = torch.randn(self.key_value_shape, dtype=self.dtype)
        s1 = (self.qseqlen + self.sparse_size - 1) // self.sparse_size
        realS2 = s1
        s2 = (realS2 + 31) // 32 * 32
        self.smask_shape = (self.batch, self.head_num, s1, s2)
        self.sct_shape = (self.batch, self.head_num, s1)

    def bsa_estimate_preprocess_input(self):
        query = self.query.clone()
        key = self.key.clone()

        if self.dtype != torch.float16:
            query = query.to(torch.float16)
            key = key.to(torch.float16)

        return query, key


    def test_bsa_estimate_mindie_sd_output_shape(self):
        query, key = self.bsa_estimate_preprocess_input()
        smask, sct = torch.ops.mindiesd.sparse_block_estimate(
            query=query.to(self.device),
            key=key.to(self.device),
            actual_seq_lengths=None, actual_seq_lengths_kv=None,
            input_layout=self.input_layout,
            stride=self.stride,
            sparse_size=self.sparse_size,
            num_heads=self.head_num,
            num_key_value_heads=self.head_num,
            scale_value=self.scale_value,
            threshold=self.threshold,
            causal=self.causal,
            keep_sink=self.keep_sink,
            keep_recent=self.keep_recent,
            row_sparse=self.row_sparse
        )

        self.assertEqual(smask.shape, self.smask_shape, "Output shape does not match expected shape.")
        self.assertEqual(sct.shape, self.sct_shape, "Output shape does not match expected shape.")

    def test_bsa_estimate_mindie_sd_vs_sparse_estimate_cpu(self):
        """对比 sparse_block_estimate_mindie_sd 与 cpu 实现的结果"""
        query, key = self.bsa_estimate_preprocess_input()
        smask, sct = torch.ops.mindiesd.sparse_block_estimate(
            query=query.to(self.device),
            key=key.to(self.device),
            actual_seq_lengths=None, actual_seq_lengths_kv=None,
            input_layout=self.input_layout,
            stride=self.stride,
            sparse_size=self.sparse_size,
            num_heads=self.head_num,
            num_key_value_heads=self.head_num,
            scale_value=self.scale_value,
            threshold=self.threshold,
            causal=self.causal,
            keep_sink=self.keep_sink,
            keep_recent=self.keep_recent,
            row_sparse=self.row_sparse
        )
        smask_cpu = sparse_estimate_cpu(query.float().numpy(), key.float().numpy(),
                                        self.causal, blocksize=self.sparse_size, stride=self.stride,
                                        threshold=self.threshold, force_sparse=self.row_sparse)

        # compare result
        smask = smask.cpu()[:, :, :, :smask_cpu.shape[-1]].reshape(self.batch,self.head_num,-1).numpy().astype(np.int32)
        smask_cpu = smask_cpu.reshape(self.batch, self.head_num,-1)
        for i in range(self.batch):
            for j in range(self.head_num):
                total_selected_blocks = smask_cpu[i,j,:].sum()
                diff_num = (smask[i,j,:] != smask_cpu[i,j,:]).sum()
                diff_num_ratio = diff_num / total_selected_blocks

                omitted_blocks = (smask[i,j,:] < smask_cpu[i,j,:]).sum()
                omitted_blocks_ratio = omitted_blocks / total_selected_blocks
                self.assertLess(diff_num_ratio, 0.02,
                                "diff_num_ratio should < 0.02.")
                self.assertLess(omitted_blocks_ratio, 0.01,
                                "omitted_blocks_ratio should < 0.01.")

    def test_invalid_layout(self):
        query, key = self.bsa_estimate_preprocess_input()
        with self.assertRaises(RuntimeError):
            smask, sct = torch.ops.mindiesd.sparse_block_estimate(
                query=query.to(self.device),
                key=key.to(self.device),
                actual_seq_lengths=None, actual_seq_lengths_kv=None,
                input_layout="TND",
                stride=self.stride,
                sparse_size=self.sparse_size,
                num_heads=self.head_num,
                num_key_value_heads=self.head_num,
                scale_value=self.scale_value,
                threshold=self.threshold,
                causal=self.causal,
                keep_sink=self.keep_sink,
                keep_recent=self.keep_recent,
                row_sparse=self.row_sparse
            )
if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)