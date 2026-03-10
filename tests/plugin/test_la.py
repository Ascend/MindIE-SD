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

# 加载自定义库
if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestLaMindieSd(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        self.batch = 1
        self.head_num = 2
        self.qseqlen = 4096
        self.kvseqlen = 128
        self.head_dim = 128
        self.dtype = torch.bfloat16

        self.query_shape = (self.batch, self.head_num, self.qseqlen, self.head_dim)
        self.key_value_shape = (self.batch, self.head_num, self.kvseqlen, self.head_dim)
        self.query = torch.randn(self.query_shape, device=self.device, dtype=self.dtype)
        self.key = torch.randn(self.key_value_shape, device=self.device, dtype=self.dtype)
        self.value = torch.randn(self.key_value_shape, device=self.device, dtype=self.dtype)

        self.scale_value = self.head_dim ** -0.5

    def la_preprocess_input(self):
        query = self.query.clone()
        key = self.key.clone()
        value = self.value.clone()

        qseqlen_pad_size = 0
        kvseqlen_pad_size = 0
        dim_pad_size = 0

        if self.qseqlen % 256 != 0:
            qseqlen_pad_size = ((self.qseqlen // 256) + 1) * 256 - self.qseqlen
            qseqlen_padding = torch.zeros([self.batch, self.head_num, qseqlen_pad_size, self.head_dim],
                                          dtype=self.dtype, device=self.device)
            query = torch.cat([query, qseqlen_padding], dim=-2).to(self.dtype)

        if self.kvseqlen % 256 != 0:
            kvseqlen_pad_size = ((self.kvseqlen // 256) + 1) * 256 - self.kvseqlen
            kvseqlen_padding = torch.zeros([self.batch, self.head_num, kvseqlen_pad_size, self.head_dim],
                                           dtype=self.dtype, device=self.device)
            key = torch.cat([key, kvseqlen_padding], dim=-2).to(self.dtype)
            value = torch.cat([value, kvseqlen_padding], dim=-2).to(self.dtype)

        if self.head_dim < 128:
            dim_pad_size = 128 - self.head_dim
            dim_padding = torch.zeros([self.batch, self.head_num, self.qseqlen + qseqlen_pad_size, dim_pad_size],
                                      dtype=self.dtype, device=self.device)
            query = torch.cat([query, dim_padding], dim=-1).to(self.dtype)
            key = torch.cat([key, dim_padding], dim=-1).to(self.dtype)
            value = torch.cat([value, dim_padding], dim=-1).to(self.dtype)

        if self.dtype != torch.float16:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        return query, key, value

    def la_postprocess_output(self, attention_out):
        # 裁剪填充部分
        attention_out = attention_out[:, :, :self.qseqlen, :self.head_dim]
        return attention_out

    def test_la_mindie_sd_output_shape(self):
        query, key, value = self.la_preprocess_input()
        _, attention_out = torch.ops.mindiesd.la(
            query, key, value, None, None, None,
            self.scale_value, self.head_num, "BNSD", 1.0, 2147483647, 1, True
        )
        attention_out = self.la_postprocess_output(attention_out)
        expected_shape = self.query_shape
        self.assertEqual(attention_out.shape, expected_shape, "Output shape does not match expected shape.")

    def test_la_mindie_sd_consistency(self):
        query, key, value = self.la_preprocess_input()
        _, output1 = torch.ops.mindiesd.la(
            query, key, value, None, None, None,
            self.scale_value, self.head_num, "BNSD", 1.0, 2147483647, 1, True
        )
        attention_out1 = self.la_postprocess_output(output1)
        _, output2 = torch.ops.mindiesd.la(
            query, key, value, None, None, None,
            self.scale_value, self.head_num, "BNSD", 1.0, 2147483647, 1, True
        )
        attention_out2 = self.la_postprocess_output(output2)
        self.assertTrue(torch.allclose(attention_out1, attention_out2),
                        "Multiple runs of the operator produce inconsistent results.")

    def test_la_mindie_sd_vs_npu_fusion_attention(self):
        """对比la_mindie_sd与npu_fusion_attention的结果"""
        query, key, value = self.la_preprocess_input()
        _, attention_out = torch.ops.mindiesd.la(
            query, key, value, None, None, None,
            self.scale_value, self.head_num, "BNSD", 1.0, 2147483647, 1, True
        )
        attention_out = self.la_postprocess_output(attention_out)

        fascore = torch_npu.npu_fusion_attention(
            self.query, self.key, self.value,
            head_num=self.head_num,
            input_layout="BNSD",
            scale=self.scale_value,
            pre_tockens=2147483647,
            next_tockens=2147483647
        )[0]

        csoine_sim = torch.cosine_similarity(
            attention_out.to("cpu").to(dtype=torch.float32).reshape(1, -1),
            fascore.to("cpu").reshape(1, -1)
        )[0]

        self.assertGreaterEqual(csoine_sim, 0.99,
                                "Cosine similarity between la_mindie_sd and npu_fusion_attention should be high.")


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)