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
import sys
import os

if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestLaPreprocessMindieSd(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        self.batch = 1
        self.head_num = 2
        self.qseqlen = 4096
        self.kvseqlen = 128
        self.head_dim = 128
        self.dtype = torch.bfloat16
        self.align_len = 256

        self.query_shape = (self.batch, self.qseqlen, self.head_num, self.head_dim)
        self.key_value_shape = (self.batch, self.kvseqlen, self.head_num, self.head_dim)
        self.query = torch.randn(self.query_shape, device=self.device, dtype=self.dtype)
        self.key = torch.randn(self.key_value_shape, device=self.device, dtype=self.dtype)
        self.value = torch.randn(self.key_value_shape, device=self.device, dtype=self.dtype)

    def _get_padded_length(self, seq_len, align_len):
        """计算对齐后的序列长度"""
        return (seq_len + align_len - 1) // align_len * align_len

    def test_la_preprocess_output_shape(self):
        out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, self.align_len
        )

        padded_qseqlen = self._get_padded_length(self.qseqlen, self.align_len)
        padded_kvseqlen = self._get_padded_length(self.kvseqlen, self.align_len)
        
        expected_query_shape = (self.batch, self.head_num, padded_qseqlen, self.head_dim)
        expected_kv_shape = (self.batch, self.head_num, padded_kvseqlen, self.head_dim)
        
        self.assertEqual(out_query.shape, expected_query_shape)
        self.assertEqual(out_key.shape, expected_kv_shape)
        self.assertEqual(out_value.shape, expected_kv_shape)

    def test_la_preprocess_consistency(self):
        out_query1, out_key1, out_value1 = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, self.align_len
        )
        
        out_query2, out_key2, out_value2 = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, self.align_len
        )
        
        self.assertTrue(torch.allclose(out_query1, out_query2))
        self.assertTrue(torch.allclose(out_key1, out_key2))
        self.assertTrue(torch.allclose(out_value1, out_value2))

    def test_la_preprocess_with_different_align_len(self):
        align_len_512 = 512
        out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, align_len_512
        )
        
        padded_qseqlen = self._get_padded_length(self.qseqlen, align_len_512)
        expected_shape = (self.batch, self.head_num, padded_qseqlen, self.head_dim)
        self.assertEqual(out_query.shape, expected_shape)

    def test_la_preprocess_with_float16(self):
        query_fp16 = self.query.to(torch.float16)
        key_fp16 = self.key.to(torch.float16)
        value_fp16 = self.value.to(torch.float16)
        
        out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
            query_fp16, key_fp16, value_fp16, self.align_len
        )
        
        self.assertEqual(out_query.dtype, torch.float16)
        self.assertEqual(out_key.dtype, torch.float16)
        self.assertEqual(out_value.dtype, torch.float16)

    def test_la_preprocess_integration_with_la_operator(self):
        processed_query, processed_key, processed_value = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, self.align_len
        )
        
        scale_value = self.head_dim ** -0.5
        _, attention_out = torch.ops.mindiesd.la(
            processed_query, processed_key, processed_value, None, None, None,
            scale_value, self.head_num, "BNSD", 1.0, 2147483647, 1, True
        )
        
        expected_shape = (self.batch, self.head_num, self.qseqlen, self.head_dim)
        self.assertEqual(attention_out.shape, expected_shape)

    def test_dtype_conversion(self):
        """测试数据类型转换 - 算子会将bfloat16转换为float16"""
        out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, self.align_len
        )
        
        # 算子会将bfloat16转换为float16
        self.assertEqual(out_query.dtype, torch.float16)
        self.assertEqual(out_key.dtype, torch.float16)
        self.assertEqual(out_value.dtype, torch.float16)

    def test_bsnd_to_bnsd_conversion(self):
        out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, self.align_len
        )
        
        # 验证格式转换: BSND -> BNSD
        self.assertEqual(out_query.shape[0], self.batch)  # Batch
        self.assertEqual(out_query.shape[1], self.head_num)  # Head num
        self.assertEqual(out_query.shape[3], self.head_dim)  # Head dim

    def test_memory_layout(self):
        out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, self.align_len
        )
        
        self.assertTrue(out_query.is_contiguous())
        self.assertTrue(out_key.is_contiguous())
        self.assertTrue(out_value.is_contiguous())

    def test_device_placement(self):
        out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
            self.query, self.key, self.value, self.align_len
        )
        
        self.assertEqual(out_query.device.type, 'npu')
        self.assertEqual(out_key.device.type, 'npu')
        self.assertEqual(out_value.device.type, 'npu')

    def test_with_different_batch_sizes(self):
        batch_sizes = [1, 2, 4]
        for batch in batch_sizes:
            with self.subTest(batch_size=batch):
                query = torch.randn((batch, self.qseqlen, self.head_num, self.head_dim), 
                                  device=self.device, dtype=self.dtype)
                key = torch.randn((batch, self.kvseqlen, self.head_num, self.head_dim), 
                                device=self.device, dtype=self.dtype)
                value = torch.randn((batch, self.kvseqlen, self.head_num, self.head_dim), 
                                  device=self.device, dtype=self.dtype)
                
                out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
                    query, key, value, self.align_len
                )
                
                padded_qseqlen = self._get_padded_length(self.qseqlen, self.align_len)
                padded_kvseqlen = self._get_padded_length(self.kvseqlen, self.align_len)
                
                expected_query_shape = (batch, self.head_num, padded_qseqlen, self.head_dim)
                expected_kv_shape = (batch, self.head_num, padded_kvseqlen, self.head_dim)
                
                self.assertEqual(out_query.shape, expected_query_shape)
                self.assertEqual(out_key.shape, expected_kv_shape)
                self.assertEqual(out_value.shape, expected_kv_shape)

    def test_with_different_head_nums(self):
        head_nums = [4, 8, 16]
        for head_num in head_nums:
            with self.subTest(head_num=head_num):
                query = torch.randn((self.batch, self.qseqlen, head_num, self.head_dim), 
                                  device=self.device, dtype=self.dtype)
                key = torch.randn((self.batch, self.kvseqlen, head_num, self.head_dim), 
                                device=self.device, dtype=self.dtype)
                value = torch.randn((self.batch, self.kvseqlen, head_num, self.head_dim), 
                                  device=self.device, dtype=self.dtype)
                
                out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
                    query, key, value, self.align_len
                )
                
                padded_qseqlen = self._get_padded_length(self.qseqlen, self.align_len)
                padded_kvseqlen = self._get_padded_length(self.kvseqlen, self.align_len)
                
                expected_query_shape = (self.batch, head_num, padded_qseqlen, self.head_dim)
                expected_kv_shape = (self.batch, head_num, padded_kvseqlen, self.head_dim)
                
                self.assertEqual(out_query.shape, expected_query_shape)
                self.assertEqual(out_key.shape, expected_kv_shape)
                self.assertEqual(out_value.shape, expected_kv_shape)

    def test_with_different_seq_lens(self):
        seq_lens = [(512, 256), (1024, 512), (2048, 1024)]
        for qseqlen, kvseqlen in seq_lens:
            with self.subTest(qseqlen=qseqlen, kvseqlen=kvseqlen):
                query = torch.randn((self.batch, qseqlen, self.head_num, self.head_dim), 
                                  device=self.device, dtype=self.dtype)
                key = torch.randn((self.batch, kvseqlen, self.head_num, self.head_dim), 
                                device=self.device, dtype=self.dtype)
                value = torch.randn((self.batch, kvseqlen, self.head_num, self.head_dim), 
                                  device=self.device, dtype=self.dtype)
                
                out_query, out_key, out_value = torch.ops.mindiesd.la_preprocess(
                    query, key, value, self.align_len
                )
                
                padded_qseqlen = self._get_padded_length(qseqlen, self.align_len)
                padded_kvseqlen = self._get_padded_length(kvseqlen, self.align_len)
                
                expected_query_shape = (self.batch, self.head_num, padded_qseqlen, self.head_dim)
                expected_kv_shape = (self.batch, self.head_num, padded_kvseqlen, self.head_dim)
                
                self.assertEqual(out_query.shape, expected_query_shape)
                self.assertEqual(out_key.shape, expected_kv_shape)
                self.assertEqual(out_value.shape, expected_kv_shape)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)