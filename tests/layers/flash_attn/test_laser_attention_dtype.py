# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

import unittest

import torch

from mindiesd.layers.flash_attn.ascend_laser_attention import AscendLaserAttention
from mindiesd.layers._custom_ops import attention_fake


class TestLaserAttentionDtype(unittest.TestCase):
    def test_fake_output_matches_kernel_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                query = torch.empty((1, 2, 256, 128), device="meta", dtype=dtype)
                value = torch.empty_like(query, dtype=torch.float16)
                stats, output = attention_fake(query, query, value)
                self.assertEqual(stats.dtype, torch.float32)
                self.assertEqual(stats.shape, (1, 2, 256))
                self.assertEqual(output.dtype, torch.bfloat16 if dtype == torch.bfloat16 else torch.float32)
                self.assertEqual(output.shape, query.shape)

    def test_preprocess_preserves_dtype_and_padding(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = torch.arange(255 * 64, dtype=torch.float32).reshape(1, 1, 255, 64).to(dtype)
                outputs = AscendLaserAttention.la_preprocess_input(source, source, source)
                for index, output in enumerate(outputs):
                    expected_dtype = torch.float16 if index == 2 else dtype
                    self.assertEqual(output.dtype, expected_dtype)
                    self.assertEqual(output.shape, (1, 1, 256, 128))
                    self.assertTrue(output.is_contiguous())
                    self.assertTrue(torch.equal(output[:, :, :255, :64], source.to(expected_dtype)))
                    self.assertEqual(torch.count_nonzero(output[:, :, 255:, :]).item(), 0)
                    self.assertEqual(torch.count_nonzero(output[:, :, :, 64:]).item(), 0)

    def test_aligned_bf16_qk_are_not_copied_and_only_value_is_cast(self):
        source = torch.full((1, 1, 256, 128), 131072.0, dtype=torch.bfloat16)
        value = torch.full_like(source, 3.25)
        query_out, key_out, value_out = AscendLaserAttention.la_preprocess_input(source, source, value)
        for output in (query_out, key_out):
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertEqual(output.data_ptr(), source.data_ptr())
            self.assertTrue(torch.equal(output, source))
        self.assertEqual(value_out.dtype, torch.float16)
        self.assertTrue(torch.equal(value_out, value.to(torch.float16)))

    def test_aligned_fp16_value_is_not_copied(self):
        query = torch.zeros((1, 1, 256, 128), dtype=torch.bfloat16)
        value = torch.ones(query.shape, dtype=torch.float16)
        _, _, value_out = AscendLaserAttention.la_preprocess_input(query, query, value)
        self.assertEqual(value_out.data_ptr(), value.data_ptr())

    def test_native_bf16_output_is_only_sliced(self):
        output = torch.full((1, 1, 256, 128), 131072.0, dtype=torch.bfloat16)
        result = AscendLaserAttention.la_postprocess_output(output, torch.bfloat16, 255, 64)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, (1, 1, 255, 64))
        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertTrue(torch.equal(result, output[:, :, :255, :64]))


if __name__ == "__main__":
    unittest.main()
