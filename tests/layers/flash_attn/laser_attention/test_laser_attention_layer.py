# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

import math
import os
import unittest
from unittest.mock import patch

import torch

from laser_attention_layer_common import LAYERS, LegacyLaserAttention, la, make_inputs
from mindiesd.layers.flash_attn.common import AttentionParam


class TestLaserAttentionLayerContract(unittest.TestCase):
    """Run the full layer on CPU, replacing only the NPU operator boundary."""

    @patch.object(la, "is_a5_device", return_value=False)
    def test_full_layer_dtype_layout_padding_and_attributes(self, _mock_a5):
        for dtype in (torch.bfloat16, torch.float16):
            for head_first in (False, True):
                for kv_len in (256, 255, 128, 1):
                    for name, forward in LAYERS.items():
                        with self.subTest(dtype=dtype, head_first=head_first, kv_len=kv_len, path=name):
                            param = AttentionParam(1, 2, 64, 255, kv_len, dtype, head_first)
                            q_shape = (1, 2, 255, 64) if head_first else (1, 255, 2, 64)
                            kv_shape = (1, 2, kv_len, 64) if head_first else (1, kv_len, 2, 64)
                            sources = [
                                (torch.arange(math.prod(shape)).float().reshape(shape) / 128 + index).to(dtype)
                                for index, shape in enumerate((q_shape, kv_shape, kv_shape))
                            ]
                            query, key, value = sources
                            operator_dtype = dtype if name == "current" else torch.float16
                            output_dtype = torch.bfloat16 if operator_dtype == torch.bfloat16 else torch.float32
                            # BF16 sentinel would overflow if accidentally routed through FP16.
                            raw_output = torch.arange(2 * 256 * 128).float().reshape(1, 2, 256, 128)
                            if output_dtype == torch.bfloat16:
                                raw_output = (131072.0 + raw_output * 1024).to(output_dtype)
                            else:
                                raw_output = raw_output / 128 + 1.00392
                            with patch.object(la.ops, "laser_attention", return_value=(None, raw_output)) as operator:
                                output = forward(param, query, key, value, scale=64**-0.5)
                            operator.assert_called_once()
                            args = operator.call_args.args
                            for index, tensor in enumerate(args[:3]):
                                self.assertEqual(tensor.shape, (1, 2, 256, 128))
                                self.assertEqual(tensor.dtype, torch.float16 if index == 2 else operator_dtype)
                                self.assertTrue(tensor.is_contiguous())
                                length = 255 if index == 0 else kv_len
                                expected_input = sources[index] if head_first else sources[index].transpose(1, 2)
                                self.assertTrue(
                                    torch.equal(tensor[:, :, :length, :64], expected_input.to(tensor.dtype))
                                )
                                self.assertEqual(torch.count_nonzero(tensor[:, :, length:, :]).item(), 0)
                                self.assertEqual(torch.count_nonzero(tensor[:, :, :, 64:]).item(), 0)
                            self.assertEqual(args[3:6], (None, None, None))
                            self.assertEqual(args[6:10], (64**-0.5, 2, "BNSD", 1.0))
                            self.assertEqual(args[10:], (la.MAX_TOKEN if kv_len == 256 else 256 - kv_len, 1, True))
                            expected = raw_output if output_dtype == dtype else raw_output.half().to(dtype)
                            expected = expected[:, :, :255, :64]
                            if not head_first:
                                expected = expected.transpose(1, 2)
                            self.assertEqual(output.shape, query.shape)
                            self.assertEqual(output.dtype, dtype)
                            self.assertTrue(torch.equal(output, expected))
                            if output_dtype == dtype:
                                self.assertEqual(output.data_ptr(), raw_output.data_ptr())

    def test_legacy_output_keeps_both_rounding_steps(self):
        raw = torch.full((1, 1, 256, 128), 1.00392, dtype=torch.float32)
        output = LegacyLaserAttention.la_postprocess_output(raw, torch.bfloat16, 255, 64)
        self.assertTrue(torch.equal(output, raw.half().bfloat16()[:, :, :255, :64]))
        self.assertFalse(torch.equal(output, raw.bfloat16()[:, :, :255, :64]))


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE") == "CPU", "NPU integration test")
class TestLaserAttentionLayerNpu(unittest.TestCase):
    def test_business_shape_current_and_legacy_vs_fas(self):
        import torch_npu

        if not torch.npu.is_available():
            self.skipTest("NPU is unavailable")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
            self.fail("Unset ASCEND_RT_VISIBLE_DEVICES; pass a physical DEVICE_ID")
        torch.npu.set_device(int(os.environ.get("DEVICE_ID", "0")))
        if la.is_a5_device():
            self.skipTest("LA is unsupported on A5")
        param, query, key, value = make_inputs(f"npu:{torch.npu.current_device()}")
        with torch.inference_mode():
            golden = (
                torch_npu.npu_fusion_attention(
                    query,
                    key,
                    value,
                    head_num=24,
                    input_layout="BSND",
                    scale=128**-0.5,
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                )[0]
                .float()
                .reshape(-1)
            )
            for name, forward in LAYERS.items():
                with self.subTest(path=name):
                    output = forward(param, query, key, value, scale=128**-0.5)
                    self.assertEqual(output.shape, query.shape)
                    self.assertEqual(output.dtype, torch.bfloat16)
                    actual = output.float().reshape(-1)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    cosine = torch.nn.functional.cosine_similarity(actual, golden, dim=0).item()
                    norm_ratio = (torch.linalg.vector_norm(actual) / torch.linalg.vector_norm(golden)).item()
                    print(f"{name}: cosine={cosine:.8f}, norm_ratio={norm_ratio:.8f}")
                    self.assertGreaterEqual(cosine, 0.99)
                    self.assertGreaterEqual(norm_ratio, 0.9)
                    self.assertLessEqual(norm_ratio, 1.1)


if __name__ == "__main__":
    unittest.main()
