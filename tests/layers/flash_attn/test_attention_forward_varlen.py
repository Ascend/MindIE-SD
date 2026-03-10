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
import time
import sys
import torch
import torch_npu

from mindiesd import attention_forward_varlen
from mindiesd.utils.exception import ParametersInvalid
from utils.utils.precision_compare import data_compare

MAX_TOKEN = 2147483647


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestAttentionForwardVarlen(unittest.TestCase):
    def setUp(self):
        if not torch_npu.npu.is_available():
            self.skipTest("NPU is not available. Skipping test.")
        self.device = "npu:0"
        self.dtype = torch.bfloat16
        self.num_heads = 3
        self.head_dim = 128
        torch.npu.set_device(0)
        self.cu_seqlens_q = [0, 76, 124]
        self.cu_seqlens_k = [0, 1640, 48608]
        self.total_q = self.cu_seqlens_q[-1]
        self.total_k = self.cu_seqlens_k[-1]
        self.q = torch.randn(self.total_q, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        self.k = torch.randn(self.total_k, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        self.v = torch.randn(self.total_k, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)

    def _check_output(self, out, expected_total_q, msg=""):
        """验证输出形状和数值合法性"""
        self.assertEqual(out.shape[0], expected_total_q, msg=f"{msg} Output total_q mismatch.")
        self.assertEqual(out.shape[1], self.num_heads, msg=f"{msg} Output num_heads mismatch.")
        self.assertEqual(out.shape[2], self.head_dim, msg=f"{msg} Output head_dim mismatch.")
        self.assertFalse(torch.isnan(out).any(), msg=f"{msg} Output contains NaN.")
        self.assertFalse(torch.isinf(out).any(), msg=f"{msg} Output contains Inf.")

    def _get_bsnd_result(self, q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False):
        """使用 npu_fusion_attention 计算参考输出"""
        out_bsnd = []
        batch_size = len(cu_seqlens_q) - 1
        head_dim = q.size(-1)
        head_num = q.size(-2)
        scale = head_dim ** -0.5

        atten_mask = None
        if causal:
            atten_mask = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(q.device)

        for i in range(batch_size):
            start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]

            q_i = q[start_q:end_q].view(1, -1, head_num, head_dim)
            k_i = k[start_k:end_k].view(1, -1, head_num, head_dim)
            v_i = v[start_k:end_k].view(1, -1, head_num, head_dim)

            sparse_mode = 3 if causal else 0

            out_i = torch_npu.npu_fusion_attention(
                q_i, k_i, v_i,
                atten_mask=atten_mask,
                input_layout="BSND",
                scale=scale,
                pre_tockens=MAX_TOKEN,
                next_tockens=MAX_TOKEN,
                sparse_mode=sparse_mode,
                head_num=head_num
            )[0].view(-1, head_num, head_dim)

            out_bsnd.append(out_i)

        return torch.cat(out_bsnd, dim=0)

    def _benchmark(self, func, inputs, warmup=5, repeat=10):
        """通用性能测试：返回平均耗时（微秒）"""
        for _ in range(warmup):
            func(*inputs)
            torch.npu.synchronize()

        times = []
        for _ in range(repeat):
            torch.npu.synchronize()
            start = time.time()
            func(*inputs)
            torch.npu.synchronize()
            end = time.time()
            times.append(end - start)

        # 取后 5 次平均
        avg_time = sum(times[-5:]) / 5 * 1e6  # 转为微秒
        return avg_time

    def _test_accuracy(self, q, k, v, cu_seqlens_q, cu_seqlens_k, causal, test_name):
        """测试精度：对比自研实现与 npu_fusion_attention"""
        print(f"➡️  Running accuracy test (causal={causal}): {test_name}")

        out = attention_forward_varlen(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            causal=causal,
        )

        # 参考实现
        out_ref = self._get_bsnd_result(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)
        result, _, max_err = data_compare(out.cpu(), out_ref.cpu())
        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    def _test_performance(self, q, k, v, cu_seqlens_q, cu_seqlens_k, test_name):
        """测试性能：对比自研 vs npu_fusion_attention"""
        print(f"➡️  Running performance test: {test_name}")

        head_num = q.size(1)
        softmax_scale = self.head_dim ** -0.5

        # 自研实现耗时
        def sd_func():
            return attention_forward_varlen(
                q=q, k=k, v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                causal=False,
                dropout_p=0.0,
            )

        sd_time = self._benchmark(sd_func, ())

        # npu_fusion_attention 耗时
        def op_func():
            return torch_npu.npu_fusion_attention(
                q, k, v,
                head_num=head_num,
                scale=softmax_scale,
                keep_prob=1.0,
                input_layout="TND",
                actual_seq_qlen=cu_seqlens_q[1:],
                actual_seq_kvlen=cu_seqlens_k[1:],
                sparse_mode=3,  # 对应 causal=False 时的 packed 模式
            )[0]

        op_time = self._benchmark(op_func, ())

        print(f"Self-implemented avg time: {sd_time:.2f} μs")
        print(f"NPU fusion attention avg time: {op_time:.2f} μs")
        print(f"Speedup: {op_time / sd_time:.2f}x" if sd_time < op_time else f"Overhead: {sd_time / op_time:.2f}x")

    def _run_test_case(self, cu_seqlens_q, cu_seqlens_k, test_name):
        """运行一个完整测试用例：精度 + 性能"""
        print("\n" + "=" * 60)
        print(f"Testing: {test_name}")
        print("=" * 60)

        total_q = cu_seqlens_q[-1]
        total_k = cu_seqlens_k[-1]

        q = torch.randn(total_q, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        k = torch.randn(total_k, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        v = torch.randn(total_k, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)

        # 精度测试：causal=False 和 True
        self._test_accuracy(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False, test_name=f"{test_name}_causal_False")
        self._test_accuracy(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True, test_name=f"{test_name}_causal_True")

        # 性能测试（causal=False）
        self._test_performance(q, k, v, cu_seqlens_q, cu_seqlens_k, test_name=test_name)

        print(f"✅ {test_name} passed.\n")

    def test_case1(self):
        """测试 Case 1: q: [124,3,128], k/v: [48608,3,128]"""
        cu_seqlens_q = [0, 76, 124]
        cu_seqlens_k = [0, 1640, 48608]
        self._run_test_case(cu_seqlens_q, cu_seqlens_k, "Case1_SmallBatch")

    def test_case2(self):
        """测试 Case 2: q/k/v: [48484,3,128], cu_seqlens 长度为16"""
        cu_seqlens = [0, 3348, 6696, 10044, 12648, 15996, 19344, 22692, 25296,
                      28644, 31992, 35340, 37944, 40734, 43524, 46314, 48484]
        self._run_test_case(cu_seqlens, cu_seqlens, "Case2_LargeBatch")

    def test_valid_input_no_unsupported_params(self):
        """测试所有参数合法时，不抛出异常"""
        try:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                dropout_p=0.0,
            )

        except Exception as e:
            self.fail(f"合法输入不应抛出异常: {e}")

    def test_invalid_q_dimension(self):
        """测试 q 不是 3D 时抛出异常"""
        q_2d = torch.randn(10, 64)
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=q_2d, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                dropout_p=0.0,
            )
        self.assertIn("Expected q to be 3D", str(cm.exception))

    def test_invalid_k_dimension(self):
        """测试 k 不是 3D 时抛出异常"""
        k_2d = torch.randn(80, 64)
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=k_2d, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                dropout_p=0.0,
            )
        self.assertIn("Expected k to be 3D", str(cm.exception))

    def test_invalid_v_dimension(self):
        """测试 v 不是 3D 时抛出异常"""
        v_2d = torch.randn(80, 64)
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=v_2d,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                dropout_p=0.0,
            )
        self.assertIn("Expected v to be 3D", str(cm.exception))

    def test_dropout_p_not_zero(self):
        """测试 dropout_p 不为 0 时报错"""
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                dropout_p=0.1,
            )
        self.assertIn("dropout_p should be set to 0.0 during evaluation", str(cm.exception))

    def test_max_seqlen_q_provided(self):
        """测试 max_seqlen_q 不为 None 时报错"""
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                max_seqlen_q=128
            )
        self.assertIn("max_seqlen_q=128", str(cm.exception))

    def test_max_seqlen_k_provided(self):
        """测试 max_seqlen_k 不为 None 时报错"""
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                max_seqlen_k=128
            )
        self.assertIn("max_seqlen_k=128", str(cm.exception))

    def test_window_size_provided(self):
        """测试 window_size 不为 None 时报错"""
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                window_size=64
            )
        self.assertIn("window_size=64", str(cm.exception))

    def test_softcap_provided(self):
        """测试 softcap 不为 None 时报错"""
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                softcap=30.0
            )
        self.assertIn("softcap=30.0", str(cm.exception))

    def test_alibi_slopes_provided(self):
        """测试 alibi_slopes 不为 None 时报错"""
        alibi = torch.randn(4)
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                alibi_slopes=alibi
            )
        self.assertIn("alibi_slopes=", str(cm.exception))

    def test_deterministic_provided(self):
        """测试 deterministic 不为 None 时报错"""
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                deterministic=True
            )
        self.assertIn("deterministic=True", str(cm.exception))

    def test_return_attn_probs_provided(self):
        """测试 return_attn_probs 不为 None 时报错"""
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                return_attn_probs=True
            )
        self.assertIn("return_attn_probs=True", str(cm.exception))

    def test_block_table_provided(self):
        """测试 block_table 不为 None 时报错"""
        block_table = torch.tensor([0, 1, 2], dtype=torch.int32)
        with self.assertRaises(ParametersInvalid) as cm:
            attention_forward_varlen(
                q=self.q, k=self.k, v=self.v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                block_table=block_table
            )
        self.assertIn("block_table=", str(cm.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)