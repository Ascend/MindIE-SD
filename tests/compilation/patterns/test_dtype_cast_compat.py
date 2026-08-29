import os
import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from mindiesd.compilation.patterns.rms_norm_pattern import (
    IS_TORCH_GE_29,
)
from mindiesd.compilation.patterns.rms_norm_pattern import (
    create as create_rmsnorm_pattern,
)
from mindiesd.compilation.patterns.rope_pattern import create as create_rope_pattern


def _trace(pattern, args):
    return make_fx(pattern)(*args)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
@unittest.skipUnless(IS_TORCH_GE_29, "dtype cast 适配分支仅对 torch >= 2.9 生效")
class TestPatternDtypeCastCompat(unittest.TestCase):
    """回归测试：torch_npu >= 2.9 移除 Tensor.to 的 NPU 过适配(MR 30358)后,
    AOT 分解中的 dtype cast 由 npu._npu_dtype_cast 变为 aten._to_copy。
    本测试验证 pattern 期望图已适配为 _to_copy, 否则 RMSNorm/RoPE
    融合在 torch 2.9+ 环境会全部失配。
    """

    def test_rmsnorm_pattern_first_cast_is_to_copy(self):
        pattern_cls = create_rmsnorm_pattern(dtype=torch.bfloat16, epsilon=1e-6)
        hidden_states = torch.empty(2, 2, 2, 2, dtype=torch.bfloat16, device="meta")
        weight = torch.empty(2, dtype=torch.bfloat16, device="meta")

        gm = _trace(pattern_cls.pattern, (hidden_states, weight))
        call_targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]

        # rms_norm 分解序列的第一个 op 必须是 _to_copy（fp32 提升）
        self.assertEqual(call_targets[0], torch.ops.aten._to_copy.default)

    def test_rope_pattern_casts_are_to_copy(self):
        pattern_cls = create_rope_pattern(torch.float32)
        x = torch.empty(2, 2, 2, 2, dtype=torch.bfloat16, device="meta")
        cos = torch.empty(1, 2, 1, 2, dtype=torch.float32, device="meta")
        sin = torch.empty(1, 2, 1, 2, dtype=torch.float32, device="meta")

        gm = _trace(pattern_cls.pattern, (x, cos, sin))
        call_targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]

        to_copy_count = sum(t == torch.ops.aten._to_copy.default for t in call_targets)
        # x 与 x_rotated 的 fp32 cast + 末尾 .to(x.dtype) cast
        self.assertGreaterEqual(to_copy_count, 3)
        # 最后一个 op 必须是 _to_copy（对应真实图 apply_rotary_emb 的 .to(x.dtype)）
        self.assertEqual(call_targets[-1], torch.ops.aten._to_copy.default)


if __name__ == "__main__":
    unittest.main()
