#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Breakable Graph 分段后端的段间绑定与回放校验（不依赖 NPU kernel）。

用回放桩替换 ``_capture_single_graph`` / ``_copy_inputs_on_stream``，因此无需真实
图捕获；用例固定跑在 CPU 上，检测到可用 NPU 即判定跑错了套件（后端把 orig 输入
搬到 NPU 的分支因此不会生效）。覆盖两处回归：

1. 单输出段导出的节点本身返回 tuple 时，整个返回值必须作为一个衔接值传给下一段；
2. 捕获期固化的标量输入在回放时取值变化必须显式报错，不能静默复用。
"""

import contextlib
import importlib
import operator
import unittest
from unittest.mock import patch

import torch


class _NoopGraph:
    """NPUGraph 替身：replay 不做事（捕获阶段的 eager 结果即视为回放结果）。"""

    def replay(self):
        pass


def _stub_capture(gm, static_inputs):
    return _NoopGraph(), gm(*static_inputs)


def _sync_copy(entry, needs_copy):
    for static_buf, new_inp in needs_copy:
        static_buf.copy_(new_inp)


class _TupleSplitModel(torch.nn.Module):
    """单节点返回 tuple：split 导出一个 tuple 值供后段的两个 getitem 消费。"""

    def forward(self, x):
        first, second = x.split(1, dim=0)
        return first + second


class _ScalarMulModel(torch.nn.Module):
    """标量图输入：k 在捕获期固化进图，回放时不得静默复用旧值。"""

    def forward(self, x, k):
        y = x * k
        return y.reshape(-1) + 1


class TestSegmentedAclGraphBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splitter = importlib.import_module("mindiesd.compilation.graph_splitter")
        cls.backend_mod = importlib.import_module("mindiesd.compilation.aclgraph_backend")
        # 纯 CPU 用例：图捕获与回放均已用桩替换，不应依赖设备；有可用 NPU 说明跑错了套件。
        npu = getattr(torch, "npu", None)
        assert npu is None or not npu.is_available(), "segmented aclgraph UT must run without an available NPU"
        cls.device = torch.device("cpu")

    def _tensor(self, *shape, dtype=torch.float32):
        count = 1
        for dim in shape:
            count *= dim
        return torch.arange(count, dtype=dtype, device=self.device).reshape(*shape)

    @staticmethod
    def _reshape_breakpoints(gm):
        return [node.name for node in gm.graph.nodes if node.name.startswith("reshape")]

    def _compile_segmented(self, model, breakpoints, args):
        """返回 ``(run, first_output)``；``run`` 始终在设备无关桩的 patch 作用域内调用。"""
        gm = torch.fx.symbolic_trace(model)
        _, seg_meta, out_spec = self.splitter.split_graph_by_breakpoints(gm, breakpoints)
        self.assertIsNotNone(seg_meta, "breakpoints should split the graph into segments")
        backend = self.backend_mod.create_segmented_aclgraph_backend(gm, seg_meta, out_spec)

        @contextlib.contextmanager
        def _stub_env():
            with (
                patch.object(self.backend_mod, "_capture_single_graph", _stub_capture),
                patch.object(self.backend_mod, "_copy_inputs_on_stream", _sync_copy),
            ):
                yield

        def run(*call_args):
            with _stub_env():
                return backend(*call_args)

        return run, run(*args)

    def test_single_tuple_output_binds_whole_tuple(self):
        """前段唯一导出节点返回 tuple 时，后段必须收到整个 tuple。"""
        model = _TupleSplitModel()
        gm = torch.fx.symbolic_trace(model)
        breakpoints = [node.name for node in gm.graph.nodes if node.target is operator.getitem][:1]
        self.assertTrue(breakpoints, "symbolic trace should emit getitem nodes for the split result")

        x = self._tensor(2, 3)
        _, out = self._compile_segmented(model, breakpoints, (x,))
        self.assertTrue(torch.allclose(out, model(x)), "segmented output must match the eager model")

    def test_scalar_input_change_raises(self):
        """同一形状、不同标量输入必须显式报错，而不是复用捕获值。"""
        model = _ScalarMulModel()
        gm = torch.fx.symbolic_trace(model)
        breakpoints = self._reshape_breakpoints(gm)
        self.assertTrue(breakpoints, "symbolic trace should emit a reshape node")

        x = self._tensor(4)
        run, out = self._compile_segmented(model, breakpoints, (x, 2))
        self.assertTrue(torch.allclose(out, model(x, 2)))

        with self.assertRaises(RuntimeError) as ctx:
            run(x, 3)
        self.assertIn("scalar input mismatch", str(ctx.exception))

    def test_scalar_input_same_value_replays(self):
        """标量值未变化时正常回放。"""
        model = _ScalarMulModel()
        gm = torch.fx.symbolic_trace(model)
        breakpoints = self._reshape_breakpoints(gm)

        x = self._tensor(4)
        run, _ = self._compile_segmented(model, breakpoints, (x, 2))
        self.assertTrue(torch.allclose(run(x, 2), model(x, 2)))

    def test_tensor_input_shape_change_raises(self):
        """分段路径不支持运行时形状变化：显式报错，不静默错算。"""
        model = _ScalarMulModel()
        gm = torch.fx.symbolic_trace(model)
        breakpoints = self._reshape_breakpoints(gm)

        run, _ = self._compile_segmented(model, breakpoints, (self._tensor(4), 2))
        with self.assertRaises(RuntimeError) as ctx:
            run(self._tensor(8), 2)
        self.assertIn("input mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
