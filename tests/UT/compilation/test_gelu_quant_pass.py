#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2 that you can use this software in accordance
# with the terms and conditions of the Mulan PSL v2. You may obtain a copy of the
# Mulan PSL at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Pure-logic unit tests for the P4 gelu→quant chain matcher (torch-free host).

Covers ``find_gelu_quant_chains`` in ``mindiesd/compilation/passes/gelu_quant_pass.py``
using hand-built duck-typed nodes: real fx graphs need torch (absent on this Windows
dev machine), and the matcher only touches ``node.op/target/args/kwargs`` — exactly
the surface the fakes provide. The graph-rewrite path (needs real ``fx.Graph``) is
covered on the NPU side by ``tests/UT/compilation/verify_gelu_quant_and_rope.py``.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _install_fake_torch():
    """gm_pass_base 在模块级 ``import torch`` 并在类注解里取 ``torch.fx.GraphModule``。"""
    if "torch" in sys.modules:
        return
    fake_torch = types.ModuleType("torch")
    fake_torch.fx = types.SimpleNamespace(GraphModule=object, Graph=object, Node=object)
    sys.modules["torch"] = fake_torch


def _load_real_module(name: str, rel_path: str):
    """按文件路径把真实仓库模块加载为 ``name``，父包用占位 ModuleType 注册。

    与 test_backend_mode.py 的策略 2 同思路：绕开 ``mindiesd/__init__.py``
    （依赖 torch/torch_npu），仅执行目标文件本身。
    """
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)
    target = _REPO_ROOT / rel_path
    if not target.exists():
        raise FileNotFoundError(target)
    # 同一文件已按本策略加载过就复用：模块级副作用不可重入，例如 logging.py 末尾的
    # init_logger() 会往全局 logger 追加 StreamHandler，重复执行会叠加 handler。
    loaded = sys.modules.get(name)
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == target.resolve():
        return loaded
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_pass_module():
    _install_fake_torch()
    _load_real_module("mindiesd.utils.env", "mindiesd/utils/env.py")
    _load_real_module("mindiesd.utils.logs.logging", "mindiesd/utils/logs/logging.py")
    _load_real_module("mindiesd.compilation.passes.gm_pass_base", "mindiesd/compilation/passes/gm_pass_base.py")
    return _load_real_module(
        "mindiesd.compilation.passes.gelu_quant_pass",
        "mindiesd/compilation/passes/gelu_quant_pass.py",
    )


def _op(name):
    """构造等价 fx 图 ``torch.ops`` 重载的假算子对象。

    真实 ``OpOverload.__name__`` 带重载后缀（``"npu_dynamic_mx_quant.default"``，
    部分版本还带 ``"aten::"`` 前缀），这里照此模拟；用裸名会掩盖匹配器必须处理的
    差异（见 ``gm_pass_base.target_op_name``）。
    """
    return type(name, (), {"__name__": f"{name}.default"})()


class _FakeNode:
    def __init__(self, op, target, args=(), kwargs=None, users=()):
        self.op = op
        self.target = target
        self.args = tuple(args)
        self.kwargs = kwargs or {}
        self.users = dict.fromkeys(users)


QUANT = "npu_dynamic_mx_quant"
DST = {"dst_type": 292}


class TestFindGeluQuantChains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_pass_module()

    def test_direct_chain_fast_gelu(self):
        x = _FakeNode("placeholder", "x")
        gelu = _FakeNode("call_function", _op("npu_fast_gelu"), args=(x,))
        quant = _FakeNode("call_function", _op(QUANT), args=(gelu,), kwargs=DST)
        chains = self.mod.find_gelu_quant_chains([x, gelu, quant])
        self.assertEqual(len(chains), 1)
        _, gelu_node, views, dst_type, approximate = chains[0]
        self.assertIs(gelu_node, gelu)
        self.assertEqual(views, ())
        self.assertEqual(dst_type, 292)
        self.assertIsNone(approximate)  # fast_gelu 链头：由 pass 配置决定

    def test_view_chain_order_preserved(self):
        x = _FakeNode("placeholder", "x")
        gelu = _FakeNode("call_function", _op("npu_fast_gelu"), args=(x,))
        view1 = _FakeNode("call_function", _op("view"), args=(gelu, (1, -1, 13824)))
        view2 = _FakeNode("call_function", _op("_unsafe_view"), args=(view1, (26350, -1)))
        quant = _FakeNode("call_function", _op(QUANT), args=(view2,), kwargs=DST)
        chains = self.mod.find_gelu_quant_chains([x, gelu, view1, view2, quant])
        self.assertEqual(len(chains), 1)
        _, _, views, _, _ = chains[0]
        # 按图内顺序（靠近 gelu 的 view 在前），改写时按此序在 gelu 输入上重建。
        self.assertEqual(views, (view1, view2))

    def test_aten_gelu_approximate_passthrough(self):
        x = _FakeNode("placeholder", "x")
        gelu_tanh = _FakeNode("call_function", _op("gelu"), args=(x,), kwargs={"approximate": "tanh"})
        gelu_plain = _FakeNode("call_function", _op("gelu"), args=(x,))
        q1 = _FakeNode("call_function", _op(QUANT), args=(gelu_tanh,), kwargs={"dst_type": 24})
        q2 = _FakeNode("call_function", _op(QUANT), args=(gelu_plain,), kwargs={"dst_type": 24})
        chains = self.mod.find_gelu_quant_chains([x, gelu_tanh, gelu_plain, q1, q2])
        self.assertEqual([c[4] for c in chains], ["tanh", "none"])

    def test_non_gelu_head_no_match(self):
        x = _FakeNode("placeholder", "x")
        add = _FakeNode("call_function", _op("add"), args=(x, x))
        already = _FakeNode("call_function", _op("npu_gelu_quant"), args=(x,), kwargs=DST)
        method_node = _FakeNode("call_method", _op("gelu"), args=(x,))
        q1 = _FakeNode("call_function", _op(QUANT), args=(add,), kwargs=DST)
        q2 = _FakeNode("call_function", _op(QUANT), args=(already,), kwargs=DST)
        q3 = _FakeNode("call_function", _op(QUANT), args=(method_node,), kwargs=DST)
        nodes = [x, add, already, method_node, q1, q2, q3]
        self.assertEqual(self.mod.find_gelu_quant_chains(nodes), [])

    def test_quant_input_placeholder_no_match(self):
        x = _FakeNode("placeholder", "x")
        quant = _FakeNode("call_function", _op(QUANT), args=(x,), kwargs=DST)
        self.assertEqual(self.mod.find_gelu_quant_chains([x, quant]), [])


if __name__ == "__main__":
    unittest.main()
