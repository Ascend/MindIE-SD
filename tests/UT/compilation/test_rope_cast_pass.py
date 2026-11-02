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

"""Pure-logic unit tests for the P2 rope-cast chain matcher (torch-free host).

Covers ``find_rope_cast_chains`` in ``mindiesd/compilation/passes/rope_cast_pass.py``
using hand-built duck-typed nodes (same loader approach as test_gelu_quant_pass.py).
The graph-rewrite path is covered on the NPU side by
``tests/UT/compilation/verify_gelu_quant_and_rope.py``.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _install_fake_torch():
    if "torch" in sys.modules:
        return
    fake_torch = types.ModuleType("torch")
    fake_torch.fx = types.SimpleNamespace(GraphModule=object, Graph=object, Node=object)
    sys.modules["torch"] = fake_torch


def _load_real_module(name: str, rel_path: str):
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
        "mindiesd.compilation.passes.rope_cast_pass",
        "mindiesd/compilation/passes/rope_cast_pass.py",
    )


def _op(name, prefix=""):
    """Simulate a real ``OpOverload``: its ``__name__`` carries the overload suffix
    (``"_to_copy.default"``) and some torch versions add a namespace prefix
    (``"aten::_to_copy.default"``). Using a bare name here would hide exactly the
    mismatch this matcher must handle.
    """
    return type(name, (), {"__name__": f"{prefix}{name}.default"})()


class _FakeNode:
    def __init__(self, op, target, args=(), kwargs=None, users=(), meta=None):
        self.op = op
        self.target = target
        self.args = tuple(args)
        self.kwargs = kwargs or {}
        self.users = dict.fromkeys(users)
        self.meta = meta if meta is not None else {}


FP32 = "torch.float32"
BF16 = "torch.bfloat16"


class TestTargetOpName(unittest.TestCase):
    """The matcher must normalize real OpOverload names, not just bare duck-typed names."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_pass_module()

    def test_overload_and_namespace_variants(self):
        cases = {
            "_to_copy": "_to_copy",
            "_to_copy.default": "_to_copy",
            "aten::_to_copy": "_to_copy",
            "aten::_to_copy.default": "_to_copy",
            "npu_rotary_mul.default": "npu_rotary_mul",
            "npu_dynamic_mx_quant.default": "npu_dynamic_mx_quant",
        }
        for raw, expected in cases.items():
            fake = type("Fake", (), {"__name__": raw})()
            self.assertEqual(self.mod.target_op_name(fake), expected, raw)

    def test_name_attr_absent_falls_back_to_str(self):
        class _NoName:
            def __str__(self):
                return "torch.ops.aten._to_copy.default"

        self.assertEqual(self.mod.target_op_name(_NoName()), "_to_copy")


class TestFindRopeCastChains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_pass_module()

    @staticmethod
    def _build_chain(src_meta=None, in_cast_target="npu_dtype_cast", in_cast_kw=True, out_dtype_token=BF16):
        x_meta = {"val": types.SimpleNamespace(dtype=src_meta)} if src_meta else {}
        x = _FakeNode("placeholder", "x", meta=x_meta)
        if in_cast_kw:  # aten._to_copy 形态：dtype 关键字
            cast_in = _FakeNode("call_function", _op("_to_copy"), args=(x,), kwargs={"dtype": FP32})
        else:  # npu_dtype_cast 形态：dtype 位置参数
            cast_in = _FakeNode("call_function", _op(in_cast_target), args=(x, FP32))
        cos = _FakeNode("placeholder", "cos")
        sin = _FakeNode("placeholder", "sin")
        rope = _FakeNode(
            "call_function",
            _op("npu_rotary_mul"),
            args=(cast_in, cos, sin, "interleave"),
        )
        cast_out = _FakeNode("call_function", _op("_to_copy"), args=(rope,), kwargs={"dtype": out_dtype_token})
        rope.users = dict.fromkeys([cast_out])
        return x, cast_in, cos, sin, rope, cast_out

    def test_basic_chain_matched(self):
        x, cast_in, cos, sin, rope, cast_out = self._build_chain(src_meta=BF16)
        chains = self.mod.find_rope_cast_chains([x, cast_in, cos, sin, rope, cast_out])
        self.assertEqual(len(chains), 1)
        rope_node, in_node, outs = chains[0]
        self.assertIs(rope_node, rope)
        self.assertIs(in_node, cast_in)
        self.assertEqual(outs, [cast_out])

    def test_npu_dtype_cast_positional_form(self):
        x, cast_in, cos, sin, rope, cast_out = self._build_chain(src_meta=BF16, in_cast_kw=False)
        chains = self.mod.find_rope_cast_chains([x, cast_in, cos, sin, rope, cast_out])
        self.assertEqual(len(chains), 1)

    def test_non_cast_user_of_rope_skipped(self):
        x, cast_in, cos, sin, rope, cast_out = self._build_chain(src_meta=BF16)
        add = _FakeNode("call_function", _op("add"), args=(rope, rope))
        rope.users[add] = None
        nodes = [x, cast_in, cos, sin, rope, cast_out, add]
        self.assertEqual(self.mod.find_rope_cast_chains(nodes), [])

    def test_out_cast_wrong_dtype_skipped(self):
        x, cast_in, cos, sin, rope, cast_out = self._build_chain(src_meta=BF16, out_dtype_token=FP32)
        self.assertEqual(self.mod.find_rope_cast_chains([x, cast_in, cos, sin, rope, cast_out]), [])

    def test_in_cast_with_layout_kwarg_skipped(self):
        x, cast_in, cos, sin, rope, cast_out = self._build_chain(src_meta=BF16)
        cast_in.kwargs = {"dtype": FP32, "memory_format": "preserve_format"}
        self.assertEqual(self.mod.find_rope_cast_chains([x, cast_in, cos, sin, rope, cast_out]), [])

    def test_src_meta_fp16_guard(self):
        x, cast_in, cos, sin, rope, cast_out = self._build_chain(src_meta="torch.float16")
        self.assertEqual(self.mod.find_rope_cast_chains([x, cast_in, cos, sin, rope, cast_out]), [])

    def test_src_meta_absent_allowed(self):
        # meta 缺失（纯鸭子类型/极简图）不设防：真实 aot/dynamo 图均有 meta，由其兜底。
        x, cast_in, cos, sin, rope, cast_out = self._build_chain(src_meta=None)
        nodes = [x, cast_in, cos, sin, rope, cast_out]
        self.assertEqual(len(self.mod.find_rope_cast_chains(nodes)), 1)

    def test_rope_input_not_cast_skipped(self):
        x = _FakeNode("placeholder", "x")
        cos = _FakeNode("placeholder", "cos")
        rope = _FakeNode("call_function", _op("npu_rotary_mul"), args=(x, cos, cos, "interleave"))
        self.assertEqual(self.mod.find_rope_cast_chains([x, cos, rope]), [])


if __name__ == "__main__":
    unittest.main()
