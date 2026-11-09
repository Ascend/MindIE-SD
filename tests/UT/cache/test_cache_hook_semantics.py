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

"""Cache hook 语义对齐测试（与具体 cache 实现无关，对应 Issue #139）。"""

import os
import threading
import unittest
from collections import namedtuple
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch import nn

import mindiesd.cache_agent.cache as cache_module
from mindiesd.cache_agent import CacheAgent, CacheConfig
from mindiesd.utils.exception import ModelExecError


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU."
)
class TestCacheHookSemantics(unittest.TestCase):
    """与具体 cache 实现无关的 hook 语义对齐测试。"""

    @staticmethod
    def _make_config():
        return CacheConfig(
            method="attention_cache",
            blocks_count=2,
            steps_count=3,
            step_start=1,
            step_interval=2,
            step_end=3,
        )

    def _run_steps(self, agent, func, x):
        """运行 3 个 step，每个 step 包含 2 个 block，返回各 step 的输出。"""
        outputs = []
        for _ in range(3):
            outputs.append([agent.apply(func, x) for _ in range(2)])
        return outputs

    def test_plain_function_reuses_cache_with_isolated_result(self):
        """非 nn.Module 的纯函数命中缓存时，返回缓存结果的隔离副本。"""
        config = self._make_config()
        agent = CacheAgent(config)

        def func(x):
            return x * 2

        x = torch.tensor([1.0, 2.0])
        outputs = self._run_steps(agent, func, x)

        self.assertTrue(torch.equal(outputs[2][0], outputs[0][0]))
        self.assertTrue(torch.equal(outputs[2][1], outputs[0][1]))

    def test_module_without_hooks_reuses_cache_with_isolated_result(self):
        """命中缓存且模块无 hooks 时，返回缓存结果的隔离副本。"""
        config = self._make_config()
        agent = CacheAgent(config)
        block = nn.Linear(4, 4, bias=False)

        x = torch.randn(1, 4)
        outputs = self._run_steps(agent, block, x)

        self.assertTrue(torch.equal(outputs[2][0], outputs[0][0]))
        self.assertTrue(torch.equal(outputs[2][1], outputs[0][1]))

    def test_global_hooks_are_called_when_cache_hit(self):
        """命中缓存时，全局注册的 forward hook 同样不应被跳过。"""
        config = self._make_config()
        agent = CacheAgent(config)
        block = nn.Linear(4, 4, bias=False)
        calls = []

        handle = torch.nn.modules.module.register_module_forward_hook(lambda mod, args, out: calls.append(out))
        try:
            x = torch.randn(1, 4)
            outputs = self._run_steps(agent, block, x)

            # step0/step1 计算 + step2 命中缓存 = 6 次
            self.assertEqual(len(calls), 6)
            self.assertTrue(torch.equal(outputs[2][0], outputs[0][0]))
            self.assertTrue(torch.equal(outputs[2][1], outputs[0][1]))
        finally:
            handle.remove()

    def test_global_with_kwargs_hooks_are_called_when_cache_hit(self):
        """命中缓存时，with_kwargs 形式的全局 forward hook 同样触发且不重复执行。"""
        config = self._make_config()
        agent = CacheAgent(config)
        block = nn.Linear(4, 4, bias=False)
        calls = []

        def global_hook_with_kwargs(mod, args, kwargs, out):
            calls.append((args, kwargs, out))
            return out

        handle = torch.nn.modules.module.register_module_forward_hook(global_hook_with_kwargs, with_kwargs=True)
        try:
            x = torch.randn(1, 4)
            outputs = self._run_steps(agent, block, x)

            # 每次调用（含命中缓存）恰好触发一次，不重复执行
            self.assertEqual(len(calls), 6)
            # 命中缓存时 with_kwargs 全局 hook 收到位置参数元组与缓存输出
            self.assertTrue(torch.equal(calls[4][0][0], x))
            self.assertTrue(torch.equal(calls[4][2], outputs[0][0]))
        finally:
            handle.remove()

    def test_global_pre_hook_when_cache_hit(self):
        """回归：全局 forward pre-hook 在命中缓存时触发，且未命中路径的输入修改生效。"""
        config = self._make_config()
        agent = CacheAgent(config)
        block = nn.Linear(4, 4, bias=False)
        calls = []

        def global_pre_hook(mod, args):
            calls.append(args)
            return (args[0] * 2,)

        x = torch.randn(1, 4)
        # 期望值需在注册全局 hook 前计算，避免回归计算本身再次被全局 hook 改写
        expected = block(x * 2)

        handle = torch.nn.modules.module.register_module_forward_pre_hook(global_pre_hook)
        try:
            outputs = self._run_steps(agent, block, x)

            # 每次调用（含命中缓存）恰好触发一次；与 PyTorch 一致，hook 收到位置参数元组
            self.assertEqual(len(calls), 6)
            self.assertIsInstance(calls[0], tuple)
            self.assertTrue(torch.equal(calls[0][0], x))
            # 未命中路径：pre-hook 对输入的修改生效
            self.assertTrue(torch.equal(outputs[0][0], expected))
            # 命中路径：复用同一结果
            self.assertTrue(torch.equal(outputs[2][0], outputs[0][0]))
        finally:
            handle.remove()

    def test_custom_call_module_keeps_call_semantics(self):
        """回归：重写 __call__ 的模块未被绕过，未命中路径保留 __call__ 中的后处理逻辑。"""
        config = self._make_config()
        agent = CacheAgent(config)
        call_count = {"n": 0}
        hook_outputs = []

        class CustomCallBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)

            def __call__(self, *args, **kwargs):
                call_count["n"] += 1
                out = super().__call__(*args, **kwargs)
                # 后处理逻辑写在 __call__ 中：绕过 __call__ 直接调用 forward 会丢失该逻辑
                return out * 2

            def forward(self, x):
                return self.linear(x)

        block = CustomCallBlock()
        block.register_forward_hook(lambda mod, args, out: hook_outputs.append(out))

        x = torch.randn(1, 4)
        outputs = self._run_steps(agent, block, x)

        # 自定义调用链不缓存，每次 apply 都走正常 __call__。
        self.assertEqual(call_count["n"], 6)
        # 未命中路径保留 __call__ 的后处理（×2）
        self.assertTrue(torch.equal(outputs[0][0], block.linear(x) * 2))
        self.assertTrue(torch.equal(outputs[2][0], outputs[0][0]))
        # 自定义调用链不使用缓存，hooks 每次调用都会触发。
        self.assertEqual(len(hook_outputs), 6)

    def test_custom_call_impl_module_keeps_call_semantics(self):
        """回归：重写 _call_impl 的模块未被绕过，未命中路径保留调用级后处理逻辑。"""
        config = self._make_config()
        agent = CacheAgent(config)
        call_count = {"n": 0}
        hook_outputs = []

        class CustomCallImplBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)

            def _call_impl(self, *args, **kwargs):
                call_count["n"] += 1
                out = super()._call_impl(*args, **kwargs)
                return out * 2

            def forward(self, x):
                return self.linear(x)

        block = CustomCallImplBlock()
        block.register_forward_hook(lambda mod, args, out: hook_outputs.append(out))

        x = torch.randn(1, 4)
        outputs = self._run_steps(agent, block, x)

        # 自定义调用链不缓存，每次 apply 都执行自定义 _call_impl。
        self.assertEqual(call_count["n"], 6)
        self.assertTrue(torch.equal(outputs[0][0], block.linear(x) * 2))
        self.assertTrue(torch.equal(outputs[2][0], outputs[0][0]))
        self.assertEqual(len(hook_outputs), 6)

    def test_instance_custom_call_impl_module_keeps_call_semantics(self):
        """回归：实例级替换 _call_impl 的模块未被绕过，未命中路径保留调用级逻辑。"""
        config = self._make_config()
        agent = CacheAgent(config)
        call_count = {"n": 0}
        hook_outputs = []

        class InstanceCallImplBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.linear(x)

        block = InstanceCallImplBlock()
        standard_call_impl = nn.Module._call_impl

        def custom_call_impl(mod, *args, **kwargs):
            call_count["n"] += 1
            return standard_call_impl(mod, *args, **kwargs) * 2

        # pylint: disable-next=attribute-defined-outside-init
        block._call_impl = custom_call_impl.__get__(block, type(block))
        block.register_forward_hook(lambda mod, args, out: hook_outputs.append(out))

        x = torch.randn(1, 4)
        outputs = self._run_steps(agent, block, x)
        expected = block.linear(x) * 2

        # 自定义调用链不缓存，每次 apply 都执行实例级 _call_impl。
        self.assertEqual(call_count["n"], 6)
        self.assertTrue(torch.equal(outputs[0][0], expected))
        self.assertTrue(torch.equal(outputs[2][0], outputs[0][0]))
        self.assertEqual(len(hook_outputs), 6)

    def test_compiled_call_impl_module_keeps_call_semantics(self):
        """回归：torch.compile 使用的 _compiled_call_impl 未被绕过。"""
        config = self._make_config()
        agent = CacheAgent(config)
        hook_outputs = []

        class CompiledCallBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.linear(x)

        block = CompiledCallBlock()
        # pylint: disable-next=attribute-defined-outside-init
        block._compiled_call_impl = lambda *args, **kwargs: block._call_impl(*args, **kwargs) * 2
        block.register_forward_hook(lambda mod, args, out: hook_outputs.append(out))

        x = torch.randn(1, 4)
        outputs = self._run_steps(agent, block, x)

        self.assertTrue(torch.equal(outputs[0][0], block.linear(x) * 2))
        self.assertTrue(torch.equal(outputs[2][0], outputs[0][0]))
        self.assertEqual(len(hook_outputs), 6)

    @unittest.skipUnless(hasattr(torch.nn.Module, "compile"), "Module.compile is required.")
    def test_torch_compile_module_does_not_skip_hooks(self):
        """真实 torch.compile 包装模块不得在缓存判定后跳过 hooks。"""
        config = self._make_config()
        agent = CacheAgent(config)
        forward_calls = []
        hook_calls = []

        class CompiledBlock(nn.Module):
            def forward(self, x):
                forward_calls.append(x)
                return x * 2

        block = CompiledBlock()
        block.register_forward_hook(lambda mod, args, out: hook_calls.append(out))
        block.compile(backend="eager")

        x = torch.randn(1, 4)
        self._run_steps(agent, block, x)

        self.assertEqual(len(forward_calls), 6)
        self.assertEqual(len(hook_calls), 6)

    def test_namedtuple_output_type_and_tensors_are_preserved_for_hooks(self):
        """命中与未命中路径均应向 hook 传递同类型的 namedtuple。"""
        config = self._make_config()
        agent = CacheAgent(config)
        HookOutput = namedtuple("HookOutput", ["hidden", "residual"])
        hook_types = []

        class TupleBlock(nn.Module):
            def forward(self, x):
                return HookOutput(hidden=x * 2, residual=x * 3)

        block = TupleBlock()

        def inplace_hook(mod, args, out):
            hook_types.append(type(out))
            out.hidden.mul_(2)
            out.residual.add_(1)

        block.register_forward_hook(inplace_hook)
        x = torch.tensor([1.0, 2.0])
        outputs = self._run_steps(agent, block, x)

        self.assertTrue(all(output_type is HookOutput for output_type in hook_types))
        self.assertTrue(torch.equal(outputs[2][0].hidden, outputs[0][0].hidden))
        self.assertTrue(torch.equal(outputs[2][0].residual, outputs[0][0].residual))
        self.assertTrue(torch.equal(outputs[2][1].hidden, outputs[0][1].hidden))
        self.assertTrue(torch.equal(outputs[2][1].residual, outputs[0][1].residual))

    def test_dict_output_values_are_isolated_for_inplace_hooks(self):
        """dict 输出的张量不得被原地修改型 hook 污染。"""
        config = self._make_config()
        agent = CacheAgent(config)

        class DictBlock(nn.Module):
            def forward(self, x):
                return {"hidden": x * 2, "residual": x * 3}

        block = DictBlock()

        def inplace_hook(mod, args, out):
            out["hidden"].mul_(2)
            out["residual"].add_(1)

        block.register_forward_hook(inplace_hook)
        x = torch.tensor([1.0, 2.0])
        outputs = self._run_steps(agent, block, x)

        self.assertTrue(torch.equal(outputs[2][0]["hidden"], outputs[0][0]["hidden"]))
        self.assertTrue(torch.equal(outputs[2][0]["residual"], outputs[0][0]["residual"]))
        self.assertTrue(torch.equal(outputs[2][1]["hidden"], outputs[0][1]["hidden"]))
        self.assertTrue(torch.equal(outputs[2][1]["residual"], outputs[0][1]["residual"]))

    def test_list_output_values_are_isolated_for_inplace_hooks(self):
        """list 输出中的张量也不得被原地修改型 hook 污染。"""
        config = self._make_config()
        agent = CacheAgent(config)

        class ListBlock(nn.Module):
            def forward(self, x):
                return [x * 2, x * 3]

        block = ListBlock()

        def inplace_hook(mod, args, out):
            out[0].mul_(2)
            out[1].add_(1)

        block.register_forward_hook(inplace_hook)
        x = torch.tensor([1.0, 2.0])
        outputs = self._run_steps(agent, block, x)

        for step in (2,):
            self.assertTrue(torch.equal(outputs[step][0][0], outputs[0][0][0]))
            self.assertTrue(torch.equal(outputs[step][0][1], outputs[0][0][1]))
            self.assertTrue(torch.equal(outputs[step][1][0], outputs[0][1][0]))
            self.assertTrue(torch.equal(outputs[step][1][1], outputs[0][1][1]))

    def test_custom_container_output_is_isolated_for_inplace_hooks(self):
        """自定义 dataclass 输出中的张量同样不得污染缓存。"""
        config = self._make_config()
        agent = CacheAgent(config)

        @dataclass
        class Box:
            hidden: torch.Tensor
            residual: torch.Tensor

        class BoxBlock(nn.Module):
            def forward(self, x):
                return Box(hidden=x * 2, residual=x * 3)

        block = BoxBlock()

        def inplace_hook(mod, args, out):
            out.hidden.mul_(2)
            out.residual.add_(1)

        block.register_forward_hook(inplace_hook)
        x = torch.tensor([1.0, 2.0])
        outputs = self._run_steps(agent, block, x)

        self.assertTrue(torch.equal(outputs[2][0].hidden, outputs[0][0].hidden))
        self.assertTrue(torch.equal(outputs[2][0].residual, outputs[0][0].residual))
        self.assertTrue(torch.equal(outputs[2][1].hidden, outputs[0][1].hidden))
        self.assertTrue(torch.equal(outputs[2][1].residual, outputs[0][1].residual))

    def test_unclonable_output_raises_instead_of_corrupting_cache(self):
        """无法安全复制的输出应显式失败，不能静默写入可能被 hook 修改的缓存。"""
        config = self._make_config()
        agent = CacheAgent(config)

        class UnclonableOutput:
            def __deepcopy__(self, memo):
                raise RuntimeError("cannot copy output")

        class UnclonableBlock(nn.Module):
            def forward(self, x):
                return UnclonableOutput()

        block = UnclonableBlock()
        block.register_forward_hook(lambda mod, args, out: None)

        with self.assertRaises(ModelExecError) as context:
            agent.apply(block, torch.tensor([1.0, 2.0]))
        self.assertIn("cannot safely copy", str(context.exception))

    def test_clone_preserves_aliases_types_and_shared_storage(self):
        """复制输出应保留 Tensor alias、容器子类和 torch.return_types 类型。"""
        tensor = torch.tensor([1.0, 2.0])
        raw = (tensor, [tensor], {"view": tensor[:]})
        cloned = cache_module._clone_result(raw)

        self.assertTrue(cloned[0] is cloned[1][0])
        self.assertEqual(cloned[0].data_ptr(), cloned[2]["view"].data_ptr())
        self.assertNotEqual(cloned[0].data_ptr(), tensor.data_ptr())

        class TupleSubclass(tuple):
            pass

        class ListSubclass(list):
            pass

        self.assertIs(type(cache_module._clone_result(TupleSubclass((tensor,)))), TupleSubclass)
        self.assertIs(type(cache_module._clone_result(ListSubclass([tensor]))), ListSubclass)
        max_result = torch.max(tensor, dim=0)
        self.assertIs(
            type(cache_module._clone_result(max_result)),
            torch.return_types.max,  # pylint: disable=no-member
        )

    def test_clone_supports_cyclic_containers(self):
        """循环容器的复制应保留循环结构而不是无限递归。"""
        tensor = torch.tensor([1.0])
        cyclic = [tensor]
        cyclic.append(cyclic)
        cloned = cache_module._clone_result(cyclic)

        self.assertIs(cloned, cloned[1])
        self.assertIsNot(cloned, cyclic)
        self.assertTrue(torch.equal(cloned[0], tensor))
        self.assertNotEqual(cloned[0].data_ptr(), tensor.data_ptr())

    def test_concurrent_agent_use_is_rejected(self):
        """同一个 agent 的并发调用应显式失败，不能静默混写缓存状态。"""
        config = self._make_config()
        agent = CacheAgent(config)
        entered = threading.Event()
        release = threading.Event()
        thread_errors = []

        class BlockingBlock(nn.Module):
            def forward(self, x):
                if threading.current_thread().name == "first-cache-call":
                    entered.set()
                    release.wait(timeout=5)
                return x * 2

        block = BlockingBlock()

        def first_call():
            try:
                agent.apply(block, torch.tensor([1.0]))
            except Exception as exc:  # pylint: disable=broad-except
                thread_errors.append(exc)

        thread = threading.Thread(target=first_call, name="first-cache-call")
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        try:
            with self.assertRaisesRegex(ModelExecError, "concurrent or reentrant"):
                agent.apply(block, torch.tensor([2.0]))
        finally:
            release.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(thread_errors, [])
        agent.apply(block, torch.tensor([3.0]))

    def test_backward_hooks_keep_native_semantics(self):
        """存在 backward hook 时应保持原生调用并禁用不兼容的缓存复用。"""
        config = self._make_config()
        agent = CacheAgent(config)
        block = nn.Linear(4, 4, bias=False)
        forward_calls = []
        backward_calls = []

        block.register_forward_hook(lambda mod, args, out: forward_calls.append(out))
        block.register_full_backward_hook(
            lambda mod, grad_input, grad_output: backward_calls.append((grad_input, grad_output))
        )

        x = torch.randn(1, 4, requires_grad=True)
        for _ in range(3):
            for _ in range(2):
                agent.apply(block, x).sum().backward()

        self.assertEqual(len(forward_calls), 6)
        self.assertEqual(len(backward_calls), 6)

    def test_always_call_hook_runs_when_forward_raises(self):
        """未命中路径的 forward 异常仍应触发 always_call forward hook。"""
        config = self._make_config()
        agent = CacheAgent(config)

        class RaisingBlock(nn.Module):
            def forward(self, x):
                raise RuntimeError("forward failed")

        block = RaisingBlock()
        hook_outputs = []
        block.register_forward_hook(lambda mod, args, out: hook_outputs.append(out), always_call=True)

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            agent.apply(block, torch.tensor([1.0, 2.0]))
        self.assertEqual(hook_outputs, [None])

    def test_always_call_hook_keeps_native_semantics_on_hit(self):
        """always_call hook 存在时保留原生调用链，命中与计算路径语义一致。"""
        config = self._make_config()
        agent = CacheAgent(config)

        class DummyBlock(nn.Module):
            def forward(self, x):
                return x * 2

        block = DummyBlock()

        def raising_hook(mod, args, out):
            raise RuntimeError("hook failed")

        block.register_forward_hook(raising_hook)
        always_calls = []
        block.register_forward_hook(lambda mod, args, out: always_calls.append(out), always_call=True)

        # always_call 触发原生 __call__，异常传播与 always_call 补跑由 PyTorch 保证
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "hook failed"):
                agent.apply(block, torch.tensor([1.0, 2.0]))

        self.assertEqual(len(always_calls), 2)
        self.assertTrue(all(out is not None for out in always_calls))

    def test_tracing_uses_native_call_without_cache_reuse(self):
        """TorchScript tracing 状态下应保持原生调用并且不缓存。"""
        config = self._make_config()
        agent = CacheAgent(config)
        call_count = {"n": 0}

        class TracingBlock(nn.Module):
            def forward(self, x):
                call_count["n"] += 1
                return x * 2

        block = TracingBlock()
        x = torch.tensor([1.0, 2.0])

        def _tracing_state():
            return object()

        with patch.object(cache_module.torch._C, "_get_tracing_state", _tracing_state):
            self._run_steps(agent, block, x)

        self.assertEqual(call_count["n"], 6)


if __name__ == "__main__":
    unittest.main()
