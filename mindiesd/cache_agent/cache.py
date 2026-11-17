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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass

import torch
from torch.nn.modules.module import (
    _global_backward_hooks,
    _global_backward_pre_hooks,
    _global_forward_hooks,
    _global_forward_hooks_always_called,
    _global_forward_hooks_with_kwargs,
    _global_forward_pre_hooks,
)

from mindiesd.utils.exception import ModelExecError


_UNSET = object()


@dataclass
class CacheConfig:
    method: str
    blocks_count: int
    steps_count: int
    step_start: int = 0
    step_interval: int = 1
    step_end: int = 10000
    block_start: int = 0
    block_end: int = 10000


def _uses_custom_call_chain(module):
    """判断模块是否使用了非标准 PyTorch 调用链。

    部分模块会把后处理、执行上下文或分发逻辑写在 ``__call__`` / ``_call_impl``
    中（包括实例级 monkey patch）；``torch.compile`` 也可能通过
    ``_compiled_call_impl`` 替换调用链。直接调用 ``forward`` 会绕过这些逻辑，
    因此这类模块必须保留正常的调用语义。
    """
    module_type = type(module)
    if module_type.__call__ is not torch.nn.Module.__call__:
        return True
    if module_type._call_impl is not torch.nn.Module._call_impl:
        return True
    if "_call_impl" in vars(module):
        return True
    return getattr(module, "_compiled_call_impl", None) is not None


def _has_forward_hooks(module):
    """判断模块自身或全局作用域是否注册了 forward pre-hook / forward hook。"""
    return bool(
        module._forward_pre_hooks
        or module._forward_hooks
        or _global_forward_pre_hooks
        or _global_forward_hooks
        or _global_forward_hooks_with_kwargs
    )


def _requires_native_call(module):
    """判断模块是否需要保留完整原生调用链，而不能安全复用缓存。"""
    tracing_state = getattr(torch._C, "_get_tracing_state", None)
    return bool(
        module._backward_hooks
        or module._backward_pre_hooks
        or _global_backward_hooks
        or _global_backward_pre_hooks
        or module._forward_hooks_always_called
        or _global_forward_hooks_always_called
        or (tracing_state and tracing_state())
    )


def _invoke_module(module, args, kwargs, cached_result=_UNSET, protect_forward_output=False):
    """通过原生调用链执行 hooks，并可按需捕获 forward 的原始输出。

    临时替换实例上的 ``forward``：未命中时仍执行真实 forward 并记录原始输出，
    命中时直接返回缓存结果。``Module.__call__`` 负责 pre-hook 和 post-hook 的顺序、
    异常及 ``with_kwargs`` 语义，避免在 CacheAgent 中复制 PyTorch 内部实现。
    """
    original_forward = module.forward
    instance_forward = vars(module).get("forward", _UNSET)
    raw_result = None

    def _replacement_forward(*forward_args, **forward_kwargs):
        nonlocal raw_result
        if cached_result is not _UNSET:
            return cached_result
        raw_result = original_forward(*forward_args, **forward_kwargs)
        return _clone_result(raw_result) if protect_forward_output else raw_result

    object.__setattr__(module, "forward", _replacement_forward)
    try:
        result = module(*args, **kwargs)
    finally:
        if instance_forward is _UNSET:
            object.__delattr__(module, "forward")
        else:
            object.__setattr__(module, "forward", instance_forward)
    return raw_result, result


def _collect_tensors(value, seen, tensors):
    """收集输出图中的 Tensor，跳过循环引用。"""
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if torch.is_tensor(value):
        tensors.append(value)
        return
    if isinstance(value, Mapping):
        # 只收集 value：key 参与哈希/相等判断，复制 key 会改变容器语义
        for item in value.values():
            _collect_tensors(item, seen, tensors)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _collect_tensors(item, seen, tensors)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect_tensors(getattr(value, field.name), seen, tensors)
        return
    if hasattr(value, "__dict__"):
        _collect_tensors(vars(value), seen, tensors)
    for cls in type(value).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if slot not in ("__dict__", "__weakref__") and hasattr(value, slot):
                _collect_tensors(getattr(value, slot), seen, tensors)


def _clone_result(result):
    """复制输出图，保留 Tensor 别名、容器类型和循环引用。

    与 ``deepcopy`` 的差异：``Mapping`` 的 key 保持原对象（key 参与哈希/相等比较，替换会改变容器语义），
    只复制 value。
    """
    try:
        tensors = []
        _collect_tensors(result, set(), tensors)
        cloned_tensors = deepcopy(tuple(tensor.detach() for tensor in tensors))
        tensor_memo = {id(tensor): cloned for tensor, cloned in zip(tensors, cloned_tensors)}
        return _deepcopy_preserving_keys(result, tensor_memo)
    except Exception as exc:
        raise ModelExecError(
            f"[CacheAgent] cannot safely copy cached output of type {type(result).__name__} "
            f"for forward hooks ({type(exc).__name__}: {exc})."
        ) from exc


def _deepcopy_preserving_keys(value, memo):
    """递归复制输出图，Mapping 的 key 不复制，保持循环引用与别名。

    只在纯 ``dict`` 上做特例（必须精确类型比较，isinstance 会误纳 Mapping 子类）；
    其余类型（tuple / namedtuple / dataclass / Tensor / 自定义容器）交给 ``deepcopy``。
    """
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if type(value) is dict:  # pylint: disable=unidiomatic-typecheck
        cloned = {}
        memo[value_id] = cloned
        for key, item in value.items():
            cloned[key] = _deepcopy_preserving_keys(item, memo)
        return cloned
    cloned = deepcopy(value, memo)
    memo[value_id] = cloned
    return cloned


class CacheBase(ABC):
    def __init__(self, config: CacheConfig):
        super().__init__()
        self._config = config
        self._cur_step = 0
        self._cur_block = 0

    def apply(self, func: callable, *args, **kwargs):
        res = self.apply_imp(func, *args, **kwargs)
        self._counter()  # 内部计数
        return res

    @abstractmethod
    def apply_imp(self, func: callable, *args, **kwargs):
        pass

    def _call_with_hooks(self, func, cached_result, args, kwargs, protect_result=False):
        """命中缓存时，仍触发 func（若为 nn.Module）上注册的 forward hooks。

        Cache 命中时不再调用真实 forward，这里以缓存结果模拟 forward 的输出，并交由
        PyTorch 原生 ``Module.__call__`` 触发 hooks，保证调用语义与未命中路径一致。

        Args:
            func: 被 cache 包装的函数或 nn.Module。
            cached_result: 命中缓存时用于模拟 forward 输出的结果（应为未经
                forward hooks 处理的原始值，避免 hook 被重复应用）。
            args: 模拟 forward 调用的位置参数。
            kwargs: 模拟 forward 调用的关键字参数。
            protect_result: cached_result 是否与缓存共享同一对象。为 True 时（命中路径
                直接复用缓存张量）会先复制再交给 forward hook，避免原地修改型 hook
                污染缓存；调用方传入新张量时可保持 False 省去复制。

        Returns:
            经 hooks 处理后的最终结果。

        Note:
            使用自定义 ``__call__`` / ``_call_impl`` 或 ``_compiled_call_impl`` 的模块无法
            在不绕过自定义调用链的前提下用缓存结果触发 hooks，这里直接执行原生
            ``func(...)``。
        """
        if not isinstance(func, torch.nn.Module):
            return _clone_result(cached_result) if protect_result else cached_result
        if _requires_native_call(func):
            return func(*args, **kwargs)
        if _uses_custom_call_chain(func):
            return func(*args, **kwargs)
        if not _has_forward_hooks(func):
            return _clone_result(cached_result) if protect_result else cached_result
        if protect_result:
            cached_result = _clone_result(cached_result)
        _, result = _invoke_module(func, args, kwargs, cached_result=cached_result)
        return result

    def _compute_with_hooks(self, func, args, kwargs):
        """计算原始 forward 输出，并恰好应用一次 hooks。

        返回 (raw, result, cacheable)：raw 为未经 forward hooks 处理的原始 forward
        输出（用于缓存，避免命中时 hook 被重复应用），result 为经 forward hooks
        处理后的最终输出。与 ``_call_with_hooks`` 配合，保证可缓存路径中每个 hook
        恰好触发一次。

        Args:
            func: 被 cache 包装的函数或 nn.Module。
            args: forward 调用的位置参数。
            kwargs: forward 调用的关键字参数。

        Returns:
            (原始 forward 输出, 经 forward hooks 处理后的最终输出, 是否可缓存)。

        Note:
            使用自定义 ``__call__`` / ``_call_impl`` 或 ``_compiled_call_impl`` 的
            模块直接走正常的 ``func(...)`` 调用（保留模块自带的调用级逻辑），并返回
            不可缓存标记。存在 forward hook 时，交给 hook 的是原始输出的副本，避免
            原地修改型 hook 污染缓存。

            注册 backward hook、``always_call`` hook 或处于 tracing 状态时，保留完整
            原生调用链并返回不可缓存标记，避免临时替换 forward 改变 PyTorch 语义。
        """
        if not isinstance(func, torch.nn.Module):
            output = func(*args, **kwargs)
            return output, output, True
        if _requires_native_call(func):
            output = func(*args, **kwargs)
            return output, output, False
        if _uses_custom_call_chain(func):
            output = func(*args, **kwargs)
            return output, output, False
        if not _has_forward_hooks(func):
            output = func(*args, **kwargs)
            return output, output, True
        # post-hook 可能原地修改输出：交给 hooks 副本，保证捕获的 raw 干净入缓存
        has_forward_hooks = bool(func._forward_hooks or _global_forward_hooks)
        raw, result = _invoke_module(func, args, kwargs, protect_forward_output=has_forward_hooks)
        return raw, result, True

    @staticmethod
    def _requires_native_call(func):
        """返回 nn.Module 是否需要完整原生调用链。"""
        return isinstance(func, torch.nn.Module) and _requires_native_call(func)

    @staticmethod
    def _clone_result(result):
        """复制缓存输出，并保留容器、别名和循环引用。"""
        return _clone_result(result)

    @abstractmethod
    def _release(self):
        pass

    def _counter(self):
        self._cur_block += 1  # 内部计数+1
        if self._cur_block == self._config.blocks_count:
            self._cur_step += 1  # 满足blocks_count时，step+1
            self._cur_block = 0
            if self._cur_step == self._config.steps_count:
                self._cur_step = 0  # 满足到step_count的时候清0，并清空内存
                self._release()
