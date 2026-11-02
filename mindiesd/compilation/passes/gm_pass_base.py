#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import hashlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

import torch


class GraphModulePass(ABC):
    """Use the same interface as Inductor's CustomGraphPass"""

    @abstractmethod
    def __call__(self, graph: torch.fx.GraphModule) -> None:
        """
        Implementation of the custom pass.
        """

    def uuid(self) -> Any:
        """
        Provide a unique identifier for the pass, used for code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        hasher = hashlib.sha256()
        src = inspect.getsource(self.__class__)
        hasher.update(src.encode("utf-8"))
        return hasher.hexdigest()


def target_op_name(target) -> str:
    """把 fx ``node.target`` 归一化为裸算子名（不含命名空间与重载后缀）。

    真实 fx 图里 ``node.target`` 是 ``OpOverload``，其 ``__name__`` 带重载后缀
    （如 ``"_to_copy.default"``），部分 torch 版本还带命名空间前缀
    （如 ``"aten::_to_copy.default"``）。直接与 ``"_to_copy"`` 这类裸名比较会恒不
    命中，导致 pass 静默空转。这里统一去掉 ``::`` 命名空间与 ``.overload`` 后缀，
    同时保留鸭子类型节点（测试中的假算子）的裸名语义。
    """
    name = getattr(target, "__name__", None)
    if not name:
        text = str(target)
        # OpOverload 无 __name__ 时 str() 形如 "torch.ops.aten._to_copy.default"，
        # 取倒数第二段（op 名）而不是最后的重载名。
        parts = text.split(".")
        name = parts[-2] if len(parts) >= 2 and "ops" in parts else text
    name = name.rsplit("::", 1)[-1]
    return name.split(".", 1)[0]
