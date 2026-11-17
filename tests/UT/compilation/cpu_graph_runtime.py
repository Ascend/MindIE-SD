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


"""CPU capture/replay double for the hardware graph boundary only."""

import contextlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class CapturedGraph(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = []
        self.replay_count = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = function(*args, **kwargs)
        self.operations.append((function, args, kwargs, result))
        return result

    def replay(self):
        self.replay_count += 1
        with torch.no_grad():
            for function, args, kwargs, original in self.operations:
                updated = function(*args, **kwargs)
                old_leaves, _ = tree_flatten(original)
                new_leaves, _ = tree_flatten(updated)
                for old, new in zip(old_leaves, new_leaves):
                    if isinstance(old, torch.Tensor) and old.data_ptr() != new.data_ptr():
                        old.copy_(new)


@contextlib.contextmanager
def capture(npu_graph, pool):
    with npu_graph:
        yield
