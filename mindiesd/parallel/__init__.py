# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Distributed attention with caller-owned state and lazy compatibility imports."""

# pylint: disable=cyclic-import  # public package exports entries that import private pipeline modules

from .distributed_sparse_attention import distributed_sparse_attention
from .state import ParallelState

__all__ = [
    "distributed_sparse_attention",
    "ParallelState",
]
__version__ = "0.2.0"


def __getattr__(name):
    # Preserve existing direct imports without expanding the public API.
    modules = {
        "QKVLayoutPlan": "qkv_layout",
        "plan_qkv_layout": "qkv_layout",
        "create_ulysses_kv_gather_groups": "hybrid_sparse_attention",
        "create_sparse_parallel_groups": "hybrid_sparse_attention",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{modules[name]}"), name)
