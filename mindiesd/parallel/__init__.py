# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Distributed sparse attention with caller-owned process state."""

# pylint: disable=cyclic-import  # public package exports the entry that imports private pipeline modules

from .distributed_sparse_attention import distributed_sparse_attention
from .qkv_layout import QKVLayoutPlan, plan_qkv_layout
from .state import ParallelState

__all__ = [
    "distributed_sparse_attention",
    "ParallelState",
    "QKVLayoutPlan",
    "plan_qkv_layout",
]

__version__ = "0.2.0"
