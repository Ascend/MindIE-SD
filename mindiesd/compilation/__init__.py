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

__all__ = [
    "MindieSDBackend",
    "CompilationConfig"
]


def __getattr__(name):
    if name == "MindieSDBackend":
        from .mindie_sd_backend import MindieSDBackend as _cls
        return _cls
    if name == "CompilationConfig":
        from .compiliation_config import CompilationConfig as _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
