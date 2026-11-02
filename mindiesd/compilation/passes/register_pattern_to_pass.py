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

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .pattern_match_pass import PatternMatchPass


class PatternBase(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError("Pass must implement the 'name' method!")

    @staticmethod
    @abstractmethod
    def inputs():
        raise NotImplementedError("Pass must implement the 'pattern' method!")

    @staticmethod
    @abstractmethod
    def pattern(*args, **kwargs):
        raise NotImplementedError("Pass must implement the 'pattern' method!")

    @staticmethod
    @abstractmethod
    def replacement(*args, **kwargs):
        raise NotImplementedError("Pass must implement the 'replacement' method!")

    @staticmethod
    def backend() -> str:
        """声明该模板注册到的后端，默认 Inductor；npugraph_ex 专用模板可覆写为 "npugraph_ex"。

        非 abstract：现有模板无需改动即可沿用默认值。
        """
        return "inductor"


patterns = PatternMatchPass()


def register_pattern_to_pass(
    pattern_or_patterns: PatternBase | type[PatternBase] | Iterable[PatternBase | type[PatternBase]],
) -> None:
    if isinstance(pattern_or_patterns, type) and issubclass(pattern_or_patterns, PatternBase):
        pattern_group = [pattern_or_patterns]
    elif isinstance(pattern_or_patterns, (list, tuple)):
        pattern_group = list(pattern_or_patterns)
    else:
        raise TypeError(f"input type is not supported! current type is：{type(pattern_or_patterns)}")

    for pattern in pattern_group:
        patterns.register_pattern(
            pattern.name(),
            pattern.pattern,
            pattern.replacement,
            pattern.inputs(),
            extra_check=getattr(pattern, "extra_check", None),
            backend=pattern.backend(),
        )
