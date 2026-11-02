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

import re
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass

from ...utils.logs.logging import logger
from .._custom_decomposition import select_custom_decomp_table
from .gm_pass_base import GraphModulePass

DEBUG_LOG_LEVEL = 10

torch_version = re.match(r"(\d+\.\d+)", torch.__version__).group(1)
IS_TORCH_21 = torch_version == "2.1"
if IS_TORCH_21:

    def mindie_inference_graph(fn, args):
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx.experimental.proxy_tensor import make_fx

        decomp_table = select_custom_decomp_table()

        def safe_to_copy(x, dtype=None, layout=None, device=None, pin_memory=False, non_blocking=False):
            if isinstance(x, FakeTensor):
                return x
            return torch.ops.aten._to_copy.default(x, dtype, layout, device, pin_memory, non_blocking)

        decomp_table[torch.ops.aten._to_copy.default] = safe_to_copy
        gm = make_fx(fn, decomposition_table=decomp_table)(*args)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm


class PatternBackend(str, Enum):
    """算子融合模板注册的目标后端。

    收敛 Inductor 与 npugraph_ex 两套 register_replacement 入口：INDUCTOR 走
    ``torch._inductor.pattern_matcher``（default/aclgraph 路径，aot 之后生效）；
    NPUGRAPH_EX 走 ``torch.npu.npugraph_ex.register_replacement``（npugraph_ex
    内部正确阶段生效）。
    """

    INDUCTOR = "inductor"
    NPUGRAPH_EX = "npugraph_ex"


class PatternMatchPass(GraphModulePass):
    def __init__(self):
        self.pattern_replacements: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {}
        try:
            self.pattern_pass: PatternMatcherPass = PatternMatcherPass(pass_name="pattern_match_pass")  # nosec B106
        except TypeError:
            self.pattern_pass: PatternMatcherPass = PatternMatcherPass()

    def __call__(self, graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
        matched_cnt = 0
        while True:
            cnt = self.pattern_pass.apply(graph)
            if cnt == 0:
                break
            matched_cnt += cnt
        if logger.isEnabledFor(DEBUG_LOG_LEVEL):
            logger.debug("PatternMatchPass replace %d patterns.", matched_cnt)
            pattern_idx = 0
            logger.debug("Patterns registered for replacement:")
            try:
                from torch._inductor.pattern_matcher import PatternPrettyPrinter

                for pattern_entry in self.pattern_pass.patterns.values():
                    for p in pattern_entry:
                        p_str = PatternPrettyPrinter.run(p.pattern)
                        logger.debug("Pattern %d: %s", pattern_idx, p_str)
                        pattern_idx += 1
            except ImportError:
                logger.debug("PatternPrettyPrinter not available, skipping pattern printing")
        return graph

    def register_pattern(
        self,
        name: str,
        pattern: Callable[..., Any],
        replacement: Callable[..., Any],
        example_inputs: list[torch.Tensor],
        extra_check: Callable[[pm.Match], bool] | None = None,
        backend: "PatternBackend | str" = PatternBackend.INDUCTOR,
    ):
        if name in self.pattern_replacements:
            logger.error(
                "[MindIE-SD/compilation] Pattern registration failed. "
                "issue=pattern name already registered, pattern_name=%s, expected=unique pattern name. "
                "possible_cause=activate_pattern_once or custom registration was called repeatedly with the same name. "
                "Troubleshooting: check pattern registration order and avoid duplicate names.",
                name,
            )
            raise ValueError(f"Pattern '{name}' is already registered.")

        self.pattern_replacements[name] = (pattern, replacement)
        logger.debug("Registering pattern: %s (backend=%s)", name, backend)

        if isinstance(backend, PatternBackend):
            resolved = backend
        else:
            try:
                resolved = PatternBackend(str(backend))
            except ValueError:
                logger.warning(
                    "[MindIE-SD/compilation] Unknown pattern backend=%s, fall back to inductor. pattern_name=%s.",
                    backend,
                    name,
                )
                resolved = PatternBackend.INDUCTOR

        if resolved == PatternBackend.NPUGRAPH_EX:
            self._register_npugraph_ex_replacement(name, pattern, replacement, example_inputs, extra_check)
        else:
            self._register_inductor_replacement(name, pattern, replacement, example_inputs, extra_check)

    def _register_inductor_replacement(
        self,
        name: str,
        pattern: Callable[..., Any],
        replacement: Callable[..., Any],
        example_inputs: list[torch.Tensor],
        extra_check: Callable[[pm.Match], bool] | None = None,
    ):
        # Inductor 路径：通过 torch._inductor.pattern_matcher.register_replacement 注册，
        # 在 default/aclgraph 路径的 graph_rewrite_before_freezing（aot 之后）生效。
        if not hasattr(pm, "fwd_only"):
            if IS_TORCH_21:
                pm.fwd_only = mindie_inference_graph
            else:
                logger.warning(
                    "[MindIE-SD/compilation] Pattern replacement preparation failed. "
                    "issue=torch._inductor.pattern_matcher.fwd_only is unavailable, torch_version=%s, "
                    "expected=fwd_only API exists or torch version is 2.1 for compatibility patch. "
                    "possible_cause=current torch version does not provide the expected inductor API. "
                    "Troubleshooting: verify torch version compatibility and pattern registration stack.",
                    torch.__version__,
                )

        def fwd_only_with_custom_decomp(
            fn: Callable[..., Any],
            args: Sequence[Any],
            *,
            run_functional_passes: bool = True,
            get_decomp_fn: Callable[..., Any] | None = select_custom_decomp_table,
        ) -> torch.fx.GraphModule:
            if IS_TORCH_21:
                return pm.fwd_only(fn=fn, args=args)
            else:
                return pm.fwd_only(
                    fn=fn, args=args, run_functional_passes=run_functional_passes, get_decomp_fn=get_decomp_fn
                )

        try:
            register_kwargs = {}
            if extra_check is not None:
                register_kwargs["extra_check"] = extra_check
            pm.register_replacement(
                pattern,
                replacement,
                example_inputs,
                fwd_only_with_custom_decomp,
                self.pattern_pass.patterns,
                **register_kwargs,
            )
            logger.debug("Successfully register pattern: %s", name)
        except RuntimeError as e:
            if "Duplicate pattern" in str(e):
                logger.debug(
                    "[MindIE-SD/compilation] Duplicate pattern registration skipped. "
                    "pattern_name=%s, possible_cause=the same pattern was activated more than once.",
                    name,
                )
            else:
                logger.error(
                    "[MindIE-SD/compilation] Pattern registration failed. "
                    "issue=torch inductor register_replacement raised RuntimeError, pattern_name=%s, "
                    "actual_error=%s. possible_cause=pattern, replacement, or example_inputs are incompatible. "
                    "Troubleshooting: inspect the pattern definition, replacement function schema, example input "
                    "shape/dtype, and torch._inductor stack.",
                    name,
                    e,
                )
                raise e

    def _register_npugraph_ex_replacement(
        self,
        name: str,
        pattern: Callable[..., Any],
        replacement: Callable[..., Any],
        example_inputs: list[torch.Tensor],
        extra_check: Callable[[pm.Match], bool] | None = None,
    ):
        # npugraph_ex 路径：借其原生 register_replacement，在 npugraph_ex 内部正确阶段
        # （aot 之后）生效；不在 compile_fx 之前手工改写，避免原始 gm 单节点失配。
        # 该 API 签名随 torch_npu 版本变化，失败时仅告警并跳过，不阻断注册流程。
        try:
            npugraph_ex = getattr(getattr(torch, "npu", None), "npugraph_ex", None)
            register = getattr(npugraph_ex, "register_replacement", None)
            if register is None:
                logger.warning(
                    "[MindIE-SD/compilation] npugraph_ex pattern registration skipped. "
                    "issue=torch.npu.npugraph_ex.register_replacement not found, pattern_name=%s. "
                    "possible_cause=torch_npu version does not provide npugraph_ex.register_replacement. "
                    "Troubleshooting: install a compatible torch_npu version.",
                    name,
                )
                return
            register_kwargs = {}
            if extra_check is not None:
                register_kwargs["extra_check"] = extra_check
            register(pattern, replacement, example_inputs, **register_kwargs)
            logger.debug("Successfully register npugraph_ex pattern: %s", name)
        except Exception as e:  # noqa: BLE001
            # 注册失败不应阻断整个编译：npugraph_ex 的 register_replacement 在各版本
            # 可能抛 ValueError/AttributeError/KeyError 等（不限于 ImportError/
            # RuntimeError/TypeError），只捕获固定几类会让异常冒泡出 torch.compile。
            logger.warning(
                "[MindIE-SD/compilation] npugraph_ex pattern registration failed, skipped. "
                "pattern_name=%s, actual_error=%s. "
                "possible_cause=pattern/replacement/example_inputs incompatible with npugraph_ex register_replacement. "
                "Troubleshooting: inspect the npugraph_ex register_replacement signature and pattern definition.",
                name,
                e,
            )
