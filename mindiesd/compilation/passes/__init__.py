#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import threading

from ...utils.logs.logging import logger

once_flag = threading.Event()
_once_lock = threading.Lock()


def activate_pattern_once():
    def activate_pattern():
        import importlib

        from ..compiliation_config import CompilationConfig

        from .register_pattern_to_pass import patterns, register_pattern_to_pass

        pattern_registry = {
            # Always register the shared residual before shorter residual-gate patterns.
            "enable_add_layer_norm": ("AddLayerNormPatternGroup", "..patterns"),
            "enable_rms_norm": ("RMSNormPatternGroup", "..patterns"),
            "enable_rope": ("RopePatternGroup", "..patterns"),
            "enable_adalayernorm": ("AdaLayerNormPatternGroup", "..patterns"),
            "enable_fast_gelu": ("GELUPatternGroup", "..patterns"),
            "enable_mul_add": ("MulAddPatternGroup", "..patterns"),
            "enable_wan_adalayernorm": ("WanAdaLayerNormPatternGroup", "..patterns"),
            # FLUX/Qwen norm_out: (1+scale)[:,None] form, NOT covered by wan_adalayernorm
            "enable_norm_out_adaln": ("NormOutAdaLayerNormPatternGroup", "..patterns"),
            # 残差+gate 需在 adaLN/rope 之后(先融合 modulation 与 4D 链, 防误匹配)
            "enable_wan_rope": ("WanRopePatternGroup", "..patterns"),
            # MiniMax-H3 RoPE 先注册:先吃掉 rope 子图, 防 wan residual_gate 误匹配(F2 教训)
            "enable_minimax_h3_rope": ("MiniMaxH3RopePatternGroup", "..patterns"),
            # Qwen-Image RoPE 同规则: 必须先于 wan_residual_gate 注册, 防止其误匹配
            # rope 的 add(mul(x,cos), mul(x_rot,sin)) 子图 (qwen 4D fallback 实测)
            "enable_qwen_rope": ("QwenRopePatternGroup", "..patterns"),
            # Register after Qwen RoPE to avoid matching RoPE add-mul subgraphs.
            "enable_qwen_residual_gate": ("QwenResidualGatePatternGroup", "..patterns"),
            "enable_wan_residual_gate": ("WanResidualGatePatternGroup", "..patterns"),
            "enable_wan_rmsnorm": ("WanRmsNormPatternGroup", "..patterns"),
            "enable_minimax_h3_gate": ("MiniMaxH3GatePatternGroup", "..patterns"),
            "enable_minimax_h3_adaln": ("MiniMaxH3AdaLnPatternGroup", "..patterns"),
            # MiniMax-H3 SwiGLU: split->silu->mul -> triton swiglu(免 cat)
            "enable_minimax_h3_swiglu": ("MiniMaxH3SwigluPatternGroup", "..patterns"),
            # MiniMax-H3 RMSNorm: torch 2.11 下 rms_norm 在 freeze 前已分解为链, before 即命中
            "enable_minimax_h3_rmsnorm": ("MiniMaxH3RmsNormPatternGroup", "..patterns"),
        }

        fusion_config = CompilationConfig.fusion_patterns
        for config_key, (pattern_group_name, pattern_module) in pattern_registry.items():
            if getattr(fusion_config, config_key, False):
                patterns_module = importlib.import_module(pattern_module, package=__package__)
                pattern_group = getattr(patterns_module, pattern_group_name)
                register_pattern_to_pass(pattern_group)

        patterns_module = importlib.import_module("..patterns", package=__package__)
        patterns_module.GroupNormSiLUPatternGroup.register(patterns.pattern_pass)

    if not once_flag.is_set():
        with _once_lock:
            if not once_flag.is_set():
                activate_pattern()
                once_flag.set()


def apply_lightweight_decomp(gm, example_inputs):
    """跨后端轻量算子展开（阶段三策略 A）。

    在跳过完整编译的路径（如 ``aclgraph_only``）上，用 ``make_fx`` + 自定义 decomp 表把
    原始 Dynamo gm 的单节点算子展开为 aten 级算子链，使按 aten 链编写的融合模板
    （rms_norm/rope 等）在该路径同样可命中。失败时原样返回 gm 并告警，不阻断流程。

    注意：展开后图节点名会变化，若同时配置 ``aclgraph_breakpoints``，断点需基于展开
    后的节点名。需在 NPU 环境验证 make_fx 对 NPU op 的 trace 兼容性。
    """
    try:
        from torch.fx.experimental.proxy_tensor import make_fx

        from .._custom_decomposition import select_custom_decomp_table

        decomp_table = select_custom_decomp_table()
        if not decomp_table:
            # 空分解表时 make_fx 不会展开任何算子，却会用真实张量把整图执行一遍
            # （编译期真实计算，且 keep side effect）。既无收益又可能引入额外开销，
            # 直接返回原图。
            logger.debug(
                "[MindIE-SD/compilation] apply_lightweight_decomp skipped: custom decomposition "
                "table is empty, returning original gm."
            )
            return gm
        new_gm = make_fx(gm, decomposition_table=decomp_table)(*example_inputs)
        new_gm.graph.eliminate_dead_code()
        new_gm.recompile()
        return new_gm
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[MindIE-SD/compilation] apply_lightweight_decomp failed, keep original gm. error=%s",
            e,
        )
        return gm
