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

import threading

once_flag = threading.Event()
_once_lock = threading.Lock()


def activate_pattern_once():

    def activate_pattern():
        import importlib

        from ..compiliation_config import CompilationConfig
        from .register_pattern_to_pass import register_pattern_to_pass

        pattern_registry = {
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

    if not once_flag.is_set():
        with _once_lock:
            if not once_flag.is_set():
                activate_pattern()
                once_flag.set()
