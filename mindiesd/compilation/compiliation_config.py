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
import dataclasses


@dataclasses.dataclass(frozen=False)
class FusionPatterns:
    enable_rms_norm: bool = True
    enable_rope: bool = True
    enable_adalayernorm: bool = True
    enable_fast_gelu: bool = True
    enable_mul_add: bool = True
    enable_wan_adalayernorm: bool = True
    # 残差+gate 融合: `x + y*gate` pattern, 注册顺序在 adaLN/rope 之后避免误匹配
    enable_wan_residual_gate: bool = True
    enable_wan_rope: bool = True
    # qk_norm(RMSNorm)融合: npu_rms_norm 在 eager GraphModule 下不会被 torch_npu
    # decomp 表分解; eps 常量须 float32 舍入(9.999999974752427e-07)才能命中
    enable_wan_rmsnorm: bool = True
    # MiniMax-H3 RMSNorm 融合(register_replacement,见 patterns/minimax_h3_rmsnorm_pattern.py):
    # torch 2.11 下 rms_norm 在 freeze 前已被 Dynamo 分解为 pow/mean/rsqrt 链,before_freezing
    # 的 pattern matcher 一次命中, 替换为 torch_npu.npu_rms_norm 单 kernel。覆盖 DiT block 内
    # norm1/norm2、qk_norm(norm_q/norm_k)、token_refiner 与 norm_out。
    enable_minimax_h3_rmsnorm: bool = True
    # MiniMax-H3 RoPE 融合(register_replacement,见 patterns/minimax_h3_rope_pattern.py):
    # 匹配 rotate_half 部分旋转链(slice/split/neg/cat/mul×2/add) → npu_rotary_mul。
    # 需在 wan_residual_gate 之前注册, 防止其误匹配 rope 子图(F2 教训)。
    enable_minimax_h3_rope: bool = True


class CompilationConfig:
    enable_freezing: bool = True
    graph_log_url: str | None = None
    fusion_patterns: FusionPatterns = FusionPatterns()
    aclgraph_only: bool = False
    aclgraph_with_compile: bool = False
    safe_output_mode: bool = True
    aclgraph_lazy_capture: bool = False
    aclgraph_max_entries: int = 0
    # 计算精度: 不区分精度类型,编译侧不做任何隐式精度转换。bf16/fp32 由模型层面
    # 决定(dummy run 的 --compute-precision: 权重 cast + 源码级分支 .float())。

    def __init__(self):
        raise RuntimeError("CompilationConfig is a static class, do not instantiate it")
