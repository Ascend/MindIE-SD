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
import dataclasses
from enum import Enum

from ..utils.logs.logging import logger


class BackendMode(str, Enum):
    """编译后端模式枚举，作为后端选择的单点入口。

    ``CompilationConfig.backend_mode`` 为空时由
    :meth:`CompilationConfig.resolved_backend_mode` 按旧布尔字段推导，保持向后兼容。
    """

    DEFAULT = "default"
    ACLGRAPH_ONLY = "aclgraph_only"
    ACLGRAPH_WITH_COMPILE = "aclgraph_with_compile"
    NPUGRAPH_EX = "npugraph_ex"

    @classmethod
    def coerce(cls, value) -> "BackendMode | None":
        """将字符串或 BackendMode 归一化为枚举；无法识别或为空时返回 None。

        同时支持枚举名（如 ``"NPUGRAPH_EX"``）与枚举值（如 ``"npugraph_ex"``）。
        """
        if value is None:
            return None
        if isinstance(value, BackendMode):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            for member in cls:
                if normalized in (member.name, member.value):
                    return member
            logger.warning(
                "[MindIE-SD/compilation] unrecognized backend_mode=%r, ignored; falling back to "
                "legacy boolean fields. valid values: %s (enum names are also accepted).",
                value,
                ", ".join(repr(member.value) for member in cls),
            )
        else:
            logger.warning(
                "[MindIE-SD/compilation] backend_mode must be a str or BackendMode, got %s; ignored.",
                type(value).__name__,
            )
        return None


@dataclasses.dataclass(frozen=False)
class FusionPatterns:
    enable_add_layer_norm: bool = True
    enable_rms_norm: bool = True
    enable_rope: bool = True
    enable_adalayernorm: bool = True
    enable_fast_gelu: bool = True
    enable_mul_add: bool = True
    enable_wan_adalayernorm: bool = True
    # FLUX/Qwen norm_out 融合: (1+scale)[:,None] 调制形态(norm1/norm2 的 modulation 不同)
    enable_norm_out_adaln: bool = True
    # Wan residual + gate fusion.  Model-specific setup disables it for Qwen.
    enable_wan_residual_gate: bool = True
    enable_wan_rope: bool = True
    # qk_norm(RMSNorm)融合: npu_rms_norm 在 eager GraphModule 下不会被 torch_npu
    # decomp 表分解; eps 常量须 float32 舍入(9.999999974752427e-07)才能命中
    enable_wan_rmsnorm: bool = True
    # MiniMax-H3 AdaLN 调制: x*(1+scale)+shift(index_select 表行) -> triton gather_scale_shift(表 [3,D] L2 驻留)
    enable_minimax_h3_adaln: bool = True
    # MiniMax-H3 SwiGLU: split->silu->mul -> triton swiglu(免 cat)
    enable_minimax_h3_swiglu: bool = True
    # MiniMax-H3 残差 gate: hidden + gate_table[idx]*attn/ff -> triton gather_residual_gate(先于 residual_gate 注册)
    enable_minimax_h3_gate: bool = True
    # MiniMax-H3 RMSNorm 融合(register_replacement,见 patterns/minimax_h3_rmsnorm_pattern.py):
    # torch 2.11 下 rms_norm 在 freeze 前已被 Dynamo 分解为 pow/mean/rsqrt 链,before_freezing
    # 的 pattern matcher 一次命中, 替换为 torch_npu.npu_rms_norm 单 kernel。覆盖 DiT block 内
    # norm1/norm2、qk_norm(norm_q/norm_k)、token_refiner 与 norm_out。
    enable_minimax_h3_rmsnorm: bool = True
    # MiniMax-H3 RoPE 融合(register_replacement,见 patterns/minimax_h3_rope_pattern.py):
    # 匹配 rotate_half 部分旋转链(slice/split/neg/cat/mul×2/add) → npu_rotary_mul。
    # 需在 wan_residual_gate 之前注册, 防止其误匹配 rope 子图(F2 教训)。
    enable_minimax_h3_rope: bool = True
    # Qwen-Image RoPE 融合(register_replacement,见 patterns/qwen_rope_pattern.py):
    # 实数域等价 complex rotary → npu_rotary_mul; 需先于 wan_residual_gate 注册(F2 同源)
    enable_qwen_rope: bool = True
    # Qwen residual + gate fusion with [B, 1, D] token-axis broadcasting.
    enable_qwen_residual_gate: bool = True
    # gelu→npu_dynamic_mx_quant 融合为 npu_gelu_quant（P4）。近似语义/输出 scale
    # 与 npu_quant_matmul 的兼容性需 NPU 验证（tests/compilation/verify_gelu_quant_and_rope.py）。
    enable_gelu_quant_fusion: bool = False
    # npu_rotary_mul 两头 fp32 cast 消除（P2）。bf16 全参已 NPU 实测可用；全程 bf16
    # 与 fp32 计算后舍入存在数值差异，开启前需质量回归。
    enable_rope_cast_fusion: bool = False
    # P4 链头为 npu_fast_gelu（无近似参数可透传）时的近似模式；aten.gelu 链头透传自身。
    gelu_quant_approximate: str = "tanh"


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
    # 决定(dummy run 的 --quant: 权重 cast + 源码级分支 .float())。

    # 后端单点路由（阶段一新增）。为 None 时由 resolved_backend_mode 按旧布尔推导，
    # 与原 mindie_sd_backend 的 if 判断顺序保持一致，保证向后兼容。
    backend_mode: "BackendMode | str | None" = None
    # 透传给 npugraph_ex 编译入口的选项字典。
    npugraph_ex_options: dict | None = None
    # 分段捕获断点（阶段二新增）。节点名列表，为空/None 时整图单段捕获。
    aclgraph_breakpoints: list | None = None
    # 图捕获路径（aclgraph_only）上是否对原始 gm 做轻量算子展开，使既有融合模板命中。
    aclgraph_enable_patterns: bool = False

    def __init__(self):
        raise RuntimeError("CompilationConfig is a static class, do not instantiate it")

    @classmethod
    def resolved_backend_mode(cls) -> "BackendMode":
        """返回最终生效的后端。

        优先级：显式 backend_mode > aclgraph_with_compile > aclgraph_only > DEFAULT。
        aclgraph_with_compile 优先于 aclgraph_only，与原 ``mindie_sd_backend`` 的
        if 判断顺序保持一致，避免改变既有行为。
        """
        explicit = BackendMode.coerce(cls.backend_mode)
        if explicit is not None:
            return explicit
        if cls.aclgraph_with_compile:
            return BackendMode.ACLGRAPH_WITH_COMPILE
        if cls.aclgraph_only:
            return BackendMode.ACLGRAPH_ONLY
        return BackendMode.DEFAULT
