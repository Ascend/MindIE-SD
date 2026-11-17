#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2 that you can use this software in accordance
# with the terms and conditions of the Mulan PSL v2. You may obtain a copy of the
# Mulan PSL at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Rope 前后 fp32 cast 消除 pass（候选 P2，默认关闭）。

Wan2.2 的 rope 链（fx dump 帧2）为::

    _to_copy(x, dtype=fp32) → npu_rotary_mul(x_f32, cos, sin, 'interleave')
                            → _to_copy(→ bf16)

x 是 [1, S, H, D] 主激活张量，两次大 cast 全为满足 npu_rotary_mul 的 fp32 入参假设。
NPU 实测 ``npu_rotary_mul`` 直接支持 bf16 全参输入（bf16 入 bf16 出），故把 x 侧
两头 cast 删掉、cos/sin（[S, D] 小张量）改为一次性 bf16 化并按图去重，主激活上的
两次全张量 cast 变为 rope 算子内部计算精度差异。

数值注意：原链在 fp32 域做旋转再舍入 bf16，新链全程 bf16，存在可测差异（这正是
收益来源），开启前需 NPU 质量回归。保守约束：仅当 rope 的全部用户都是纯 dtype
（无 layout/memory_format 附加语义）的 bf16 cast 时才改写，否则跳过该节点。
"""

from ...utils.logs.logging import logger
from .gm_pass_base import GraphModulePass, target_op_name

_COPY_NAMES = ("_to_copy", "npu_dtype_cast", "_npu_dtype_cast")
_ROPE_NAME = "npu_rotary_mul"


def _dtype_token(dtype) -> str:
    """torch.float32 / 'float32' / 裸枚举统一成 ``float32`` 这类 token（不依赖 torch 对象）。"""
    return str(dtype).rsplit(".", 1)[-1].lower()


def _cast_dtype(node):
    """取 cast 节点目标 dtype：关键字优先（aten._to_copy），其次位置参数 2（npu_dtype_cast）。"""
    if "dtype" in node.kwargs:
        return node.kwargs["dtype"]
    if len(node.args) >= 2:
        return node.args[1]
    return None


def _is_pure_dtype_copy(node, dtype_token: str) -> bool:
    """target 是 cast 形态、目标 dtype 匹配、且未携带 layout/memory_format 等附加语义。"""
    if getattr(node, "op", None) != "call_function" or target_op_name(node.target) not in _COPY_NAMES:
        return False
    if not set(node.kwargs) <= {"dtype"}:
        return False
    dtype = _cast_dtype(node)
    return dtype is not None and _dtype_token(dtype) == dtype_token


def _meta_dtype(node):
    val = getattr(node, "meta", None)
    if not isinstance(val, dict):
        return None
    return getattr(val.get("val"), "dtype", None)


def find_rope_cast_chains(nodes) -> list:
    """找 ``_to_copy(→fp32) → npu_rotary_mul → _to_copy(→bf16)…`` 链（纯函数，可脱离 torch 测试）。

    Args:
        nodes: 任意可迭代的 fx.Node（真实图传 ``list(graph.nodes)``；单元测试可传鸭子类型）。

    Returns:
        list[tuple]: ``(rope_node, cast_in, cast_outs)``。匹配条件：rope 第 1 参的生产者
        是纯 fp32 cast；rope 的全部用户都是纯 bf16 cast（至少 1 个）。x 侧源 dtype 经
        ``node.meta`` 可得时必须为 bf16（fp16 等其他源不与其后 bf16 cast 构成无损往返，
        跳过）；meta 不可得时不设防（真实 aot/dynamo 图均有 meta）。
    """
    chains = []
    for node in nodes:
        if getattr(node, "op", None) != "call_function" or target_op_name(node.target) != _ROPE_NAME:
            continue
        if not node.args:
            continue
        cast_in = node.args[0]
        if not _is_pure_dtype_copy(cast_in, "float32"):
            continue
        src = cast_in.args[0] if cast_in.args else None
        src_dtype = _meta_dtype(src)
        if src_dtype is not None and _dtype_token(src_dtype) != "bfloat16":
            continue
        cast_outs = []
        matched = True
        for user in node.users:
            if _is_pure_dtype_copy(user, "bfloat16"):
                cast_outs.append(user)
            else:
                matched = False
                break
        if not matched or not cast_outs:
            continue
        chains.append((node, cast_in, cast_outs))
    return chains


class RopeCastFusionPass(GraphModulePass):
    """消除 npu_rotary_mul 两头主激活 cast、cos/sin 小张量 bf16 化。

    开关：FusionPatterns.enable_rope_cast_fusion（默认关闭）。
    """

    def __call__(self, gm) -> None:
        try:
            import torch

            copy_op = torch.ops.aten._to_copy.default
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "[MindIE-SD/compilation] RopeCastFusionPass skipped: torch.ops.aten._to_copy "
                "unavailable on this environment. error=%s",
                exc,
            )
            return

        graph = gm.graph
        chains = find_rope_cast_chains(list(graph.nodes))
        if not chains:
            return

        fused = 0
        cos_cast_cache = {}
        for rope, cast_in, cast_outs in chains:
            # x 侧：bf16 直入（npu_rotary_mul bf16 全参已 NPU 实测支持）。
            x_bf16 = cast_in.args[0] if cast_in.args else cast_in.kwargs.get("self")
            with graph.inserting_before(rope):
                new_args = [x_bf16]
                for extra in rope.args[1:]:
                    new_args.append(self._bf16_arg(graph, extra, copy_op, torch.bfloat16, cos_cast_cache))
                new_rope = graph.call_function(rope.target, args=tuple(new_args), kwargs=dict(rope.kwargs))
            for out in cast_outs:
                out.replace_all_uses_with(new_rope)
            fused += 1

        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info(
            "[MindIE-SD/compilation] RopeCastFusionPass fused %d rope cast chains (cos/sin casts=%d).",
            fused,
            len(cos_cast_cache),
        )

    @staticmethod
    def _bf16_arg(graph, arg, copy_op, bf16_dtype, cache: dict):
        """rope 余参 bf16 化：张量节点且 meta 非 bf16 时插入去重的小 cast，非节点参数原样透传。"""
        if not hasattr(arg, "op"):
            return arg  # 'interleave' 等字符串/常量参数
        dtype = _meta_dtype(arg)
        if dtype is not None and _dtype_token(dtype) == "bfloat16":
            return arg
        if arg in cache:
            return cache[arg]
        # 必须插在 arg 之后：新 cast 以 arg 为输入，插到 arg 之前会破坏拓扑序，
        # 末尾 eliminate_dead_code() 内部的 lint() 会抛 "used before it has been defined"。
        with graph.inserting_after(arg):
            cast = graph.call_function(copy_op, args=(arg,), kwargs={"dtype": bf16_dtype})
        cache[arg] = cast
        return cast
