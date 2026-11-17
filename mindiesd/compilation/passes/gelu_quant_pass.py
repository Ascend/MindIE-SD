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
"""Gelu→DynamicMXQuant 融合 pass（候选 P4，默认关闭）。

把 FFN 里的 ``gelu(x) → [view/reshape …] → npu_dynamic_mx_quant(dst_type)`` 链
替换为单算子 ``npu_gelu_quant``（NPU 实测 schema：
``npu_gelu_quant(Tensor input, *, str approximate="none", str quant_mode="dynamic",
int? dst_type=None, str round_mode="rint") -> (Tensor, Tensor)``），省掉 gelu 输出
整张中间张量的写回与再读（fc1 输出 [M, N_ffn] 量级，全链一趟遍历变两趟合一）。

匹配按 ``node.target`` 的算子名，不依赖 torch.ops 对象恒等；替换算子惰性解析，
torch_npu 未提供时跳过改写并告警，不影响编译。

正确性要点：
- gelu 逐元素，与 view/reshape 可交换。改写在 gelu 的输入上按原顺序重建同一 view
  序列后送入 npu_gelu_quant，与原链（gelu→view→quant）逐元素一致。
- 链头是 ``aten.gelu`` 时透传其 ``approximate``（none/tanh 数学一致）；链头是
  ``npu_fast_gelu``（fast-gelu 近似，无近似参数）时使用配置值，两者非严格同一
  近似，需 NPU 质量回归后再默认开启。
- ``npu_gelu_quant`` 与 ``npu_dynamic_mx_quant`` 同为 (payload, scale) 二元组输出，
  下游 ``operator.getitem`` 消费方式不变；但两者的 scale 语义并不自动等价：
  ``npu_dynamic_mx_quant`` 是 MX 分块 e8m0 scale，``npu_gelu_quant`` 为 per-row fp32
  scale。融合后是否与 ``npu_quant_matmul(pertoken_scale=...)`` 兼容，必须经
  ``verify_gelu_quant_and_rope.py`` 在 NPU 上做数值回归后才能开启默认开关。
- ``dst_type`` 无法从链上确定时直接跳过该链（不猜 int8 默认）；链上若携带
  ``axis``/``scale_alg``/``dst_type_max`` 等 ``npu_gelu_quant`` 无法表达的参数，
  同样跳过，避免静默改变量化语义。
"""

from ...utils.logs.logging import logger
from .gm_pass_base import GraphModulePass, target_op_name

_GELU_HEAD_NAMES = ("gelu", "npu_gelu", "npu_fast_gelu")
_VIEW_NAMES = ("view", "reshape", "_unsafe_view")
_QUANT_NAME = "npu_dynamic_mx_quant"
_MAX_VIEW_WALK = 4


def _is_gelu_head(node) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    return target_op_name(node.target) in _GELU_HEAD_NAMES


def find_gelu_quant_chains(nodes) -> list:
    """在节点序列里找 ``gelu → [view…] → npu_dynamic_mx_quant`` 链（纯函数，可脱离 torch 测试）。

    Args:
        nodes: 任意可迭代的 fx.Node（真实图传 ``list(graph.nodes)``；单元测试可传鸭子类型）。

    Returns:
        list[tuple]: ``(quant_node, gelu_node, view_nodes, dst_type, approximate)``；
        ``view_nodes`` 按图内顺序（靠近 gelu 者在前）；``dst_type`` 从 quant 节点的
        ``dst_type`` 关键字透传（实际 dump 中均为关键字形态）；``approximate`` 为
        ``aten.gelu`` 链头的近似参数，``npu_fast_gelu`` 链头为 None（由调用方配置）。
    """
    chains = []
    for node in nodes:
        if getattr(node, "op", None) != "call_function":
            continue
        if target_op_name(node.target) != _QUANT_NAME:
            continue
        if not node.args:
            continue
        src = node.args[0]
        views = []
        for _ in range(_MAX_VIEW_WALK):
            if (
                src is not None
                and getattr(src, "op", None) == "call_function"
                and target_op_name(src.target) in _VIEW_NAMES
            ):
                views.append(src)
                src = src.args[0] if src.args else None
            else:
                break
        if src is None or not _is_gelu_head(src):
            continue
        approximate = None
        if target_op_name(src.target) == "gelu":
            approximate = src.kwargs.get("approximate", "none")
        chains.append((node, src, tuple(reversed(views)), node.kwargs.get("dst_type"), approximate))
    return chains


class GeluQuantFusionPass(GraphModulePass):
    """把 gelu+量化链替换为 ``npu_gelu_quant``（开关见 FusionPatterns）。"""

    def __init__(self, approximate: str = "tanh"):
        """
        Args:
            approximate: 链头为 ``npu_fast_gelu``（无近似参数可透传）时使用的近似模式，
                仅接受 ``"tanh"``/``"none"``（npu_gelu_quant schema 同 aten.gelu 约定）。
        """
        if approximate not in ("tanh", "none"):
            raise ValueError(f"GeluQuantFusionPass approximate must be 'tanh' or 'none', got {approximate!r}")
        self.approximate = approximate

    def __call__(self, gm) -> None:
        try:
            import torch

            gelu_quant_op = torch.ops.npu.npu_gelu_quant.default
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "[MindIE-SD/compilation] GeluQuantFusionPass skipped: torch.ops.npu.npu_gelu_quant "
                "unavailable on this environment. error=%s",
                exc,
            )
            return

        graph = gm.graph
        chains = find_gelu_quant_chains(list(graph.nodes))
        if not chains:
            return

        fused = 0
        skipped = 0
        for quant_node, gelu_node, view_nodes, dst_type, gelu_approx in chains:
            # npu_gelu_quant 的 dst_type 没有安全默认值：取不到时 torch_npu 侧按 int8
            # 处理，与 MXFP8/MXFP4 的显式 dst_type 不等价。无法确定即跳过该链。
            if dst_type is None:
                skipped += 1
                logger.warning(
                    "[MindIE-SD/compilation] GeluQuantFusionPass skipped chain at node '%s': "
                    "npu_dynamic_mx_quant has no dst_type kwarg; refusing to guess a default.",
                    quant_node.name,
                )
                continue
            # npu_gelu_quant 只能表达 approximate/quant_mode/dst_type/round_mode；
            # axis/scale_alg/dst_type_max 等语义参数无法等价传递，宁可不融合也不静默丢弃。
            unsupported = {
                key: value for key, value in quant_node.kwargs.items() if key not in ("dst_type", "round_mode")
            }
            if unsupported:
                skipped += 1
                logger.warning(
                    "[MindIE-SD/compilation] GeluQuantFusionPass skipped chain at node '%s': "
                    "unsupported npu_dynamic_mx_quant kwargs %s cannot be expressed by npu_gelu_quant.",
                    quant_node.name,
                    sorted(unsupported),
                )
                continue

            approx = gelu_approx if gelu_approx is not None else self.approximate
            fused_kwargs = {"approximate": approx, "quant_mode": "dynamic", "dst_type": dst_type}
            if "round_mode" in quant_node.kwargs:
                fused_kwargs["round_mode"] = quant_node.kwargs["round_mode"]
            # 在 gelu 输入上按原顺序重建 view 序列（gelu 逐元素，与 view 可交换）。
            with graph.inserting_before(quant_node):
                new_src = gelu_node.args[0]
                for view in view_nodes:
                    new_src = graph.call_function(view.target, args=(new_src, *view.args[1:]), kwargs=dict(view.kwargs))
                fused_node = graph.call_function(
                    gelu_quant_op,
                    args=(new_src,),
                    kwargs=fused_kwargs,
                )
            # 两个算子都是 (payload, scale) 元组输出，getitem 用户无需逐一改接。
            quant_node.replace_all_uses_with(fused_node)
            fused += 1

        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info(
            "[MindIE-SD/compilation] GeluQuantFusionPass fused %d gelu→dynamic_mx_quant chains "
            "(approximate=%s, skipped=%d).",
            fused,
            self.approximate,
            skipped,
        )
