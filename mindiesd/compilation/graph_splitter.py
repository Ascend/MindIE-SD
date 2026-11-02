#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""计算图断点切分工具（Breakable Graph）。

按显式断点节点名将 ``fx.GraphModule`` 切分为多个顺序子图，供分段捕获后端使用。
采用 ``torch.fx.Graph.node_copy`` 做子图复制，段间通过 placeholder 衔接前段输出，
不引入 ``split_module`` 的子模块间接，避免破坏既有 Pattern Pass。

语义：``breakpoints`` 为节点名列表，每个匹配的节点作为新一段的起点。例如节点序列
``[n0, n1, n2, n3]``、断点 ``["n2"]`` 切分为 ``[[n0, n1], [n2, n3]]``。断点为空或全部
不匹配时返回单个原图（不切分），调用方据此退化为整图单段捕获。

注意：本模块需在 NPU 环境验证切分与段间衔接的正确性。
"""

import dataclasses

from torch import fx


@dataclasses.dataclass
class GraphSegment:
    """一个切分后的子图段及其段间衔接元数据。

    Attributes:
        gm: 子图 GraphModule，其 placeholder 顺序与 ``inputs`` 一致。
        inputs: 每个 placeholder 的来源。``("orig", name)`` 表示原始模型输入；
            ``("prev", src_seg_idx, out_idx)`` 表示前一段 ``src_seg_idx`` 的第
            ``out_idx`` 个输出，分段捕获后端据此将前段输出缓冲绑定到该输入。
        num_outputs: 该段输出张量个数，供后端分配段间衔接缓冲。
    """

    gm: fx.GraphModule
    inputs: list[tuple]
    num_outputs: int


@dataclasses.dataclass
class OutputSpec:
    """整图返回值结构，供分段捕获后端按原始 output 结构组装返回。

    ``layout`` 是原 ``output`` 节点的嵌套结构模板（tuple/list/dict 递归），叶子为
    ``("leaf", index)``，常量叶子为 ``("const", value)``；``leaves[index]`` 给出该叶子
    对应的 ``(段号, 段内输出序号)``。这样即使某个图输出由非末段产生，也能被正确定位
    并按原顺序、原容器类型返回。
    """

    layout: tuple
    leaves: list[tuple[int, int]]


def _build_output_layout(value, leaves: list) -> tuple:
    """把原 ``output`` 参数递归转成结构模板，并把 Node 叶子登记到 ``leaves``。"""
    if isinstance(value, fx.Node):
        index = len(leaves)
        leaves.append(value)
        return ("leaf", index)
    if isinstance(value, tuple):
        return ("tuple", [_build_output_layout(v, leaves) for v in value])
    if isinstance(value, list):
        return ("list", [_build_output_layout(v, leaves) for v in value])
    if isinstance(value, dict):
        return ("dict", {k: _build_output_layout(v, leaves) for k, v in value.items()})
    return ("const", value)


def split_graph_by_breakpoints(
    gm: fx.GraphModule, breakpoints: list[str] | None = None
) -> tuple[list[fx.GraphModule], list[GraphSegment] | None, OutputSpec | None]:
    """按断点节点名切分 ``gm``。

    Args:
        gm: 待切分的 GraphModule。
        breakpoints: 断点节点名列表。

    Returns:
        ``(sub_graphs, segments_meta, output_spec)``。无切分时返回
        ``([gm], None, None)``；切分后返回子图列表、对应的段元数据列表，以及
        整图返回值的结构描述（用于跨段组装并按原顺序/原容器类型返回）。
    """
    if not breakpoints:
        return [gm], None, None

    breakpoint_set = set(breakpoints)
    non_output_nodes = [n for n in gm.graph.nodes if n.op != "output"]

    # 1. 按断点分段：断点节点作为新一段的起点。
    segments: list[list[fx.Node]] = []
    current: list[fx.Node] = []
    for node in non_output_nodes:
        if node.name in breakpoint_set and current:
            segments.append(current)
            current = []
        current.append(node)
    if current:
        segments.append(current)

    if len(segments) <= 1:
        return [gm], None, None

    # 2. 节点 → 段号映射。
    node_seg = {}
    for i, seg in enumerate(segments):
        for n in seg:
            node_seg[n] = i

    # 3. 原始 output 节点（最终结果消费方）及其返回值结构。
    output_node = None
    for n in gm.graph.nodes:
        if n.op == "output":
            output_node = n
            break
    output_leaves: list[fx.Node] = []
    if output_node is not None and output_node.args:
        output_layout = _build_output_layout(output_node.args[0], output_leaves)
    else:
        output_layout = ("const", None)

    # 4. 每段对外输出 = 被后段节点引用、或被最终 output 引用的段内节点。
    seg_output_sets = [set() for _ in segments]
    for i, seg in enumerate(segments):
        for n in seg:
            for user in n.users:
                useg = node_seg.get(user)
                if useg is None:
                    useg = len(segments)  # output 节点视为最终消费
                if useg > i:
                    seg_output_sets[i].add(n)

    # 按节点顺序确定每段输出的稳定顺序（段内顺序仅供 backend 定位，对外返回顺序
    # 由下面的 OutputSpec 按原 output 参数序决定）。
    seg_output_ordered = [[n for n in seg if n in seg_output_sets[i]] for i, seg in enumerate(segments)]

    # 整图返回值：记录每个输出叶子来自 (段号, 段内输出序号)。非末段产生的图输出
    # 也必须出现在这里，否则后端只会回放末段、静默丢输出。
    leaves_spec: list[tuple[int, int]] = []
    for leaf in output_leaves:
        owner = node_seg.get(leaf)
        if owner is None:
            raise RuntimeError(f"breakpoint split: graph output node '{leaf.name}' is not part of any segment")
        try:
            pos = seg_output_ordered[owner].index(leaf)
        except ValueError as exc:
            raise RuntimeError(
                f"breakpoint split: graph output node '{leaf.name}' is not exported by segment {owner}"
            ) from exc
        leaves_spec.append((owner, pos))
    output_spec = OutputSpec(layout=output_layout, leaves=leaves_spec)

    # 5. 用 node_copy 构造每段子图，段外引用转化为 placeholder。
    sub_graphs: list[fx.GraphModule] = []
    seg_meta: list[GraphSegment] = []

    for i, seg in enumerate(segments):
        new_graph = fx.Graph()
        env = {}
        input_specs: list[tuple] = []

        # 默认参数在 def 时绑定本轮循环的 env/input_specs/new_graph（每轮循环
        # 新建绑定、无跨调用共享的闭包惯用法），抑制 pylint 对可变默认值的误报。
        # pylint: disable=dangerous-default-value
        def arg_transform(orig_node, _env=env, _specs=input_specs, _new_graph=new_graph):
            if orig_node in _env:
                return _env[orig_node]
            # 段外引用 → 创建 placeholder 并记录来源。
            ph = _new_graph.placeholder(orig_node.name)
            _env[orig_node] = ph
            src_seg = node_seg.get(orig_node)
            if src_seg is None:
                _specs.append(("orig", orig_node.name))
            else:
                out_idx = seg_output_ordered[src_seg].index(orig_node)
                _specs.append(("prev", src_seg, out_idx))
            return ph

        for node in seg:
            if node.op == "output":
                continue
            new_node = new_graph.node_copy(node, arg_transform)
            env[node] = new_node
            if node.op == "placeholder":
                # 原始模型输入：node_copy 已复制为 placeholder，补充来源标记以保持与
                # input_specs 顺序一致。
                input_specs.append(("orig", node.name))

        outs = seg_output_ordered[i]
        if len(outs) == 1:
            new_graph.output(env[outs[0]])
        elif len(outs) > 1:
            new_graph.output([env[o] for o in outs])
        else:
            new_graph.output(None)

        new_graph.lint()
        sub_gm = fx.GraphModule(gm, new_graph, class_name=f"Segment{i}")
        sub_graphs.append(sub_gm)
        seg_meta.append(GraphSegment(gm=sub_gm, inputs=input_specs, num_outputs=len(outs)))

    return sub_graphs, seg_meta, output_spec
