#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import operator

import torch
import torch_npu
from torch import fx
from torch._inductor.pattern_matcher import Arg, CallFunction, GraphPatternEntry, Ignored, MULTIPLE


def _chain_nodes(silu_node: fx.Node, native_group_norm: bool):
    if not silu_node.args or not isinstance(silu_node.args[0], fx.Node):
        return None, None

    norm_value_node = silu_node.args[0]
    if not native_group_norm:
        if norm_value_node.target == torch.ops.aten.group_norm.default:
            return norm_value_node, norm_value_node
        return None, None

    if (
        norm_value_node.target != operator.getitem
        or len(norm_value_node.args) != 2
        or norm_value_node.args[1] != 0
    ):
        return None, None
    norm_node = norm_value_node.args[0]
    if isinstance(norm_node, fx.Node) and norm_node.target == torch.ops.aten.native_group_norm.default:
        return norm_node, norm_value_node
    return None, None


def _fused_args(norm_node: fx.Node, native_group_norm: bool):
    args = norm_node.args
    if native_group_norm and len(args) >= 8:
        return args[0], args[1], args[2], args[6], args[7]
    if not native_group_norm and len(args) >= 5:
        return args[0], args[2], args[3], args[1], args[4]
    return None


def _meta_value(value):
    if not isinstance(value, fx.Node):
        return value
    meta_value = value.meta.get("val")
    return meta_value if meta_value is not None else value.meta.get("example_value")


def _fake_outputs(args):
    if args is None:
        return None
    meta_args = tuple(_meta_value(value) for value in args)
    input_meta = meta_args[0]
    if not isinstance(input_meta, torch.Tensor):
        return None
    if input_meta.device.type != "meta" and not hasattr(input_meta, "fake_mode"):
        return None
    if any(value is not None and not isinstance(value, torch.Tensor) for value in meta_args[1:3]):
        return None
    try:
        outputs = torch_npu.npu_group_norm_silu(*meta_args)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        return None
    if not isinstance(outputs, (tuple, list)) or not outputs or not isinstance(outputs[0], torch.Tensor):
        return None
    return outputs


def _extra_check(match, native_group_norm: bool):
    silu_node = match.output_node()
    norm_node, norm_value_node = _chain_nodes(silu_node, native_group_norm)
    if norm_node is None or norm_value_node is None:
        return False
    if set(norm_value_node.users) != {silu_node}:
        return False

    if native_group_norm:
        for user in norm_node.users:
            if user is norm_value_node:
                continue
            if (
                user.target == operator.getitem
                and len(user.args) == 2
                and user.args[1] in (1, 2)
                and not user.users
            ):
                continue
            return False
    elif set(norm_node.users) != {silu_node}:
        return False

    return _fake_outputs(_fused_args(norm_node, native_group_norm)) is not None


def _handler(match, native_group_norm: bool):
    graph = match.graph
    silu_node = match.output_node()
    norm_node, norm_value_node = _chain_nodes(silu_node, native_group_norm)
    args = _fused_args(norm_node, native_group_norm)
    outputs = _fake_outputs(args)
    if outputs is None:
        raise RuntimeError("GroupNorm+SiLU GraphPatternEntry lost its fake output metadata after matching.")

    with graph.inserting_before(silu_node):
        fused_result = graph.call_function(torch_npu.npu_group_norm_silu, args=args)
        fused_output = graph.call_function(operator.getitem, args=(fused_result, 0))
    fused_result.meta["val"] = outputs
    fused_result.meta["example_value"] = outputs
    fused_output.meta = silu_node.meta.copy()
    fused_output.meta["val"] = outputs[0]
    fused_output.meta["example_value"] = outputs[0]

    silu_node.replace_all_uses_with(fused_output)
    graph.erase_node(silu_node)
    if norm_value_node is not norm_node:
        graph.erase_node(norm_value_node)
    for user in list(norm_node.users):
        if (
            user.target == operator.getitem
            and len(user.args) == 2
            and user.args[1] in (1, 2)
            and not user.users
        ):
            graph.erase_node(user)
    if not norm_node.users:
        graph.erase_node(norm_node)


class GroupNormSiLUPatternGroup:
    """Register GroupNorm+SiLU graph patterns with the Inductor matcher."""

    @staticmethod
    def register(pattern_pass):
        """Register entries for both GroupNorm ATen graph forms."""
        if not hasattr(torch_npu, "npu_group_norm_silu"):
            return

        group_norm = CallFunction(
            torch.ops.aten.group_norm.default,
            Arg(),
            Ignored(),
            Arg(),
            Arg(),
            Ignored(),
            Ignored(),
        )
        group_norm_silu = CallFunction(torch.ops.aten.silu.default, group_norm)
        GraphPatternEntry(
            pattern=group_norm_silu,
            extra_check=lambda match: _extra_check(match, native_group_norm=False),
            handler=lambda match, *args, **kwargs: _handler(match, native_group_norm=False),
        ).register(pattern_pass)

        native_group_norm = CallFunction(
            torch.ops.aten.native_group_norm.default,
            Arg(),
            Arg(),
            Arg(),
            Ignored(),
            Ignored(),
            Ignored(),
            Ignored(),
            Ignored(),
            _users=MULTIPLE,
        )
        native_group_norm_output = CallFunction(operator.getitem, native_group_norm, 0)
        native_group_norm_silu = CallFunction(torch.ops.aten.silu.default, native_group_norm_output)
        GraphPatternEntry(
            pattern=native_group_norm_silu,
            extra_check=lambda match: _extra_check(match, native_group_norm=True),
            handler=lambda match, *args, **kwargs: _handler(match, native_group_norm=True),
        ).register(pattern_pass)
