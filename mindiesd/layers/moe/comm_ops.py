#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import torch.distributed as dist

from ...utils import ParametersInvalid


def all_gather(tensor, group, dim=0):
    """All-gather across group, concatenating along dim."""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    if dim < 0:
        dim += tensor.dim()

    input_size = tensor.size()
    output_flat = torch.empty(
        input_size[0] * world_size,
        *input_size[1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(output_flat, tensor.contiguous(), group=group)

    if dim == 0:
        return output_flat

    output_flat = output_flat.reshape((world_size,) + input_size)
    output_flat = output_flat.movedim(0, dim)
    return output_flat.reshape(input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :])


def reduce_scatter(tensor, group, dim=0):
    """Reduce-scatter across group along dim."""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    if dim < 0:
        dim += tensor.dim()

    if tensor.shape[dim] % world_size != 0:
        raise ParametersInvalid(
            f"reduce_scatter requires tensor dim {dim} to be divisible by group size, "
            f"but got {tensor.shape[dim]}, world_size={world_size}."
        )

    input_tensor = tensor.movedim(dim, 0).contiguous()
    chunk_size = input_tensor.shape[0] // world_size
    output = torch.empty(
        (chunk_size,) + input_tensor.shape[1:],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.reduce_scatter_tensor(output, input_tensor, group=group)
    return output.movedim(0, dim).contiguous()


def all_reduce(tensor, group):
    """All-reduce across group in-place."""
    dist.all_reduce(tensor, group=group)
    return tensor


def all_to_all_single(input_tensor, output_split_sizes, input_split_sizes, group):
    """All-to-all single with output allocation."""
    if output_split_sizes is None:
        output = torch.empty_like(input_tensor)
    else:
        output = input_tensor.new_empty(
            size=[sum(output_split_sizes)] + list(input_tensor.size()[1:]),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
    dist.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=False,
    )
    return output
