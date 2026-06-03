#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import itertools

import torch
from torch.nn import ModuleList

from .utils.logs.logging import logger


def _log_offload_param_error(issue, expected, actual, troubleshooting):
    logger.error(
        "[MindIE-SD/offload] Offload parameter validation failed. "
        "issue=%s, expected=%s, actual=%s. possible_cause=caller passed invalid offload configuration. "
        "Troubleshooting: %s",
        issue,
        expected,
        actual,
        troubleshooting,
    )


def enable_offload(model, blocks, min_reserved_blocks_count=2):
    """
    启用 DiT 模型层级的显存换入换出（offload）机制。

    该函数通过将暂时不使用的层卸载到 CPU，并在需要时异步拷贝回 NPU，
    从而显著降低 NPU 显存峰值，支持更大模型或更长序列的推理。

    Args:
        model (torch.nn.Module): 需要启用 offload 的目标模型。
        blocks (List[torch.nn.Module]): 模型中按顺序排列的 blocks 列表，
            通常对应 Transformer 的各层。
        min_reserved_blocks_count (int, optional): 始终保留在 NPU 上的 block 数量。
            其余 block 的权重将在 CPU 与 NPU 之间动态换入换出。默认值为 2。

    Returns:
        None: 该函数为原地修改，不返回任何值。

    Raises:
        RuntimeError: 当 NPU 相关资源初始化失败时抛出。
        TypeError: 当输入参数类型不符合要求时抛出。
        ValueError: 当输入参数值不符合要求时抛出。

    Note:
        1. 该函数会注册两个 hook：
           - 前向预 Hook：在 block 前向计算前，将其后续 block 的权重从 CPU 异步拷贝到 NPU。
           - 前向 Hook：在 block 前向计算后，将其权重从 NPU 卸载（resize 为 0）以释放显存。
        2. 拷贝流 (`h2d_stream`, `d2h_stream`) 与计算流分离，实现拷贝与计算
           并行，降低延迟。

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from mindiesd.offload import enable_dit_offload
        >>>
        >>> # 定义一个简单的 DiT 模型
        >>> class SimpleDiTBlock(nn.Module):
        ...     def __init__(self, dim):
        ...         super().__init__()
        ...         self.norm = nn.LayerNorm(dim)
        ...         self.mlp = nn.Sequential(
        ...             nn.Linear(dim, dim * 4),
        ...             nn.GELU(),
        ...             nn.Linear(dim * 4, dim)
        ...         )
        ...     def forward(self, x):
        ...         return x + self.mlp(self.norm(x))
        >>>
        >>> class SimpleDiT(nn.Module):
        ...     def __init__(self, num_blocks=12, dim=768):
        ...         super().__init__()
        ...         self.blocks = nn.ModuleList([SimpleDiTBlock(dim) for _ in range(num_blocks)])
        ...     def forward(self, x):
        ...         for block in self.blocks:
        ...             x = block(x)
        ...         return x
        >>>
        >>> # 创建模型实例
        >>> model = SimpleDiT(num_blocks=12, dim=768)
        >>>
        >>> # 启用 offload 机制
        >>> enable_offload(model, model.blocks, min_reserved_blocks_count=2)
        >>> model.to("npu")  # 将模型移动到 NPU
        >>>
        >>> # 准备输入数据
        >>> x = torch.randn(1, 16, 768).to("npu")  # (batch_size, seq_len, dim)
        >>>
        >>> # 执行推理
        >>> with torch.no_grad():
        ...     output = model(x)
        >>>
    """
    if not isinstance(model, torch.nn.Module):
        _log_offload_param_error(
            "model type mismatch",
            "torch.nn.Module",
            type(model).__name__,
            "pass a torch.nn.Module instance as model",
        )
        raise TypeError(f"model must be torch.nn.Module type, current type: {type(model).__name__}")

    if not isinstance(blocks, ModuleList):
        _log_offload_param_error(
            "blocks type mismatch",
            "torch.nn.ModuleList",
            type(blocks).__name__,
            "pass model blocks as torch.nn.ModuleList",
        )
        raise TypeError(f"blocks must be ModuleList, current type: {type(blocks).__name__}")

    if not blocks:
        _log_offload_param_error(
            "blocks is empty",
            "len(blocks)>0",
            len(blocks),
            "provide at least one block before enabling offload",
        )
        raise ValueError("blocks cannot be empty list")

    for i, block in enumerate(blocks):
        if not isinstance(block, torch.nn.Module):
            _log_offload_param_error(
                "block type mismatch",
                "torch.nn.Module",
                f"blocks[{i}]={type(block).__name__}",
                "ensure every item in blocks is a torch.nn.Module",
            )
            raise TypeError(f"blocks[{i}] must be torch.nn.Module type, current type: {type(block).__name__}")

    if not isinstance(min_reserved_blocks_count, int):
        _log_offload_param_error(
            "min_reserved_blocks_count type mismatch",
            "int",
            type(min_reserved_blocks_count).__name__,
            "pass an integer min_reserved_blocks_count",
        )
        raise TypeError(
            f"min_reserved_blocks_count must be int type, current type: {type(min_reserved_blocks_count).__name__}"
        )
    if min_reserved_blocks_count < 0:
        _log_offload_param_error(
            "min_reserved_blocks_count is negative",
            "min_reserved_blocks_count>=0",
            min_reserved_blocks_count,
            "set min_reserved_blocks_count to a non-negative integer",
        )
        raise ValueError(f"min_reserved_blocks_count must be >= 0, current value: {min_reserved_blocks_count}")

    if min_reserved_blocks_count >= len(blocks):
        _log_offload_param_error(
            "min_reserved_blocks_count exceeds block count",
            "min_reserved_blocks_count<len(blocks)",
            f"min_reserved_blocks_count={min_reserved_blocks_count}, len(blocks)={len(blocks)}",
            "reduce min_reserved_blocks_count or provide more blocks",
        )
        raise ValueError(
            f"min_reserved_blocks_count must be < len(blocks), "
            f"current value: {min_reserved_blocks_count}, blocks length: {len(blocks)}"
        )
    if hasattr(model, "mindiesd_offload_enabled") and model.mindiesd_offload_enabled:
        return
    model.mindiesd_offload_enabled = True
    model.h2d_stream = torch.npu.Stream()
    model.d2h_stream = torch.npu.Stream()
    model.min_reserved_blocks_count = min_reserved_blocks_count
    model.event = []
    model.blocks = blocks
    for i, block in enumerate(model.blocks):
        block.index = i
        model.event.append(torch.npu.Event())

    def parameter_to_device_hook(block, _input):
        to_device_index = block.index + model.min_reserved_blocks_count
        forward_event = torch.npu.Event()
        forward_event.record()
        if to_device_index < len(model.blocks):
            with torch.npu.stream(model.h2d_stream):
                model.h2d_stream.wait_event(forward_event)

                for _, p in itertools.chain(
                    model.blocks[to_device_index].named_parameters(), model.blocks[to_device_index].named_buffers()
                ):
                    p.data.untyped_storage().resize_(p.storage_size)
                    if p.is_slice_tensor:
                        p.data.copy_(p.p_cpu, non_blocking=True)
                    else:
                        p.data.untyped_storage().copy_(p.p_cpu.untyped_storage(), non_blocking=True)

                model.event[to_device_index].record()
        torch.npu.current_stream().wait_event(model.event[block.index])

    def parameter_to_resize_hook(block, _input, _output):
        if block.index >= model.min_reserved_blocks_count:
            forward_event = torch.npu.Event()
            forward_event.record()
            with torch.npu.stream(model.d2h_stream):
                model.d2h_stream.wait_event(forward_event)

                for _, p in itertools.chain(block.named_parameters(), block.named_buffers()):
                    p.data.untyped_storage().resize_(0)
        torch.npu.current_stream().wait_stream(model.d2h_stream)

    with torch.npu.stream(model.h2d_stream):
        for i, block in enumerate(model.blocks):
            block.to("npu")
            if i >= model.min_reserved_blocks_count:
                for _, p in itertools.chain(block.named_parameters(), block.named_buffers()):
                    p_cpu = torch.empty(p.data.shape, dtype=p.dtype, pin_memory=True, device="cpu")
                    setattr(p, "p_cpu", p_cpu)

                    expected_storage_size = p.data.numel() * p.data.element_size()
                    is_slice_tensor = p.data.untyped_storage().size() != expected_storage_size
                    storage_size = p.data.untyped_storage().size()
                    if is_slice_tensor:
                        p.p_cpu.copy_(p.data, non_blocking=True)
                    else:
                        p.p_cpu.untyped_storage().copy_(p.data.untyped_storage(), non_blocking=True)

                    setattr(p, "storage_size", storage_size)
                    setattr(p, "is_slice_tensor", is_slice_tensor)

                    p.data.untyped_storage().resize_(0)
    torch.npu.current_stream().wait_stream(model.h2d_stream)

    for block_idx, block in enumerate(model.blocks):
        block.register_forward_pre_hook(parameter_to_device_hook)
        if block_idx >= model.min_reserved_blocks_count:
            block.register_forward_hook(parameter_to_resize_hook)
