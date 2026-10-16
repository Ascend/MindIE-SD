#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""compile 图层性能问题的模型层补丁（与精度/量化正交，纯性能）。

这类问题共性是：eager 无开销/开销小，但 compile 图保留了无意义节点或热点生成链，
Inductor 仍调度对应 kernel，造成纯浪费。解法统一为**模型层模块替换**（Dynamo trace
直接透传，图中不再出现问题节点）。

- ``replace_zero_dropout``: p=0 的 ``nn.Dropout`` → ``nn.Identity``
  （FLUX/Qwen 的 dropout 配置 p=0.0；compile 图保留 ``aten.dropout`` 节点，
  Inductor 仍调度 DropoutV3 kernel，实测 qwen/flux 各 ~1.43ms/step 纯浪费。
  曾尝试 pattern matcher 方案（``aten.dropout(x,0.0,True)`` → ``x.contiguous()``）
  会触发 pattern matcher 死循环（实测卡死），故走模型层替换）。
- ``replace_pos_embed_with_buffers``: qwen pos_embed 输出预计算 + 模块实例替换
  （qwen 的 complex freqs 生成链 split/slice/view/expand/cat/clone 在 NPU 走
  AiCpu kernel，实测 BroadcastToAiCpu 1.70ms/step（14%，第二热点）；dummy 固定
  分辨率下每 forward 结果相同，预计算为 buffer 后冻结为图常量。qwen rope pattern
  的 freqs 输入绑定 buffer 常量，不受影响）。
"""

import logging

import torch

logger = logging.getLogger(__name__)


def replace_zero_dropout(module):
    """递归把 p=0 的 nn.Dropout 替换为 nn.Identity。"""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Dropout):
            if child.p == 0.0:
                setattr(module, name, torch.nn.Identity())
                replaced += 1
        else:
            replaced += replace_zero_dropout(child)
    if replaced:
        logger.warning("Replaced %d zero-dropout modules with Identity under %s",
                       replaced, type(module).__name__)
    return replaced


_MAX_TXT = 512


class FixedPosEmbed(torch.nn.Module):
    """qwen pos_embed 替身: img_freqs 固定 buffer + txt_freqs 动态 slice。"""

    def __init__(self, img_freqs, txt_freqs_table):
        super().__init__()
        self.register_buffer("img_freqs", img_freqs, persistent=False)
        self.register_buffer("txt_freqs_table", txt_freqs_table, persistent=False)

    def forward(self, img_shapes, max_txt_seq_len, device):
        return self.img_freqs, self.txt_freqs_table[:max_txt_seq_len]


def replace_pos_embed_with_buffers(transformer, sample_args, sample_kwargs=None):
    """预计算 qwen pos_embed 输出并替换模块实例。

    Args:
        transformer: 目标 transformer（含 pos_embed 子模块）。
        sample_args: 预计算示例输入（img_shapes 与真实 forward 一致）。
        sample_kwargs: 示例关键字参数（max_txt_seq_len 预计算用 MAX_TXT）。
    """
    sample_kwargs = dict(sample_kwargs or {})
    sample_kwargs.setdefault("max_txt_seq_len", _MAX_TXT)
    pos_embed = getattr(transformer, "pos_embed", None)
    if pos_embed is None or not hasattr(pos_embed, "forward"):
        logger.warning("pos_embed not found, skip replacement")
        return
    with torch.no_grad():
        img_freqs, txt_freqs = pos_embed.forward(*sample_args, **sample_kwargs)
    transformer.pos_embed = FixedPosEmbed(img_freqs, txt_freqs)
    logger.warning("pos_embed replaced with FixedPosEmbed (img=%s txt_table=%s)",
                   tuple(img_freqs.shape), tuple(txt_freqs.shape))
