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


import logging
import os
import time

import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

logger = logging.getLogger(__name__)


def _from_config_cpu(cls, cfg, npu_device):
    model = cls.from_config(cfg, torch_dtype=torch.bfloat16)
    model.to(npu_device)
    return model


def build_qwen_image_pipeline(
    config_or_dir, num_layers=None, num_text_encoder_layers=None, device=None, timer=None
):
    t_start = time.time()
    npu_device = torch.device(device) if device else torch.device("npu:0")

    transformer_cfg = QwenImageTransformer2DModel.load_config(
        config_or_dir, subfolder="transformer"
    )
    if num_layers is not None:
        transformer_cfg["num_layers"] = num_layers
    t0 = time.time()
    transformer = _from_config_cpu(
        QwenImageTransformer2DModel, transformer_cfg, npu_device
    )
    if timer:
        timer.record_build("Transformer", time.time() - t0)

    vae_cfg = AutoencoderKLQwenImage.load_config(config_or_dir, subfolder="vae")
    t0 = time.time()
    vae = _from_config_cpu(AutoencoderKLQwenImage, vae_cfg, npu_device)
    if timer:
        timer.record_build("VAE", time.time() - t0)

    scheduler_cfg = FlowMatchEulerDiscreteScheduler.load_config(
        config_or_dir, subfolder="scheduler"
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_cfg)

    text_encoder_cfg = AutoConfig.from_pretrained(
        os.path.join(config_or_dir, "text_encoder")
    )
    if num_text_encoder_layers is not None:
        text_encoder_cfg.num_hidden_layers = num_text_encoder_layers
        if hasattr(text_encoder_cfg, "text_config"):
            text_encoder_cfg.text_config.num_hidden_layers = num_text_encoder_layers
        if hasattr(text_encoder_cfg, "vision_config"):
            text_encoder_cfg.vision_config.depth = num_text_encoder_layers
    t0 = time.time()
    with torch.device("meta"):
        text_encoder = Qwen2_5_VLForConditionalGeneration(text_encoder_cfg)
    text_encoder.to_empty(device=torch.device("cpu"))
    text_encoder.to(npu_device, dtype=torch.bfloat16)

    tokenizer = Qwen2Tokenizer.from_pretrained(
        os.path.join(config_or_dir, "tokenizer")
    )

    if timer:
        timer.record_build(
            "Text encoder + scheduler + tokenizer", time.time() - t0
        )

    pipe = QwenImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )

    total = 0
    for attr_name in ("transformer", "text_encoder", "vae"):
        t = getattr(pipe, attr_name, None)
        if t is not None:
            n = sum(p.numel() for p in t.parameters())
            total += n
            logger.warning("%s params: %.2f B", attr_name, n / 1e9)
    logger.warning("Total params: %.2f B", total / 1e9)
    logger.warning(
        "Estimated memory (bfloat16): %.1f GB", total * 2 / (1024 ** 3)
    )
    logger.warning("Build time: %.2f ms", (time.time() - t_start) * 1000)

    return pipe
