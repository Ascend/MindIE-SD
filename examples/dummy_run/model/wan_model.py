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


import json
import logging
import os
import time

import torch

from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanPipeline,
    WanTransformer3DModel,
)
from transformers import AutoConfig, T5TokenizerFast, UMT5EncoderModel

from . import _PhaseTimer

logger = logging.getLogger(__name__)


def _from_config_meta(cls, cfg, device):
    """Build model with meta device then to_empty on target device."""
    with torch.device("meta"):
        model = cls.from_config(cfg, torch_dtype=torch.bfloat16)
    return model.to_empty(device=device)


def build_wan_pipeline(config_or_dir, num_layers=None, num_layers_2=None,
                       device=None, timer=None):
    t_start = time.time()
    npu_device = torch.device(device) if device else torch.device("npu:0")

    # Build Transformer (high-noise stage)
    logger.warning("  Loading transformer config ...")
    transformer_cfg = WanTransformer3DModel.load_config(config_or_dir, subfolder="transformer")
    if num_layers is not None:
        transformer_cfg["num_layers"] = num_layers
    logger.warning("  Building transformer (meta->to_empty) ...")
    t0 = time.time()
    transformer = _from_config_meta(WanTransformer3DModel, transformer_cfg, npu_device)
    if timer:
        timer.record_build("Transformer", time.time() - t0)

    # Build Transformer_2 (low-noise stage)
    logger.warning("  Loading transformer_2 config ...")
    transformer2_cfg = WanTransformer3DModel.load_config(config_or_dir, subfolder="transformer_2")
    if num_layers_2 is not None:
        transformer2_cfg["num_layers"] = num_layers_2
    logger.warning("  Building transformer_2 (meta->to_empty) ...")
    t0 = time.time()
    transformer_2 = _from_config_meta(WanTransformer3DModel, transformer2_cfg, npu_device)
    if timer:
        timer.record_build("Transformer_2", time.time() - t0)

    # VAE
    logger.warning("  Loading VAE config ...")
    vae_cfg = AutoencoderKLWan.load_config(config_or_dir, subfolder="vae")
    logger.warning("  Building VAE (meta->to_empty) ...")
    t0 = time.time()
    vae = _from_config_meta(AutoencoderKLWan, vae_cfg, npu_device)
    if timer:
        timer.record_build("VAE", time.time() - t0)

    # Scheduler
    scheduler_cfg = UniPCMultistepScheduler.load_config(config_or_dir, subfolder="scheduler")
    scheduler = UniPCMultistepScheduler.from_config(scheduler_cfg)

    # Text encoder
    logger.warning("  Loading text encoder config ...")
    text_encoder_cfg = AutoConfig.from_pretrained(os.path.join(config_or_dir, "text_encoder"))
    logger.warning("  Building text encoder (meta->to_empty) ...")
    t0 = time.time()
    with torch.device("meta"):
        text_encoder = UMT5EncoderModel(text_encoder_cfg)
    text_encoder.to_empty(device=npu_device).to(torch.bfloat16)

    # Tokenizer
    logger.warning("  Loading tokenizer ...")
    tokenizer = T5TokenizerFast.from_pretrained(os.path.join(config_or_dir, "tokenizer"))

    # Boundary ratio
    with open(os.path.join(config_or_dir, "model_index.json"), "r") as fh:
        idx = json.load(fh)
    boundary_ratio = idx.get("boundary_ratio", 0.875)

    if timer:
        timer.record_build("Text encoder + scheduler + tokenizer", time.time() - t0)

    pipe = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        transformer_2=transformer_2,
        vae=vae,
        boundary_ratio=boundary_ratio,
    )

    total = 0
    for attr_name in ("transformer", "transformer_2", "text_encoder", "vae"):
        t = getattr(pipe, attr_name, None)
        if t is not None:
            n = sum(p.numel() for p in t.parameters())
            total += n
            logger.warning("%s params: %.2f B", attr_name, n / 1e9)
    logger.warning("Total params: %.2f B", total / 1e9)
    logger.warning("Estimated memory (bfloat16): %.1f GB", total * 2 / (1024 ** 3))
    logger.warning("Build time: %.1f s", time.time() - t_start)

    return pipe
