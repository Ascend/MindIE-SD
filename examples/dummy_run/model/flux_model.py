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
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
)

logger = logging.getLogger(__name__)

_K_NUM_SINGLE_BLOCKS = "num_single_transformer_blocks"
_K_NUM_JOINT_BLOCKS = "num_joint_transformer_blocks"
_ATTR_VAE = "vae"
_ATTR_TEXT_ENCODER = "text_encoder"
_ATTR_TEXT_ENCODER_2 = "text_encoder_2"


def _from_config_npu(cls, cfg, npu_device):
    with torch.device("meta"):
        model = cls.from_config(cfg, torch_dtype=torch.bfloat16)
    model.to_empty(device=torch.device("cpu"))
    model.to(npu_device)
    return model


def build_flux_pipeline(
    config_or_dir,
    num_layers=None,
    num_clip_layers=1,
    num_t5_layers=2,
    device=None,
    timer=None,
):
    t_start = time.time()
    npu_device = torch.device(device) if device else torch.device("npu:0")

    transformer_cfg = FluxTransformer2DModel.load_config(
        config_or_dir, subfolder="transformer"
    )
    if num_layers is not None:
        transformer_cfg["num_layers"] = num_layers
        if _K_NUM_SINGLE_BLOCKS in transformer_cfg:
            transformer_cfg[_K_NUM_SINGLE_BLOCKS] = num_layers
        if _K_NUM_JOINT_BLOCKS in transformer_cfg:
            transformer_cfg[_K_NUM_JOINT_BLOCKS] = num_layers
    t0 = time.time()
    transformer = _from_config_npu(
        FluxTransformer2DModel, transformer_cfg, npu_device
    )
    if timer:
        timer.record_build("Transformer", time.time() - t0)

    vae_cfg = AutoencoderKL.load_config(config_or_dir, subfolder=_ATTR_VAE)
    t0 = time.time()
    vae = _from_config_npu(AutoencoderKL, vae_cfg, npu_device)
    if timer:
        timer.record_build("VAE", time.time() - t0)

    scheduler_cfg = FlowMatchEulerDiscreteScheduler.load_config(
        config_or_dir, subfolder="scheduler"
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_cfg)

    text_encoder_cfg = CLIPTextConfig.from_pretrained(
        os.path.join(config_or_dir, _ATTR_TEXT_ENCODER)
    )
    text_encoder_cfg.num_hidden_layers = num_clip_layers
    t0 = time.time()
    text_encoder = CLIPTextModel(text_encoder_cfg)
    text_encoder.to(torch.bfloat16)

    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(config_or_dir, "tokenizer")
    )

    text_encoder_2_cfg = T5Config.from_pretrained(
        os.path.join(config_or_dir, _ATTR_TEXT_ENCODER_2)
    )
    text_encoder_2_cfg.num_hidden_layers = num_t5_layers
    text_encoder_2 = T5EncoderModel(text_encoder_2_cfg)
    text_encoder_2.to(torch.bfloat16)

    tokenizer_2 = T5Tokenizer.from_pretrained(
        os.path.join(config_or_dir, "tokenizer_2")
    )

    if timer:
        timer.record_build(
            "Text encoders + scheduler + tokenizers", time.time() - t0
        )

    pipe = FluxPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
    )

    total = 0
    for attr_name in ("transformer", _ATTR_TEXT_ENCODER, _ATTR_TEXT_ENCODER_2, _ATTR_VAE):
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
