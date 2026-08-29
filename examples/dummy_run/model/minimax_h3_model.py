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
from diffusers import MiniMaxH3Scheduler
from diffusers.models import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Transformer3DModel,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3Blocks,
    MiniMaxH3DecodeStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
from transformers import (
    AutoConfig,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

logger = logging.getLogger(__name__)


class MiniMaxH3BlocksNoDecode(MiniMaxH3Blocks):
    """MiniMax-H3 blocks without the decode step.

    The t2va dummy run verifies the conditioning/denoising path and stops at the denoised latents,
    so the video/audio VAE decode (custom remote code) is never invoked.
    """

    block_classes = [b for b in MiniMaxH3Blocks.block_classes if b is not MiniMaxH3DecodeStep]
    block_names = [n for n in MiniMaxH3Blocks.block_names if n != "decode"]


class MiniMaxH3DummyPipeline(MiniMaxH3ModularPipeline):
    @property
    def text_encoder_layer(self):
        # MiniMax-H3 conditions on hidden_states[layer] of its Qwen3-VL conditioner (layer 50
        # with the full checkpoint). A dummy run truncates the conditioner to 2 decoder layers,
        # whose last state is post-norm and not a valid conditioning, so condition on the first
        # decoder layer instead.
        return 1


def _from_config_meta(cls, cfg, device):
    """Build model with meta device then to_empty on target device."""
    with torch.device("meta"):
        model = cls.from_config(cfg, torch_dtype=torch.bfloat16)
    return model.to_empty(device=device)


def build_minimax_h3_pipeline(config_dir, num_layers=2, device=None, timer=None):
    t_start = time.time()
    npu_device = torch.device(device) if device else torch.device("npu:0")

    # Transformer (the joint video + audio DiT over one packed sequence)
    transformer_cfg = MiniMaxH3Transformer3DModel.load_config(config_dir, subfolder="transformer")
    transformer_cfg["num_layers"] = num_layers
    transformer_cfg["num_refiner_layers"] = 1
    t0 = time.time()
    transformer = _from_config_meta(MiniMaxH3Transformer3DModel, transformer_cfg, npu_device)
    if timer:
        timer.record_build("Transformer", time.time() - t0)

    # Text encoder (Qwen3-VL conditioner, truncated)
    text_encoder_cfg = AutoConfig.from_pretrained(os.path.join(config_dir, "text_encoder"))
    text_encoder_cfg.text_config.num_hidden_layers = 2
    text_encoder_cfg.vision_config.depth = 1
    t0 = time.time()
    with torch.device("meta"):
        text_encoder = Qwen3VLForConditionalGeneration(text_encoder_cfg)
    text_encoder.to_empty(device=npu_device).to(torch.bfloat16)
    if timer:
        timer.record_build("Text encoder", time.time() - t0)

    # Video VAE / audio VAE: built from config for parameter accounting only. The t2va workflow
    # never invokes them in forward (no keyframe/reference encoding, and the decode step is
    # pruned), so the custom remote code of the released checkpoints is not exercised.
    vae_cfg = AutoencoderKLMiniMaxH3.load_config(config_dir, subfolder="vae")
    t0 = time.time()
    vae = _from_config_meta(AutoencoderKLMiniMaxH3, vae_cfg, npu_device)
    if timer:
        timer.record_build("Video VAE", time.time() - t0)

    audio_vae_cfg = AutoencoderKLMiniMaxH3Audio.load_config(config_dir, subfolder="audio_vae")
    t0 = time.time()
    audio_vae = _from_config_meta(AutoencoderKLMiniMaxH3Audio, audio_vae_cfg, npu_device)
    if timer:
        timer.record_build("Audio VAE", time.time() - t0)

    # Tokenizer / processor (real vocab files, KB-scale)
    t0 = time.time()
    tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(config_dir, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(config_dir, "processor"))
    if timer:
        timer.record_build("Tokenizer + processor", time.time() - t0)

    # Schedulers (video shift=12.0, audio shift=3.0)
    scheduler_cfg = MiniMaxH3Scheduler.load_config(config_dir, subfolder="scheduler")
    audio_scheduler_cfg = MiniMaxH3Scheduler.load_config(config_dir, subfolder="audio_scheduler")
    t0 = time.time()
    scheduler = MiniMaxH3Scheduler.from_config(scheduler_cfg)
    audio_scheduler = MiniMaxH3Scheduler.from_config(audio_scheduler_cfg)
    if timer:
        timer.record_build("Schedulers", time.time() - t0)

    # Assemble the modular pipeline: component specs are read from the repo's model_index.json, then
    # the dummy (random-weight) components are registered over the from_pretrained placeholders.
    pipe = MiniMaxH3DummyPipeline(
        blocks=MiniMaxH3BlocksNoDecode(), pretrained_model_name_or_path=config_dir
    )
    pipe.register_components(
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        audio_vae=audio_vae,
        tokenizer=tokenizer,
        processor=processor,
        scheduler=scheduler,
        audio_scheduler=audio_scheduler,
    )

    total = 0
    for attr_name in ("transformer", "text_encoder", "vae", "audio_vae"):
        t = getattr(pipe, attr_name, None)
        if t is not None:
            n = sum(p.numel() for p in t.parameters())
            total += n
            logger.warning("%s params: %.2f B", attr_name, n / 1e9)
    logger.warning("Total params: %.2f B", total / 1e9)
    logger.warning("Estimated memory (bfloat16): %.1f GB", total * 2 / (1024 ** 3))
    logger.warning("Build time: %.2f ms", (time.time() - t_start) * 1000)

    return pipe
