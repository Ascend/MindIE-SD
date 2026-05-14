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
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def _get_custom_classes():
    for name in sorted(sys.modules.keys()):
        if "tokenizer_wrapper" in name and "hunyuan" in name.lower():
            mod = sys.modules[name]
            return mod.TokenizerEncodeOutput, mod.ImageInfo
    raise RuntimeError("Cannot find TokenizerEncodeOutput/ImageInfo")


def build_dummy_tokenizer_output(model, token_length=4096):
    tkw = model._tkwrapper  # pylint: disable=protected-access
    tok = tkw.tokenizer
    coi = tok.convert_tokens_to_ids

    text_ids = tok.encode("test", add_special_tokens=False)
    dummy = text_ids[0] if text_ids else 0

    bos = tkw.bos_token_id or dummy
    boi = tkw.boi_token_id or dummy
    eoi = tkw.eoi_token_id or dummy
    img = tkw.img_token_id or dummy

    tokens = (
        [bos]
        + text_ids
        + [boi]
        + [coi("<img_ratio_16>") or dummy, coi("<img_size_1024>") or dummy, coi("<timestep>") or dummy]
        + [img] * token_length
        + [eoi]
    )

    for i, t in enumerate(tokens):
        if t is None:
            logger.warning("Token at position %d is None", i)

    input_ids = torch.tensor([tokens], dtype=torch.int64)

    img_start = 1 + len(text_ids) + 1 + 3
    img_end = img_start + token_length
    img_slice = slice(img_start, img_end)
    timestep_idx = img_start - 1  # last meta token

    TokenizerEncodeOutput, ImageInfo = _get_custom_classes()  # pylint: disable=invalid-name

    mask = torch.zeros(1, len(tokens), dtype=torch.bool)
    mask[0, img_start:img_end] = True

    output = TokenizerEncodeOutput(
        tokens=input_ids,
        gen_image_slices=[[img_slice]],
        all_image_slices=[[img_slice]],
        gen_image_mask=mask,
        joint_image_slices=[[]],
        cond_vae_image_slices=[[]],
        cond_vit_image_slices=[[]],
        text_slices=[],
        gen_timestep_scatter_index=torch.tensor([[timestep_idx]]),
    )

    img_info = ImageInfo(
        image_type="gen_image",
        image_width=1024,
        image_height=1024,
        token_width=64,
        token_height=64,
        image_token_length=token_length,
        base_size=1024,
        ratio_index=16,
    )

    return output, img_info, img_start


def build_hunyuan_image3_model(config_dir, num_layers=2, device=None, timer=None):
    t_start = time.time()
    npu_device = torch.device(device) if device else torch.device("npu:0")

    logger.warning("  Loading config ...")
    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    config.num_hidden_layers = num_layers
    config.vit["num_hidden_layers"] = 1

    logger.warning("  Building model (from_config, bfloat16) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if timer:
        timer.record_build("Model (AR+MoE+ViT+VAE)", time.time() - t0)

    logger.warning("  Loading tokenizer ...")
    t0 = time.time()
    model.load_tokenizer(config_dir)
    if timer:
        timer.record_build("Tokenizer", time.time() - t0)

    logger.warning("  Moving to NPU ...")
    t0 = time.time()
    model.to(npu_device)
    if timer:
        timer.record_build("Move to device", time.time() - t0)

    total = sum(p.numel() for p in model.parameters())
    logger.warning("Total params: %.2f B", total / 1e9)
    logger.warning("Estimated memory (bfloat16): %.1f GB", total * 2 / (1024**3))
    logger.warning("Build time: %.2f ms", (time.time() - t_start) * 1000)

    return model
