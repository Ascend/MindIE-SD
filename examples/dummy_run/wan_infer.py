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


import argparse
import logging
import os
import sys
import time

import torch
import torch_npu

from model import check_npu, resolve_config_path, _PhaseTimer
from model.wan_model import build_wan_pipeline

os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
FAST_LAYERS = 2
HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 81
PROMPT = "test"
PROFILE_DIR = "./profile_l1"

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Wan2.2 NPU dummy weight verification"
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--config_cache", type=str, default=None)
    parser.add_argument("--compile", action="store_true",
                        help="Enable MindieSDBackend compilation")
    parser.add_argument("--profile", action="store_true",
                        help="Enable NPU profiling (level=l1, with_stack=False)")
    parser.add_argument("--skip-vae", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip VAE decode (default). Use --no-skip-vae to enable decode.")
    return parser.parse_args()


def _apply_mindie_compile(pipe):
    from mindiesd.compilation import MindieSDBackend

    for attr in ("transformer", "transformer_2"):
        t = getattr(pipe, attr, None)
        if t is not None:
            compiled = torch.compile(t, backend=MindieSDBackend())
            setattr(pipe, attr, compiled)
            logger.warning("%s compiled with MindieSDBackend", attr)


def _start_profile():
    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(PROFILE_DIR),
        with_stack=False,
    )
    prof.start()
    logger.warning("Profiling started (dir=%s, level=l1)", PROFILE_DIR)
    return prof


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    args = _parse_args()
    check_npu()

    device_id = args.device_id
    torch.npu.set_device(device_id)
    torch.npu.empty_cache()
    device = "npu:%d" % device_id
    logger.warning("Using device: %s", device)

    config_dir = resolve_config_path(args.config_cache, MODEL_ID)
    logger.warning("Using config from: %s", config_dir)

    timer = _PhaseTimer(device_id=device_id)
    timer.start_build()

    logger.warning("Building Wan2.2 (%d blocks) ...", FAST_LAYERS)

    pipe = build_wan_pipeline(config_dir, num_layers=FAST_LAYERS,
                              num_layers_2=FAST_LAYERS, device=device,
                              timer=timer)

    t0 = time.time()
    pipe.to(device)
    timer.record_build("Move to device", time.time() - t0)

    if args.compile:
        t0 = time.time()
        _apply_mindie_compile(pipe)
        timer.record_build("Compilation", time.time() - t0)

    prof = None
    if args.profile:
        prof = _start_profile()

    timer.install(pipe)

    logger.warning("Warmup inference (1 step) ...")
    with torch.no_grad():
        pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
             num_frames=NUM_FRAMES, num_inference_steps=1,
             guidance_scale=1.0,
             output_type="latent" if args.skip_vae else "pil")
    torch.npu.synchronize()
    timer.capture_warmup()

    logger.warning("Timed inference (1 step) ...")
    torch.npu.synchronize()
    t0 = time.time()
    with torch.no_grad():
        pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
             num_frames=NUM_FRAMES, num_inference_steps=1,
             guidance_scale=1.0,
             output_type="latent" if args.skip_vae else "pil")
    torch.npu.synchronize()
    logger.warning("Inference time: %.1f s", time.time() - t0)
    timer.capture_timed()

    if prof is not None:
        prof.stop()
        logger.warning("Profile saved to %s", PROFILE_DIR)

    timer.summary()
    logger.warning("Wan2.2 dummy weight verification PASSED")


if __name__ == "__main__":
    main()
