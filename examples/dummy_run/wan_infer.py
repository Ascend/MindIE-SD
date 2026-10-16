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

sys.path.append(os.path.dirname(__file__))
from model import _PhaseTimer, check_npu, resolve_config_path
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
    parser.add_argument("--num_layers", type=int, default=FAST_LAYERS,
                        help="Number of transformer layers (default: %d)" % FAST_LAYERS)
    parser.add_argument("--compile", action="store_true",
                        help="Enable MindieSDBackend compilation")
    parser.add_argument("--profile", action="store_true",
                        help="Enable NPU profiling (level=l1, with_stack=False)")
    parser.add_argument("--skip-vae", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip VAE decode (default). Use --no-skip-vae to enable decode.")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                        help="Video frames per generation (Wan requires (N-1) %% 4 == 0; "
                             "17 frames ~= 1s at 16fps). Vary to validate pass generality "
                             "across sequence lengths.")
    parser.add_argument("--height", type=int, default=HEIGHT,
                        help="Video height (latent S scales with frames and resolution).")
    parser.add_argument("--width", type=int, default=WIDTH,
                        help="Video width.")
    parser.add_argument("--compute-precision", type=str, default="bf16",
                        choices=["fp32", "bf16"],
                        help="Compute precision (default bf16). bf16 = Wan activations and "
                             "weights both run natively in bf16 (the model's .float() "
                             "conversions are branched to .to(bf16) at the code level; no "
                             "implicit conversion happens in compilation). fp32 = original "
                             "fp32 compute (slower, ~14x on GEMMs).")
    return parser.parse_args()


def _apply_mindie_compile(pipe):
    from mindiesd.compilation import MindieSDBackend

    for attr in ("transformer", "transformer_2"):
        t = getattr(pipe, attr, None)
        if t is not None:
            compiled = torch.compile(t, backend=MindieSDBackend())
            setattr(pipe, attr, compiled)
            logger.warning("%s compiled with MindieSDBackend", attr)


# ---------------------------------------------------------------------------
# Compute precision (default bf16): transformer 权重 cast bf16 + 模型 `.float()`
# 源码级分支为 `.to(bf16)`, 编译侧零隐式精度转换。
# ---------------------------------------------------------------------------


def _rewrite_wan_float_conversions(compute_dtype):
    """把 Wan forwards 里的 `.float()` 改写为 `.to(_mindie_compute_dtype)`(源码级)。

    Dynamo trace 时绕过 `torch.Tensor.float` patch, 必须改写源码才能让图真正 bf16。
    """
    import inspect
    import textwrap
    import typing as _typing

    from diffusers.models import normalization
    from diffusers.models.transformers import transformer_wan as tw

    normalization._mindie_compute_dtype = compute_dtype
    tw._mindie_compute_dtype = compute_dtype

    for cls, hint in ((normalization.FP32LayerNorm, "FP32LayerNorm"),
                      (tw.WanTransformerBlock, "WanTransformerBlock"),
                      (tw.WanTransformer3DModel, "WanTransformer3DModel")):
        try:
            src = textwrap.dedent(inspect.getsource(cls.forward))
        except (OSError, TypeError):
            continue
        if ".float()" not in src:
            continue
        new_src = src.replace(".float()", ".to(_mindie_compute_dtype)")
        # exec 进真实模块 dict(Dynamo guard 要求 __globals__ 是模块);
        # 模块启用 from __future__ import annotations, exec 求值注解需注入 typing 名
        module_ns = cls.forward.__globals__
        module_ns["_mindie_compute_dtype"] = compute_dtype
        for _name in ("Any", "Optional", "Tuple", "List", "Dict", "Union",
                      "Callable", "Sequence", "Mapping", "Type"):
            module_ns.setdefault(_name, getattr(_typing, _name, None))
        exec(compile(new_src, f"<wan-{hint}-compute-dtype>", "exec"), module_ns)
        setattr(cls, "forward", module_ns["forward"])


def _apply_compute_precision(pipe, precision):
    """bf16: transformer 权重 cast + `.float()` 分支; fp32: 原行为。"""
    if precision != "bf16":
        return
    for attr in ("transformer", "transformer_2"):
        t = getattr(pipe, attr, None)
        if t is not None:
            t.to(torch.bfloat16)
    _rewrite_wan_float_conversions(torch.bfloat16)
    logger.warning("Compute precision: bf16 (model-level, no implicit conversion in compilation)")


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

    logger.warning("Building Wan2.2 (%d blocks) ...", args.num_layers)

    pipe = build_wan_pipeline(config_dir, num_layers=args.num_layers,
                               num_layers_2=args.num_layers, device=device,
                               timer=timer)

    t0 = time.time()
    pipe.to(device)
    timer.record_build("Move to device", time.time() - t0)

    # compute precision (default bf16): model-level weights cast + .float() branching
    _apply_compute_precision(pipe, args.compute_precision)

    if args.compile:
        t0 = time.time()
        _apply_mindie_compile(pipe)
        timer.record_build("Compilation", time.time() - t0)

    timer.install(pipe)

    logger.warning("Warmup (1 step):")
    with torch.no_grad():
        pipe(prompt=PROMPT, height=args.height, width=args.width,
             num_frames=args.num_frames, num_inference_steps=1,
             guidance_scale=1.0,
             output_type="latent" if args.skip_vae else "pil")
    torch.npu.synchronize()
    timer.capture_warmup()

    prof = None
    if args.profile:
        prof = _start_profile()

    logger.warning("Timed (1 step):")
    torch.npu.synchronize()
    t0 = time.time()
    with torch.no_grad():
        pipe(prompt=PROMPT, height=args.height, width=args.width,
             num_frames=args.num_frames, num_inference_steps=1,
             guidance_scale=1.0,
             output_type="latent" if args.skip_vae else "pil")
    torch.npu.synchronize()
    logger.warning("Inference time: %.2f ms", (time.time() - t0) * 1000)
    timer.capture_timed()

    if prof is not None:
        torch.npu.synchronize()
        prof.stop()
        logger.warning("Profile saved to %s", PROFILE_DIR)

    timer.summary()
    logger.warning("Wan2.2 dummy weight verification PASSED")


if __name__ == "__main__":
    main()
