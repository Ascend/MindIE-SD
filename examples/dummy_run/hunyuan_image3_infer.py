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
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.append(os.path.dirname(__file__))
from model import check_npu, resolve_config_path, _PhaseTimer
from model.hunyuan_image3_model import build_hunyuan_image3_model, build_dummy_tokenizer_output

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "tencent/HunyuanImage-3.0"
FAST_LAYERS = 2
IMAGE_SIZE = [1024, 1024]
TOKEN_LENGTH = 4096
PROFILE_DIR = "./profile_l1"

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0 NPU dummy weight verification")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--config_cache", type=str, default=None)
    parser.add_argument(
        "--num_layers",
        type=int,
        default=FAST_LAYERS,
        help="Number of transformer layers (default: %d)" % FAST_LAYERS,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable MindieSDBackend compilation",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable NPU profiling (level=l1, with_stack=False)",
    )
    parser.add_argument(
        "--skip-vae",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


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

    # Required: HunyuanImage-3.0 custom code hardcodes torch.cuda calls.
    # Patch them for NPU execution. Use allow_in_graph for torch.compile compat.
    import torch.cuda as _cuda
    import contextlib

    @torch._dynamo.allow_in_graph  # pylint: disable=protected-access
    def _npu_set_device(d):
        if isinstance(d, int):
            torch.npu.set_device(d)
        else:
            torch.npu.set_device(d.index)

    @torch._dynamo.allow_in_graph  # pylint: disable=protected-access
    def _nop_nvtx(*_args, **_kw):
        return contextlib.nullcontext()

    _cuda.set_device = _npu_set_device

    def _npu_current_device():
        return torch.npu.current_device()

    _cuda.current_device = _npu_current_device
    _cuda.nvtx.range = _nop_nvtx

    logger.warning(
        "Building HunyuanImage-3.0 (%d transformer layers) ...",
        args.num_layers,
    )

    model = build_hunyuan_image3_model(
        config_dir,
        num_layers=args.num_layers,
        device=device,
        timer=timer,
    )

    if args.compile:
        from mindiesd.compilation import MindieSDBackend

        t0 = time.time()
        torch._dynamo.config.capture_scalar_outputs = True  # pylint: disable=protected-access
        model = torch.compile(model, backend=MindieSDBackend(), fullgraph=False)
        timer.record_build("Compilation", time.time() - t0)

    # Initialize pipeline (HF cached from_config may not set it)
    if model._pipeline is None:  # pylint: disable=protected-access
        _pipe_cls = None
        for _n in sys.modules:
            if "hunyuan_image_3_pipeline" in _n:
                _pipe_cls = sys.modules[_n].HunyuanImage3Text2ImagePipeline
                _sched_cls = sys.modules[_n].FlowMatchDiscreteScheduler
                break
        if _pipe_cls is not None:
            model._pipeline = _pipe_cls(  # pylint: disable=protected-access
                model=model,
                scheduler=_sched_cls(shift=3.0, reverse=True, solver="euler"),
                vae=model.vae,
            )

    # Patch VAE decode to handle float32→bfloat16 on NPU
    _vae_dec = model.vae.decode

    def _patched_vae_decode(*args, **kwargs):
        args = list(args)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            args[0] = args[0].to(torch.bfloat16)
        return _vae_dec(*args, **kwargs)

    model.vae.decode = _patched_vae_decode

    tokenizer_output, img_info, img_start = build_dummy_tokenizer_output(model, TOKEN_LENGTH)
    input_ids = tokenizer_output.tokens.to(device)
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    if tokenizer_output.gen_image_mask is not None:
        tokenizer_output.gen_image_mask = tokenizer_output.gen_image_mask.to(device)
    if tokenizer_output.gen_timestep_scatter_index is not None:
        tokenizer_output.gen_timestep_scatter_index = tokenizer_output.gen_timestep_scatter_index.to(device)

    _rope_fn = None
    for _n in sys.modules:
        if _n.endswith(".hunyuan") or _n == "hunyuan":
            _rope_fn = sys.modules[_n].build_batch_2d_rope
            break
    cos, sin = _rope_fn(
        seq_len=input_ids.shape[1],
        n_elem=model.config.attention_head_dim,
        image_infos=[[(slice(img_start, img_start + TOKEN_LENGTH), (64, 64))]],
        device=device,
        base=model.config.rope_theta,
    )
    cos, sin = cos.to(device), sin.to(device)

    prof = None
    if args.profile:
        prof = _start_profile()

    def _make_pipeline_kwargs():
        return {
            "batch_size": 1,
            "image_size": IMAGE_SIZE,
            "num_inference_steps": 1,
            "guidance_scale": 1.0,
            "model_kwargs": {
                "input_ids": input_ids,
                "tokenizer_output": tokenizer_output,
                "batch_gen_image_info": [img_info],
                "position_ids": position_ids,
                "mode": "gen_image",
                "do_sample": False,
                "custom_pos_emb": (cos, sin),
                "gen_timestep_scatter_index": tokenizer_output.gen_timestep_scatter_index,
                "image_mask": tokenizer_output.gen_image_mask,
            },
            "output_type": "latent" if args.skip_vae else "pil",
        }

    logger.warning("Warmup (1 step):")
    _ctx = sdpa_kernel([SDPBackend.MATH]) if args.compile else contextlib.nullcontext()
    with torch.no_grad(), _ctx:
        _ = model._pipeline(**_make_pipeline_kwargs())  # pylint: disable=protected-access
    torch.npu.synchronize()

    if args.profile:
        torch.npu.synchronize()
        prof.start()

    logger.warning("Timed (1 step):")
    torch.npu.synchronize()
    t0 = time.time()
    _ctx2 = sdpa_kernel([SDPBackend.MATH]) if args.compile else contextlib.nullcontext()
    with torch.no_grad(), _ctx2:
        _ = model._pipeline(**_make_pipeline_kwargs())  # pylint: disable=protected-access
    torch.npu.synchronize()
    logger.warning("Inference time: %.2f ms", (time.time() - t0) * 1000)

    if prof is not None:
        torch.npu.synchronize()
        prof.stop()
        logger.warning("Profile saved to %s", PROFILE_DIR)

    timer.summary()
    logger.warning("HunyuanImage-3.0 dummy weight verification PASSED")


if __name__ == "__main__":
    main()
