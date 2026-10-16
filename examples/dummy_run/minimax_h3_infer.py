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
from model.minimax_h3_model import build_minimax_h3_pipeline

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID_HF = "MiniMaxAI/MiniMax-H3"
MODEL_ID_MS = "MiniMax/MiniMax-H3"
FAST_LAYERS = 2
# 单个 NPU 上 dummy run 使用小画布：MiniMax-H3 对 packed 序列做全自注意力（O(seq^2)），
# 768x1344x124 帧的序列在单卡上不可行；256x384 已足够验证完整架构。
HEIGHT = 256
WIDTH = 384
NUM_FRAMES = 124          # 17 * 7 + 5 = 124，最短 5s @ 24fps
NUM_INFERENCE_STEPS = 2   # MiniMaxH3Scheduler 要求 >= 2（2 步 = 1 次 transformer 前向）
PROMPT = "test"
PROFILE_DIR = "./profile_l1"

_CONFIG_ALLOW = ["*.json", "*.txt", "*.model", "*.py", "tokenizer*"]
_CONFIG_IGNORE = ["*.safetensors", "*.bin", "*.msgpack", "*.ckpt", "*.pth", "*.index.json"]

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description="MiniMax-H3 NPU dummy weight verification")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--config_cache", type=str, default=None)
    parser.add_argument("--num_layers", type=int, default=FAST_LAYERS,
                        help=f"Number of transformer layers (default: {FAST_LAYERS})")
    parser.add_argument("--height", type=int, default=HEIGHT,
                        help=f"Video height, a multiple of 32 (default: {HEIGHT})")
    parser.add_argument("--width", type=int, default=WIDTH,
                        help=f"Video width, a multiple of 32 (default: {WIDTH})")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES,
                        help="Frames to generate at 24fps; snapped to 17*n+5, 5-15s "
                             f"(default: {NUM_FRAMES})")
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS,
                        help=f"Denoising steps, at least 2 (default: {NUM_INFERENCE_STEPS})")
    parser.add_argument("--compile", action="store_true",
                        help="Enable MindieSDBackend compilation")
    parser.add_argument("--profile", action="store_true",
                        help="Enable NPU profiling (level=l1, with_stack=False)")
    parser.add_argument("--compute-precision", type=str, default="bf16",
                        choices=["fp32", "bf16"],
                        help="Compute precision (default bf16). bf16 = MiniMax-H3 DiT "
                             "weights are cast to bf16; the transformer has no fp32 "
                             "forcing (.float()) islands, so the whole DiT block stack "
                             "then runs natively in bf16 (GEMM ~15x faster than fp32), "
                             "with no implicit conversion in compilation. fp32 = original "
                             "fp32 compute.")
    return parser.parse_args()


def _resolve_config(config_cache):
    if config_cache and os.path.isdir(config_cache):
        return config_cache
    # modelscope 优先：MiniMax-H3 在 HF 上为 gated 模型，modelscope 镜像无需鉴权
    try:
        from modelscope import snapshot_download

        cache_dir = snapshot_download(
            MODEL_ID_MS,
            allow_patterns=_CONFIG_ALLOW,
            ignore_patterns=_CONFIG_IGNORE,
            max_workers=1,
        )
        logger.warning("Config downloaded from modelscope: %s", cache_dir)
        return cache_dir
    except Exception as exc:
        logger.warning("modelscope download failed (%s), falling back to HF", exc)
    return resolve_config_path(None, MODEL_ID_HF)


class _CompiledDiT(torch.nn.Module):
    """Keep the original forward signature on a torch.compile'd DiT.

    ``torch.compile`` wraps ``forward`` as ``(*args, **kwargs)``; the MiniMax-H3
    denoise block filters ``denoiser_input_fields`` through
    ``inspect.signature(transformer.forward)``, so without this wrapper the five
    row-index arguments (``token_tags``/``position_ids``/``video_indices``/
    ``audio_indices``/``text_indices``) would be dropped and forward fails.
    """

    def __init__(self, compiled):
        super().__init__()
        self._compiled = compiled
        # Pipeline properties (patch_size, canvas_multiple, ...) read
        # `transformer.config`; expose it so they keep working after registration.
        self.config = compiled.config

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
    ):
        return self._compiled(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )


def _apply_mindie_compile(pipe):
    from mindiesd.compilation import MindieSDBackend

    try:
        compiled = torch.compile(pipe.transformer, backend=MindieSDBackend())
        pipe.register_components(transformer=_CompiledDiT(compiled))
        logger.warning("transformer compiled with MindieSDBackend")
    except Exception as exc:  # pragma: no cover - best effort on a new architecture
        logger.warning("Compilation failed (%s), running eager", exc)


# ---------------------------------------------------------------------------
# Compute precision (default bf16).
#
# MiniMax-H3's DiT has no fp32-forcing `.float()` islands: every projection and
# norm aligns its inputs with `get_parameter_dtype(...)`, and `_apply_rotary_emb`
# casts the rope cos/sin to the hidden dtype. So with --compute-precision bf16
# the precision is handled at the MODEL level (no implicit conversion in
# compilation):
#   * transformer / text_encoder / VAE weights and buffers are cast to bf16;
#   * a `Tensor.float` patch keeps the eager parts from materializing fp32.
# Dynamo then traces a genuinely bf16 graph and MindieSDBackend performs no
# implicit precision conversion (ComputeDtypePass is a no-op on a clean bf16
# graph; per compilation-dev skill, compile-side casting would insert _to_copy
# between ops and break pattern matching).
# ---------------------------------------------------------------------------

_MINDIE_COMPUTE_DTYPE = torch.bfloat16
_ORIG_TENSOR_FLOAT = torch.Tensor.float


def _install_float_dtype_patch():
    """Make every `Tensor.float()` return bf16 for the eager parts."""
    def _patched_float(self):
        if torch.bfloat16 == _MINDIE_COMPUTE_DTYPE:
            return self.to(torch.bfloat16)
        return _ORIG_TENSOR_FLOAT(self)

    torch.Tensor.float = _patched_float


def _apply_compute_precision(pipe, precision):
    """Apply the configured compute precision at the model level.

    bf16: cast the DiT/text-encoder/VAE weights to bf16 and patch `Tensor.float`
    for the eager parts. fp32: leave weights untouched.
    """
    global _MINDIE_COMPUTE_DTYPE
    _MINDIE_COMPUTE_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    if precision == "bf16":
        for attr in ("transformer", "text_encoder", "vae", "audio_vae"):
            t = getattr(pipe, attr, None)
            if t is not None:
                t.to(torch.bfloat16)
        _install_float_dtype_patch()
    logger.warning("Compute precision: %s (model-level; DiT weights %s, "
                   "no implicit conversion in compilation)",
                   precision, _MINDIE_COMPUTE_DTYPE)


_COMPUTE_OP_KEYWORDS = ("addmm", "mm.", "bmm", "linear", "fusion_attention",
                        "layer_norm", "gelu", "softmax", "matmul", "convolution",
                        "dot", "rms_norm")


def _verify_compute_precision_graph():
    """Walk every compiled graph and flag compute ops with fp32/int32 inputs."""
    from mindiesd.compilation import MindieSDBackend

    findings = []

    def node_dtype(node):
        meta = node.meta.get("tensor_meta") or node.meta.get("val")
        return getattr(meta, "dtype", None)

    _orig_call = MindieSDBackend.__call__

    def patched_call(self, graph, example_inputs):
        for node in graph.graph.nodes:
            if node.op != "call_function":
                continue
            tgt = str(node.target)
            if not any(k in tgt for k in _COMPUTE_OP_KEYWORDS):
                continue
            for arg in node.args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                dt = node_dtype(arg)
                if dt in (torch.float32, torch.int32):
                    findings.append((node.name, tgt[:50], str(dt)))
        return _orig_call(self, graph, example_inputs)

    MindieSDBackend.__call__ = patched_call
    return findings


def _hook_text_encoder_model(pipe, timer):
    """Hook the text-encoder submodule.

    The MiniMax-H3 text-encoder block drives `text_encoder.model` (submodule) directly
    instead of the top-level forward, so the phase timer's hooks on `pipe.text_encoder`
    never fire.
    """
    mod = pipe.text_encoder.model
    timer._handles.append(  # pylint: disable=protected-access
        mod.register_forward_pre_hook(lambda _m, _in, _n="text_encoder": timer._on_pre(_n))  # noqa: B023
    )
    timer._handles.append(  # pylint: disable=protected-access
        mod.register_forward_hook(lambda _m, _in, _out, _n="text_encoder": timer._on_post(_n))  # noqa: B023
    )


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
    device = f"npu:{device_id}"
    logger.warning("Using device: %s", device)

    config_dir = _resolve_config(args.config_cache)
    logger.warning("Using config from: %s", config_dir)

    timer = _PhaseTimer(device_id=device_id)
    timer.start_build()

    logger.warning("Building MiniMax-H3 (%d blocks) ...", args.num_layers)
    pipe = build_minimax_h3_pipeline(config_dir, num_layers=args.num_layers,
                                     device=device, timer=timer)

    t0 = time.time()
    pipe.to(device)
    timer.record_build("Move to device", time.time() - t0)

    # compute precision (default bf16): model-level weights cast; the compiled
    # transformer then traces a genuinely bf16 graph (no implicit conversion).
    _cp_findings = _verify_compute_precision_graph()
    t0 = time.time()
    _apply_compute_precision(pipe, args.compute_precision)
    timer.record_build(f"Apply compute precision ({args.compute_precision})", time.time() - t0)

    if args.compile:
        t0 = time.time()
        _apply_mindie_compile(pipe)
        timer.record_build("Compilation", time.time() - t0)

    timer.install(pipe)
    _hook_text_encoder_model(pipe, timer)

    def _run():
        return pipe(
            prompt=PROMPT,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            output=["latents", "audio_latents"],
        )

    logger.warning("Warmup (%d steps):", args.num_inference_steps)
    with torch.no_grad():
        state = _run()
    torch.npu.synchronize()
    timer.capture_warmup()

    if args.compile and args.compute_precision != "fp32":
        if _cp_findings:
            logger.warning("Compute-precision verification FAILED: %d fp32/int32 "
                           "compute-input violation(s), first 10=%s",
                           len(_cp_findings), _cp_findings[:10])
        else:
            logger.warning("Compute-precision verification PASSED: "
                           "no fp32/tf32/int32 compute nodes in the compiled graph")

    prof = None
    if args.profile:
        prof = _start_profile()

    logger.warning("Timed (%d steps):", args.num_inference_steps)
    torch.npu.synchronize()
    t0 = time.time()
    with torch.no_grad():
        state = _run()
    torch.npu.synchronize()
    logger.warning("Inference time: %.2f ms", (time.time() - t0) * 1000)
    timer.capture_timed()

    latents = state["latents"]
    audio_latents = state["audio_latents"]
    logger.warning("Video latents: %s %s", tuple(latents.shape), latents.dtype)
    logger.warning("Audio latents: %s %s", tuple(audio_latents.shape), audio_latents.dtype)

    if prof is not None:
        torch.npu.synchronize()
        prof.stop()
        logger.warning("Profile saved to %s", PROFILE_DIR)

    timer.summary()
    logger.warning("MiniMax-H3 dummy weight verification PASSED")


if __name__ == "__main__":
    main()
