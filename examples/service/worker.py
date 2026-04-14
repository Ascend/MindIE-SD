#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import argparse
import datetime
import logging
import os
import sys
import time
from typing import Optional

from PIL import Image
import ray
import torch
import torch.distributed as dist
import torch_npu
from pydantic import BaseModel


import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.distributed.parallel_mgr import ParallelConfig, init_parallel_env
from wan.distributed.tp_applicator import TensorParallelApplicator
from wan.utils.utils import save_video
from mindiesd import CacheConfig, CacheAgent
from request import GeneratorRequest

torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False

# 常量定义
ATTENTION_CACHE_METHOD = 'attention_cache'

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt": (
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
            "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
            "Blurred beach scenery forms the background featuring crystal-clear waters, "
            "distant green hills, and a blue sky dotted with white clouds. "
            "The cat assumes a naturally relaxed posture, as if savoring the sea breeze "
            "and warm sunlight. A close-up shot highlights the feline's intricate details "
            "and the refreshing atmosphere of the seaside."
        ),
        "image": "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


@ray.remote(resources={"NPU": 1})
class GeneratorWorker:
    def __init__(self, args, rank: int, world_size: int):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["ALGO"] = "1"
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = 'expandable_segments:True'
        os.environ["TASK_QUEUE_ENABLE"] = "2"
        os.environ["CPU_AFFINITY_CONF"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
                
        self.rank = rank
        self.world_size = world_size
        self._init_logging(rank)
        self.initialize_model(args)


    @classmethod
    def _init_logging(cls, rank):
        if rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)])
        else:
            logging.basicConfig(level=logging.ERROR)


    def initialize_model(self, args):
        self.args = args
        cfg = self._init_parallel_env(args)

        rainfusion_config = {
            "sparsity": args.sparsity,
            "skip_timesteps": args.sparse_start_step,
            "grid_size": None,
            "atten_mask_all": None,
            "type": args.rainfusion_type
        }

        if dist.is_initialized():
            base_seed = [args.base_seed] if self.rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0]
        logging.info("Model initialization completed")

        if "t2v" in args.task:
            self._init_t2v_pipeline(args, cfg, rainfusion_config)
        else:
            self._init_i2v_pipeline(args, cfg, rainfusion_config)

    def generate(self, request: GeneratorRequest):
        stream = torch.npu.Stream()
        stream.synchronize()
        start_time = time.time()
        request_info = {
            "task": request.task,
            "prompt": request.prompt,
            "size": request.size,
            "steps": request.sample_steps,
            "frame_num": request.frame_num,
            "shift": request.sample_shift,
            "sample_solver": request.sample_solver,
            "sampling_steps": request.sample_steps,
            "guide_scale": request.sample_guide_scale,
            "seed": request.base_seed,
            "offload_model": request.offload_model
        }
        logging.info(f"request: {request_info}")
        img = None
        if request.image is not None:
            img = Image.open(request.image).convert("RGB")
            logging.info(f"Input image: {request.image}")
        
        rainfusion_config = {
            "sparsity": self.sparsity,
            "skip_timesteps": self.sparse_start_step,
            "grid_size": None,
            "atten_mask_all": None,
            "type": self.rainfusion_type
        }
        if self.use_rainfusion:
            if self.dit_fsdp:
                self.pipe.low_noise_model._fsdp_wrapped_module.rainfusion_config = rainfusion_config
                self.pipe.high_noise_model._fsdp_wrapped_module.rainfusion_config = rainfusion_config
            else:
                self.pipe.low_noise_model.rainfusion_config = rainfusion_config
                self.pipe.high_noise_model.rainfusion_config = rainfusion_config
        self.pipe.low_noise_model.freqs_list = None
        self.pipe.high_noise_model.freqs_list = None
        logging.info(f"freqs_list: {self.pipe.low_noise_model.freqs_list}")
        if "t2v" in request.task:
            video = self.pipe.generate(
                request.prompt,
                size=SIZE_CONFIGS[request.size],
                frame_num=request.frame_num,
                shift=request.sample_shift,
                sample_solver=request.sample_solver,
                sampling_steps=request.sample_steps,
                guide_scale=request.sample_guide_scale,
                seed=request.base_seed,
                offload_model=request.offload_model
            )
        else:
            if img is None:
                raise ValueError("Image is required for i2v generation.")
            video = self.pipe.generate(
                request.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[request.size],
                frame_num=request.frame_num,
                shift=request.sample_shift,
                sample_solver=request.sample_solver,
                sampling_steps=request.sample_steps,
                guide_scale=request.sample_guide_scale,
                seed=request.base_seed,
                offload_model=request.offload_model)
        stream.synchronize()
        elapsed_time = time.time() - start_time
        if self.rank == 0:
            if request.save_disk_path is None:
                formatted_time = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
                formatted_prompt = request.prompt.replace(" ", "_").replace("/", "_")[:50]
                suffix = '.mp4'
                size_format = request.size.replace('*', 'x') if sys.platform == 'win32' else request.size
                request.save_disk_path = f"{size_format}_{formatted_prompt}_{formatted_time}{suffix}"

                logging.info(f"Saving generated video to {request.save_disk_path}")
            save_video(
                tensor=video[None],
                save_file=request.save_disk_path,
                fps=request.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        del video
        return {
            "message": "Video generated successfully",
            "elapsed_time": f"{elapsed_time:.2f} sec",
            "output": request.save_disk_path
        }


    def _init_common_pipeline(self, args, pipe, rainfusion_config):
        transformer_low = pipe.low_noise_model
        transformer_high = pipe.high_noise_model
        if args.use_rainfusion:
            if args.dit_fsdp:
                transformer_low._fsdp_wrapped_module.rainfusion_config = rainfusion_config
                transformer_high._fsdp_wrapped_module.rainfusion_config = rainfusion_config
            else:
                transformer_low.rainfusion_config = rainfusion_config
                transformer_high.rainfusion_config = rainfusion_config
        self.use_rainfusion = args.use_rainfusion
        self.sparsity = args.sparsity
        self.sparse_start_step = args.sparse_start_step
        self.rainfusion_type = rainfusion_config["type"]

        if args.tp_size > 1:
            logging.info("Initializing Tensor Parallel ...")
            applicator = TensorParallelApplicator(args.tp_size, device_map="cpu")
            applicator.apply_to_model(transformer_low)
            applicator.apply_to_model(transformer_high)
        if args.use_attentioncache:
            config_high = CacheConfig(
                method=ATTENTION_CACHE_METHOD,
                blocks_count=len(transformer_high.blocks),
                steps_count=args.sample_steps,
                step_start=args.start_step,
                step_interval=args.attentioncache_interval,
                step_end=args.end_step
            )
            config_low = CacheConfig(
                method=ATTENTION_CACHE_METHOD,
                blocks_count=len(transformer_low.blocks),
                steps_count=args.sample_steps,
                step_start=args.start_step,
                step_interval=args.attentioncache_interval,
                step_end=args.end_step
            )
        else:
            config_high = CacheConfig(
                method=ATTENTION_CACHE_METHOD,
                blocks_count=len(transformer_high.blocks),
                steps_count=args.sample_steps
            )
            config_low = CacheConfig(
                method=ATTENTION_CACHE_METHOD,
                blocks_count=len(transformer_low.blocks),
                steps_count=args.sample_steps
            )
        cache_high = CacheAgent(config_high)
        cache_low = CacheAgent(config_low)
        if args.dit_fsdp:
            for block in transformer_high._fsdp_wrapped_module.blocks:
                block._fsdp_wrapped_module.cache = cache_high
                block._fsdp_wrapped_module.args = args
            for block in transformer_low._fsdp_wrapped_module.blocks:
                block._fsdp_wrapped_module.cache = cache_low
                block._fsdp_wrapped_module.args = args
        else:
            for block in transformer_high.blocks:
                block.cache = cache_high
                block.args = args
            for block in transformer_low.blocks:
                block.cache = cache_low
                block.args = args


    def _init_t2v_pipeline(self, args, cfg, rainfusion_config):
        logging.info("Creating WanT2V pipeline.")
        self.pipe = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=self.rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_vae_parallel=args.vae_parallel,
        )
        self._init_common_pipeline(args, self.pipe, rainfusion_config)
        logging.info("Warm up 2 steps ...")
        self.pipe.generate(
            EXAMPLE_PROMPT["t2v-A14B"]["prompt"],
            size=SIZE_CONFIGS["1280*720"],
            frame_num=81,
            shift=None,
            sample_solver='unipc',
            sampling_steps=2,
            guide_scale=args.sample_guide_scale,
            seed=0,
            offload_model=args.offload_model)
        logging.info("T2V warmup finished.")


    def _init_i2v_pipeline(self, args, cfg, rainfusion_config):
        logging.info("Creating WanI2V pipeline.")
        self.pipe = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=self.rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_vae_parallel=args.vae_parallel,
        )
        self._init_common_pipeline(args, self.pipe, rainfusion_config)
        logging.info("Warm up 2 steps ...")
        img = Image.open(EXAMPLE_PROMPT["i2v-A14B"]["image"]).convert("RGB")
        self.pipe.generate(
            EXAMPLE_PROMPT["i2v-A14B"]["prompt"],
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=2,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

        logging.info("I2V warmup finished.")


    def _init_parallel_env(self, args):
        if self.world_size > 1:
            torch.npu.set_device(0)

            dist.init_process_group(
                backend="hccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size)

        need_parallel_env = (
            args.cfg_size > 1 or 
            args.ulysses_size > 1 or 
            args.ring_size > 1 or 
            args.tp_size > 1
        )
        if need_parallel_env:
            if args.cfg_size * args.ulysses_size * args.ring_size * args.tp_size != self.world_size:
                product = args.cfg_size * args.ulysses_size * args.ring_size * args.tp_size
                raise ValueError(
                    f"The number of cfg_size, ulysses_size, ring_size and tp_size should be equal to the world size. "
                    f"Got {args.cfg_size} * {args.ulysses_size} * {args.ring_size} * {args.tp_size} = {product}, \
                    expected {self.world_size}"
                )
            sp_degree = args.ulysses_size * args.ring_size
            parallel_config = ParallelConfig(
                sp_degree=sp_degree,
                ulysses_degree=args.ulysses_size,
                ring_degree=args.ring_size,
                tp_degree=args.tp_size,
                use_cfg_parallel=(args.cfg_size == 2),
                world_size=self.world_size,
            )
            init_parallel_env(parallel_config)
        if args.tp_size > 1 and args.dit_fsdp:
            logging.info("DiT using Tensor Parallel, disabled dit_fsdp")
            args.dit_fsdp = False
        self.dit_fsdp = args.dit_fsdp
        cfg = WAN_CONFIGS[args.task]
        if args.ulysses_size > 1:
            if cfg.num_heads % args.ulysses_size != 0:
                raise ValueError(f"`{cfg.num_heads}` cannot be divided evenly by `{args.ulysses_size}`")
        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")
        return cfg


def _parse_args():
    from generate import _parse_args
    return _parse_args()


def _validate_args(request: GeneratorRequest):
    from generate import _validate_args
    _validate_args(request)