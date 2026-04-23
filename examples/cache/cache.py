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
import os
import time
from PIL import Image
import torch
from diffusers.utils import logging
from mindiesd import CacheConfig, CacheAgent

# 缓存配置开关（从环境变量读取）
COND_CACHE = bool(int(os.environ.get('COND_CACHE', 0)))
UNCOND_CACHE = bool(int(os.environ.get('UNCOND_CACHE', 0)))

logger = logging.get_logger(__name__)


#解决 diffuser 0.35.1 torch2.1 报错
def custom_op(
    name,
    fn=None,
    /,
    *,
    mutates_args,
    device_types=None,
    schema=None,
    tags=None,
):
    def decorator(func):
        return func

    if fn is not None:
        return decorator(fn)

    return decorator


def register_fake(
    op,
    fn=None,
    /,
    *,
    lib=None,
    _stacklevel: int = 1,
    allow_override: bool = False,
):
    def decorator(func):
        return func

    if fn is not None:
        return decorator(fn)

    return decorator

torch.library.custom_op = custom_op
torch.library.register_fake = register_fake

from qwenimage_edit.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage_edit.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline


def read_prompts(file_path):
    """读取提示词文件（每行一个提示词）"""
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"提示词文件不存在: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if not prompts:
        raise ValueError(f"提示词文件内容为空: {file_path}")
    return prompts


def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用 Qwen-Image-Edit-2509 模型生成编辑图像")

    # 模型配置
    parser.add_argument("--model_path", type=str, default="/home/weight/Qwen-Image-Edit-2509/",
                        help="模型本地路径")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"],
                        help="模型数据类型")
    parser.add_argument("--device", type=str, default="npu", help="运行设备（npu/cuda/cpu）")
    parser.add_argument("--device_id", type=int, default=0, help="设备ID（如昇腾芯片索引）")

    # 输入配置（多图支持，用逗号分隔路径）
    parser.add_argument("--img_paths", type=str, required=True,
                        help="输入图像路径（多图用逗号分隔，如 'img1.png,img2.png'）")
    parser.add_argument("--prompt_file", type=str, default="edit_prompts.txt",
                        help="提示词文件路径（每行一个提示词）")
    parser.add_argument("--negative_prompt_file", type=str, default=None,
                        help="负面提示词文件路径（每行一个）")

    # 推理配置
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="推理步数")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                        help="真实CFG缩放系数")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="引导缩放系数（Qwen特有）")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（确保 reproducibility）")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="每个提示词生成的图像数量")

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="output_images",
                        help="生成图像保存目录")
    # 量化文件
    parser.add_argument(
        "--quant_desc_path",
        type=str,
        default=None,
        help="Path to quantization description file (e.g., quant_model_description_*.json). "
             "Enables quantization if provided (applies to Text Encoder and Transformer)."
    )

    args = parser.parse_args()

    if args.quant_desc_path:
        # 校验文件是否存在
        if not os.path.exists(args.quant_desc_path):
            raise FileNotFoundError(f"Quantization description file not found: {args.quant_desc_path}")
        # 校验文件格式
        if not args.quant_desc_path.endswith(".json") or "quant_model_description" not in args.quant_desc_path:
            raise ValueError(f"Invalid quantization file: {args.quant_desc_path}. "
                                "Expected format: 'quant_model_description_*.json'")

    return args


def main():
    args = _parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备配置
    device = f"{args.device}:{args.device_id}"
    torch.npu.set_device(args.device_id)  # 昇腾设备绑定
    logger.warning(f"使用设备: {device}")

    # 数据类型配置
    torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32

    # 加载模型
    logger.warning(f"从 {args.model_path} 加载模型...")

    transformer = QwenImageTransformer2DModel.from_pretrained(
        os.path.join(args.model_path, 'transformer'),
        torch_dtype=torch_dtype,
        device_map=None,               # 禁用自动设备映射，昇腾环境下默认加载到CPU
        low_cpu_mem_usage=True         # 启用CPU低内存模式，避免加载时CPU内存溢出
    )

    if args.quant_desc_path:
        from mindiesd import quantize
        logger.warning("Quantizing Transformer (单独量化核心组件)...")
        quantize(
            model=transformer,
            quant_des_path=args.quant_desc_path,
            use_nz=True,
        )
        torch.npu.empty_cache()  # 清理NPU显存缓存

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True
    )

    # VAE优化配置（避免显存溢出）
    pipeline.vae.use_slicing = True
    pipeline.vae.use_tiling = True

    # 移动模型到目标设备
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=None)  # 显示进度条

    # 加载并预处理输入图像（多图支持，转换为RGB）
    img_path_list = [p.strip() for p in args.img_paths.split(",")]
    images = []
    for img_path in img_path_list:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        img = Image.open(img_path).convert("RGB")  # 强制转为3通道
        images.append(img)
    logger.warning(f"加载完成 {len(images)} 张输入图像")

    if COND_CACHE or UNCOND_CACHE:
        # 使能DitCache
        cache_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=60,
            steps_count=args.num_inference_steps,
            step_start=10,
            step_interval=3,
            step_end=35,
            block_start=10,
            block_end=50
        )
        pipeline.transformer.cache_cond = CacheAgent(cache_config) if COND_CACHE else None
        pipeline.transformer.cache_uncond = CacheAgent(cache_config) if UNCOND_CACHE else None
        logger.warning("启用缓存配置")

    # 读取提示词和负面提示词
    prompts = read_prompts(args.prompt_file)
    neg_prompts = read_prompts(args.negative_prompt_file) if args.negative_prompt_file else [" "] * len(prompts)
    logger.warning(f"加载完成 {len(prompts)} 个提示词")

    # 推理循环
    total_time = 0.0
    for prompt_idx, (prompt, neg_prompt) in enumerate(zip(prompts, neg_prompts)):
        # 准备输入参数
        inputs = {
            "image": images,
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "generator": torch.Generator(device=device).manual_seed(args.seed),
            "true_cfg_scale": args.true_cfg_scale,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "num_images_per_prompt": args.num_images_per_prompt,
        }

        # 执行推理并计时
        torch.npu.synchronize()  # 昇腾设备同步
        start_time = time.time()

        with torch.inference_mode():
            output = pipeline(**inputs)

        torch.npu.synchronize()
        end_time = time.time()
        infer_time = end_time - start_time
        logger.warning(f"提示词 {prompt_idx + 1}/{len(prompts)} 推理完成，耗时: {infer_time:.2f}秒")

        # 保存生成结果
        for img_idx, img in enumerate(output.images):
            save_path = os.path.join(
                args.output_dir,
                f"edit_result_{prompt_idx}_{img_idx}.png"
            )
            img.save(save_path)
            logger.warning(f"图像保存至: {save_path}")

        # 累计时间（跳过前3次预热）
        if prompt_idx >= 3:
            total_time += infer_time

    # 计算平均时间（排除前3次）
    if len(prompts) > 3:
        avg_time = total_time / (len(prompts) - 3)
        logger.warning(f"排除前3次预热后，平均推理时间: {avg_time:.2f}秒")


if __name__ == "__main__":
    main()