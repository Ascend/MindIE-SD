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

"""Shared W8A8 online-quantization helpers for dummy runs (--quant w8a8).

对 dummy run 的 transformer 应用 **W8A8 在线量化**（复用 `mindiesd.quantize()`
在线路径，quantization 模块零改动）。矩阵乘按设备选择量化格式：

- **A5（如 950PR）** -> W8A8-MXFP8（激活 `npu_dynamic_mx_quant` + 权重 MXFP8 +
  `npu_quant_matmul`）
- **A2 / A3（如 910B / 910C）** -> W8A8-INT8（动态 int8 在线量化）

量化范围（Matmul-only，其余向量运算保持 bf16）:

- **Matmul**: 所有 `nn.Linear` 替换为对应的 W8A8 在线量化 Linear
- **GroupMatmul**: 框架层支持 —— MoE grouped MLP 的量化路径
  （`mindiesd.layers.moe`，`set_moe_quant_algo(...)` 触发）；dummy run 各模型无
  MoE 层，不强制命中（以 MoE 单测覆盖）
- **FA**: 暂不处理（dummy run 不挂载 FA 量化）

用法（各 `*_infer.py` 统一）::

    from model.common import apply_w8a8_quant, report_quant_layers
    apply_w8a8_quant(pipe, attrs=("transformer",))
    report_quant_layers(pipe)
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _resolve_w8a8_algorithm():
    """按 NPUDevice 解析 --quant w8a8 的默认算法。

    A5 -> W8A8_MXFP8；A2/A3（910B/910C）-> W8A8_DYNAMIC（int8）。
    """
    from mindiesd.quantization.mode import QuantAlgorithm
    from mindiesd.utils.get_platform import NPUDevice, get_npu_device

    device = get_npu_device()
    if device == NPUDevice.A5:
        logger.warning("  [quant] NPUDevice=%s -> W8A8-MXFP8 (A5)", device.name)
        return QuantAlgorithm.W8A8_MXFP8
    logger.warning("  [quant] NPUDevice=%s -> W8A8-INT8 (dynamic)", device.name)
    return QuantAlgorithm.W8A8_DYNAMIC


def _make_online_config(dtype, algorithm):
    """构造 W8A8(Matmul) 在线量化配置（不挂载 FA 量化）。"""
    from mindiesd.quantization.config import OnlineQuantConfig

    return OnlineQuantConfig(
        quant_type=algorithm,
        timestep_config=None,
    )


def apply_w8a8_quant(pipe, attrs=("transformer",), dtype=torch.bfloat16,
                     fallback_layers=None, algorithm=None):
    """对 pipe 的指定组件应用 W8A8 在线量化（Matmul；FA 暂不处理）。

    Args:
        pipe: diffusers pipeline 或任意 nn.Module 容器。
        attrs: 需要量化的组件属性名（默认只量化 transformer）；
               传入空元组/None 时直接量化 pipe 自身（适用于非 diffusers 容器）。
        dtype: 反量化输出 dtype（默认 bf16，与 --quant bf16 对齐）。
        fallback_layers: 回退到 W16A16 的层模式映射
            （如 Wan `time_embedder` 用 `next(iter(parameters()))` 探测 dtype，
            量化后该模块参数为空会 StopIteration，需回退）。
        algorithm: 显式指定 QuantAlgorithm；None 时按 NPUDevice 自动选择
            （A5 -> W8A8_MXFP8，A2/A3 -> W8A8_DYNAMIC int8）。
    """
    from mindiesd import quantize
    from mindiesd.quantization.mode import QuantAlgorithm

    if algorithm is None:
        algorithm = _resolve_w8a8_algorithm()

    online_config = _make_online_config(dtype, algorithm)
    if fallback_layers:
        online_config.fallback_layers = dict(fallback_layers)

    targets = [(attr, getattr(pipe, attr, None)) for attr in attrs] if attrs else [("self", pipe)]

    quantized = 0
    for attr, mod in targets:
        if mod is None:
            logger.warning("  [quant] component '%s' not found, skipped", attr)
            continue
        quantize(mod, online_config=online_config, dtype=dtype)
        # 稳定性修复: 量化层 __init__ 把 bias 转成 fp32（精度考虑），与 bf16 基座的
        # Dynamo guard 不一致，导致 compile 每次调用重编译（实测 ~1.8s/次，wall 里
        # 出现单个 ~1.8s 设备空闲间隙）。对齐回 bf16 后 guard 稳定。
        _align_bias_dtype(mod, dtype)
        quantized += sum(1 for _ in mod.modules() if _is_quant_linear(_))

    logger.warning(
        "W8A8 quantization applied (%s): %d linear layer(s) quantized "
        "(FA quantization not enabled)",
        algorithm.value if isinstance(algorithm, QuantAlgorithm) else algorithm,
        quantized,
    )
    return quantized


def apply_mxfp8_quant(pipe, attrs=("transformer",), dtype=torch.bfloat16, fallback_layers=None):
    """兼容入口：强制 W8A8-MXFP8（历史调用方/实验脚本使用）。"""
    from mindiesd.quantization.mode import QuantAlgorithm

    return apply_w8a8_quant(pipe, attrs=attrs, dtype=dtype,
                            fallback_layers=fallback_layers,
                            algorithm=QuantAlgorithm.W8A8_MXFP8)


def _align_bias_dtype(module, dtype):
    """把量化层模块树的 bias 对齐到 dtype（bf16），保证 Dynamo guard 稳定。

    量化层 __init__ 会把 bias 转成 fp32；若 base dtype 是 bf16，compile 图 guard
    期望 bf16，fp32 bias 会在每次调用时触发重编译（~1.8s/次）。
    """
    aligned = 0
    for m in module.modules():
        b = getattr(m, "bias", None)
        if b is not None and b.dtype != dtype:
            m.bias = b.to(dtype)
            aligned += 1
    if aligned:
        logger.warning("  [quant] aligned %d bias buffer(s) to %s", aligned, dtype)


def _is_quant_linear(module):
    from mindiesd.quantization.layer import (
        W8A8MXFP8OnlineQuantLinear,
        W8A8OnlineQuantLinear,
        _OnlineQuantLinearBase,
    )

    return isinstance(module, (W8A8MXFP8OnlineQuantLinear, W8A8OnlineQuantLinear,
                               _OnlineQuantLinearBase))


def report_quant_layers(pipe, attrs=("transformer",)):
    """汇总量化命中情况（日志输出，供验证确认量化层生效）。"""
    for attr in attrs:
        mod = getattr(pipe, attr, None) if attr != "self" else pipe
        if mod is None:
            continue
        linear_total = sum(1 for _ in mod.modules() if isinstance(_, torch.nn.Linear))
        quant_total = sum(1 for _ in mod.modules() if _is_quant_linear(_))
        logger.warning(
            "  [quant] %s: %d quant linear / %d remaining nn.Linear",
            attr,
            quant_total,
            linear_total,
        )
