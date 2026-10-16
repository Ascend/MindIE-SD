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

"""Shared compute-precision helpers for dummy runs (--quant bf16|fp32).

机制（与 wan/minimax 一致，模型级精度、编译侧零隐式转换）:
- bf16: 权重 cast 到 bf16 + 把 diffusers 前向里的 `.float()` 精度岛改写为 `.to(bf16)`
  （apply_rotary_emb 的 x/cos/sin、apply_rotary_emb_qwen 的 complex 旋转改实数域等价），
  保证 Dynamo trace 出的图是真正 bf16，编译侧不再有隐式 fp32 提升/降回
  （否则 pattern 匹配会被 _to_copy 打断）。
- fp32: 原行为（权重 fp32，不做任何改写）。
"""

import inspect
import logging

import torch

logger = logging.getLogger(__name__)

_MINDIE_COMPUTE_DTYPE = torch.bfloat16
_ORIG_TENSOR_FLOAT = torch.Tensor.float


def compute_dtype_from_precision(precision):
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]


def _rewrite_apply_rotary_emb(compute_dtype):
    """把 diffusers `apply_rotary_emb` 的 fp32 精度岛改写为目标 dtype（源码级）。

    Dynamo trace 时绕过 `torch.Tensor.float` patch，必须改写源码才能让图真正 bf16
    （wan 时期教训）。FLUX / Qwen-Image 共用该函数（`out = (x.float() * cos +
    x_rotated.float() * sin).to(x.dtype)`），cos/sin 在 NPU 上为 fp32（FluxPosEmbed
    freqs_dtype=fp32），若只改 x 不改 cos/sin，乘法仍会被提升回 fp32。
    """
    from diffusers.models import embeddings

    embeddings._mindie_compute_dtype = compute_dtype

    src = inspect.getsource(embeddings.apply_rotary_emb)
    if ".float()" not in src:
        logger.debug("apply_rotary_emb has no .float() site, skip rewrite")
        return

    # 只改写 use_real 分支（FLUX/Qwen 路径）；view_as_complex 分支依赖 fp32 复数，保持原样。
    new_src = src.replace(
        "cos, sin = cos.to(x.device), sin.to(x.device)",
        "cos, sin = cos.to(x.device), sin.to(x.device)\n"
        "        cos = cos.to(_mindie_compute_dtype)\n"
        "        sin = sin.to(_mindie_compute_dtype)",
    ).replace(
        "out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)",
        "out = (x * cos + x_rotated * sin).to(x.dtype)",
    )

    module_ns = embeddings.apply_rotary_emb.__globals__
    module_ns["_mindie_compute_dtype"] = compute_dtype

    # 命中校验：diffusers 源码布局变化时 `.replace` 会静默 no-op，导致 cos/sin 未 cast、
    # fp32 精度岛残留且验证工具检测不到。必须确认两个替换都实际生效。
    if new_src == src or ".to(_mindie_compute_dtype)" not in new_src:
        raise RuntimeError(
            "apply_rotary_emb rewrite did not match installed diffusers source "
            "(cos/sin cast not inserted; .float() removal not applied). "
            "Update _rewrite_apply_rotary_emb for the installed diffusers version."
        )
    exec(compile(new_src, "<mindie-compute-dtype-apply_rotary_emb>", "exec"), module_ns)  # noqa: S102
    embeddings.apply_rotary_emb = module_ns["apply_rotary_emb"]


def _rewrite_apply_rotary_emb_qwen(compute_dtype):
    """Qwen-Image 的 complex 旋转(use_real=False)改写为实数域等价形式(bf16 安全)。

    原实现 `view_as_complex(x.float()) * freqs_cis` 依赖 fp32 复数;
    Tensor.float patch 会让 `x.float()` 返回 bf16, view_as_complex(bf16) 直接报错。
    实数域等价: (xr+i*xi)*(cos+i*sin) -> out_real=xr*cos-xi*sin, out_imag=xr*sin+xi*cos,
    cos/sin 显式对齐 x.dtype, 全部 bf16 计算, 同时为 qwen rope 融合 pattern 铺路。
    """
    from diffusers.models.transformers import transformer_qwenimage as tq

    func = tq.apply_rotary_emb_qwen
    src = inspect.getsource(func)
    old_block = (
        "        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))\n"
        "        freqs_cis = freqs_cis.unsqueeze(1)\n"
        "        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)\n"
        "\n"
        "        return x_out.type_as(x)"
    )
    if old_block not in src:
        # bf16 模式下该改写是必需项：不改写则 Tensor.float patch 使 view_as_complex(bf16) 崩溃，
        # 或 fp32 复数岛残留。diffusers 版本变化导致失配时必须显式失败，而非静默跳过。
        raise RuntimeError(
            "apply_rotary_emb_qwen complex block not found in installed diffusers; "
            "rewrite is required for bf16. "
            "Update _rewrite_apply_rotary_emb_qwen for the installed diffusers version."
        )
    new_block = (
        "        xr, xi = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)\n"
        "        cos = freqs_cis.real.unsqueeze(1).to(x.dtype)\n"
        "        sin = freqs_cis.imag.unsqueeze(1).to(x.dtype)\n"
        "        out_real = xr * cos - xi * sin\n"
        "        out_imag = xr * sin + xi * cos\n"
        "        x_out = torch.stack([out_real, out_imag], dim=-1).flatten(3)\n"
        "\n"
        "        return x_out.type_as(x)"
    )
    new_src = src.replace(old_block, new_block)
    module_ns = func.__globals__
    exec(compile(new_src, "<mindie-compute-dtype-apply_rotary_emb_qwen>", "exec"), module_ns)  # noqa: S102
    tq.apply_rotary_emb_qwen = module_ns["apply_rotary_emb_qwen"]


def _install_tensor_float_patch(compute_dtype):
    """eager 路径兜底: 让 `Tensor.float()` 返回 compute_dtype（minimax 方案）。"""
    global _MINDIE_COMPUTE_DTYPE
    _MINDIE_COMPUTE_DTYPE = compute_dtype

    def _patched_float(self):
        if torch.bfloat16 == _MINDIE_COMPUTE_DTYPE:
            return self.to(torch.bfloat16)
        return _ORIG_TENSOR_FLOAT(self)

    torch.Tensor.float = _patched_float


def apply_compute_precision(
    pipe, precision, attrs=("transformer", "transformer_2", "text_encoder")
):
    """应用模型级计算精度。

    bf16: 组件权重 cast bf16 + `.float()` 岛改写（apply_rotary_emb 源码级 + Tensor.float 兜底）。
    fp32: 原行为，不改写。
    """
    compute_dtype = compute_dtype_from_precision(precision)
    if precision == "bf16":
        for attr in attrs:
            mod = getattr(pipe, attr, None)
            if mod is not None:
                mod.to(compute_dtype)
        _rewrite_apply_rotary_emb(compute_dtype)
        _rewrite_apply_rotary_emb_qwen(compute_dtype)
        _install_tensor_float_patch(compute_dtype)
    logger.warning(
        "Compute precision: %s (model-level; weights %s, no implicit conversion in compilation)",
        precision,
        compute_dtype,
    )
    return compute_dtype


_COMPUTE_OP_KEYWORDS = ("addmm", "mm.", "bmm", "linear", "fusion_attention",
                        "layer_norm", "gelu", "softmax", "matmul", "convolution",
                        "dot", "rms_norm", "mul", "add", "sub", "div", "pow",
                        "sqrt", "rsqrt")


def verify_compute_precision_graph():
    """Walk every compiled graph and flag compute ops with fp32/int32 inputs.

    返回 findings 列表; 空列表 = 编译图无 fp32 计算节点（图真正 bf16）。
    """
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
