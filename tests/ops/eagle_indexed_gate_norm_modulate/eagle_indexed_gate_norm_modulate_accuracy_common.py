#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import os
import random

import torch

SEED = 20261024
NAMES = ("residual", "gate", "branch", "weight", "shift", "scale", "indices")
OUTPUT_NAMES = ("residual_out", "modulated_out")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
PROD_SHAPES = {13640: 3, 5672: 12}  # Number of residual rows -> number of indexed rows.
# Explicit cases include distribution; do not expand another Cartesian product.
PROD_SCENARIOS = (
    (13640, 3, 2048, "bf16", "bf16", "normal"),
    (13640, 3, 5376, "bf16", "bf16", "uniform"),
    (13640, 3, 6656, "bf16", "bf16", "normal"),
    (13640, 3, 7168, "bf16", "bf16", "uniform"),
    (5672, 12, 2048, "bf16", "bf16", "uniform"),
    (5672, 12, 5376, "bf16", "bf16", "normal"),
    (5672, 12, 6656, "bf16", "bf16", "uniform"),
    (5672, 12, 7168, "bf16", "bf16", "normal"),
    (5672, 12, 6656, "fp16", "bf16", "uniform"),
    (13640, 3, 7168, "bf16", "fp16", "normal"),
)


def init_device(device_id):
    if device_id < 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("Use a nonnegative physical device ID; unset ASCEND_RT_VISIBLE_DEVICES")
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    import mindiesd  # noqa: F401

    return f"npu:{device_id}"


def make_inputs(
    rows, hidden, dtype, device, distribution, seed=SEED, index_dtype=torch.int64, strided=False, indexed_rows=3
):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    shapes = (
        (rows, hidden),
        (indexed_rows, hidden),
        (rows, hidden),
        (hidden,),
        (indexed_rows, hidden),
        (indexed_rows, hidden),
    )
    result = []
    for name, shape in zip(NAMES, shapes):
        # Directly allocate the requested physical stride, then fill the actual input view.
        stride = (hidden + 7, 1) if len(shape) == 2 else (2,)
        value = (
            torch.empty_strided(shape, stride, dtype=dtype, device=device)
            if strided
            else torch.empty(shape, dtype=dtype, device=device)
        )
        if distribution == "normal":
            mean, std = rng.uniform(-5, 5), rng.uniform(0.1, 2)
            value.normal_(mean, std)
            print(f"{name}: distribution=normal seed={seed} mean={mean:.9g} std={std:.9g} unclamped=True")
        elif distribution == "uniform":
            value.uniform_(-5, 5)
            print(f"{name}: distribution=uniform seed={seed} low=-5 high=5")
        else:
            raise ValueError(f"Unknown distribution {distribution}")
        result.append(value)
    indices = torch.empty_strided((rows,), (2 if strided else 1,), dtype=index_dtype, device=device)
    indices.copy_(torch.arange(rows, dtype=index_dtype, device=device) % indexed_rows)
    print(f"indices: cyclic row%{indexed_rows}; includes first/last indexed row when rows>=K, repeats when rows>K")
    return result[0], result[1], result[2], result[3], result[4], result[5], indices


def call_dut(inputs, output_dtype, eps=1e-6, log=True):
    if log:
        for name, value in zip(NAMES, inputs):
            print(
                f"{name}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device} "
                f"stride={value.stride()} contiguous={value.is_contiguous()}"
            )
        print(
            f"layout=ND norm_type=0 eps={eps} output_dtype={output_dtype} "
            "DUT=torch.ops.mindiesd.eagle_indexed_gate_norm_modulate"
        )
    residual, gate, branch, weight, shift, scale, indices = inputs
    outputs = torch.ops.mindiesd.eagle_indexed_gate_norm_modulate(
        residual, gate, branch, shift, scale, indices, weight=weight, norm_type=0, eps=eps, output_dtype=output_dtype
    )
    assert len(outputs) == 2, "Expected exactly two DUT outputs"
    expected_dtype = output_dtype if output_dtype is not None else inputs[0].dtype
    for name, value in zip(OUTPUT_NAMES, outputs):
        assert value.shape == inputs[0].shape and value.dtype == expected_dtype, (name, value.shape, value.dtype)
        assert value.is_contiguous(), f"{name}: DUT output must be contiguous"
        if log:
            print(
                f"{name}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device} "
                f"stride={value.stride()} contiguous={value.is_contiguous()}"
            )
    return outputs


def golden(inputs, output_dtype, eps=1e-6):
    """FP32 phases on the input device; low precision only at the two output stores."""
    residual, gate, branch, weight, shift, scale, indices = inputs
    index = indices.long()
    updated = residual.float() + gate.float()[index] * branch.float()
    variance = (updated * updated).sum(-1, keepdim=True) / residual.shape[-1]
    normalized = updated * torch.rsqrt(variance + eps) * weight.float()
    output = normalized * (1.0 + scale.float()[index]) + shift.float()[index]
    return updated.to(output_dtype), output.to(output_dtype)


def _coordinate(index, shape):
    coords = []
    for size in reversed(shape):
        index, remainder = divmod(index, size)
        coords.append(remainder)
    return tuple(reversed(coords))


def compare(actual, expected, large=False):
    assert len(actual) == len(expected) == 2, "Expected exactly two outputs"
    for name, out, ref in zip(OUTPUT_NAMES, actual, expected):
        assert out.shape == ref.shape and out.dtype == ref.dtype, (name, out.shape, out.dtype, ref.shape, ref.dtype)
        assert out.is_contiguous(), f"{name}: output must be contiguous"
        if not out.numel():
            print(f"{name}: empty output shape/dtype/contiguity PASS")
            continue
        # Only completed outputs move to CPU: large-shape golden computation stays on NPU.
        a, g = out.cpu().to(torch.float64).flatten(), ref.cpu().to(torch.float64).flatten()
        assert bool(torch.isfinite(a).all() & torch.isfinite(g).all()), f"{name}: unexpected NaN/Inf"
        atol = rtol = 2 ** (-10 if out.dtype == torch.float16 else -7)
        diff = (a - g).abs()
        worst = int(diff.argmax().item())
        max_abs, g_worst = float(diff[worst].item()), float(g[worst].item())
        info = torch.finfo(out.dtype)
        exponent = math.floor(math.log2(abs(g_worst))) if abs(g_worst) >= info.tiny else int(math.log2(info.tiny))
        ulp = math.ldexp(info.eps, exponent)
        limit = max(0.1, 32 * ulp)
        matched = float((diff <= atol + rtol * g.abs()).float().mean().item())
        an, gn = float(torch.linalg.vector_norm(a).item()), float(torch.linalg.vector_norm(g).item())
        ratio = an / gn if gn else (1.0 if an == 0 else math.inf)
        cosine = float(torch.dot(a, g).item()) / (an * gn) if an and gn else float(an == gn)
        mixed = matched >= 0.99 and max_abs <= limit
        print(
            f"{name}: output_dtype={out.dtype} tolerance_dtype={out.dtype} metrics=CPU_FP64 "
            f"reason=FP32_compute_then_output_round atol={atol} rtol={rtol} required_matched_ratio=0.99 "
            f"absolute_floor=0.1 matched={matched:.9g} matched_ok={matched >= 0.99} max_abs={max_abs:.9g} "
            f"max_abs_ok={max_abs <= limit} index={_coordinate(worst, out.shape)} "
            f"actual={float(a[worst].item()):.9g} golden={g_worst:.9g} "
            f"ULP={ulp:.9g} ULP32={32 * ulp:.9g} limit={limit:.9g} "
            f"mixed_{'soft' if large else 'hard'}={mixed} cosine={cosine:.9g} cosine_min=0.99 "
            f"cosine_hard={cosine >= 0.99} norm_ratio={ratio:.9g} norm_range=[0.9,1.1] "
            f"norm_hard={0.9 <= ratio <= 1.1}"
        )
        assert cosine >= 0.99, f"{name}: cosine={cosine} < 0.99"
        assert 0.9 <= ratio <= 1.1, f"{name}: norm_ratio={ratio} outside [0.9,1.1]"
        if not large:
            assert matched >= 0.99, f"{name}: matched_ratio={matched} < 0.99"
            assert max_abs <= limit, f"{name}: max_abs={max_abs} > {limit}"


def scenario_name(hidden, input_dtype, output_dtype, rows=13640, indexed_rows=3):
    return f"fused_n{rows}_k{indexed_rows}_d{hidden}_{input_dtype}_to_{output_dtype}_normal"
