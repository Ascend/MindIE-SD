#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared inputs, FP32 golden, and tolerance gates for rms_norm_indexed_scale_shift accuracy."""

from __future__ import annotations

import math
import os
import random

import torch

# Support namespace-package imports and direct script execution.
# pylint: disable=relative-beyond-top-level
if __package__:
    from .index_modulation_test_config import EPS, SEED
else:
    from index_modulation_test_config import EPS, SEED

PROD_DTYPE = torch.bfloat16

# The output is BF16 with an FP32 reduction/affine in between, so it is judged
# on the BFLOAT16 tier: rtol=atol=2^-7, matched ratio >= 0.99, absolute floor 0.1.
_TOLERANCES = {
    torch.bfloat16: ("BFLOAT16", 2**-7, 2**-7, 0.99, 0.1),
}


def init_device(device_id):
    """Validate device id and set_device before any NPU init."""
    if device_id < 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("Use a nonnegative physical device ID; unset ASCEND_RT_VISIBLE_DEVICES")
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def make_inputs(
    rows,
    hidden_size,
    num_groups,
    dtype,
    device,
    distribution,
    seed=SEED,
    *,
    index_dtype=torch.int64,
    strided=False,
):
    """Build fixed-seed inputs, optionally using offset views of padded/chunked storage."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    print(f"distribution={distribution} seed={seed}")

    def sample(name, dims):
        if distribution == "uniform":
            result = torch.empty(dims, dtype=dtype, device=device).uniform_(-5, 5)
        elif distribution == "normal":
            mean, std = rng.uniform(-5, 5), rng.uniform(0.1, 2)
            print(f"{name}: mean={mean:.9g} std={std:.9g}")
            result = torch.empty(dims, dtype=dtype, device=device).normal_(mean, std)
        else:
            raise ValueError(distribution)
        return result

    x = sample("x", (rows, hidden_size))
    weight = sample("weight", (hidden_size,))
    shift = sample("shift", (num_groups, hidden_size))
    scale = sample("scale", (num_groups, hidden_size))
    indices = torch.randint(num_groups, (rows,), dtype=index_dtype, device=device)
    if strided:
        x_storage = torch.full((rows, hidden_size + 7), float("nan"), dtype=dtype, device=device)
        x_view = x_storage[:, 3 : 3 + hidden_size]
        x_view.copy_(x)
        x = x_view
        tables = []
        for value, chunk in ((shift, 1), (scale, 4)):
            storage = torch.full((num_groups, 6 * hidden_size), float("nan"), dtype=dtype, device=device)
            view = storage.chunk(6, dim=1)[chunk]
            view.copy_(value)
            tables.append(view)
        shift, scale = tables[0], tables[1]
        weight_storage = torch.full((hidden_size + 2,), float("nan"), dtype=dtype, device=device)
        weight_storage[1:-1].copy_(weight)
        weight = weight_storage[1:-1]
        index_storage = torch.full((2 * rows + 1,), -1, dtype=index_dtype, device=device)
        index_storage[1::2].copy_(indices)
        indices = index_storage[1::2]
    return x, weight, shift, scale, indices


def log_inputs(x, weight, shift, scale, indices, eps=EPS):
    """Print the actual state read back from the tensors handed to the DUT."""
    for name, value in (
        ("x", x),
        ("weight", weight),
        ("shift", shift),
        ("scale", scale),
        ("indices", indices),
    ):
        print(
            f"{name}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device} "
            f"layout={value.layout} stride={value.stride()} contiguous={value.is_contiguous()}",
            flush=True,
        )
    print(f"eps={eps} num_groups={shift.shape[0]}", flush=True)


def golden(x, weight, shift, scale, indices, eps=EPS, out_dtype=None):
    """FP32 reference matching the kernel's numeric stages, on the inputs' device."""
    print(f"golden=FP32_RMSNorm_indexed_affine device={x.device} output_dtype={out_dtype or x.dtype}")
    out_dtype = out_dtype or x.dtype
    xf = x.float()
    wf = weight.float()
    indexed_shift = shift.float().index_select(0, indices)
    indexed_scale = scale.float().index_select(0, indices)
    variance = (xf * xf).mean(dim=-1, keepdim=True)
    normalized = xf * torch.rsqrt(variance + eps) * wf
    output = normalized * (1.0 + indexed_scale) + indexed_shift
    return output.to(out_dtype)


def _coordinate(flat_index, shape):
    result = []
    for size in reversed(shape):
        flat_index, index = divmod(flat_index, size)
        result.append(index)
    return tuple(reversed(result))


def compare(actual, expected, *, large=False):
    """Judge output against golden: cosine >= 0.99 and norm ratio in [0.9, 1.1]."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, (
        f"output contract: actual shape={actual.shape} dtype={actual.dtype}; "
        f"expected shape={expected.shape} dtype={expected.dtype}"
    )
    assert actual.is_contiguous(), f"output must be contiguous; stride={actual.stride()}"
    tolerance_dtype, atol, rtol, required, absolute_floor = _TOLERANCES[actual.dtype]
    # CPU FP64 for small tests; large metrics stay on NPU in FP32.
    metric_dtype = torch.float32 if large else torch.float64
    a = actual.to(metric_dtype).flatten()
    g = expected.to(metric_dtype).flatten()
    for name, values in (("golden", g), ("actual", a)):
        finite = torch.isfinite(values)
        if not bool(finite.all()):
            invalid = ~finite
            count = int(invalid.sum().item())
            first = int(invalid.to(torch.int32).argmax().item())
            raise AssertionError(
                f"{name} output: nonfinite valid element; "
                f"shape={tuple(actual.shape)} count={count}/{values.numel()} "
                f"first_index={_coordinate(first, actual.shape)} "
                f"actual={float(a[first].item()):.9g} golden={float(g[first].item()):.9g}; "
                "only logical output elements are checked, not input or storage outside the output view"
            )
    diff = (a - g).abs()
    worst = int(diff.argmax().item())
    max_abs = float(diff[worst].item())
    value = abs(float(g[worst].item()))
    # ULP of the actual output format at the maximum-error reference element.
    info = torch.finfo(actual.dtype)
    exponent = math.floor(math.log2(value)) if value >= info.tiny else math.log2(info.tiny)
    ulp = math.ldexp(info.eps, int(exponent))
    limit = max(absolute_floor, 32 * ulp)
    matched = float((diff <= atol + rtol * g.abs()).to(torch.float32).mean().item())
    an = float(torch.linalg.vector_norm(a).item())
    gn = float(torch.linalg.vector_norm(g).item())
    ratio = an / gn if gn else (1.0 if an == 0 else math.inf)
    cosine = float(torch.dot(a, g).item()) / (an * gn) if an and gn else float(an == gn)
    matched_ok = matched >= required
    max_abs_ok = max_abs <= limit
    mixed_ok = matched_ok and max_abs_ok
    detail = (
        f"output: index={_coordinate(worst, actual.shape)} "
        f"actual={float(a[worst].item()):.9g} golden={float(g[worst].item()):.9g}"
    )
    print(
        f"output: tolerance_dtype={tolerance_dtype} output_dtype={actual.dtype} "
        f"reason='RMSNorm reduction and affine in FP32, final output rounds to {actual.dtype}' "
        f"atol={atol} rtol={rtol} required_matched_ratio={required} absolute_floor={absolute_floor}"
    )
    print(
        f"{detail} ulp={ulp:.9g} ulp32={32 * ulp:.9g} limit={limit:.9g} "
        f"matched={matched:.9g} matched_ok={matched_ok} max_abs={max_abs:.9g} max_abs_ok={max_abs_ok} "
        f"mixed_{'soft' if large else 'hard'}={mixed_ok} "
        f"cosine={cosine:.9g} cosine_min=0.99 cosine_hard={cosine >= 0.99} "
        f"norm_ratio={ratio:.9g} norm_range=[0.9,1.1] norm_hard={0.9 <= ratio <= 1.1}"
    )
    failures = []
    if not matched_ok:
        failures.append(f"matched={matched:.9g} < {required}")
    if not max_abs_ok:
        failures.append(f"max_abs={max_abs:.9g} > {limit:.9g}")
    if failures:
        print(f"{detail}; {'; '.join(failures)}")
    assert cosine >= 0.99, f"{detail}; cosine={cosine:.9g} < 0.99"
    assert 0.9 <= ratio <= 1.1, f"{detail}; norm_ratio={ratio:.9g} outside [0.9, 1.1]"
    if not large:
        assert mixed_ok, f"{detail}; {'; '.join(failures)}"


def check_output_contract(output, inputs):
    """Check the raw DUT tensor before any CPU transfer or metric conversion."""
    x = inputs[0]
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"output: expected one Tensor, got {type(output).__name__}")
    if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device or not output.is_contiguous():
        raise AssertionError(
            f"output: expected shape={tuple(x.shape)} dtype={x.dtype} device={x.device} contiguous=True; "
            f"actual shape={tuple(output.shape)} dtype={output.dtype} device={output.device} "
            f"stride={output.stride()} contiguous={output.is_contiguous()}"
        )


def call_dut(x, weight, shift, scale, indices, eps=EPS):
    """Log inputs, call the standalone kernel, and validate its raw output."""
    import mindiesd.layers.index_modulation as dut

    print(f"DUT=rms_norm_indexed_scale_shift source={dut.__file__}")
    log_inputs(x, weight, shift, scale, indices, eps=eps)
    output = dut.rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)
    check_output_contract(output, (x, weight, shift, scale, indices))
    print(
        f"output: shape={tuple(output.shape)} dtype={output.dtype} device={output.device} "
        f"layout={output.layout} stride={output.stride()} contiguous={output.is_contiguous()}"
    )
    return output
