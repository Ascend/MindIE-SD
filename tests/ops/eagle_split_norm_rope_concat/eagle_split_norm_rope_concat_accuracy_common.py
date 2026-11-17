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

import math
import os
import random

import torch

if __package__:
    from .eagle_split_norm_rope_concat_cases import PROD_HEADS, PROD_SHAPE  # pylint: disable=relative-beyond-top-level
else:
    from eagle_split_norm_rope_concat_cases import PROD_HEADS, PROD_SHAPE

SEED = 20261017
PROD_ATTRS = dict(norm_type=3, rope_type=1, rotary_dim=96, eps=1e-5, output_bnsd=False)

# Q/K stay FP32 through RMSNorm and RoPE, then round once to the output dtype.
# Each row is (tolerance_dtype, atol, rtol, required_matched_ratio, absolute_floor).
_QK_TOLERANCES = {
    torch.float16: ("FLOAT16", 2**-10, 2**-10, 0.99, 0.1),
    torch.bfloat16: ("BFLOAT16", 2**-7, 2**-7, 0.99, 0.1),
}


def init_device(device_id):
    if device_id < 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("Use a nonnegative physical device ID; unset ASCEND_RT_VISIBLE_DEVICES")
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    import mindiesd  # noqa: F401

    return f"npu:{device_id}"


def make_inputs(shape, heads, rotary_dim, dtype, device, distribution, seed=SEED, rope_rank=2, compact=False):
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

    batch = shape[0] if len(shape) == 4 else 1
    seq, dim = shape[-3], shape[-1]
    width = rotary_dim // 2 if compact else rotary_dim
    rope_shape = {2: (seq, width), 3: (batch, seq, width), 4: (batch, seq, 1, width)}[rope_rank]
    return (
        sample("qkv", shape),
        sample("norm_query_weight", (dim,)),
        sample("norm_key_weight", (dim,)),
        sample("rope_cos", rope_shape),
        sample("rope_sin", rope_shape),
        list(heads),
    )


def log_inputs(inputs, rotary_dim=96, rope_type=1, eps=1e-5, output_bnsd=False, norm_type=3):
    names = ("qkv", "norm_query_weight", "norm_key_weight", "rope_cos", "rope_sin", "head_counts")
    for name, value in zip(names, inputs):
        if name == "head_counts":
            print(f"{name}: {value}")
            continue
        if value is None:
            print(f"{name}: None")
            continue
        print(
            f"{name}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device} "
            f"stride={value.stride()} contiguous={value.is_contiguous()}"
        )
    print(
        f"heads={inputs[5]} layout={'SND' if inputs[0].ndim == 3 else 'BSND'} "
        f"norm_type={norm_type} rotary_dim={rotary_dim} rope_type={rope_type} eps={eps} output_bnsd={output_bnsd}"
    )


def inputs_to_cpu(inputs):
    return tuple(value.cpu() if isinstance(value, torch.Tensor) else value for value in inputs)


def golden(inputs, rotary_dim=96, rope_type=1, eps=1e-5, output_bnsd=False, norm_type=3):
    """CPU small-shape simulation or NPU composition, on the inputs' current device."""
    packed, qw, kw, cos, sin, heads = inputs
    if norm_type not in (2, 3):
        raise ValueError("golden supports norm_type 2 and 3")
    if rope_type not in (0, 1):
        raise ValueError("golden supports rope_type 0 and 1")
    rotary_dim = packed.shape[-1] if rotary_dim is None else rotary_dim
    rank = packed.ndim
    packed4 = packed.unsqueeze(0) if rank == 3 else packed
    q, k, v = torch.split(packed4, heads, dim=2)
    width = cos.shape[-1]
    cos = cos.reshape(-1, packed4.shape[1], 1, width).float()
    sin = sin.reshape(-1, packed4.shape[1], 1, width).float()
    if width == rotary_dim // 2:
        if rope_type == 1:
            cos, sin = torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
        else:
            cos, sin = cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)
    elif width != rotary_dim:
        raise ValueError("cache width must equal R or R/2")
    result = []
    for x, weight in ((q, qw), (k, kw)):
        xf = x.float()
        # Match the Vector Sqrt + Div path without intermediate low-precision rounding.
        variance = (xf * xf).mean(dim=-1, keepdim=True)
        inverse = torch.ones_like(variance) / torch.sqrt(variance + eps)
        norm = xf * inverse
        if norm_type == 3:
            norm = norm * weight.float()
        prefix = norm[..., :rotary_dim]
        if rope_type == 1:
            left, right = prefix.chunk(2, dim=-1)
            rotated = torch.cat((-right, left), dim=-1)
        else:
            pairs = prefix.reshape(*prefix.shape[:-1], rotary_dim // 2, 2)
            rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        y = torch.cat((prefix * cos + rotated * sin, norm[..., rotary_dim:]), dim=-1).to(x.dtype)
        result.append(y)
    result.append(v)
    if output_bnsd:
        return tuple(x.transpose(1, 2).contiguous() for x in result)
    return tuple((x.squeeze(0) if rank == 3 else x).contiguous() for x in result)


def _coordinate(flat_index, shape):
    result = []
    for size in reversed(shape):
        flat_index, index = divmod(flat_index, size)
        result.append(index)
    return tuple(reversed(result))


def compare(actual, expected, *, large=False):
    for name, out, ref in zip(("q", "k", "v"), actual, expected):
        assert out.shape == ref.shape and out.dtype == ref.dtype, (name, out.shape, out.dtype)
        assert out.is_contiguous()
        if name == "v":
            exact = torch.equal(out.view(torch.int16), ref.view(torch.int16))
            print(
                f"{name}: output_dtype={out.dtype} tolerance_dtype=EXACT reason=copy_or_layout_only exact_hard={exact}"
            )
            assert exact, "V must be copied exactly"
            continue
        tolerance_dtype, atol, rtol, required, absolute_floor = _QK_TOLERANCES[out.dtype]
        # CPU FP64 metrics for small tests; large metrics stay on NPU in FP32.
        metric_dtype = torch.float32 if large else torch.float64
        a, g = out.to(metric_dtype).flatten(), ref.to(metric_dtype).flatten()
        assert bool(torch.isfinite(a).all() & torch.isfinite(g).all()), f"{name}: nonfinite result"
        diff = (a - g).abs()
        worst = int(diff.argmax().item())
        max_abs = float(diff[worst].item())
        value = abs(float(g[worst].item()))
        # ULP of the actual output format at the maximum-error reference element.
        info = torch.finfo(out.dtype)
        exponent = math.floor(math.log2(value)) if value >= info.tiny else math.log2(info.tiny)
        ulp = math.ldexp(info.eps, int(exponent))
        limit = max(absolute_floor, 32 * ulp)
        matched = float((diff <= atol + rtol * g.abs()).to(torch.float32).mean().item())
        an, gn = float(torch.linalg.vector_norm(a).item()), float(torch.linalg.vector_norm(g).item())
        ratio = an / gn if gn else (1.0 if an == 0 else math.inf)
        cosine = float(torch.dot(a, g).item()) / (an * gn) if an and gn else float(an == gn)
        matched_ok = matched >= required
        max_abs_ok = max_abs <= limit
        mixed_ok = matched_ok and max_abs_ok
        detail = (
            f"{name}: index={_coordinate(worst, out.shape)} "
            f"actual={float(a[worst].item()):.9g} golden={float(g[worst].item()):.9g}"
        )
        print(
            f"{name}: tolerance_dtype={tolerance_dtype} output_dtype={out.dtype} "
            f"reason='FP32 RMSNorm and RoPE, final output rounds to {out.dtype}' "
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


def call_dut(inputs, **attrs):
    # Default to R=96, weighted HALF and the input layout; cases override attributes explicitly.
    attrs = {**PROD_ATTRS, **attrs}
    log_inputs(inputs, **attrs)
    return torch.ops.mindiesd.eagle_split_norm_rope_concat(*inputs, **attrs)
