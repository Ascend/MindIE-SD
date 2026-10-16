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

"""Shared Laser Attention helpers: uniform/normal inputs, CPU/NPU golden, gates."""

from __future__ import annotations

import math

import torch

SEED = 20260921
ACCURACY_DISTRIBUTIONS = ("uniform", "normal")
LAYOUT = "BNSD"
SEQ_ALIGN = 256
DIM_ALIGN = 128
MAX_TOKEN = 2147483647

# FP16 inputs select tiling key 0; BF16 Q/K with FP16 V select tiling key 1.
COMPUTE_DTYPE = torch.float16

SMALL_BATCH = 1
SMALL_NUM_HEADS = 2
SMALL_HEAD_DIM = 128
SMALL_ALIGNED_Q_SEQ = 256
SMALL_ALIGNED_KV_SEQ = 512
SMALL_MASKED_Q_SEQ = 512
SMALL_MASKED_KV_SEQ = 768
SMALL_MASKED_PRE_TOKENS = 255

PROD_BATCH = 1
PROD_NUM_HEADS = 24
PROD_Q_SEQ = 5376
PROD_KV_SEQ = 8960
PROD_HEAD_DIM = 128
PROD_DTYPE = torch.bfloat16

COSINE_MIN = 0.99
NORM_RATIO_LO = 0.9
NORM_RATIO_HI = 1.1

# mixed_tolerance_standard: (rtol, atol, required_matched_ratio, fallback_max_abs).
# https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
_MIXED_TOLERANCE = {
    "float16": (2**-10, 2**-10, 0.99, 1e-1),
    "bfloat16": (2**-7, 2**-7, 0.99, 1e-1),
    "float32": (2**-13, 2**-13, 0.99, 1e-2),
}
_ULP_FACTOR = 32


def softmax_scale(head_dim):
    return head_dim**-0.5


def generate_opbase_tensor(shape, dtype, generator, distribution="mixed", tensor_name=None):
    """Separate accuracy distributions; retain mixed fill for performance callers."""
    if distribution not in (*ACCURACY_DISTRIBUTIONS, "mixed"):
        raise ValueError(f"unsupported input distribution: {distribution}")
    if distribution == "uniform":
        data = torch.empty(shape, dtype=torch.float32).uniform_(-5.0, 5.0, generator=generator)
        if tensor_name is not None:
            print(f"input={tensor_name} distribution=uniform low=-5 high=5")
        return data.to(dtype)
    if distribution == "normal":
        mu = torch.empty((), dtype=torch.float32).uniform_(-5.0, 5.0, generator=generator).item()
        sigma = torch.empty((), dtype=torch.float32).uniform_(0.1, 2.0, generator=generator).item()
        data = torch.empty(shape, dtype=torch.float32).normal_(mu, sigma, generator=generator)
        if tensor_name is not None:
            print(f"input={tensor_name} distribution=normal mu={mu!r} sigma={sigma!r}")
        return data.to(dtype)
    numel = 1
    for dim in shape:
        numel *= dim
    uniform = torch.empty(numel, dtype=torch.float32)
    uniform.uniform_(-5.0, 5.0, generator=generator)
    mu = torch.empty((), dtype=torch.float32)
    mu.uniform_(-5.0, 5.0, generator=generator)
    sigma = torch.empty((), dtype=torch.float32)
    sigma.uniform_(0.1, 2.0, generator=generator)
    normal = torch.empty(numel, dtype=torch.float32)
    normal.normal_(float(mu.item()), float(sigma.item()), generator=generator)
    pick = torch.empty(numel, dtype=torch.float32)
    pick.uniform_(0.0, 1.0, generator=generator)
    mixed = torch.where(pick < 0.5, uniform, normal)
    return mixed.reshape(shape).to(dtype)


def validate_shape(batch, num_heads, query_seq_len, kv_seq_len, head_dim):
    if min(batch, num_heads, query_seq_len, kv_seq_len, head_dim) <= 0:
        raise ValueError("batch, heads, sequence lengths and head dimension must be positive")
    if query_seq_len % SEQ_ALIGN or kv_seq_len % SEQ_ALIGN:
        raise ValueError(f"operator sequence lengths must be multiples of {SEQ_ALIGN}; no automatic padding")
    if head_dim % DIM_ALIGN:
        raise ValueError(f"operator head dimension must be a multiple of {DIM_ALIGN}; no automatic padding")


def build_laser_attention_inputs(
    batch,
    num_heads,
    query_seq_len,
    kv_seq_len,
    head_dim,
    dtype,
    generator,
    distribution="mixed",
    log_distribution=False,
):
    """Generate contiguous operator-ready BNSD tensors; dtype specifies Q/K, V is FP16."""
    validate_shape(batch, num_heads, query_seq_len, kv_seq_len, head_dim)
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q/K dtype must be FP16 or BF16")
    query = generate_opbase_tensor(
        (batch, num_heads, query_seq_len, head_dim), dtype, generator, distribution, "Q" if log_distribution else None
    )
    key = generate_opbase_tensor(
        (batch, num_heads, kv_seq_len, head_dim), dtype, generator, distribution, "K" if log_distribution else None
    )
    value = generate_opbase_tensor(
        (batch, num_heads, kv_seq_len, head_dim),
        torch.float16,
        generator,
        distribution,
        "V" if log_distribution else None,
    )
    return query, key, value


def kernel_output_dtype(query_dtype):
    return torch.bfloat16 if query_dtype == torch.bfloat16 else torch.float32


def print_operator_inputs(query, key, value, scale, pre_tokens=MAX_TOKEN):
    valid_kv_seq_len = key.shape[2] if pre_tokens == MAX_TOKEN else key.shape[2] - pre_tokens
    print(
        f"op=torch.ops.mindiesd.la layout={LAYOUT} head_num={query.shape[1]} "
        f"scale={scale} keep_prob=1.0 pre_tokens={pre_tokens} next_tokens=1 is_highPrecision=True "
        f"valid_q_seq_len={query.shape[2]} valid_kv_seq_len={valid_kv_seq_len}"
    )
    for name, tensor in (("Q", query), ("K", key), ("V", value)):
        print(
            f"{name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} stride={tensor.stride()} contiguous={tensor.is_contiguous()}"
        )
    print(f"expected_output shape={tuple(query.shape)} dtype={kernel_output_dtype(query.dtype)}")


def _compose_attention_golden(query, key, value, scale, pre_tokens, chunk_rows):
    """FP32 reference on the input device, chunked in Q to bound score storage."""
    key_t = key.float().transpose(-1, -2)
    value_f = value.float()
    output = torch.empty(query.shape, dtype=kernel_output_dtype(query.dtype), device=query.device)
    for start in range(0, query.shape[2], chunk_rows):
        scores = torch.matmul(query[:, :, start : start + chunk_rows, :].float(), key_t) * scale
        if pre_tokens != MAX_TOKEN:
            scores[..., key.shape[2] - pre_tokens :] = float("-inf")
        probs = torch.softmax(scores, dim=-1)
        output[:, :, start : start + chunk_rows, :] = torch.matmul(probs, value_f).to(output.dtype)
    return output


def _cpu_laser_rowsum(exp_scores):
    """Mirror VectorForward::process_cacl_wrap's FP16 128-column block folding."""
    block_count = exp_scores.shape[-1] // 128
    # get_padding_info_for_row_max pads the remainder after complete groups of 16.
    full_blocks, tail_blocks = divmod(block_count, 16)
    padded_tail = 0 if tail_blocks == 0 else 1 << (tail_blocks - 1).bit_length()
    padded_blocks = full_blocks * 16 + padded_tail
    blocks = torch.zeros((*exp_scores.shape[:-1], padded_blocks, 128), dtype=torch.float16)
    blocks[..., :block_count, :] = exp_scores.reshape(*exp_scores.shape[:-1], block_count, 128)

    def fold(part):
        while part.shape[-2] > 1:
            half = part.shape[-2] // 2
            part = (part[..., :half, :].float() + part[..., half:, :].float()).half()
        return part[..., 0, :]

    if 16 < padded_blocks < 32:
        folded = (fold(blocks[..., :16, :]).float() + fold(blocks[..., 16:, :]).float()).half()
    else:
        folded = fold(blocks)
    # Approximate vcadd's internal reduction with FP32 accumulation, then its FP16 output.
    return folded.float().sum(dim=-1, keepdim=True).half().float()


def cpu_laser_attention_golden(query, key, value, scale, pre_tokens=MAX_TOKEN):
    """CPU stage simulation for dense, single-KV-section small shapes (KV <= 4096).

    Cube MMA and vector exp/vcadd internals are approximated on CPU, not bit-exact.
    Explicit FP16 stage boundaries follow CubeForward.h and VectorForward.h.
    """
    validate_shape(*query.shape[:2], query.shape[2], key.shape[2], query.shape[3])
    if key.shape[2] > 4096:
        raise ValueError("CPU golden supports one KV section (<= 4096); use NPU golden for large shapes")
    if pre_tokens != MAX_TOKEN and not 0 <= pre_tokens < SEQ_ALIGN:
        raise ValueError("pre_tokens must be MAX_TOKEN or a masked KV suffix length in [0, 255]")
    query, key, value = (tensor.detach().cpu() for tensor in (query, key, value))
    key_t = key.float().transpose(-1, -2)
    value_f = value.float()
    # The tiling attribute is FP32; vmuls casts it to INPUT_T=half.
    scale_half = torch.tensor(scale, dtype=torch.float32).half().float()
    output = torch.empty(query.shape, dtype=kernel_output_dtype(query.dtype))
    for start in range(0, query.shape[2], 256):
        # Cube QK -> F322F16, followed by FP16 vmuls (not FP32 QK * scale).
        scores = torch.matmul(query[:, :, start : start + 256, :].float(), key_t).half()
        scores = (scores.float() * scale_half).half()
        if pre_tokens != MAX_TOKEN and pre_tokens:
            scores[..., key.shape[2] - pre_tokens :] = -60000.0
        rowmax = scores.amax(dim=-1, keepdim=True)
        shifted = (scores.float() - rowmax.float()).half()
        exp_scores = shifted.float().exp().half()
        rowsum = _cpu_laser_rowsum(exp_scores)
        # PV consumes unnormalized FP16 exponentials and FP16 V, accumulates FP32.
        weighted = torch.matmul(exp_scores.float(), value_f)
        # Vector vdiv produces an FP32 reciprocal; Cube diagonal matmul applies it.
        output[:, :, start : start + 256, :] = (weighted * rowsum.reciprocal()).to(output.dtype)
    return output


def npu_laser_attention_golden(query, key, value, scale, pre_tokens=MAX_TOKEN):
    """Large-shape NPU compose; preserve mixed Q/K/V input values by promoting each to FP32."""
    if any(tensor.device.type != "npu" for tensor in (query, key, value)):
        raise ValueError("large-shape golden requires NPU inputs")
    return _compose_attention_golden(query, key, value, scale, pre_tokens, 128)


def call_laser_attention(query, key, value, scale=None, pre_tokens=MAX_TOKEN):
    """Call the operator with unchanged BNSD inputs and return its raw attention output."""
    import mindiesd  # noqa: F401 - register torch.ops.mindiesd.la

    batch, num_heads, q_seq, head_dim = query.shape
    validate_shape(batch, num_heads, q_seq, key.shape[2], head_dim)
    if key.shape != value.shape or key.shape[:2] != query.shape[:2] or key.shape[3] != head_dim:
        raise ValueError("Q/K/V must have matching B/N/D and matching K/V shapes")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != torch.float16:
        raise ValueError("operator requires FP16/BF16 Q/K with matching dtype and FP16 V")
    if any(not tensor.is_contiguous() or tensor.device != query.device for tensor in (query, key, value)):
        raise ValueError("Q/K/V must be contiguous and on the same device")
    if pre_tokens != MAX_TOKEN and not 0 <= pre_tokens < SEQ_ALIGN:
        raise ValueError("pre_tokens must be MAX_TOKEN or a masked KV suffix length in [0, 255]")
    if scale is None:
        scale = softmax_scale(head_dim)
    _, attention_out = torch.ops.mindiesd.la(
        query,
        key,
        value,
        None,
        None,
        None,
        scale,
        num_heads,
        LAYOUT,
        1.0,
        pre_tokens,
        1,
        True,
    )
    expected_dtype = kernel_output_dtype(query.dtype)
    if attention_out.shape != query.shape or attention_out.dtype != expected_dtype:
        raise RuntimeError(
            f"kernel output mismatch: shape={tuple(attention_out.shape)} dtype={attention_out.dtype}, "
            f"expected={tuple(query.shape)}/{expected_dtype}"
        )
    return attention_out


def cosine_metrics(reference, actual, chunk_elements=1_048_576):
    """Chunked FP64 cosine / max_abs / norm_ratio."""
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}")
    reference_flat = reference.detach().cpu().reshape(-1)
    actual_flat = actual.detach().cpu().reshape(-1)
    dot = reference_squared = actual_squared = 0.0
    max_abs_error = 0.0
    for start in range(0, reference_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, reference_flat.numel())
        reference_chunk = reference_flat[start:end].double()
        actual_chunk = actual_flat[start:end].double()
        dot += torch.dot(reference_chunk, actual_chunk).item()
        reference_squared += torch.dot(reference_chunk, reference_chunk).item()
        actual_squared += torch.dot(actual_chunk, actual_chunk).item()
        max_abs_error = max(max_abs_error, (actual_chunk - reference_chunk).abs().max().item())
    denominator = math.sqrt(reference_squared * actual_squared)
    if not math.isfinite(dot) or not math.isfinite(denominator) or denominator == 0.0:
        raise ValueError("cosine inputs produced a non-finite or zero norm")
    cosine = max(-1.0, min(1.0, dot / denominator))
    return {
        "cosine": cosine,
        "max_abs_error": max_abs_error,
        "norm_ratio": math.sqrt(actual_squared / reference_squared),
    }


def _mixed_tolerance_dtype_key(dtype):
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    text = str(dtype).lower().replace(" ", "").replace("_", "").replace("torch.", "")
    if text in _MIXED_TOLERANCE:
        return text
    raise ValueError(f"unsupported mixed-tolerance dtype: {dtype}")


def _ulp_at_value(value, dtype_key):
    """Binade spacing in the output format, including zero/subnormal spacing."""
    fraction_bits, min_exponent = {
        "float16": (10, -14),
        "bfloat16": (7, -126),
        "float32": (23, -126),
    }[dtype_key]
    if not math.isfinite(value):
        raise ValueError("ULP requires a finite narrowed golden value")
    exponent = math.frexp(abs(value))[1] - 1 if value else min_exponent
    return math.ldexp(1.0, max(exponent, min_exponent) - fraction_bits)


def check_mixed_tolerance(actual, golden, dtype):
    """opbase mixed tolerance: matched_ratio and max_abs_error.

    Element pass: |actual - golden| <= atol + rtol * |golden|.
    Case pass: matched_ratio >= 0.99 and finite max_abs_error <= max(fallback, 32 * ULP(g_low)).
    dtype selects the compute row; actual.dtype selects the ULP format at the worst-error point.
    """
    key = _mixed_tolerance_dtype_key(dtype)
    rtol, atol, required_ratio, fixed_limit = _MIXED_TOLERANCE[key]
    output_key = _mixed_tolerance_dtype_key(actual.dtype)
    max_abs_limit = fixed_limit
    if actual.shape != golden.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(golden.shape)}")
    actual_flat = actual.detach().cpu().double().reshape(-1)
    golden_flat = golden.detach().cpu().double().reshape(-1)
    if actual_flat.numel() == 0:
        return {
            "result": "Pass",
            "dtype_key": key,
            "matched_ratio": 1.0,
            "max_abs_error": 0.0,
            "rtol": rtol,
            "atol": atol,
            "required_matched_ratio": required_ratio,
            "max_abs_error_limit": max_abs_limit,
        }
    abs_error = (actual_flat - golden_flat).abs()
    threshold = atol + rtol * golden_flat.abs()
    finite = torch.isfinite(actual_flat) & torch.isfinite(golden_flat)
    narrowed = golden_flat.to(actual.dtype).double()
    # Standard section 2.3: finite golden may overflow to a matching signed infinity.
    matching_inf = torch.isinf(actual_flat) & (actual_flat == narrowed)
    matched = (finite & (abs_error <= threshold)) | matching_inf
    matched_ratio = float(matched.sum().item()) / float(actual_flat.numel())
    max_abs_error = 0.0
    golden_at_max = 0.0
    ulp_at_max = 0.0
    valid_anchor = True
    if finite.any().item():
        # argmax selects the first point if several finite points tie.
        index = int(abs_error.masked_fill(~finite, -1.0).argmax().item())
        max_abs_error = float(abs_error[index].item())
        golden_at_max = float(narrowed[index].item())
        valid_anchor = math.isfinite(golden_at_max)
        if valid_anchor:
            ulp_at_max = _ulp_at_value(golden_at_max, output_key)
            max_abs_limit = max(fixed_limit, _ULP_FACTOR * ulp_at_max)
    # Undefined NaN / nonmatching infinity cases must not disappear into the 1% allowance.
    valid_nonfinite = bool((finite | matching_inf).all().item())
    passed = valid_anchor and valid_nonfinite and matched_ratio >= required_ratio and max_abs_error <= max_abs_limit
    return {
        "result": "Pass" if passed else "Failed",
        "dtype_key": key,
        "matched_ratio": matched_ratio,
        "max_abs_error": max_abs_error,
        "rtol": rtol,
        "atol": atol,
        "required_matched_ratio": required_ratio,
        "max_abs_error_limit": max_abs_limit,
        "output_dtype_key": output_key,
        "golden_at_max_error": golden_at_max,
        "ulp_at_max_error": ulp_at_max,
    }


def print_accuracy_report(golden_name, dut_name, metrics, gate):
    print(f"golden={golden_name}  dut={dut_name}")
    print(
        f"cosine={metrics['cosine']:.8f}  "
        f"max_abs_error={metrics['max_abs_error']:.6f}  "
        f"norm_ratio={metrics['norm_ratio']:.6f}"
    )
    print(
        f"mixed_tolerance={gate['result']}  dtype={gate['dtype_key']}  "
        f"matched_ratio={gate['matched_ratio'] * 100.0:.4f}%  "
        f"max_abs={gate['max_abs_error']:.6f}  "
        f"rtol={gate['rtol']} atol={gate['atol']}  "
        f"required_ratio={gate['required_matched_ratio']}  "
        f"max_abs_limit={gate['max_abs_error_limit']}  "
        f"output_dtype={gate.get('output_dtype_key', 'n/a')}  "
        f"golden_at_max={gate.get('golden_at_max_error', 0.0)}  "
        f"ulp_at_max={gate.get('ulp_at_max_error', 0.0)}"
    )


def collect_gate_failures(metrics, gate, require_mixed=True):
    failures = []
    if require_mixed and gate["result"] != "Pass":
        failures.append(
            f"mixed tolerance {gate['result']}: dtype={gate['dtype_key']} "
            f"matched_ratio={gate['matched_ratio'] * 100.0:.4f}% "
            f"max_abs={gate['max_abs_error']:.6f} limit={gate['max_abs_error_limit']} "
            f"rtol={gate['rtol']} atol={gate['atol']}"
        )
    if metrics["cosine"] < COSINE_MIN:
        failures.append(
            f"cosine {metrics['cosine']:.8f} < {COSINE_MIN}  "
            f"max_abs={metrics['max_abs_error']:.6f}  "
            f"norm_ratio={metrics['norm_ratio']:.6f}"
        )
    if not (NORM_RATIO_LO <= metrics["norm_ratio"] <= NORM_RATIO_HI):
        failures.append(
            f"norm_ratio {metrics['norm_ratio']:.6f} is outside "
            f"[{NORM_RATIO_LO}, {NORM_RATIO_HI}]  "
            f"cosine={metrics['cosine']:.8f}  "
            f"max_abs={metrics['max_abs_error']:.6f}"
        )
    return failures
