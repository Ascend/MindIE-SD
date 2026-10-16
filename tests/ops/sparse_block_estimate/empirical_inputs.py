#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Build in-memory Q/K test inputs from statistics of a real ``data.pt`` snapshot."""

import hashlib
import math
from pathlib import Path
from typing import Any

import torch


SUPPORTED_MODES = ("realistic", "low", "typical", "high", "tail")
CASE_SEED_STRIDE = 1_000_003
PROFILE_QUANTILES = (0.0001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.9999)


def _sample_flat(flat: torch.Tensor, sample_size: int, generator: torch.Generator) -> torch.Tensor:
    if flat.numel() <= sample_size:
        return flat.float()
    indices = torch.randint(0, flat.numel(), (sample_size,), generator=generator)
    return flat[indices].float()


def _streaming_stats(tensor: torch.Tensor, chunk_elements: int) -> dict[str, float | int]:
    flat = tensor.contiguous().view(-1)
    count = 0
    sum_value = 0.0
    sum_square = 0.0
    minimum = math.inf
    maximum = -math.inf
    for start in range(0, flat.numel(), chunk_elements):
        values = flat[start : start + chunk_elements].float()
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            continue
        finite64 = finite.double()
        count += finite.numel()
        sum_value += finite64.sum().item()
        sum_square += (finite64 * finite64).sum().item()
        minimum = min(minimum, finite.min().item())
        maximum = max(maximum, finite.max().item())
    if count != flat.numel():
        raise ValueError(f"snapshot tensor contains {flat.numel() - count} NaN/Inf values")
    mean = sum_value / count
    variance = max(sum_square / count - mean * mean, 0.0)
    return {"count": count, "min": minimum, "max": maximum, "mean": mean, "std": math.sqrt(variance)}


def _profile_tensor(
    tensor: torch.Tensor,
    sample_size: int,
    row_sample_size: int,
    chunk_elements: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    if tensor.ndim != 4:
        raise ValueError(f"snapshot tensor must be rank 4 BNSD, got shape {tuple(tensor.shape)}")
    global_stats = _streaming_stats(tensor, chunk_elements)
    sample = _sample_flat(tensor.contiguous().view(-1), sample_size, generator)
    quantiles = torch.quantile(sample, torch.tensor(PROFILE_QUANTILES))
    global_stats["quantiles"] = {
        "p0.01": quantiles[0].item(),
        "p1": quantiles[1].item(),
        "p5": quantiles[2].item(),
        "p50": quantiles[3].item(),
        "p95": quantiles[4].item(),
        "p99": quantiles[5].item(),
        "p99.99": quantiles[6].item(),
    }

    head_dim = tensor.shape[-1]
    rows = tensor.contiguous().view(-1, head_dim)
    if rows.shape[0] > row_sample_size:
        indices = torch.randint(0, rows.shape[0], (row_sample_size,), generator=generator)
        rows = rows[indices]
    rows = rows.float()
    row_rms = torch.sqrt((rows * rows).mean(dim=-1))
    row_quantiles = torch.quantile(row_rms, torch.tensor((0.01, 0.05, 0.5, 0.95, 0.99)))
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "global": global_stats,
        "per_head": [_streaming_stats(tensor[:, head : head + 1], chunk_elements) for head in range(tensor.shape[1])],
        "sampled_rows_and_channels": {
            "sampled_rows": rows.shape[0],
            "sampled_channel_mean": rows.mean(dim=0).tolist(),
            "sampled_channel_std": rows.std(dim=0, unbiased=False).tolist(),
            "row_rms": {
                "quantiles": {
                    "p1": row_quantiles[0].item(),
                    "p5": row_quantiles[1].item(),
                    "p50": row_quantiles[2].item(),
                    "p95": row_quantiles[3].item(),
                    "p99": row_quantiles[4].item(),
                }
            },
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot_profile(
    path: Path,
    expected_q_heads: int,
    expected_kv_heads: int,
    expected_head_dim: int,
    seed: int,
    sample_size: int = 1_000_000,
    row_sample_size: int = 16_384,
    chunk_elements: int = 1_000_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the real snapshot once, profile Q/K, then release its tensors."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("data.pt must contain a dictionary")
    missing = [name for name in ("query_bf16", "key_bf16") if name not in payload]
    if missing:
        raise ValueError(f"data.pt is missing required tensors: {missing}")
    query = payload["query_bf16"]
    key = payload["key_bf16"]
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
        raise ValueError("query_bf16 and key_bf16 in data.pt must be tensors")
    for name, tensor, heads in (
        ("query_bf16", query, expected_q_heads),
        ("key_bf16", key, expected_kv_heads),
    ):
        if (
            tensor.ndim != 4
            or tensor.shape[0] != 1
            or tensor.shape[1] != heads
            or tensor.shape[-1] != expected_head_dim
        ):
            raise ValueError(
                f"{name} must have BNSD shape [1, {heads}, S, {expected_head_dim}], got {tuple(tensor.shape)}"
            )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    profile = {
        "query_bf16": _profile_tensor(query, sample_size, row_sample_size, chunk_elements, generator),
        "key_bf16": _profile_tensor(key, sample_size, row_sample_size, chunk_elements, generator),
    }
    source = {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "profile_seed": seed,
        "sample_size": sample_size,
        "row_sample_size": row_sample_size,
    }
    del payload, query, key
    return profile, source


def _row_scale(tensor_profile: dict[str, Any], sequence: int, mode: str, generator: torch.Generator) -> torch.Tensor:
    row_rms = tensor_profile["sampled_rows_and_channels"]["row_rms"]["quantiles"]
    median = max(float(row_rms["p50"]), torch.finfo(torch.float32).tiny)
    low = max(float(row_rms["p1"]) / median, 0.01)
    high = max(float(row_rms["p99"]) / median, low)
    if mode == "low":
        return torch.full((sequence, 1), low)
    if mode == "typical":
        return torch.ones((sequence, 1))
    if mode == "high":
        return torch.full((sequence, 1), high)
    p5 = max(float(row_rms["p5"]) / median, 0.01)
    p95 = max(float(row_rms["p95"]) / median, p5)
    sigma = max((math.log(p95) - math.log(p5)) / (2.0 * 1.6448536269514722), 0.01)
    scale = torch.exp(torch.randn((sequence, 1), generator=generator) * sigma)
    return scale.clamp_(min=low, max=high)


def make_tensor(
    tensor_profile: dict[str, Any],
    heads: int,
    sequence: int,
    head_dim: int,
    mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Match the super_test empirical generator without writing generated tensors."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported generation mode: {mode}")
    row_profile = tensor_profile["sampled_rows_and_channels"]
    channel_mean = torch.tensor(row_profile["sampled_channel_mean"], dtype=torch.float32)
    channel_std = torch.tensor(row_profile["sampled_channel_std"], dtype=torch.float32)
    if channel_mean.numel() != head_dim or len(tensor_profile["per_head"]) != heads:
        raise ValueError("requested generated shape does not match the snapshot profile")
    channel_mean_variance = channel_mean.var(unbiased=False).item()
    channel_noise_rms = math.sqrt((channel_std * channel_std).mean().item())
    lower = float(tensor_profile["global"]["quantiles"]["p0.01"])
    upper = float(tensor_profile["global"]["quantiles"]["p99.99"])
    output = torch.empty((1, heads, sequence, head_dim), dtype=torch.bfloat16)
    for head in range(heads):
        target_std = float(tensor_profile["per_head"][head]["std"])
        noise_target = math.sqrt(max(target_std * target_std - channel_mean_variance, 0.0))
        noise_factor = noise_target / max(channel_noise_rms, torch.finfo(torch.float32).tiny)
        noise = torch.randn((sequence, head_dim), generator=generator)
        scale = _row_scale(tensor_profile, sequence, mode, generator)
        values = channel_mean + noise * channel_std * noise_factor * scale
        values.clamp_(min=lower, max=upper)
        output[0, head] = values.to(torch.bfloat16)
    if mode == "tail":
        flat = output.view(-1)
        injection_count = max(16, flat.numel() // 100_000)
        indices = torch.randint(0, flat.numel(), (injection_count,), generator=generator)
        flat[indices[0::2]] = float(tensor_profile["global"]["min"])
        flat[indices[1::2]] = float(tensor_profile["global"]["max"])
    return output


def make_query_key(
    profile: dict[str, Any],
    q_heads: int,
    kv_heads: int,
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    mode: str,
    enhance_mode: float,
    seed: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = make_tensor(profile["query_bf16"], q_heads, q_seq, head_dim, mode, generator)
    key = make_tensor(profile["key_bf16"], kv_heads, kv_seq, head_dim, mode, generator)
    if enhance_mode != 1.0:
        query = (query.float() * enhance_mode).to(torch.bfloat16)
        key = (key.float() * enhance_mode).to(torch.bfloat16)
    return query.to(dtype), key.to(dtype)
