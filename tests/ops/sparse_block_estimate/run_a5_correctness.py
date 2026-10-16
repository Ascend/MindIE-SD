#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Standalone correctness runner for the A5 sparse_block_estimate kernel.

Small shapes compare the int8 mask with a CPU matmul + FP32 softmax + block-sum
golden. The production target shape uses the same formula on NPU. Neither path
calls another sparse-estimate implementation.

Examples:
    python tests/ops/sparse_block_estimate/run_a5_correctness.py --preset smoke --dtype both
    python tests/ops/sparse_block_estimate/run_a5_correctness.py --preset target --device-id 0
    python tests/ops/sparse_block_estimate/run_a5_correctness.py --reference-only
"""

import argparse
import dataclasses
import gc
import importlib
import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from empirical_inputs import CASE_SEED_STRIDE, SUPPORTED_MODES, load_snapshot_profile, make_query_key

STRIDE = 8
HEAD_DIM = 128
Q_BLOCK = 128
KV_BLOCK = 128
PACKED_Q_ROWS_PER_BLOCK = Q_BLOCK // STRIDE
PACKED_K_COLS_PER_BLOCK = KV_BLOCK // STRIDE
PACKED_K_TILE_COLUMNS = 512 // STRIDE


@dataclasses.dataclass(frozen=True)
class Case:
    batch: int
    q_heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int


# Q tile is 256 and K tile is 512. smoke_align is the aligned pair; smoke and
# smoke_q_tail keep the +1 tails (KV 2049, and Q 257).
PRESETS = {
    "smoke_align": Case(1, 8, 1, 256, 2048),
    "smoke": Case(1, 8, 1, 256, 2049),
    "smoke_q_tail": Case(1, 8, 1, 257, 2049),
    "multitile": Case(1, 8, 1, 512, 4097),
    "target": Case(1, 32, 4, 2304, 30757),
}
SMALL_PRESETS = ("smoke_align", "smoke", "smoke_q_tail", "multitile")


def _torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _make_inputs(
    case: Case,
    dtype: torch.dtype,
    seed: int,
    profile: dict | None = None,
    mode: str = "realistic",
    enhance_mode: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if profile is not None:
        return make_query_key(
            profile,
            case.q_heads,
            case.kv_heads,
            case.q_seq,
            case.kv_seq,
            HEAD_DIM,
            mode,
            enhance_mode,
            seed,
            dtype,
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    query = _experimental_standard((case.batch, case.q_heads, case.q_seq, HEAD_DIM), generator).to(dtype)
    key = _experimental_standard((case.batch, case.kv_heads, case.kv_seq, HEAD_DIM), generator).to(dtype)
    return query, key


def _experimental_standard(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """50% U[-5, 5] and 50% Normal(mu, sigma), mu in [-5, 5], sigma in [0.1, 2]."""
    uniform = torch.rand(shape, generator=generator, dtype=torch.float32).mul_(10.0).sub_(5.0)
    mu = torch.rand((1,), generator=generator).mul_(10.0).sub_(5.0)
    sigma = torch.rand((1,), generator=generator).mul_(1.9).add_(0.1)
    normal = torch.randn(shape, generator=generator, dtype=torch.float32).mul_(sigma).add_(mu)
    choose_uniform = torch.rand(shape, generator=generator) < 0.5
    return torch.where(choose_uniform, uniform, normal)


def _sample_rows(case: Case, limit: int) -> list[tuple[int, int, int]]:
    q_blocks = math.ceil(case.q_seq / Q_BLOCK)
    rows = [
        (batch, q_head, q_block)
        for batch in range(case.batch)
        for q_head in range(case.q_heads)
        for q_block in range(q_blocks)
    ]
    if limit <= 0 or limit >= len(rows):
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    indices = torch.linspace(0, len(rows) - 1, steps=limit).round().to(torch.int64).tolist()
    return [rows[index] for index in dict.fromkeys(indices)]


def _pack_query_block(query_block: torch.Tensor) -> torch.Tensor:
    if query_block.ndim != 2 or query_block.shape[1] != HEAD_DIM or not 0 < query_block.shape[0] <= Q_BLOCK:
        raise ValueError(f"query block must be [1, {Q_BLOCK}] x {HEAD_DIM}, got {tuple(query_block.shape)}")
    if query_block.shape[0] < Q_BLOCK:
        query_block = torch.nn.functional.pad(query_block, (0, 0, 0, Q_BLOCK - query_block.shape[0]))
    return (
        query_block.reshape(PACKED_Q_ROWS_PER_BLOCK, STRIDE, HEAD_DIM)
        .flip(1)
        .reshape(PACKED_Q_ROWS_PER_BLOCK, STRIDE * HEAD_DIM)
    )


def _pack_key(key: torch.Tensor) -> torch.Tensor:
    padding = (-key.shape[0]) % STRIDE
    if padding:
        key = torch.nn.functional.pad(key, (0, 0, 0, padding))
    return key.reshape(-1, STRIDE * HEAD_DIM)


def _select_mask(
    block_scores: torch.Tensor,
    threshold: float,
    row_sparse: float,
    keep_sink: bool,
    keep_recent: bool,
) -> torch.Tensor:
    """Reproduce the estimator's block selection, independent of its QK path."""
    candidates = block_scores.clone()
    if keep_sink:
        candidates[0] = 0.0

    # The existing non-causal operator contract always protects the final block;
    # keep_recent is retained in the signature for API parity.
    _ = keep_recent
    protect_last = True
    if protect_last:
        candidates[-1] = 0.0

    if threshold >= 1.0:
        mask = torch.ones_like(candidates, dtype=torch.bool)
    else:
        sorted_scores = torch.sort(candidates, descending=True).values
        target = candidates.sum(dtype=torch.float32) * threshold
        selected = int((sorted_scores.cumsum(0, dtype=torch.float32) < target).sum().item()) + 1
        if 0.0 < row_sparse < 1.0:
            selected = min(selected, max(1, math.floor(candidates.numel() * row_sparse)))
        selected = min(selected, candidates.numel())
        guard = sorted_scores[selected - 1].float() - 1.0e-6
        mask = block_scores.float() >= guard

    if keep_sink:
        mask[0] = True
    if protect_last:
        mask[-1] = True
    return mask


def _weight_query_tail(
    probabilities: torch.Tensor,
    q_blocks: list[int],
    q_len: int,
) -> torch.Tensor:
    """Match the kernel Q-tail denominator: a partial pack scales by valid/8, padded packs drop out."""
    if probabilities.shape[0] != len(q_blocks) * PACKED_Q_ROWS_PER_BLOCK:
        raise ValueError("packed query rows do not match the requested Q blocks")
    weights = torch.ones(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype)
    for index, q_block in enumerate(q_blocks):
        for local in range(PACKED_Q_ROWS_PER_BLOCK):
            raw0 = q_block * Q_BLOCK + local * STRIDE
            slot = index * PACKED_Q_ROWS_PER_BLOCK + local
            if raw0 >= q_len:
                weights[slot] = 0
            elif raw0 + STRIDE > q_len:
                weights[slot] = (q_len - raw0) / STRIDE
    if torch.count_nonzero(weights != 1).item() == 0:
        return probabilities
    return probabilities * weights.unsqueeze(-1)


def reference_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    rows: Iterable[tuple[int, int, int]],
    scale_value: float,
    threshold: float,
    row_sparse: float,
    keep_sink: bool,
    keep_recent: bool,
) -> tuple[
    dict[tuple[int, int, int], torch.Tensor],
    dict[tuple[int, int, int], torch.Tensor],
]:
    """Compute pre-selection block scores and masks for requested rows."""
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
    head_ratio = q_heads // kv_heads
    kv_blocks = math.ceil(key.shape[2] / KV_BLOCK)
    packed_key_cache: dict[tuple[int, int], torch.Tensor] = {}
    score_result = {}
    mask_result = {}

    grouped_rows: dict[tuple[int, int], list[int]] = {}
    for batch, q_head, q_block in rows:
        grouped_rows.setdefault((batch, q_head), []).append(q_block)

    for (batch, q_head), q_blocks in grouped_rows.items():
        kv_head = q_head // head_ratio
        cache_key = (batch, kv_head)
        if cache_key not in packed_key_cache:
            packed_key_cache[cache_key] = _pack_key(key[batch, kv_head].float())
        packed_key = packed_key_cache[cache_key]
        packed_query = torch.cat(
            [
                _pack_query_block(
                    query[
                        batch,
                        q_head,
                        q_block * Q_BLOCK : (q_block + 1) * Q_BLOCK,
                    ].float()
                )
                for q_block in q_blocks
            ],
            dim=0,
        )
        logits = packed_query @ packed_key.transpose(0, 1)
        logits.mul_(scale_value)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        probabilities = _weight_query_tail(probabilities, q_blocks, query.shape[2])

        padding = kv_blocks * PACKED_K_COLS_PER_BLOCK - probabilities.shape[-1]
        if padding:
            probabilities = torch.nn.functional.pad(probabilities, (0, padding))
        block_scores = probabilities.reshape(
            len(q_blocks), PACKED_Q_ROWS_PER_BLOCK, kv_blocks, PACKED_K_COLS_PER_BLOCK
        ).sum(dim=(1, 3))
        for index, q_block in enumerate(q_blocks):
            row = (batch, q_head, q_block)
            score = block_scores[index]
            if score.device.type != "cpu":
                score = score.cpu()
            score_result[row] = score
            mask_result[row] = _select_mask(score, threshold, row_sparse, keep_sink, keep_recent)
    return score_result, mask_result


def _load_custom_operator(device_id: int) -> None:
    try:
        torch_npu = importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for NPU execution; use --reference-only on a CPU host") from exc

    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU is not available")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        raise SystemExit(
            f"ERROR: ASCEND_RT_VISIBLE_DEVICES is set ({visible!r}). Unset it and pass --device-id (npu-smi NPU ID)."
        )
    torch.npu.set_device(device_id)
    soc_version = torch_npu.npu.get_soc_version()
    if soc_version != 260:
        raise RuntimeError(f"device {device_id} is SoC version {soc_version}, expected A5 (260)")
    print("torch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    print("device:", device_id, torch_npu.npu.get_device_name(device_id))
    print("soc_version:", soc_version)

    from mindiesd.layers.register_ops import (  # pylint: disable=import-outside-toplevel
        _load_mindie_ops_library,
    )

    _load_mindie_ops_library()
    try:
        _operator = torch.ops.mindiesd.sparse_block_estimate
    except AttributeError as exc:
        raise RuntimeError("mindiesd::sparse_block_estimate was not registered by libPTAExtensionOPS.so") from exc
    del _operator


def _compare(
    actual_mask: torch.Tensor,
    actual_count: torch.Tensor,
    golden: dict[tuple[int, int, int], torch.Tensor],
    valid_kv_blocks: int,
    max_diff_ratio: float,
    max_omitted_ratio: float,
) -> dict[str, float]:
    compared = 0
    different = 0
    omitted = 0
    expected_true = 0
    count_mismatch = 0

    for row, expected in golden.items():
        batch, q_head, q_block = row
        actual = actual_mask[batch, q_head, q_block, :valid_kv_blocks].bool().cpu()
        expected = expected.bool().cpu()
        compared += expected.numel()
        different += torch.ne(actual, expected).sum().item()
        omitted += torch.logical_and(expected, torch.logical_not(actual)).sum().item()
        expected_true += expected.sum().item()
        count_mismatch += int(actual_count[batch, q_head, q_block].item() != actual.sum().item())

    padding_nonzero = actual_mask[..., valid_kv_blocks:].bool().sum().item()
    diff_ratio = different / max(expected_true, 1)
    omitted_ratio = omitted / max(expected_true, 1)
    metrics = {
        "compared_elements": compared,
        "different_elements": different,
        "omitted_elements": omitted,
        "expected_selected": expected_true,
        "diff_ratio": diff_ratio,
        "omitted_ratio": omitted_ratio,
        "count_mismatch_rows": count_mismatch,
        "padding_nonzero": padding_nonzero,
    }
    if diff_ratio >= max_diff_ratio:
        raise AssertionError(f"mask diff ratio {diff_ratio:.6f} >= {max_diff_ratio:.6f}")
    if omitted_ratio >= max_omitted_ratio:
        raise AssertionError(f"omitted ratio {omitted_ratio:.6f} >= {max_omitted_ratio:.6f}")
    if count_mismatch:
        raise AssertionError(f"sparse_count_table mismatched mask sums in {count_mismatch} sampled rows")
    if padding_nonzero:
        raise AssertionError(f"mask padding contains {padding_nonzero} non-zero values")
    return metrics


def run(args: argparse.Namespace, dtype_name: str, row_offset: int | None = None) -> dict:
    case = PRESETS[args.preset]
    dtype = _torch_dtype(dtype_name)
    query, key = _make_inputs(
        case,
        dtype,
        args.seed,
        getattr(args, "input_profile", None),
        getattr(args, "case_mode", "realistic"),
        args.enhance_mode,
    )
    sample_rows = args.sample_rows
    if sample_rows is None:
        sample_rows = 0
    rows = _sample_rows(case, sample_rows)
    use_npu_golden = args.preset == "target" and not args.reference_only
    if use_npu_golden:
        query = query.to(f"npu:{args.device}")
        key = key.to(f"npu:{args.device}")
    if row_offset is not None and rows:
        all_rows = _sample_rows(case, 0)
        if sample_rows == 1:
            rows = [all_rows[row_offset % len(all_rows)]]
        elif sample_rows and sample_rows < len(all_rows):
            rows = [all_rows[(all_rows.index(row) + row_offset) % len(all_rows)] for row in rows]
    scale_value = float(os.environ.get("SPARSE_ESTIMATE_SCALE_VALUE", HEAD_DIM**-0.5 / STRIDE))
    _, golden_masks = reference_rows(
        query,
        key,
        rows,
        scale_value,
        args.threshold,
        args.row_sparse,
        args.keep_sink,
        args.keep_recent,
    )
    result = {
        "preset": args.preset,
        "dtype": dtype_name,
        "shape_q": list(query.shape),
        "shape_k": list(key.shape),
        "sampled_rows": len(rows),
        "scale_value": scale_value,
        "golden": "npu_matmul_softmax_blocksum" if use_npu_golden else "cpu_matmul_softmax_blocksum",
        "input_distribution": (
            "empirical_snapshot" if getattr(args, "input_profile", None) is not None else "experimental_standard_50_50"
        ),
    }
    if args.reference_only:
        result["reference_selected_min"] = min(mask.sum().item() for mask in golden_masks.values())
        result["reference_selected_max"] = max(mask.sum().item() for mask in golden_masks.values())
        return result

    if use_npu_golden:
        query_npu = query
        key_npu = key
    else:
        query_npu = query.to(f"npu:{args.device}")
        key_npu = key.to(f"npu:{args.device}")
    common_kwargs = {
        "query": query_npu,
        "key": key_npu,
        "input_layout": "BNSD",
        "stride": STRIDE,
        "sparse_size": Q_BLOCK,
        "num_heads": case.q_heads,
        "num_key_value_heads": case.kv_heads,
        "scale_value": scale_value,
        "threshold": args.threshold,
        "causal": False,
        "keep_sink": args.keep_sink,
        "keep_recent": args.keep_recent,
        "row_sparse": args.row_sparse,
    }
    actual_mask, actual_count = torch.ops.mindiesd.sparse_block_estimate(**common_kwargs)
    torch.npu.synchronize()
    actual_mask = actual_mask.cpu()
    actual_count = actual_count.cpu()

    expected_shape = (
        case.batch,
        case.q_heads,
        math.ceil(case.q_seq / Q_BLOCK),
        math.ceil(math.ceil(case.kv_seq / KV_BLOCK) / 32) * 32,
    )
    if tuple(actual_mask.shape) != expected_shape:
        raise AssertionError(f"mask shape {tuple(actual_mask.shape)} != {expected_shape}")
    if tuple(actual_count.shape) != expected_shape[:-1]:
        raise AssertionError(f"count shape {tuple(actual_count.shape)} != {expected_shape[:-1]}")
    if actual_mask.dtype != torch.int8 or actual_count.dtype != torch.int32:
        raise AssertionError(f"unexpected output dtypes: {actual_mask.dtype}, {actual_count.dtype}")

    valid_kv_blocks = math.ceil(case.kv_seq / KV_BLOCK)
    expected_counts = actual_mask[..., :valid_kv_blocks].to(torch.int32).sum(dim=-1)
    all_row_count_mismatch = torch.ne(actual_count, expected_counts).sum().item()
    result["all_row_count_mismatch"] = all_row_count_mismatch
    if all_row_count_mismatch:
        raise AssertionError(f"sparse_count_table mismatched mask sums in {all_row_count_mismatch} output rows")
    result.update(
        _compare(
            actual_mask,
            actual_count,
            golden_masks,
            valid_kv_blocks,
            args.max_diff_ratio,
            args.max_omitted_ratio,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="smoke")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--device", "--device-id", type=int, default=0, dest="device")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--data",
        type=Path,
        help="Real snapshot data.pt; multi-case mode defaults to data.pt beside this script.",
    )
    parser.add_argument("--modes", default="realistic,low,high,tail")
    parser.add_argument(
        "--enhance-mode",
        type=float,
        default=1.0,
        help="Multiply every empirically generated Q/K value by this positive factor.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=1,
        help="Number of independent input seeds; multi-case mode writes only an aggregate report.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print multi-case progress every N input groups; 0 disables progress output.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Golden rows to check; default 0 checks every Q-block row.",
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--row-sparse", type=float, default=1.0)
    parser.add_argument("--keep-sink", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-recent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-diff-ratio", type=float, default=0.02)
    parser.add_argument("--max-omitted-ratio", type=float, default=0.01)
    parser.add_argument("--report", help="Write all per-dtype metrics to a JSON report.")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if not 0.0 < args.row_sparse <= 1.0:
        parser.error("--row-sparse must be in (0, 1]")
    if args.sample_rows is not None and args.sample_rows < 0:
        parser.error("--sample-rows must be non-negative")
    if args.cases <= 0:
        parser.error("--cases must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    args.mode_list = [item.strip() for item in args.modes.split(",") if item.strip()]
    if not args.mode_list:
        parser.error("--modes must contain at least one generation mode")
    invalid_modes = sorted(set(args.mode_list) - set(SUPPORTED_MODES))
    if invalid_modes:
        parser.error(f"unsupported --modes values: {invalid_modes}; expected {SUPPORTED_MODES}")
    if not math.isfinite(args.enhance_mode) or args.enhance_mode <= 0.0:
        parser.error("--enhance-mode must be finite and greater than zero")
    for name in (
        "max_diff_ratio",
        "max_omitted_ratio",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def _update_stress_summary(summary: dict, result: dict, case_index: int, seed: int) -> None:
    context = {"case_index": case_index, "seed": seed, "dtype": result["dtype"]}
    for name in (
        "diff_ratio",
        "omitted_ratio",
    ):
        value = result.get(name)
        previous = summary["worst_metrics"].get(name)
        if value is not None and (previous is None or value > previous["value"]):
            summary["worst_metrics"][name] = {"value": value, **context}
    for name in (
        "compared_elements",
        "different_elements",
        "omitted_elements",
        "count_mismatch_rows",
        "padding_nonzero",
        "all_row_count_mismatch",
    ):
        summary["totals"][name] = summary["totals"].get(name, 0) + int(result.get(name, 0))


def _write_report(path: str | None, payload: dict) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_stress(args: argparse.Namespace, dtype_names: tuple[str, ...]) -> None:
    started = time.monotonic()
    case = PRESETS[args.preset]
    data_path = args.data or Path(__file__).with_name("data.pt")
    if not data_path.is_file():
        raise FileNotFoundError(
            f"real snapshot not found: {data_path}; copy it beside this script as data.pt or pass --data"
        )
    profile, profile_source = load_snapshot_profile(
        data_path,
        case.q_heads,
        case.kv_heads,
        HEAD_DIM,
        args.seed,
    )
    effective_sample_rows = args.sample_rows
    if effective_sample_rows is None:
        effective_sample_rows = 0
    summary = {
        "status": "RUNNING",
        "preset": args.preset,
        "cases_requested": args.cases,
        "cases_completed": 0,
        "dtype_executions_completed": 0,
        "dtypes": list(dtype_names),
        "base_seed": args.seed,
        "seed_rule": f"base_seed + case_index * {CASE_SEED_STRIDE}",
        "input_distribution": "empirical per-channel/per-head/row-RMS profile from data.pt",
        "profile_source": profile_source,
        "profile_ranges": {
            name: {
                "min": tensor_profile["global"]["min"],
                "max": tensor_profile["global"]["max"],
                "p0.01": tensor_profile["global"]["quantiles"]["p0.01"],
                "p99.99": tensor_profile["global"]["quantiles"]["p99.99"],
            }
            for name, tensor_profile in profile.items()
        },
        "modes": args.mode_list,
        "enhance_mode": args.enhance_mode,
        "sample_rows_per_execution": effective_sample_rows,
        "row_sampling": (
            "all logical output rows" if effective_sample_rows == 0 else "rotating across all logical output rows"
        ),
        "saved_input_or_score_tensors": False,
        "worst_metrics": {},
        "totals": {},
    }
    summary["shape_q"] = [case.batch, case.q_heads, case.q_seq, HEAD_DIM]
    summary["shape_k"] = [case.batch, case.kv_heads, case.kv_seq, HEAD_DIM]

    for case_index in range(args.cases):
        case_args = argparse.Namespace(**vars(args))
        case_args.seed = args.seed + case_index * CASE_SEED_STRIDE
        case_args.case_mode = args.mode_list[case_index % len(args.mode_list)]
        case_args.input_profile = profile
        case_args.report = None
        try:
            for dtype_name in dtype_names:
                result = run(case_args, dtype_name, row_offset=case_index)
                _update_stress_summary(summary, result, case_index, case_args.seed)
                summary["dtype_executions_completed"] += 1
                del result
        except Exception as exc:
            summary["status"] = "FAIL"
            summary["failed_case"] = {
                "case_index": case_index,
                "seed": case_args.seed,
                "dtype": dtype_name,
                "mode": case_args.case_mode,
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary["elapsed_seconds"] = time.monotonic() - started
            _write_report(args.report, summary)
            raise
        summary["cases_completed"] += 1
        if (case_index + 1) % 10 == 0:
            gc.collect()
        if args.progress_every and ((case_index + 1) % args.progress_every == 0 or case_index + 1 == args.cases):
            print(
                json.dumps(
                    {
                        "progress": f"{case_index + 1}/{args.cases}",
                        "seed": case_args.seed,
                        "mode": case_args.case_mode,
                        "enhance_mode": args.enhance_mode,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary["status"] = "PASS"
    summary["elapsed_seconds"] = time.monotonic() - started
    _write_report(args.report, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print("PASS")


def main() -> None:
    args = parse_args()
    if not args.reference_only:
        _load_custom_operator(args.device)
    dtype_names = ("fp16", "bf16") if args.dtype == "both" else (args.dtype,)
    if args.cases > 1:
        _run_stress(args, dtype_names)
        return
    if args.data is not None:
        if not args.data.is_file():
            raise FileNotFoundError(f"real snapshot not found: {args.data}")
        case = PRESETS[args.preset]
        profile, _profile_source = load_snapshot_profile(
            args.data,
            case.q_heads,
            case.kv_heads,
            HEAD_DIM,
            args.seed,
        )
        args.input_profile = profile
    results = []
    for dtype_name in dtype_names:
        result = run(args, dtype_name)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.report:
        _write_report(args.report, {"status": "PASS", "results": results})
    print("PASS")


if __name__ == "__main__":
    main()
