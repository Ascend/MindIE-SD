#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""One CPU-prepared FIA forward for hardware profiling or simulation."""

import argparse
import importlib.util
import os
import random
from pathlib import Path

from tests.ops.fused_infer_attention_score.fia_quant_common import FIA_QUANT_CASE_BY_NAME, add_shape_arguments


def operator_package_dir():
    spec = importlib.util.find_spec("mindiesd")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("MindIE-SD package is not available")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def prepare_operator_environment():
    # CANN reads custom OPP paths on initialization, before MindIE-SD is imported.
    vendors = operator_package_dir() / "ops/vendors"
    paths = [str(path) for path in vendors.iterdir() if path.is_dir()]
    if not (vendors / "aie_ascendc").is_dir():
        raise RuntimeError(f"Missing built FIA operator package: {vendors}")
    previous = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([*paths, previous])


def make_inputs(args):
    import torch

    from tests.ops.fused_infer_attention_score.fia_accuracy_common import synthesize_bf16

    case = FIA_QUANT_CASE_BY_NAME[args.path]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    parameter_rng = random.Random(args.seed)
    shapes = (
        (args.batch_size, args.num_query_heads, args.query_seq_len, args.head_dim),
        (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim),
        (args.batch_size, args.num_kv_heads, args.kv_seq_len, args.head_dim),
    )
    originals, quantized, scales = [], [], []
    blocks = ((128, 128), (256, 128), (case.value_token_block, case.value_channel_block))
    for role, shape, block in zip(("query", "key", "value"), shapes, blocks):
        print(
            f"stage=cpu_input role={role} shape={shape} distribution={args.distribution} seed={args.seed}", flush=True
        )
        if args.distribution == "snapshot":
            source = synthesize_bf16(role, shape, generator, args.enhance_mode)
        elif args.distribution == "uniform":
            source = torch.empty(shape).uniform_(-5, 5, generator=generator).to(torch.bfloat16)
        else:
            mean, std = parameter_rng.uniform(-5, 5), parameter_rng.uniform(0.1, 2)
            print(f"role={role} mean={mean:.9g} std={std:.9g} clamp=none", flush=True)
            source = torch.empty(shape).normal_(mean, std, generator=generator).to(torch.bfloat16)
        encoded, scale = quantize_cpu(source, *block)
        originals.append(source)
        quantized.append(encoded)
        scales.append(scale)
    return originals, quantized, scales


def quantize_cpu(source, row_block, column_block):
    """Encode independent blocks with FP32 max-abs scales and FP8 RNE.

    Slice legal tail blocks directly: the operator sees the requested unpadded
    physical shape. Process one head at a time to bound CPU temporary memory.
    """
    import torch

    batch, heads, rows, columns = source.shape
    encoded = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (batch, heads, (rows + row_block - 1) // row_block, (columns + column_block - 1) // column_block),
        dtype=torch.float32,
    )
    for batch_index in range(batch):
        for head in range(heads):
            for row in range(0, rows, row_block):
                for column in range(0, columns, column_block):
                    values = source[batch_index, head, row : row + row_block, column : column + column_block].float()
                    scale = values.abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / 448.0
                    scales[batch_index, head, row // row_block, column // column_block] = scale
                    encoded[batch_index, head, row : row + row_block, column : column + column_block] = (
                        (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                    )
    return encoded, scales


def call_fia(args, qkv, scales):
    import torch

    case = FIA_QUANT_CASE_BY_NAME[args.path]
    return torch.ops.mindiesd.fused_infer_attention_score_v2(
        *qkv,
        num_query_heads=args.num_query_heads,
        num_key_value_heads=args.num_kv_heads,
        softmax_scale=args.head_dim**-0.5,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="BNSD",
        query_quant_mode=7,
        key_quant_mode=7,
        value_quant_mode=case.value_quant_mode,
        inner_precise=case.inner_precise,
        dequant_scale_query=scales[0],
        dequant_scale_key=scales[1],
        dequant_scale_value=scales[2],
        out_dtype=torch.bfloat16,
    )


def describe_inputs(qkv, scales):
    for name, tensor in zip(("Q", "K", "V", "Q_scale", "K_scale", "V_scale"), (*qkv, *scales)):
        print(
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
            f"stride={tensor.stride()} contiguous={tensor.is_contiguous()}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run exactly one FP8 FIA workload; omitted dimensions use the first preset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--path", choices=FIA_QUANT_CASE_BY_NAME, default="original")
    parser.add_argument("--device-id", type=int, default=0)
    add_shape_arguments(parser, single=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--enhance-mode", type=float, default=2.0)
    parser.add_argument("--distribution", choices=("snapshot", "normal", "uniform"), default="normal")
    parser.add_argument("--msprof-mode", action="store_true", help="Compatibility flag; always one forward.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if args.device_id < 0 or args.batch_size != 1 or args.head_dim != 128:
        raise ValueError("Require device-id >= 0, batch-size=1 and head-dim=128")
    if min(args.num_query_heads, args.num_kv_heads, args.query_seq_len, args.kv_seq_len) <= 0:
        raise ValueError("Head counts and sequence lengths must be positive")
    if args.num_query_heads % args.num_kv_heads:
        raise ValueError("Query head count must be divisible by KV head count")


def main():
    args = parse_args()
    print(
        f"BNSD Q={(1, args.num_query_heads, args.query_seq_len, 128)} "
        f"K=V={(1, args.num_kv_heads, args.kv_seq_len, 128)} path={args.path} "
        "QKV=FP8_E4M3 scales=FP32 output=BF16 calls=1",
        flush=True,
    )
    if args.dry_run:
        return
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("Unset ASCEND_RT_VISIBLE_DEVICES; use --device-id")
    prepare_operator_environment()
    import torch
    import torch_npu

    print(f"stage=device_init device_id={args.device_id}", flush=True)
    torch.npu.set_device(args.device_id)
    import mindiesd  # noqa: F401 pylint: disable=unused-import

    originals, qkv, scales = make_inputs(args)
    del originals
    print("stage=device_copy", flush=True)
    qkv = [tensor.to(f"npu:{args.device_id}") for tensor in qkv]
    scales = [tensor.to(f"npu:{args.device_id}") for tensor in scales]
    torch_npu.npu.synchronize()
    describe_inputs(qkv, scales)
    print("stage=target_call kernel=EagleFusedInferAttentionScore", flush=True)
    with torch.inference_mode():
        output, lse = call_fia(args, qkv, scales)
        print("stage=target_return", flush=True)
        torch_npu.npu.synchronize()
    print("stage=target_synchronized", flush=True)
    expected = (args.batch_size, args.num_query_heads, args.query_seq_len, args.head_dim)
    if tuple(output.shape) != expected or output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise RuntimeError(f"Invalid output metadata: {output.shape}, {output.dtype}, {output.stride()}")
    if lse is not None and lse.numel():
        raise RuntimeError("Expected empty softmax_lse")
    print("RESULT forward synchronized; metadata PASS. Numerical accuracy is tested separately.", flush=True)


if __name__ == "__main__":
    main()
