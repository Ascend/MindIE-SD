#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Shared FIA FP8 quant path definitions used by accuracy and performance tests."""

from __future__ import annotations

from dataclasses import dataclass

PER_BLOCK_MODE = 7
V512_D128_MODE = 11
V512_D64_MODE = 12
Q_TOKEN_BLOCK = 128
K_TOKEN_BLOCK = 256
V256_TOKEN_BLOCK = 256
V512_TOKEN_BLOCK = 512
D128_CHANNEL_BLOCK = 128
D64_CHANNEL_BLOCK = 64


@dataclass(frozen=True)
class FiaQuantCase:
    name: str
    description: str
    value_quant_mode: int
    value_token_block: int
    value_channel_block: int
    inner_precise: int

    @property
    def quant_modes(self):
        return PER_BLOCK_MODE, PER_BLOCK_MODE, self.value_quant_mode


FIA_QUANT_CASES = (
    FiaQuantCase(
        "original",
        "K256/V256 original",
        PER_BLOCK_MODE,
        V256_TOKEN_BLOCK,
        D128_CHANNEL_BLOCK,
        0,
    ),
    FiaQuantCase(
        "c8v16",
        "K256/V256 C8V16",
        PER_BLOCK_MODE,
        V256_TOKEN_BLOCK,
        D128_CHANNEL_BLOCK,
        4,
    ),
    FiaQuantCase(
        "v512",
        "K256/V512x128 C8V16",
        V512_D128_MODE,
        V512_TOKEN_BLOCK,
        D128_CHANNEL_BLOCK,
        4,
    ),
    FiaQuantCase(
        "v512_d64",
        "K256/V512x64 C8V16",
        V512_D64_MODE,
        V512_TOKEN_BLOCK,
        D64_CHANNEL_BLOCK,
        4,
    ),
)

FIA_QUANT_CASE_BY_NAME = {case.name: case for case in FIA_QUANT_CASES}


def parse_case_names(value):
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in FIA_QUANT_CASE_BY_NAME]
    if unknown:
        supported = ", ".join(FIA_QUANT_CASE_BY_NAME)
        raise ValueError(f"unsupported FIA path(s) {unknown}; expected: {supported}")
    if not names:
        raise ValueError("at least one FIA path must be selected")
    return [FIA_QUANT_CASE_BY_NAME[name] for name in names]


@dataclass(frozen=True)
class FiaProdShape:
    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    query_seq_len: int
    kv_seq_len: int
    head_dim: int

    @property
    def query_shape(self):
        return self.batch_size, self.num_query_heads, self.query_seq_len, self.head_dim

    @property
    def kv_shape(self):
        return self.batch_size, self.num_kv_heads, self.kv_seq_len, self.head_dim

    def arguments(self):
        return [item for key, value in vars(self).items() for item in ("--" + key.replace("_", "-"), str(value))]


PROD_SHAPES = (
    FiaProdShape(1, 32, 4, 2304, 30757, 128),
    FiaProdShape(1, 40, 40, 75600, 75600, 128),
)


def shape_help():
    return "\n".join(
        f"Q={shape.query_shape} K=V={shape.kv_shape}, BNSD, FP8 E4M3/FP32 scales/BF16 output; "
        + " ".join(shape.arguments())
        for shape in PROD_SHAPES
    )


def add_shape_arguments(parser, *, single=False):
    for key, value in vars(PROD_SHAPES[0]).items():
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=int,
            default=value if single else None,
            help=f"{'Default: ' + str(value) if single else 'Omitted: select all matching presets'}. See complete shapes below.",
        )
    parser.epilog = shape_help()


def select_shapes(args):
    selected = [
        shape
        for shape in PROD_SHAPES
        if all(getattr(args, key) is None or getattr(args, key) == value for key, value in vars(shape).items())
    ]
    if not selected:
        raise ValueError("No supported shape matches the filters.\n" + shape_help())
    return selected


def selected_paths(value):
    paths = list(FIA_QUANT_CASE_BY_NAME) if value is None else [value]
    if any(path not in FIA_QUANT_CASE_BY_NAME for path in paths):
        raise ValueError(f"Unsupported path: {value}")
    return paths
