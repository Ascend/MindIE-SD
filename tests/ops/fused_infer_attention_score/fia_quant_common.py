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
