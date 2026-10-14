#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Small-shape accuracy: CPU golden, device fixed at npu:0."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_a5_correctness import SMALL_PRESETS, _load_custom_operator, run  # noqa: E402

pytest.importorskip("torch_npu")


@pytest.fixture(scope="module", autouse=True)
def _load_operator():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available")
    _load_custom_operator(0)


@pytest.mark.parametrize("preset", SMALL_PRESETS)
@pytest.mark.parametrize("dtype_name", ("fp16", "bf16"))
def test_small_shape_vs_cpu_golden(preset, dtype_name):
    args = argparse.Namespace(
        preset=preset,
        device=0,
        seed=2026,
        input_profile=None,
        case_mode="realistic",
        enhance_mode=1.0,
        sample_rows=0,
        threshold=0.85,
        row_sparse=1.0,
        keep_sink=True,
        keep_recent=True,
        max_diff_ratio=0.02,
        max_omitted_ratio=0.01,
        reference_only=False,
    )
    result = run(args, dtype_name)
    assert result["golden"] == "cpu_matmul_softmax_blocksum"
    assert result["diff_ratio"] < args.max_diff_ratio
    assert result["omitted_ratio"] < args.max_omitted_ratio
    assert result["count_mismatch_rows"] == 0
    assert result["padding_nonzero"] == 0
