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

"""Unit tests for common.schema tables.

The schema tables are the single contract shared by the runtime op_defs layer
and the offline report tool; keep them structurally consistent so adding or
renaming an op argument cannot silently break report generation.
"""

from common.schema import (
    BASELINE_METRICS,
    COMPARE_METRICS,
    OP_SEQ_AXIS,
    OP_SERIES_KEY,
    OP_SLOT_ARGS,
    SLOT_OMIT_WHEN_DEFAULT,
)

OPS = ("fa", "bsa", "gmm", "mm")


def test_every_op_has_all_tables():
    for op in OPS:
        assert op in OP_SLOT_ARGS, f"{op} missing OP_SLOT_ARGS"
        assert op in OP_SEQ_AXIS, f"{op} missing OP_SEQ_AXIS"
        assert op in OP_SERIES_KEY, f"{op} missing OP_SERIES_KEY"


def test_slot_args_non_empty_and_no_duplicates():
    for op in OPS:
        keys = OP_SLOT_ARGS[op]
        assert keys, f"{op} has empty OP_SLOT_ARGS"
        assert len(keys) == len(set(keys)), f"{op} has duplicate slot args"


def test_series_key_is_a_slot_arg():
    # The series grouping key must be one of the slot args, otherwise
    # benchmark_report._aggregate_cases would never find it in the slot string.
    for op in OPS:
        assert OP_SERIES_KEY[op] in OP_SLOT_ARGS[op], f"{op} series key not a slot arg"


def test_seq_axis_is_a_slot_arg():
    for op in OPS:
        assert OP_SEQ_AXIS[op] in OP_SLOT_ARGS[op], f"{op} seq axis not a slot arg"


def test_compare_metrics_are_baseline_metrics():
    for metric in COMPARE_METRICS:
        assert metric in BASELINE_METRICS


def test_baseline_metrics_include_latency():
    assert "latency(us)" in BASELINE_METRICS


def test_omit_when_default_references_existing_keys():
    for op, pairs in SLOT_OMIT_WHEN_DEFAULT.items():
        assert op in OP_SLOT_ARGS
        for omit_key, default_key in pairs:
            assert omit_key in OP_SLOT_ARGS[op]
            assert default_key in OP_SLOT_ARGS[op]


def test_fa_bsa_slot_args_carry_heads_dim_func():
    # Reports must be able to show head count / dim / tested kernel function.
    for op in ("fa", "bsa"):
        for key in ("num_heads", "head_dim", "func", "batch_size"):
            assert key in OP_SLOT_ARGS[op], f"{op} missing {key} in OP_SLOT_ARGS"


def test_gmm_slot_args_carry_shape_params():
    # Varying hidden_size/moe_inter/experts across runs must not collide slots.
    for key in ("hidden_size", "moe_inter", "experts"):
        assert key in OP_SLOT_ARGS["gmm"], f"gmm missing {key} in OP_SLOT_ARGS"


def test_display_metrics_per_op():
    from common.schema import OP_DISPLAY_METRICS

    # compute-bound ops show MFU only; gmm also shows MBU
    assert OP_DISPLAY_METRICS["fa"] == ("MFU",)
    assert OP_DISPLAY_METRICS["bsa"] == ("MFU",)
    assert OP_DISPLAY_METRICS["mm"] == ("MFU",)
    assert "MBU" in OP_DISPLAY_METRICS["gmm"]
    assert "MFU" in OP_DISPLAY_METRICS["gmm"]


def test_omit_pairs_are_only_for_kv_len():
    # Current convention: only kv_len is omitted when equal to q_len.
    for op, pairs in SLOT_OMIT_WHEN_DEFAULT.items():
        for omit_key, _ in pairs:
            assert omit_key == "kv_len"
