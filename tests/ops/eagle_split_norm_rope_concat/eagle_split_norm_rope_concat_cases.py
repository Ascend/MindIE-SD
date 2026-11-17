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

"""Eight crossed configurations covering R=96/128 without a 16-case product."""

PROD_SHAPE = (1, 5672, 168, 128)
PROD_HEADS = (56, 56, 56)
CASE_IDS = (
    "half-bsnd",
    "half-bnsd-r128",
    "interleave-bsnd-r128",
    "interleave-bnsd",
    "half-bsnd-compact-r128",
    "half-bnsd-compact",
    "interleave-bsnd-compact",
    "interleave-bnsd-compact-r128",
)


def is_compact(case_id):
    if case_id not in CASE_IDS:
        raise ValueError(f"Unknown production case: {case_id}")
    return "-compact" in case_id


def accuracy_distributions(case_id):
    # Eight configurations, four uniform and four normal; no 16-case Cartesian product.
    index = CASE_IDS.index(case_id)
    return ("uniform" if index in (0, 3, 5, 6) else "normal",)


def case_attrs(case_id):
    if case_id not in CASE_IDS:
        raise ValueError(f"Unknown production case: {case_id}")
    rope, layout = case_id.split("-")[:2]
    return dict(
        norm_type=3,
        rope_type=1 if rope == "half" else 0,
        rotary_dim=128 if case_id.endswith("-r128") else 96,
        eps=1e-5,
        output_bnsd=layout == "bnsd",
    )


def output_shapes(case_id):
    attrs = case_attrs(case_id)
    batch, seq, _, dim = PROD_SHAPE
    return tuple((batch, heads, seq, dim) if attrs["output_bnsd"] else (batch, seq, heads, dim) for heads in PROD_HEADS)


def add_profile_selection(parser):
    parser.add_argument("--case", choices=CASE_IDS, help="Internal capture key; prefer semantic selectors")
    parser.add_argument("--shape", type=int, nargs=4, default=PROD_SHAPE, metavar=("B", "S", "Htotal", "D"))
    parser.add_argument("--rope-type", choices=("half", "interleave"))
    parser.add_argument("--cache-width", choices=("full", "half"))
    parser.add_argument("--output-layout", choices=("bsnd", "bnsd"))
    parser.add_argument("--rotary-dim", type=int, choices=(96, 128))


def resolve_profile_selection(parser, args):
    if tuple(args.shape) != PROD_SHAPE:
        parser.error("Performance supports --shape 1 5672 168 128 only")
    filters = (args.rope_type, args.cache_width, args.output_layout, args.rotary_dim)
    if args.case is not None:
        if any(value is not None for value in filters):
            parser.error("Do not combine internal --case with semantic selectors")
        return args.case
    if all(value is None for value in filters):
        return CASE_IDS[0]
    matches = [
        key
        for key in CASE_IDS
        if (args.rope_type is None or key.split("-", maxsplit=1)[0] == args.rope_type)
        and (args.output_layout is None or key.split("-")[1] == args.output_layout)
        and (args.cache_width is None or is_compact(key) == (args.cache_width == "half"))
        and (args.rotary_dim is None or case_attrs(key)["rotary_dim"] == args.rotary_dim)
    ]
    if len(matches) != 1:
        parser.error(
            "Single-call profile needs exactly one representative scenario; refine selectors or use shell --dry-run"
        )
    return matches[0]
