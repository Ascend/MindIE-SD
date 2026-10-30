#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Dependency-free CLI configuration shared by accuracy and performance."""

import copy
import math
import os
import struct

SEED = 20261017
PROD_ROWS = 13640
PROD_HIDDEN_SIZE = 5376
PROD_NUM_GROUPS = 3
EPS = 1e-6
PRODUCTION_SHAPES = tuple(
    (rows, hidden_size, groups) for rows, groups in ((13640, 3), (5672, 12)) for hidden_size in (2048, 5376, 6656, 7168)
)


PRODUCTION_ACCURACY_DISTRIBUTIONS = (
    "normal",
    "uniform",
    "normal",
    "uniform",
    "uniform",
    "normal",
    "uniform",
    "normal",
)


def scene_distributions(args, scene):
    if args.distribution != "all":
        return (args.distribution,)
    if args.suite == "production":
        key = (scene.rows, scene.hidden_size, scene.num_groups)
        return (PRODUCTION_ACCURACY_DISTRIBUTIONS[PRODUCTION_SHAPES.index(key)],)
    return ("uniform", "normal")


def add_scene_arguments(parser):
    parser.add_argument("--device-id", type=int, default=0, help="Physical NPU ID from npu-smi")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None, help="Logical hidden size / block_n in [1, 8192]")
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument(
        "--suite",
        choices=("default", "production"),
        default="default",
        help="production: 8 shapes, each run with the selected distributions",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="Print rows/hidden/groups for each scene without loading NPU dependencies",
    )
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and print the scene without loading NPU dependencies",
    )


def validate_scene_arguments(parser, args):
    if args.suite == "production" and any(
        value is not None for value in (args.rows, args.hidden_size, args.num_groups)
    ):
        parser.error("--suite production cannot be combined with --rows/--hidden-size/--num-groups")
    args.rows = PROD_ROWS if args.rows is None else args.rows
    args.hidden_size = PROD_HIDDEN_SIZE if args.hidden_size is None else args.hidden_size
    args.num_groups = PROD_NUM_GROUPS if args.num_groups is None else args.num_groups
    if args.device_id < 0:
        parser.error("--device-id must be a nonnegative physical NPU ID")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Unset ASCEND_RT_VISIBLE_DEVICES; use --device-id with a physical NPU ID")
    if args.rows <= 0 or args.hidden_size <= 0 or args.num_groups <= 0:
        parser.error("--rows, --hidden-size and --num-groups must be positive")
    if args.hidden_size > 8192:
        parser.error(f"--hidden-size (block_n) must be <= 8192; got {args.hidden_size}")
    try:
        eps_fp32 = struct.unpack("f", struct.pack("f", args.eps))[0]
    except OverflowError:
        eps_fp32 = math.inf
    if not math.isfinite(eps_fp32) or eps_fp32 <= 0:
        parser.error("--eps must be finite and positive in FP32")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2^32)")
    if not __debug__:
        parser.error("Run without Python optimization: accuracy checks require assertions")


def scene_label(args):
    return (
        f"rms_norm_indexed_scale_shift/rows={args.rows}/hidden={args.hidden_size}"
        f"/groups={args.num_groups}/bf16/eps={args.eps:g}/seed={args.seed}"
    )


def iter_scenes(args):
    shapes = PRODUCTION_SHAPES if args.suite == "production" else ((args.rows, args.hidden_size, args.num_groups),)
    for rows, hidden_size, groups in shapes:
        scene = copy.copy(args)
        scene.rows, scene.hidden_size, scene.num_groups = rows, hidden_size, groups
        yield scene


def print_scene_configuration(args, *, accuracy=False):
    if not (args.validate_only or args.list_scenes):
        return False
    for scene in iter_scenes(args):
        if args.list_scenes:
            line = f"{scene.rows} {scene.hidden_size} {scene.num_groups}"
            distribution = scene_distributions(args, scene)[0] if accuracy and args.suite == "production" else "all"
            print(f"{line} {distribution}" if accuracy else line)
        else:
            print(scene_label(scene))
    return True
