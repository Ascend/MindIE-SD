#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""One direct QuantFlashAttn call; metadata and MXFP4 encoding precede capture."""

import argparse


def parser(default_layout="BNSD"):
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--device-id", type=int, default=0)
    result.add_argument("--batch", type=int, default=1)
    result.add_argument("--q-heads", type=int, default=32)
    result.add_argument("--kv-heads", type=int, default=4)
    result.add_argument("--q-seq", type=int, default=2048)
    result.add_argument("--kv-seq", type=int, default=32768)
    result.add_argument(
        "--layout",
        choices=("all", "BNSD", "BSND"),
        default=default_layout,
        help="Suite layout selection; a single profiling call requires BNSD or BSND",
    )
    result.add_argument("--seed", type=int, default=21)
    return result


def case_from_args(args, layout=None):
    from qfa_accuracy_common import Case

    return Case(
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        q_seq=args.q_seq,
        kv_seq=args.kv_seq,
        layout=layout or args.layout,
    )


def cases_from_args(args):
    layouts = ("BNSD", "BSND") if args.layout == "all" else (args.layout,)
    return [case_from_args(args, layout) for layout in layouts]


def main():
    argument_parser = parser()
    argument_parser.add_argument("--distribution", choices=("uniform", "normal"), default="uniform")
    argument_parser.add_argument(
        "--msprof-mode", action="store_true", help="Issue exactly one target operator call for msprof op replay"
    )
    args = argument_parser.parse_args()
    if args.layout == "all":
        argument_parser.error(
            "a single profiling call requires --layout BNSD or BSND; use run_qfa_msprof_op.sh for all"
        )
    case = case_from_args(args)
    from qfa_accuracy_common import bind, check_outputs, initialize, make_inputs

    torch, torch_npu, device = initialize(args.device_id)
    with torch.inference_mode():
        native, _ = make_inputs(case, args.distribution, args.seed, torch, device)
        invoke, _ = bind(case, native, torch, torch_npu, device)
        output = invoke()
        torch.npu.synchronize()
        check_outputs(output, case, torch, device)


if __name__ == "__main__":
    main()
