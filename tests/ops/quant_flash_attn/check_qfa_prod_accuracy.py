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

"""Large QFA accuracy on NPU and fixed-input, bitwise stability stress."""

import json
from datetime import datetime
from pathlib import Path

from tests.ops.quant_flash_attn.profile_qfa_prod import cases_from_args, parser


def main():
    argument_parser = parser(default_layout="all")
    argument_parser.description = __doc__
    argument_parser.add_argument("--mode", choices=("all", "accuracy", "stress"), default="all")
    argument_parser.add_argument("--stress-runs", type=int, default=1000)
    argument_parser.add_argument("--output-dir", type=Path)
    argument_parser.add_argument("--validate-only", action="store_true", help="Validate arguments without loading NPU")
    args = argument_parser.parse_args()
    if args.stress_runs <= 0 or args.device_id < 0:
        argument_parser.error("stress-runs must be positive and device-id nonnegative")
    cases = cases_from_args(args)
    if args.validate_only:
        return
    from tests.ops.quant_flash_attn.qfa_accuracy_common import (
        DISTRIBUTIONS,
        REPO_ROOT,
        bind,
        check_outputs,
        compare,
        initialize,
        make_inputs,
        npu_golden,
        stress,
    )
    from tests.ops.quant_flash_attn.qfa_tolerance import BUDGET

    output_dir = (
        args.output_dir or REPO_ROOT / "log/quant_flash_attn" / datetime.now().strftime("accuracy_%Y%m%d_%H%M%S_%f")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    for case in cases:
        (output_dir / case.label).mkdir()
    torch, torch_npu, device = initialize(args.device_id)
    completed = []
    modes = ("accuracy", "stress") if args.mode == "all" else (args.mode,)
    with torch.inference_mode():
        for mode in modes:
            for case in cases:
                for distribution in DISTRIBUTIONS:
                    native, encodings = make_inputs(case, distribution, args.seed, torch, device)
                    del encodings
                    invoke, inputs = bind(case, native, torch, torch_npu, device)
                    print(f"START {mode} {case.label} distribution={distribution} seed={args.seed}", flush=True)
                    if mode == "accuracy":
                        reference = npu_golden(case, native, torch)
                        outputs = invoke()
                        torch.npu.synchronize()
                        check_outputs(outputs, case, torch, device)
                        report = compare(outputs[0], reference, BUDGET, small=False)
                        (output_dir / case.label / f"accuracy_{distribution}.json").write_text(
                            json.dumps(report, indent=2), encoding="utf-8"
                        )
                        del outputs, reference
                        line = f"PASS accuracy {case.label} distribution={distribution} (cosine/norm hard gates)"
                    else:
                        stress(invoke, inputs, case, args.stress_runs, torch, device)
                        line = f"PASS stress {case.label} distribution={distribution} runs={args.stress_runs}"
                    completed.append(line)
                    del invoke, inputs, native
    summary = "\n".join(completed) + "\n"
    (output_dir / "final_summary.txt").write_text(summary, encoding="utf-8")
    print(f"Results: {output_dir}", flush=True)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
