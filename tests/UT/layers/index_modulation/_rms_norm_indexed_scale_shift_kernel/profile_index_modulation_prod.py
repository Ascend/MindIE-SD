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

"""Single forward of rms_norm_indexed_scale_shift for ``msprof op`` collection."""

import argparse

# Support namespace-package imports and direct script execution.
# pylint: disable=relative-beyond-top-level
if __package__:
    from .index_modulation_test_config import (
        add_scene_arguments,
        print_scene_configuration,
        scene_label,
        validate_scene_arguments,
    )
else:
    from index_modulation_test_config import (
        add_scene_arguments,
        print_scene_configuration,
        scene_label,
        validate_scene_arguments,
    )


def main():
    parser = argparse.ArgumentParser()
    add_scene_arguments(parser)
    parser.add_argument("--distribution", choices=("uniform", "normal"), default="uniform")
    parser.add_argument("--msprof-mode", action="store_true")
    args = parser.parse_args()
    validate_scene_arguments(parser, args)
    if print_scene_configuration(args):
        return
    if args.suite == "production":
        parser.error("Use run_index_modulation_perf.sh --suite production; each profiler process runs one scene")
    import torch

    if __package__:
        from .index_modulation_accuracy_common import (
            PROD_DTYPE,
            check_output_contract,
            init_device,
            log_inputs,
            make_inputs,
        )
    else:
        from index_modulation_accuracy_common import (
            PROD_DTYPE,
            check_output_contract,
            init_device,
            log_inputs,
            make_inputs,
        )

    device = init_device(args.device_id)
    from mindiesd import rms_norm_indexed_scale_shift

    inputs = make_inputs(
        args.rows,
        args.hidden_size,
        args.num_groups,
        PROD_DTYPE,
        device,
        args.distribution,
        seed=args.seed,
    )
    print(f"DUT=rms_norm_indexed_scale_shift scene={scene_label(args)} distribution={args.distribution}")
    log_inputs(*inputs, eps=args.eps)
    torch.npu.synchronize()
    output = rms_norm_indexed_scale_shift(*inputs, eps=args.eps)  # One target call; msprof owns warmup/replay.
    torch.npu.synchronize()
    check_output_contract(output, inputs)
    print("forward complete")


if __name__ == "__main__":
    main()
