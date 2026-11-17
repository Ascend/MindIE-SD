#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse

import torch

from eagle_indexed_gate_norm_modulate_accuracy_common import DTYPES, PROD_SHAPES, call_dut, init_device, make_inputs


def main():
    parser = argparse.ArgumentParser(description="One fused EagleIndexedGateNormModulate invocation for msprof op")
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, choices=(2048, 5376, 6656, 7168), default=2048)
    parser.add_argument("--rows", type=int, choices=PROD_SHAPES, default=13640)
    parser.add_argument("--input-dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--output-dtype", choices=("bf16",), default="bf16")
    parser.add_argument(
        "--msprof-mode", action="store_true", help="One target invocation; warm-up is owned by msprof op"
    )
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be nonnegative")
    device = init_device(args.device_id)
    # Host preparation keeps random-generation kernels outside the profiling target.
    host_inputs = make_inputs(
        args.rows, args.hidden_size, DTYPES[args.input_dtype], "cpu", "normal", indexed_rows=PROD_SHAPES[args.rows]
    )
    inputs = tuple(x.to(device) for x in host_inputs)
    torch.npu.synchronize()
    outputs = call_dut(inputs, DTYPES[args.output_dtype])
    torch.npu.synchronize()
    assert len(outputs) == 2
    for out in outputs:
        assert out.shape == inputs[0].shape and out.dtype == DTYPES[args.output_dtype] and out.is_contiguous()


if __name__ == "__main__":
    main()
