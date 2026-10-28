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

import os

import pytest
import torch

from tests.ops.eagle_indexed_gate_norm_modulate.eagle_indexed_gate_norm_modulate_accuracy_common import (
    call_dut,
    compare,
    golden,
    init_device,
    make_inputs,
)


@pytest.fixture(name="device", scope="module")
def npu_device():
    if os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU":
        pytest.skip("eagle_indexed_gate_norm_modulate requires an NPU")
    return init_device(0)


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,hidden,index_dtype,strided",
    [
        (0, 129, torch.int32, False),
        (1, 1, torch.int64, False),
        (3, 63, torch.int32, False),
        (3, 64, torch.int64, False),
        (81, 65, torch.int64, False),
        (3, 127, torch.int32, False),
        (3, 128, torch.int64, False),
        (81, 129, torch.int32, False),
        (3, 255, torch.int64, True),
        (3, 3968, torch.int32, False),
        (3, 3969, torch.int64, False),
        (3, 4095, torch.int32, False),
        (3, 4096, torch.int64, False),
        (81, 4097, torch.int32, False),
        (3, 5376, torch.int64, False),
        (3, 6656, torch.int32, True),
        (3, 6144, torch.int32, False),
        (3, 6145, torch.int64, False),
        (3, 6655, torch.int32, True),
        (81, 6656, torch.int64, False),
        (3, 6657, torch.int32, False),
        (3, 7167, torch.int32, True),
        (81, 7168, torch.int64, False),
    ],
)
def test_accuracy(device, distribution, input_dtype, output_dtype, rows, hidden, index_dtype, strided):
    inputs = make_inputs(rows, hidden, input_dtype, device, distribution, index_dtype=index_dtype, strided=strided)
    print("golden=CPU FP32 phase simulation; metrics=CPU FP64")
    expected = golden(tuple(x.cpu() for x in inputs), output_dtype)
    actual = call_dut(inputs, output_dtype)
    compare(tuple(x.cpu() for x in actual), expected)


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_default_output_dtype(device, distribution, dtype, eps):
    inputs = make_inputs(3, 129, dtype, device, distribution)
    print("golden=CPU FP32 phase simulation; metrics=CPU FP64")
    expected = golden(tuple(x.cpu() for x in inputs), dtype, eps)
    actual = call_dut(inputs, None, eps)
    compare(tuple(x.cpu() for x in actual), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("explicit_bias", [False, True])
def test_default_eps(device, dtype, distribution, explicit_bias):
    inputs = make_inputs(3, 129, dtype, device, distribution)
    expected = golden(tuple(x.cpu() for x in inputs), dtype, 1e-6)
    residual, gate, branch, weight, shift, scale, indices = inputs
    kwargs = {"bias": None} if explicit_bias else {}
    actual = torch.ops.mindiesd.eagle_indexed_gate_norm_modulate(
        residual, gate, branch, shift, scale, indices, weight=weight, **kwargs
    )
    assert len(actual) == 2
    for value in actual:
        assert value.shape == residual.shape and value.dtype == dtype and value.is_contiguous()
    compare(tuple(x.cpu() for x in actual), expected)


@pytest.mark.parametrize("invalid", ["omitted_weight", "none_weight", "norm_type", "bias"])
def test_norm_contract(device, invalid):
    residual, gate, branch, weight, shift, scale, indices = make_inputs(3, 129, torch.float16, device, "uniform")
    kwargs = {} if invalid == "omitted_weight" else {"weight": weight}
    if invalid == "none_weight":
        kwargs["weight"] = None
    elif invalid == "norm_type":
        kwargs["norm_type"] = 1
    elif invalid == "bias":
        kwargs["bias"] = torch.zeros_like(weight)
    message = "only norm_type=0" if invalid == "norm_type" else "weight is required"
    if invalid == "bias":
        message = "bias must be None"
    with pytest.raises((RuntimeError, ValueError), match=message):
        torch.ops.mindiesd.eagle_indexed_gate_norm_modulate(residual, gate, branch, shift, scale, indices, **kwargs)


@pytest.mark.parametrize("invalid", ["hidden", "eps", "dtype", "shape", "indices"])
def test_invalid_contract(device, invalid):
    inputs = list(make_inputs(3, 129, torch.float16, device, "uniform"))
    eps, output_dtype = 1e-6, torch.float16
    if invalid == "hidden":
        inputs = list(make_inputs(3, 7169, torch.float16, device, "uniform"))
    elif invalid == "eps":
        eps = 0.0
    elif invalid == "dtype":
        output_dtype = torch.float32
    elif invalid == "shape":
        inputs[2] = inputs[2][:2]
    else:
        inputs[-1] = inputs[-1].to(torch.float32)
    with pytest.raises((RuntimeError, ValueError)):
        call_dut(tuple(inputs), output_dtype, eps)
