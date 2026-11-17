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

from tests.ops.device_support import init_test_device, require_support
from tests.ops.eagle_indexed_gate_norm_modulate.accuracy_cases import (
    ACCURACY_CASES,
    DEFAULT_OUTPUT_CASES,
    DEFAULT_EPS_CASES,
)

from tests.ops.eagle_indexed_gate_norm_modulate.eagle_indexed_gate_norm_modulate_accuracy_common import (
    call_dut,
    compare,
    golden,
    make_inputs,
)


@pytest.fixture(name="device", scope="module")
def npu_device():
    if os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU":
        pytest.skip("eagle_indexed_gate_norm_modulate requires an NPU")
    selected = init_test_device()
    import mindiesd  # noqa: F401

    return selected


@pytest.fixture(name="supported_soc", scope="module")
def supported_soc_fixture(device):
    require_support("eagle_indexed_gate_norm_modulate", kind="positive")


@pytest.mark.parametrize("distribution,input_dtype,output_dtype,rows,hidden,index_dtype,strided", ACCURACY_CASES)
@pytest.mark.usefixtures("supported_soc")
def test_accuracy(device, distribution, input_dtype, output_dtype, rows, hidden, index_dtype, strided):
    inputs = make_inputs(rows, hidden, input_dtype, device, distribution, index_dtype=index_dtype, strided=strided)
    print("golden=CPU FP32 phase simulation; metrics=CPU FP64")
    expected = golden(tuple(x.cpu() for x in inputs), output_dtype)
    actual = call_dut(inputs, output_dtype)
    compare(tuple(x.cpu() for x in actual), expected)


@pytest.mark.parametrize("distribution,dtype,eps", DEFAULT_OUTPUT_CASES)
@pytest.mark.usefixtures("supported_soc")
def test_default_output_dtype(device, distribution, dtype, eps):
    inputs = make_inputs(3, 129, dtype, device, distribution)
    print("golden=CPU FP32 phase simulation; metrics=CPU FP64")
    expected = golden(tuple(x.cpu() for x in inputs), dtype, eps)
    actual = call_dut(inputs, None, eps)
    compare(tuple(x.cpu() for x in actual), expected)


@pytest.mark.parametrize("dtype,distribution,explicit_bias", DEFAULT_EPS_CASES)
@pytest.mark.usefixtures("supported_soc")
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
