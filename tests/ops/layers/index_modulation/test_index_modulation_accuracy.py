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

"""Small-shape accuracy for rms_norm_indexed_scale_shift against a CPU FP32 golden."""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch

# Support namespace-package imports and direct script execution.
# pylint: disable=relative-beyond-top-level
if __package__:
    from .index_modulation_accuracy_common import EPS, call_dut, compare, golden, make_inputs
else:
    from index_modulation_accuracy_common import EPS, call_dut, compare, golden, make_inputs


@pytest.fixture(name="device", scope="module")
def fixture_device():
    if os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU":
        pytest.skip("Kernel accuracy requires an NPU")
    from tests.ops.device_support import init_test_device

    device = init_test_device()
    import mindiesd  # noqa: F401

    return device


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize(
    "rows,hidden_size,num_groups",
    [
        pytest.param(1, 1, 1, id="min-hidden1-g1"),
        pytest.param(1, 17, 3, id="single-row-width17-g3"),
        pytest.param(2, 3072, 3, id="width3072-g3"),
        pytest.param(2, 5376, 3, id="production-width5376-g3"),
        pytest.param(2, 8192, 3, id="width8192-g3"),
    ],
)
def test_accuracy(device, distribution, rows, hidden_size, num_groups):
    inputs = make_inputs(rows, hidden_size, num_groups, torch.bfloat16, device, distribution)
    expected = golden(*(t.cpu() for t in inputs))
    actual = call_dut(*inputs)
    compare(actual.cpu(), expected)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_input_on_noncurrent_device(device, index_dtype):
    if torch.npu.device_count() < 2:
        pytest.skip("Device-context regression requires two NPUs")
    from mindiesd import rms_norm_indexed_scale_shift

    target = torch.device(f"npu:{1 if device.index == 0 else 0}")
    with torch.npu.device(target):
        inputs = make_inputs(3, 17, 3, torch.bfloat16, target, "normal", index_dtype=index_dtype)
        expected = golden(*(value.cpu() for value in inputs))
        baseline = rms_norm_indexed_scale_shift(*inputs, eps=EPS).cpu()
    with torch.npu.device(device):
        caller_device = torch.npu.current_device()
        assert caller_device != target.index
        actual = rms_norm_indexed_scale_shift(*inputs, eps=EPS)
        assert torch.npu.current_device() == caller_device
        assert actual.device == target
        torch.npu.synchronize(target)
        actual_cpu = actual.cpu()
    compare(actual_cpu, expected)
    assert torch.equal(actual_cpu.view(torch.int16), baseline.view(torch.int16))


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_multi_row_loop(device, distribution):
    # Fixed shape from the PR report; crosses the 56-core boundary on Ascend 950.
    rows = 57
    inputs = make_inputs(rows, 17, 3, torch.bfloat16, device, distribution)
    expected = golden(*(t.cpu() for t in inputs))
    actual = call_dut(*inputs)
    compare(actual.cpu(), expected)


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_production_width_nan_row_gap(device, distribution):
    # The logical block width is 5376; the physical input row stride is 8192.
    # NaNs in the storage gap must not be loaded or included in the reduction.
    # Use the same reported shape on both hardware models.
    rows, hidden_size, storage_width = 113, 5376, 8192
    x, weight, shift, scale, indices = make_inputs(rows, hidden_size, 3, torch.bfloat16, device, distribution)
    backing = torch.full((rows, storage_width), float("nan"), dtype=torch.bfloat16, device=device)
    backing[:, :hidden_size].copy_(x)
    inputs = (backing[:, :hidden_size], weight, shift, scale, indices)
    expected = golden(*(t.cpu() for t in inputs))
    first = None
    for _ in range(3):
        actual = call_dut(*inputs).cpu()
        compare(actual, expected)
        if first is None:
            first = actual
        else:
            assert torch.equal(first.view(torch.int16), actual.view(torch.int16)), (
                "Strided production-width outputs must be bitwise equal across calls"
            )


def test_empty_rows(device):
    x = torch.empty((0, 16), dtype=torch.bfloat16, device=device)
    weight = torch.empty(16, dtype=torch.bfloat16, device=device)
    shift = torch.empty((1, 16), dtype=torch.bfloat16, device=device)
    scale = torch.empty((1, 16), dtype=torch.bfloat16, device=device)
    indices = torch.empty(0, dtype=torch.int64, device=device)
    inputs = (x, weight, shift, scale, indices)
    out = call_dut(*inputs, eps=EPS)
    expected = golden(*(t.cpu() for t in inputs))
    assert torch.equal(out.cpu(), expected), "Empty output must match the empty golden"


@pytest.fixture(name="contract_dut")
def fixture_contract_dut(monkeypatch):
    """Load real host validation without initializing Triton or the NPU package."""
    utils_name = "_index_modulation_contract.triton_utils"
    utils = types.ModuleType(utils_name)
    utils._HAS_TRITON = True
    utils.triton = types.SimpleNamespace(jit=lambda function: function)
    utils.tl = types.SimpleNamespace(constexpr=object)
    runtime = types.ModuleType("triton.runtime")
    runtime.driver = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, utils_name, utils)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    path = Path(__file__).resolve().parents[4] / "mindiesd/layers/index_modulation.py"
    spec = importlib.util.spec_from_file_location("_index_modulation_contract.index_modulation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forbid_launch(monkeypatch, dut):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid inputs must be rejected before allocation, device query or launch")

    monkeypatch.setattr(dut.torch, "empty", unexpected)
    monkeypatch.setattr(dut, "_get_num_vector_cores", unexpected)
    monkeypatch.setattr(dut, "_rms_norm_indexed_scale_shift_kernel", unexpected, raising=False)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("hidden_size", [5376])
def test_chunked_tables(device, index_dtype, distribution, hidden_size):
    inputs = make_inputs(
        25, hidden_size, 12, torch.bfloat16, device, distribution, index_dtype=index_dtype, strided=True
    )
    before = [value.cpu().clone() for value in inputs]
    expected = golden(*before)
    actual = call_dut(*inputs).cpu()
    compare(actual, expected)
    assert torch.equal(actual.view(torch.int16), call_dut(*inputs).cpu().view(torch.int16))
    for value, original in zip(inputs, before):
        assert torch.equal(value.cpu(), original), "Input modified by DUT"


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("hidden_size", [63, 64, 65, 4097])
def test_width_boundaries(device, distribution, hidden_size):
    inputs = make_inputs(3, hidden_size, 3, torch.bfloat16, device, distribution)
    compare(call_dut(*inputs).cpu(), golden(*(value.cpu() for value in inputs)))
