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
    from .index_modulation_accuracy_common import EPS, call_dut, compare, golden, init_device, make_inputs
else:
    from index_modulation_accuracy_common import EPS, call_dut, compare, golden, init_device, make_inputs


@pytest.fixture(name="device", scope="module")
def fixture_device():
    if os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU":
        pytest.skip("Kernel accuracy requires an NPU")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        pytest.fail("Unset ASCEND_RT_VISIBLE_DEVICES; small-shape tests use physical NPU 0")
    pytest.importorskip("torch_npu")
    return init_device(0)


@pytest.mark.parametrize("rows", [0, 2], ids=["empty-rows", "nonempty-rows"])
@pytest.mark.parametrize("hidden_size", [0, 8193, 16384])
def test_invalid_hidden_size(rows, hidden_size, monkeypatch, contract_dut):
    dut = contract_dut

    # Invalid block widths must be rejected even when no rows need processing.
    x = torch.empty((rows, hidden_size), dtype=torch.bfloat16)
    weight = torch.empty(hidden_size, dtype=torch.bfloat16)
    shift = torch.empty((1, hidden_size), dtype=torch.bfloat16)
    scale = torch.empty((1, hidden_size), dtype=torch.bfloat16)
    indices = torch.zeros(rows, dtype=torch.int64)

    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid hidden_size must be rejected before output allocation or device queries")

    monkeypatch.setattr(dut.torch, "empty", unexpected_call)
    monkeypatch.setattr(dut, "_get_num_vector_cores", unexpected_call)
    with pytest.raises(ValueError, match="hidden_size"):
        dut.rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, EPS)


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
    path = Path(__file__).resolve().parents[5] / "mindiesd/layers/index_modulation.py"
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


@pytest.mark.parametrize("launch_fails", [False, True])
def test_launch_device_context(contract_dut, monkeypatch, launch_fails):
    """Check dispatch and restoration on CPU, including a kernel launch error."""
    active_device = 0
    target = types.SimpleNamespace(type="npu", index=1)
    tensor = types.SimpleNamespace(shape=(3, 16), dtype=torch.bfloat16, device=target, stride=lambda dim: 16)
    output = object()
    launches = []

    @contextmanager
    def device_context(device):
        nonlocal active_device
        previous = active_device
        active_device = device.index
        try:
            yield
        finally:
            active_device = previous

    def core_count(device):
        assert active_device == device.index == 1
        return 2

    class Kernel:
        def __getitem__(self, grid):
            assert grid == (2,)
            return self.launch

        @staticmethod
        def launch(**kwargs):
            assert active_device == 1
            assert kwargs["x_ptr"] is tensor and kwargs["output_ptr"] is output
            launches.append(kwargs)
            if launch_fails:
                raise RuntimeError("simulated launch failure")

    # Input validation is covered separately; isolate device dispatch here.
    monkeypatch.setattr(contract_dut, "_check_inputs", lambda *args: None)
    monkeypatch.setattr(contract_dut.torch, "empty", lambda *args, **kwargs: output)
    monkeypatch.setattr(contract_dut.torch, "npu", types.SimpleNamespace(device=device_context), raising=False)
    monkeypatch.setattr(contract_dut, "_get_num_vector_cores", core_count)
    monkeypatch.setattr(contract_dut, "_rms_norm_indexed_scale_shift_kernel", Kernel())
    if launch_fails:
        with pytest.raises(RuntimeError, match="simulated launch failure"):
            contract_dut.rms_norm_indexed_scale_shift(tensor, tensor, tensor, tensor, tensor, EPS)
    else:
        assert contract_dut.rms_norm_indexed_scale_shift(tensor, tensor, tensor, tensor, tensor, EPS) is output
    assert len(launches) == 1
    assert active_device == 0


@pytest.mark.parametrize(
    "invalid",
    [
        "x_rank",
        "weight_shape",
        "shift_rank",
        "scale_shape",
        "table_width",
        "empty_groups",
        "indices_rank",
        "indices_length",
        "indices_dtype",
        "x_dtype",
        "weight_dtype",
        "shift_dtype",
        "scale_dtype",
        "device",
        "zero_eps",
        "negative_eps",
        "nan_eps",
        "inf_eps",
        "overflow_eps",
        "underflow_eps",
    ],
)
def test_invalid_contract(contract_dut, monkeypatch, invalid):
    inputs = list(make_inputs(3, 16, 3, torch.bfloat16, "cpu", "uniform"))
    eps = EPS
    if invalid == "x_rank":
        inputs[0] = inputs[0].reshape(1, 3, 16)
    elif invalid == "weight_shape":
        inputs[1] = inputs[1][:-1]
    elif invalid == "shift_rank":
        inputs[2] = inputs[2].flatten()
    elif invalid == "scale_shape":
        inputs[3] = inputs[3][:-1]
    elif invalid == "table_width":
        inputs[2], inputs[3] = inputs[2][:, :-1], inputs[3][:, :-1]
    elif invalid == "empty_groups":
        inputs[2], inputs[3] = inputs[2][:0], inputs[3][:0]
    elif invalid == "indices_rank":
        inputs[4] = inputs[4].reshape(3, 1)
    elif invalid == "indices_length":
        inputs[4] = inputs[4][:-1]
    elif invalid == "indices_dtype":
        inputs[4] = inputs[4].float()
    elif invalid in ("x_dtype", "weight_dtype", "shift_dtype", "scale_dtype"):
        index = ("x_dtype", "weight_dtype", "shift_dtype", "scale_dtype").index(invalid)
        inputs[index] = inputs[index].half()
    elif invalid == "device":
        inputs[1] = inputs[1].to("meta")
    else:
        eps = {
            "zero_eps": 0.0,
            "negative_eps": -1.0,
            "nan_eps": float("nan"),
            "inf_eps": float("inf"),
            "overflow_eps": 1e40,
            "underflow_eps": 1e-50,
        }[invalid]
    forbid_launch(monkeypatch, contract_dut)
    with pytest.raises(ValueError):
        contract_dut.rms_norm_indexed_scale_shift(*inputs, eps)


@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("index", range(4), ids=["x", "weight", "shift", "scale"])
def test_invalid_last_stride(contract_dut, monkeypatch, rows, index):
    inputs = list(make_inputs(rows, 16, 3, torch.bfloat16, "cpu", "uniform"))
    shape = (*inputs[index].shape[:-1], 32)
    inputs[index] = torch.empty(shape, dtype=torch.bfloat16)[..., ::2]
    forbid_launch(monkeypatch, contract_dut)
    with pytest.raises(ValueError, match="last dimension"):
        contract_dut.rms_norm_indexed_scale_shift(*inputs, EPS)


def test_reject_cpu_execution(contract_dut, monkeypatch):
    inputs = make_inputs(3, 16, 3, torch.bfloat16, "cpu", "uniform")
    forbid_launch(monkeypatch, contract_dut)
    with pytest.raises(ValueError, match="NPU"):
        contract_dut.rms_norm_indexed_scale_shift(*inputs, EPS)
