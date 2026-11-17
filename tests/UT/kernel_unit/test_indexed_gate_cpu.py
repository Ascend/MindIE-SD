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

"""CPU contracts and numerical interpretation of the indexed gate kernel body.

These tests exercise host decisions and the original Python kernel function.
Compiler allocation, device arithmetic and concurrent launches remain NPU tests.
"""

from contextlib import contextmanager
import importlib
from types import SimpleNamespace

import pytest
import torch

from UT.kernel_unit.triton_cpu import install


@pytest.fixture(name="dut")
def fixture_dut():
    return importlib.import_module("mindiesd.layers.index_modulation")


def gate_inputs(rows=3, width=8):
    return {
        "output": torch.full((rows, width), torch.nan, dtype=torch.bfloat16),
        "x": torch.ones(rows, width, dtype=torch.bfloat16),
        "gate": torch.ones(5, width, dtype=torch.bfloat16),
        "other": torch.ones(rows, width, dtype=torch.bfloat16),
        "indices": torch.arange(rows, dtype=torch.int64) % 5,
    }


@pytest.mark.parametrize(
    "invalid,message",
    [
        ("rank", "x must have shape"),
        ("empty_rows", "N > 0 and H > 0"),
        ("empty_width", "N > 0 and H > 0"),
        ("output_shape", "same.*shape"),
        ("other_shape", "same.*shape"),
        ("gate_rank", "gate must have shape"),
        ("gate_empty", "gate must have shape"),
        ("gate_width", "gate must have shape"),
        ("indices_rank", "indices must have shape"),
        ("indices_length", "indices must have shape"),
        ("indices_dtype", "indices must have shape"),
        ("output_overlap", "output rows must not overlap"),
    ],
)
def test_gate_rejects_invalid_shapes(dut, invalid, message):
    inputs = gate_inputs()
    if invalid == "rank":
        inputs["x"] = inputs["x"].flatten()
    elif invalid == "empty_rows":
        inputs = gate_inputs(rows=0)
    elif invalid == "empty_width":
        inputs = gate_inputs(width=0)
    elif invalid in ("output_shape", "other_shape"):
        name = invalid.split("_")[0]
        inputs[name] = inputs[name][:-1]
    elif invalid == "gate_rank":
        inputs["gate"] = inputs["gate"].flatten()
    elif invalid == "gate_empty":
        inputs["gate"] = inputs["gate"][:0]
    elif invalid == "gate_width":
        inputs["gate"] = inputs["gate"][:, :-1]
    elif invalid == "indices_rank":
        inputs["indices"] = inputs["indices"][:, None]
    elif invalid == "indices_length":
        inputs["indices"] = inputs["indices"][:-1]
    elif invalid == "indices_dtype":
        inputs["indices"] = inputs["indices"].float()
    else:
        inputs["output"] = inputs["output"][:1].expand(3, 8)
    with pytest.raises(ValueError, match=message):
        dut._check_indexed_gate_inputs(**inputs)


@pytest.mark.parametrize("name", ["output", "x", "gate", "other"])
@pytest.mark.parametrize("invalid", ["dtype", "stride"])
def test_gate_rejects_floating_tensor_contracts(dut, name, invalid):
    inputs = gate_inputs()
    if invalid == "dtype":
        inputs[name] = inputs[name].float()
        message = "must all be bfloat16"
    else:
        rows, width = inputs[name].shape
        inputs[name] = torch.empty(rows, width * 2, dtype=torch.bfloat16)[:, ::2]
        message = "contiguous along the last dimension"
    with pytest.raises(ValueError, match=message):
        dut._check_indexed_gate_inputs(**inputs)


@pytest.mark.parametrize("name", ["output", "gate", "other", "indices"])
def test_gate_rejects_mixed_devices(dut, name):
    inputs = gate_inputs()
    inputs[name] = inputs[name].to("meta")
    with pytest.raises(ValueError, match="same device"):
        dut._check_indexed_gate_inputs(**inputs)


@pytest.mark.parametrize("name", ["x", "gate", "other", "indices"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gate_rejects_output_input_storage_alias(dut, name, index_dtype):
    inputs = gate_inputs()
    inputs["indices"] = inputs["indices"].to(index_dtype)
    if name == "indices":
        inputs[name] = inputs["output"].view(index_dtype).flatten()[:3]
    else:
        inputs[name] = inputs["output"]
    with pytest.raises(ValueError, match=f"output must not share storage with {name}"):
        dut._check_indexed_gate_inputs(**inputs)


def test_gate_rejects_even_disjoint_views_of_output_storage(dut):
    inputs = gate_inputs()
    storage = torch.empty(48, dtype=torch.bfloat16)
    inputs["output"] = storage[:24].reshape(3, 8)
    inputs["x"] = storage[24:].reshape(3, 8)
    with pytest.raises(ValueError, match="output must not share storage with x"):
        dut._check_indexed_gate_inputs(**inputs)


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gate_accepts_read_only_aliases_and_row_gaps(dut, rows, index_dtype):
    inputs = gate_inputs(rows=rows)
    storage = torch.empty(rows, 15, dtype=torch.bfloat16)
    inputs["x"] = storage[:, 3:11]
    inputs["gate"] = inputs["x"]
    inputs["other"] = inputs["x"]
    inputs["output"] = torch.empty(rows, 19, dtype=torch.bfloat16)[:, 4:12]
    inputs["indices"] = torch.zeros(rows * 2, dtype=index_dtype)[::2]
    dut._check_indexed_gate_inputs(**inputs)


@pytest.mark.parametrize(
    "name,vector_count,expected,error",
    [
        ("Ascend910B2", 40, (40, 183 * 1024), None),
        ("Ascend910_9392", 48, (48, 183 * 1024), None),
        ("Ascend950", 64, (64, 216 * 1024), None),
        ("ascend950pr_9599", 64, (64, 216 * 1024), None),
        ("Ascend310P3", 8, None, "Unsupported NPU device"),
        ("Ascend950", 0, None, "Invalid NPU resources"),
        ("Ascend910B2", -1, None, "Invalid NPU resources"),
    ],
)
def test_gate_resource_mapping_and_cache(dut, monkeypatch, name, vector_count, expected, error):
    calls = []
    active = []

    @contextmanager
    def device_context(index):
        active.append(index)
        calls.append(("enter", index))
        try:
            yield
        finally:
            calls.append(("exit", active.pop()))

    def cores():
        assert active == [2]
        calls.append(("cores", 2))
        return vector_count

    def device_name(index):
        assert active == [2] and index == 2
        return name

    dut._get_npu_resources.cache_clear()
    monkeypatch.setattr(torch.npu, "device", device_context)
    monkeypatch.setattr(torch.npu, "get_device_name", device_name)
    monkeypatch.setattr(dut, "get_vectorcore_num", cores)
    try:
        if error:
            with pytest.raises(RuntimeError, match=error):
                dut._get_npu_resources(2)
        else:
            assert dut._get_npu_resources(2) == expected
            assert dut._get_npu_resources(2) == expected
        assert not active
        assert calls == [("enter", 2), ("cores", 2), ("exit", 2)]
    finally:
        dut._get_npu_resources.cache_clear()


@pytest.mark.parametrize(
    "width,ub_size,expected",
    [
        (1, 183 * 1024, (180, 32)),
        (32, 183 * 1024, (180, 32)),
        (33, 183 * 1024, (90, 64)),
        (2912, 183 * 1024, (2, 2912)),
        (2944, 183 * 1024, (1, 2944)),
        (8192, 183 * 1024, (1, 5824)),
        (1, 216 * 1024, (212, 32)),
        (5376, 216 * 1024, (1, 5376)),
        (8192, 216 * 1024, (1, 6880)),
    ],
)
def test_gate_tile_boundaries_and_budget(dut, width, ub_size, expected):
    block_rows, block_cols = dut._tile_shape(width, ub_size)
    assert (block_rows, block_cols) == expected
    assert block_rows > 0 and block_cols > 0 and block_cols % 32 == 0
    # The public tiling allowance includes 16 bytes per element and one int64 index per row, in two stages.
    assert 2 * block_rows * (16 * block_cols + 8) <= ub_size
    if block_rows > 1:
        assert block_cols >= width


def padded_tensor(rows, width, padding, offset):
    storage = torch.full((rows, width + padding), torch.nan, dtype=torch.bfloat16)
    return storage, storage[:, offset : offset + width]


@pytest.mark.parametrize(
    "rows,width,block_rows,block_cols,programs",
    [(3, 17, 1, 32, 2), (2, 65, 1, 32, 4), (5, 17, 2, 32, 2), (3, 65, 4, 96, 1)],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gate_original_kernel_math_and_masks(
    dut, monkeypatch, rows, width, block_rows, block_cols, programs, index_dtype
):
    language = install(monkeypatch, dut)
    # This kernel also needs the scalar minimum primitive, which other vector bodies do not use.
    monkeypatch.setattr(
        language,
        "minimum",
        lambda left, right: torch.minimum(torch.as_tensor(left), torch.as_tensor(right)),
        raising=False,
    )
    x_storage, x = padded_tensor(rows, width, 7, 3)
    gate_storage, gate = padded_tensor(5, width, 11, 4)
    other_storage, other = padded_tensor(rows, width, 13, 5)
    output_storage, output = padded_tensor(rows, width, 9, 4)
    x.copy_((torch.arange(rows * width).reshape(rows, width) % 17 - 8) / 8)
    gate.copy_((torch.arange(5 * width).reshape(5, width) % 13 - 6) / 4)
    other.copy_((torch.arange(rows * width).reshape(rows, width) % 11 - 5) / 8)
    indices_storage = torch.full((rows * 2,), 99, dtype=index_dtype)
    indices = indices_storage[::2]
    indices.copy_(torch.arange(rows) % 5)
    before = [tensor.clone() for tensor in (x_storage, gate_storage, other_storage, indices_storage)]
    expected = x.float() + gate.index_select(0, indices.long()).float() * other.float()
    columns = (width + block_cols - 1) // block_cols
    tiles = ((rows + block_rows - 1) // block_rows) * columns
    dut._indexed_gate_kernel[(programs,)](
        output,
        x,
        gate,
        other,
        indices,
        stride_output_row=output.stride(0),
        stride_x_row=x.stride(0),
        stride_gate_row=gate.stride(0),
        stride_other_row=other.stride(0),
        stride_indices=indices.stride(0),
        rows=rows,
        width=width,
        column_tiles=columns,
        tiles_per_program=tiles // programs,
        extra_tiles=tiles % programs,
        block_rows=block_rows,
        block_cols=block_cols,
    )
    torch.testing.assert_close(output, expected.to(output.dtype), rtol=0, atol=0)
    assert torch.isnan(output_storage[:, :4]).all() and torch.isnan(output_storage[:, width + 4 :]).all()
    for actual, saved in zip((x_storage, gate_storage, other_storage, indices_storage), before):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0, equal_nan=True)


def test_gate_public_backend_rejection(dut, monkeypatch):
    inputs = gate_inputs()
    monkeypatch.setattr(dut, "_HAS_TRITON", False)
    with pytest.raises(RuntimeError, match="requires triton"):
        dut.indexed_gate(**inputs)
    monkeypatch.setattr(dut, "_HAS_TRITON", True)
    with pytest.raises(ValueError, match="requires Ascend NPU inputs"):
        dut.indexed_gate(**inputs)


@pytest.mark.parametrize("vector_count,launch_fails", [(4, False), (20, False), (4, True)])
def test_gate_public_schedule_and_context(dut, monkeypatch, vector_count, launch_fails):
    target = SimpleNamespace(type="npu", index=1)
    active = [0]
    launches = []
    checked = []
    inputs = gate_inputs(rows=5, width=65)

    class DeviceTensor:
        def __init__(self, value):
            self.cpu_tensor = value
            self.device = target

        def __getattr__(self, name):
            return getattr(self.cpu_tensor, name)

    proxy = {key: DeviceTensor(value) for key, value in inputs.items()}

    @contextmanager
    def device_context(device):
        previous = active[0]
        active[0] = device.index
        try:
            yield
        finally:
            active[0] = previous

    def validate(*values):
        assert all(actual is expected for actual, expected in zip(values, proxy.values()))
        checked.append(True)

    def resources(index):
        assert index == target.index
        return vector_count, 183 * 1024

    def tile_shape(width, ub_size):
        assert width == 65 and ub_size == 183 * 1024
        return 2, 32

    class Kernel:
        def __getitem__(self, grid):
            assert grid == (min(vector_count, 9),)
            return self.launch

        @staticmethod
        def launch(*args, **kwargs):
            assert active == [1]
            assert all(actual is expected for actual, expected in zip(args, proxy.values()))
            launches.append(kwargs)
            if launch_fails:
                raise RuntimeError("simulated indexed gate launch failure")

    monkeypatch.setattr(dut, "_HAS_TRITON", True)
    monkeypatch.setattr(dut, "_check_indexed_gate_inputs", validate)
    monkeypatch.setattr(dut, "_get_npu_resources", resources)
    monkeypatch.setattr(dut, "_tile_shape", tile_shape)
    monkeypatch.setattr(dut, "_indexed_gate_kernel", Kernel())
    monkeypatch.setattr(torch.npu, "device", device_context)
    if launch_fails:
        with pytest.raises(RuntimeError, match="simulated indexed gate launch failure"):
            dut.indexed_gate(**proxy)
    else:
        assert dut.indexed_gate(**proxy) is proxy["output"]
    assert checked == [True] and active == [0]
    assert len(launches) == 1
    launch = launches[0]
    programs = min(vector_count, 9)
    assert launch["rows"] == 5 and launch["width"] == 65
    assert launch["column_tiles"] == 3
    assert launch["tiles_per_program"] == 9 // programs and launch["extra_tiles"] == 9 % programs
    assert launch["block_rows"] == 2 and launch["block_cols"] == 32
    assert launch["stride_indices"] == inputs["indices"].stride(0)
    for name in ("output", "x", "gate", "other"):
        assert launch[f"stride_{name}_row"] == inputs[name].stride(0)
