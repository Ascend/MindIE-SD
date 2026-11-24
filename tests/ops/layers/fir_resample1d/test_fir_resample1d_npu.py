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

"""Ascend NPU compilation and accuracy tests for FIR Resample1D."""

import importlib
import os

import pytest
import torch

from mindiesd.utils import ParametersInvalid


def _fir_golden(x: torch.Tensor, fir_filter: torch.Tensor) -> torch.Tensor:
    batch, channels, input_length = x.shape
    output = torch.zeros((batch, channels, 2 * input_length), dtype=torch.float32)
    for batch_index in range(batch):
        for channel_index in range(channels):
            for output_index in range(2 * input_length):
                for tap in range(12):
                    source_position = output_index + tap - 5
                    if source_position % 2 == 0:
                        source_index = max(0, min(input_length - 1, source_position // 2))
                        output[batch_index, channel_index, output_index] += (
                            x[batch_index, channel_index, source_index] * fir_filter[tap]
                        )
    return output


@pytest.fixture(name="device", scope="module")
def fixture_device():
    if os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU":
        pytest.skip("FIR Resample1D kernel tests require an NPU")
    from tests.ops.device_support import init_test_device

    return init_test_device()


def _require_optimized_backend(x: torch.Tensor, fir_filter: torch.Tensor) -> None:
    from mindiesd import is_fir_resample1d_supported

    if is_fir_resample1d_supported(x, fir_filter):
        return
    if os.environ.get("MINDIESD_REQUIRE_FIR_TRITON") == "1":
        pytest.fail("FIR Resample1D strict test requires the Triton-Ascend backend")
    pytest.skip("FIR Resample1D Triton-Ascend backend is unavailable on this NPU host")


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 7), (2, 3, 257)])
def test_fir_resample1d_npu_jit_and_accuracy(device, shape):
    from mindiesd import fir_resample1d

    generator = torch.Generator().manual_seed(2027)
    x_cpu = torch.randn(shape, generator=generator, dtype=torch.float32)
    fir_filter_cpu = torch.linspace(-0.25, 0.3, 12, dtype=torch.float32)
    expected = _fir_golden(x_cpu, fir_filter_cpu)
    x = x_cpu.to(device)
    fir_filter = fir_filter_cpu.to(device)

    _require_optimized_backend(x, fir_filter)
    actual = fir_resample1d(x, fir_filter)
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("special_value", [float("inf"), float("-inf"), float("nan")])
def test_fir_resample1d_npu_nonfinite_tap_matches_golden(device, special_value):
    from mindiesd import fir_resample1d

    x_cpu = torch.tensor([[[1.0, 2.0, 4.0]]], dtype=torch.float32)
    fir_filter_cpu = torch.ones(12, dtype=torch.float32)
    fir_filter_cpu[0] = special_value
    expected = _fir_golden(x_cpu, fir_filter_cpu)
    x = x_cpu.to(device)
    fir_filter = fir_filter_cpu.to(device)
    _require_optimized_backend(x, fir_filter)

    actual = fir_resample1d(x, fir_filter)
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)


def test_private_fir_npu_op_rejects_short_filter_on_npu(device):
    from mindiesd.layers.fir_resample1d import _fir_resample1d_npu

    x = torch.ones((1, 1, 3), device=device)
    with pytest.raises(ParametersInvalid, match="exactly 12"):
        _fir_resample1d_npu(x, torch.ones(11, device=device))


def test_fir_resample1d_input_on_noncurrent_device(device):
    if torch.npu.device_count() < 2:
        pytest.skip("Device-context regression requires two NPUs")
    from mindiesd import fir_resample1d

    target = torch.device(f"npu:{1 if device.index == 0 else 0}")
    x_cpu = torch.linspace(-1.0, 1.0, 2 * 3 * 17).reshape(2, 3, 17)
    fir_filter_cpu = torch.linspace(-0.1, 0.1, 12)
    expected = _fir_golden(x_cpu, fir_filter_cpu)
    x = x_cpu.to(target)
    fir_filter = fir_filter_cpu.to(target)
    _require_optimized_backend(x, fir_filter)

    with torch.npu.device(device):
        caller_device = torch.npu.current_device()
        output = fir_resample1d(x, fir_filter)
        assert torch.npu.current_device() == caller_device
    assert output.device == target
    torch.npu.synchronize(target)
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_fir_resample1d_npu_reference_fallback(device, monkeypatch):
    fir_module = importlib.import_module("mindiesd.layers.fir_resample1d")

    x_cpu = torch.linspace(-1.0, 1.0, 2 * 3 * 17).reshape(2, 3, 17)
    fir_filter_cpu = torch.linspace(-0.1, 0.1, 12)
    expected = _fir_golden(x_cpu, fir_filter_cpu)
    x = x_cpu.to(device)
    fir_filter = fir_filter_cpu.to(device)
    monkeypatch.setattr(fir_module, "_HAS_TRITON", True)
    monkeypatch.setattr(fir_module, "_is_triton_ascend_available", lambda _device: False)
    custom_op_called = False
    original_custom_op = fir_module._fir_resample1d_npu

    def traced_custom_op(input_x, input_filter):
        nonlocal custom_op_called
        custom_op_called = True
        return original_custom_op(input_x, input_filter)

    monkeypatch.setattr(fir_module, "_fir_resample1d_npu", traced_custom_op)
    assert not fir_module.is_fir_resample1d_supported(x, fir_filter)
    output = fir_module.fir_resample1d(x, fir_filter)
    assert custom_op_called
    torch.npu.synchronize(device)
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_fir_resample1d_compile_fullgraph_dynamic_shapes(device):
    from mindiesd import fir_resample1d

    fir_filter = torch.linspace(-0.1, 0.1, 12, device=device)
    probe = torch.ones((1, 1, 7), device=device)
    _require_optimized_backend(probe, fir_filter)
    compiled = torch.compile(fir_resample1d, backend="eager", fullgraph=True, dynamic=True)
    for length in (17, 19):
        x = torch.linspace(-1.0, 1.0, 2 * 3 * length, device=device).reshape(2, 3, length)
        torch.testing.assert_close(compiled(x, fir_filter), fir_resample1d(x, fir_filter))


def test_fir_resample1d_inductor_fullgraph(device):
    from mindiesd import fir_resample1d

    x = torch.linspace(-1.0, 1.0, 2 * 3 * 17, device=device).reshape(2, 3, 17)
    fir_filter = torch.linspace(-0.1, 0.1, 12, device=device)
    _require_optimized_backend(x, fir_filter)
    try:
        compiled = torch.compile(fir_resample1d, fullgraph=True)
        actual = compiled(x, fir_filter)
    except Exception as error:
        messages = []
        current = error
        while current is not None:
            messages.append(str(current))
            current = current.__cause__
        if "triton_key" in "\n".join(messages):
            pytest.skip("Installed PyTorch and Triton do not provide a compatible triton_key API")
        raise
    torch.testing.assert_close(actual, fir_resample1d(x, fir_filter))
