#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import contextlib
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import mindiesd
from mindiesd import fir_resample1d
from mindiesd.layers import fir_resample1d as layers_fir_resample1d
from mindiesd.layers import is_fir_resample1d_supported
from mindiesd.layers.fir_resample1d import _fir_resample1d_npu
from mindiesd.utils import ParametersInvalid


def _independent_reference(x: torch.Tensor, fir_filter: torch.Tensor) -> torch.Tensor:
    batch, channels, input_length = x.shape
    output = torch.zeros((batch, channels, 2 * input_length), dtype=x.dtype)
    for batch_index in range(batch):
        for channel_index in range(channels):
            for output_index in range(2 * input_length):
                value = 0.0
                for tap in range(12):
                    source_position = output_index + tap - 5
                    if source_position % 2 == 0:
                        source_index = max(0, min(input_length - 1, source_position // 2))
                        value += float(x[batch_index, channel_index, source_index]) * float(fir_filter[tap])
                output[batch_index, channel_index, output_index] = value
    return output


def test_public_exports_are_stable():
    assert mindiesd.fir_resample1d is layers_fir_resample1d
    assert "fir_resample1d" in mindiesd.__all__
    assert "is_fir_resample1d_supported" in mindiesd.__all__


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 7), (2, 2, 31)])
def test_fir_resample1d_matches_independent_reference(shape):
    generator = torch.Generator().manual_seed(2027)
    x = torch.randn(shape, generator=generator, dtype=torch.float32)
    fir_filter = torch.linspace(-0.25, 0.3, 12, dtype=torch.float32)

    expected = _independent_reference(x, fir_filter)
    actual = fir_resample1d(x, fir_filter)

    assert actual.shape == (shape[0], shape[1], 2 * shape[2])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("tap", "expected"),
    [
        (0, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        (5, [1.0, 0.0, 2.0, 0.0, 4.0, 0.0]),
        (11, [4.0, 0.0, 4.0, 0.0, 4.0, 0.0]),
    ],
)
def test_fir_resample1d_locks_tap_order_and_phase(tap, expected):
    x = torch.tensor([[[1.0, 2.0, 4.0]]])
    fir_filter = torch.zeros(12)
    fir_filter[tap] = 1.0

    expected_tensor = torch.tensor([[expected]])
    torch.testing.assert_close(fir_resample1d(x, fir_filter), expected_tensor, rtol=0, atol=0)


def test_fir_resample1d_does_not_normalize_or_rescale_filter():
    x = torch.ones((1, 1, 3), dtype=torch.float32)
    fir_filter = torch.full((12,), 2.0, dtype=torch.float32)

    expected = _independent_reference(x, fir_filter)
    torch.testing.assert_close(fir_resample1d(x, fir_filter), expected, rtol=0, atol=0)


@pytest.mark.parametrize("special_value", [float("inf"), float("-inf"), float("nan")])
def test_fir_resample1d_nonfinite_tap_affects_only_its_phase(special_value):
    x = torch.tensor([[[1.0, 2.0, 4.0]]], dtype=torch.float32)
    fir_filter = torch.ones(12, dtype=torch.float32)
    fir_filter[0] = special_value

    expected = _independent_reference(x, fir_filter)
    actual = fir_resample1d(x, fir_filter)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.isfinite(actual[..., ::2]).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"up": 1}, "up must be"),
        ({"up": True}, "up must be"),
        ({"down": 2}, "down must be"),
        ({"padding": "reflect"}, "padding must be"),
    ],
)
def test_fir_resample1d_rejects_unsupported_options(kwargs, message):
    with pytest.raises(ParametersInvalid, match=message):
        fir_resample1d(torch.ones(1, 1, 3), torch.ones(12), **kwargs)


@pytest.mark.parametrize(
    ("x", "fir_filter", "message"),
    [
        (torch.ones(2, 3), torch.ones(12), "shape \\[B, C, T\\]"),
        (torch.ones(1, 1, 3), torch.ones(11), "exactly 12"),
        (torch.ones(1, 1, 3), torch.ones(1, 12), "exactly 12"),
        (
            torch.ones(1, 1, 3, dtype=torch.float16),
            torch.ones(12),
            "torch.float32",
        ),
        (
            torch.ones(1, 3, 4).transpose(1, 2),
            torch.ones(12),
            "contiguous",
        ),
    ],
)
def test_fir_resample1d_rejects_unsupported_metadata(x, fir_filter, message):
    with pytest.raises(ParametersInvalid, match=message):
        fir_resample1d(x, fir_filter)


def test_fir_resample1d_rejects_autograd():
    x = torch.ones(1, 2, 3, requires_grad=True)
    with pytest.raises(ParametersInvalid, match="inference-only"):
        fir_resample1d(x, torch.ones(12))


def test_fir_resample1d_rejects_non_tensor_and_empty_dimensions():
    with pytest.raises(ParametersInvalid, match="torch.Tensor"):
        fir_resample1d([1.0], torch.ones(12))
    with pytest.raises(ParametersInvalid, match="positive"):
        fir_resample1d(torch.empty(1, 1, 0), torch.ones(12))


def test_fir_resample1d_rejects_mismatched_devices():
    with pytest.raises(ParametersInvalid, match="same device"):
        fir_resample1d(torch.ones(1, 1, 1), torch.ones(12, device="meta"))


def test_fir_cpu_is_not_reported_as_optimized():
    assert not is_fir_resample1d_supported(torch.ones(1, 2, 3), torch.ones(12))
    assert not is_fir_resample1d_supported(torch.ones(1, 2, 3), torch.ones(11))


def test_triton_ascend_capability_uses_target_device(monkeypatch):
    fir_module = importlib.import_module("mindiesd.layers.fir_resample1d")
    triton_utils = importlib.import_module("mindiesd.layers.triton_utils")
    calls = []

    def get_device_properties(index):
        calls.append(index)
        return {"num_vectorcore": 48}

    fake_triton = SimpleNamespace(
        runtime=SimpleNamespace(
            driver=SimpleNamespace(
                active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=get_device_properties))
            )
        )
    )
    monkeypatch.setattr(fir_module, "_HAS_TRITON", True)
    monkeypatch.setattr(triton_utils, "triton", fake_triton)
    monkeypatch.setattr(torch.npu, "is_available", lambda: True)
    monkeypatch.setattr(torch.npu, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.npu, "current_device", lambda: 3)
    fir_module._TRITON_ASCEND_DEVICES.clear()
    fir_module._VECTORCORE_COUNTS.clear()

    device = type("FakeNpuDevice", (), {"type": "npu", "index": 3})()
    assert fir_module._is_triton_ascend_available(device)
    assert fir_module._get_vectorcore_num(device) == 48
    assert calls == [3]
    fir_module._TRITON_ASCEND_DEVICES.clear()
    fir_module._VECTORCORE_COUNTS.clear()


def test_triton_ascend_capability_rejects_non_ascend_properties(monkeypatch):
    fir_module = importlib.import_module("mindiesd.layers.fir_resample1d")
    triton_utils = importlib.import_module("mindiesd.layers.triton_utils")
    fake_triton = SimpleNamespace(
        runtime=SimpleNamespace(
            driver=SimpleNamespace(
                active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda _index: {}))
            )
        )
    )
    monkeypatch.setattr(fir_module, "_HAS_TRITON", True)
    monkeypatch.setattr(triton_utils, "triton", fake_triton)
    monkeypatch.setattr(torch.npu, "is_available", lambda: True)
    monkeypatch.setattr(torch.npu, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    fir_module._TRITON_ASCEND_DEVICES.clear()
    fir_module._VECTORCORE_COUNTS.clear()

    device = type("FakeNpuDevice", (), {"type": "npu", "index": 0})()
    assert not fir_module._is_triton_ascend_available(device)
    assert device not in fir_module._TRITON_ASCEND_DEVICES
    fir_module._TRITON_ASCEND_DEVICES.clear()
    fir_module._VECTORCORE_COUNTS.clear()


def test_fir_fake_shape_dtype_and_device():
    x = torch.empty((2, 3, 7), dtype=torch.float32, device="meta")
    result = _fir_resample1d_npu(x, torch.empty(12, device="meta"))
    assert result.shape == (2, 3, 14)
    assert result.dtype == x.dtype
    assert result.device.type == "meta"


def test_fir_fake_tensor_mode_preserves_metadata():
    with FakeTensorMode():
        x = torch.empty((2, 3, 7), dtype=torch.float32)
        result = _fir_resample1d_npu(x, torch.empty(12))
    assert result.shape == (2, 3, 14)
    assert result.dtype == x.dtype


def test_fir_fake_supports_total_output_larger_than_int32_index_range():
    x = torch.empty((3, 1, 2**29), dtype=torch.float32, device="meta")
    result = _fir_resample1d_npu(x, torch.empty(12, device="meta"))
    assert result.shape == (3, 1, 2**30)
    assert result.numel() > 2**31


def test_fir_accepts_maximum_safe_row_output_length():
    max_output_length = (1 << 31) - 12
    x = torch.empty((1, 1, max_output_length // 2), dtype=torch.float32, device="meta")
    result = fir_resample1d(x, torch.empty(12, device="meta"))
    assert result.shape == (1, 1, max_output_length)


def test_fir_rejects_row_output_beyond_safe_int32_index_domain():
    max_output_length = (1 << 31) - 12
    x = torch.empty((1, 1, max_output_length // 2 + 1), dtype=torch.float32, device="meta")
    with pytest.raises(ParametersInvalid, match=r"2\*T must not exceed"):
        fir_resample1d(x, torch.empty(12, device="meta"))


def test_private_fir_npu_op_rejects_invalid_filter_metadata():
    x = torch.empty((1, 1, 3), device="meta")
    with pytest.raises(ParametersInvalid, match="exactly 12"):
        _fir_resample1d_npu(x, torch.empty(11, device="meta"))


def test_private_fir_npu_op_rejects_cpu():
    with pytest.raises(RuntimeError, match="requires Ascend NPU"):
        _fir_resample1d_npu(torch.ones(1, 1, 1), torch.ones(12))
