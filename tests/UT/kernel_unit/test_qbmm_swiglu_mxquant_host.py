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

"""CPU contract tests for the qbmm_swiglu_mxquant host entry point.

The device CATLASS DSL bodies are excluded from CPU coverage; these tests use a
minimal fake ``catlass.tla`` to exercise validation, tensor binding, artifact
caching and launch planning in ``kernel.py`` without an NPU.
"""

import importlib
import sys
import types

import pytest
import torch

KERNEL_MODULE = "mindiesd.layers.qbmm_swiglu_mxquant.kernel"


class FakeArtifact:
    """Callable stand-in for a compiled CATLASS artifact."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeTlaTensor:
    """Minimal TLA tensor returned by the fake ``from_dlpack``."""

    def __init__(self, value, layout_tag=None, origin_shape=None, element_type=None):
        self.value = value
        self.layout_tag = layout_tag
        self.origin_shape = origin_shape
        self.element_type = element_type

    def mark_layout_dynamic(self):
        return self


def _identity(function):
    return function


def _const_expr(value):
    return bool(value)


def _from_dlpack(value, **kwargs):
    return FakeTlaTensor(value, **kwargs)


def _install_fake_catlass(monkeypatch):
    catlass = types.ModuleType("catlass")
    catlass.__path__ = []
    tla = types.ModuleType("catlass.tla")
    tla.__path__ = []
    runtime = types.ModuleType("catlass.tla.runtime")

    tla.Tensor = object
    tla.Constexpr = object
    tla.kernel = _identity
    tla.compile = lambda *args, **kwargs: FakeArtifact()
    tla.const_expr = _const_expr
    for name in (
        "Float4E2M1",
        "Float8E4M3FN",
        "Float8E8M0",
        "Int8",
        "Int32",
        "Float32",
    ):
        setattr(tla, name, name)
    tla.arch = types.SimpleNamespace(
        RowMajor="RowMajor",
        ColumnMajor="ColumnMajor",
        RowMajorMxScaleA="RowMajorMxScaleA",
        RowMajorMxScaleB="RowMajorMxScaleB",
    )
    runtime.from_dlpack = _from_dlpack
    tla.runtime = runtime
    catlass.tla = tla

    monkeypatch.setitem(sys.modules, "catlass", catlass)
    monkeypatch.setitem(sys.modules, "catlass.tla", tla)
    monkeypatch.setitem(sys.modules, "catlass.tla.runtime", runtime)


@pytest.fixture(name="kernel_module")
def kernel_module_fixture(monkeypatch):
    _install_fake_catlass(monkeypatch)
    sys.modules.pop(KERNEL_MODULE, None)
    module = importlib.import_module(KERNEL_MODULE)
    module._ARTIFACTS.clear()
    module._SPLITK_ARTIFACTS.clear()
    props = types.SimpleNamespace(cube_core_num=24, vector_core_num=48)
    monkeypatch.setattr(torch.npu, "set_device", lambda index: None)
    monkeypatch.setattr(torch.npu, "get_device_properties", lambda index: props, raising=False)
    yield module
    module._ARTIFACTS.clear()
    module._SPLITK_ARTIFACTS.clear()


def _inputs(m, k, n, in_fp4, dtype=torch.uint8):
    if in_fp4:
        a = torch.zeros((m, k // 2), dtype=dtype)
        b = torch.zeros((n, k // 2), dtype=dtype)
    else:
        a = torch.zeros((m, k), dtype=dtype)
        b = torch.zeros((k, n), dtype=dtype)
    scale_a = torch.zeros((m, k // 64, 2), dtype=dtype)
    scale_b = torch.zeros((k // 64, n, 2), dtype=dtype)
    return a, scale_a, b, scale_b


def test_fp8_fused_run_binds_and_launches(kernel_module):
    a, sa, b, sb = _inputs(m=33, k=320, n=384, in_fp4=0)
    y, sy = kernel_module.run(a, sa, b, sb, 0, 0)

    assert y.shape == (33, 192)
    assert sy.shape == (33, 3, 2)
    assert len(kernel_module._ARTIFACTS) == 1
    artifact = next(iter(kernel_module._ARTIFACTS.values()))
    assert artifact.calls[0][1]["block_num"] == 2  # grid_m=1, grid_n=2, 24 cores
    assert kernel_module._ARTIFACTS[(0, 0, 64, 256, 1)] is artifact


def test_fp4_run_reuses_cached_artifact(kernel_module):
    a, sa, b, sb = _inputs(m=16, k=512, n=256, in_fp4=1)
    y, sy = kernel_module.run(a, sa, b, sb, 1, 1)
    assert y.shape == (16, 64)
    assert sy.shape == (16, 2, 2)

    artifact = next(iter(kernel_module._ARTIFACTS.values()))
    kernel_module.run(a, sa, b, sb, 1, 1)
    assert len(kernel_module._ARTIFACTS) == 1
    assert len(artifact.calls) == 2


def test_splitk_run_compiles_two_kernels(kernel_module, monkeypatch):
    monkeypatch.setattr(kernel_module, "_ENABLE_SPLITK", True)
    a, sa, b, sb = _inputs(m=1, k=2048, n=1024, in_fp4=0)
    y, sy = kernel_module.run(a, sa, b, sb, 0, 0)

    assert y.shape == (1, 512)
    assert sy.shape == (1, 8, 2)
    assert kernel_module._ARTIFACTS == {}
    assert len(kernel_module._SPLITK_ARTIFACTS) == 1
    partial, reduce_epi = next(iter(kernel_module._SPLITK_ARTIFACTS.values()))
    assert len(partial.calls) == 1
    assert len(reduce_epi.calls) == 1


def test_splitk_plan_is_gated(kernel_module, monkeypatch):
    assert kernel_module._splitk_plan(1, 4096, 512, 256) == 0
    monkeypatch.setattr(kernel_module, "_ENABLE_SPLITK", True)
    assert kernel_module._splitk_plan(33, 4096, 512, 256) == 0
    assert kernel_module._splitk_plan(1, 1024, 512, 256) == 0
    assert kernel_module._splitk_plan(1, 4096, 0, 256) == 0
    assert kernel_module._splitk_plan(1, 4096, 2048, 256) == 0
    # k_tiles=8, grid_n=8 -> cap=4; the largest divisor of 8 within cap is 4.
    assert kernel_module._splitk_plan(1, 2048, 1024, 256) == 4


@pytest.mark.parametrize(
    "case",
    [
        dict(in_fp4=2, out_fp4=0),
        dict(in_fp4=0, out_fp4=-1),
    ],
)
def test_rejects_invalid_switches(kernel_module, case):
    a, sa, b, sb = _inputs(m=1, k=64, n=128, in_fp4=0)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        kernel_module.run(a, sa, b, sb, case["in_fp4"], case["out_fp4"])
    assert kernel_module._ARTIFACTS == {}


def test_rejects_non_byte_tensors(kernel_module):
    a, sa, b, sb = _inputs(m=1, k=64, n=128, in_fp4=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="must be uint8"):
        kernel_module.run(a, sa, b, sb, 0, 0)


def test_rejects_mixed_devices(kernel_module):
    a, sa, b, sb = _inputs(m=1, k=64, n=128, in_fp4=0)
    sb = sb.to("meta")
    with pytest.raises(ValueError, match="must be on"):
        kernel_module.run(a, sa, b, sb, 0, 0)


def test_rejects_wrong_rank(kernel_module):
    a, sa, b, sb = _inputs(m=1, k=64, n=128, in_fp4=0)
    with pytest.raises(ValueError, match="rank-2"):
        kernel_module.run(a.unsqueeze(0), sa, b, sb, 0, 0)
    with pytest.raises(ValueError, match="rank-3"):
        kernel_module.run(a, sa.unsqueeze(0), b, sb, 0, 0)


def test_rejects_fp8_b_width_mismatch(kernel_module):
    # Reviewer example: A [1, 128] cannot pair with B [64, 128].
    a = torch.zeros((1, 128), dtype=torch.uint8)
    sa = torch.zeros((1, 2, 2), dtype=torch.uint8)
    b = torch.zeros((64, 128), dtype=torch.uint8)
    sb = torch.zeros((2, 128, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match="b must be"):
        kernel_module.run(a, sa, b, sb, 0, 0)


def test_rejects_non_multiple_k_and_n(kernel_module):
    a, sa, b, sb = _inputs(m=1, k=64, n=128, in_fp4=0)
    with pytest.raises(ValueError, match="K must be"):
        kernel_module.run(a[:, :32], torch.zeros((1, 1, 2), dtype=torch.uint8), b[:32], sb[:1], 0, 0)
    with pytest.raises(ValueError, match="N must be"):
        kernel_module.run(a, sa, b[:, :64].contiguous(), torch.zeros((1, 64, 2), dtype=torch.uint8), 0, 0)


def test_rejects_scale_shape_mismatch(kernel_module):
    a, sa, b, sb = _inputs(m=1, k=64, n=128, in_fp4=0)
    with pytest.raises(ValueError, match="scale_a must be"):
        kernel_module.run(a, torch.zeros((1, 2, 2), dtype=torch.uint8), b, sb, 0, 0)
    with pytest.raises(ValueError, match="scale_b must be"):
        kernel_module.run(a, sa, b, torch.zeros((2, 128, 2), dtype=torch.uint8), 0, 0)


def test_package_export_runs_wrapper(kernel_module):
    import mindiesd

    a, sa, b, sb = _inputs(m=1, k=64, n=128, in_fp4=0)
    y, sy = mindiesd.qbmm_swiglu_mxquant(a, sa, b, sb, 0, 0)
    assert y.shape == (1, 64)
    assert sy.shape == (1, 1, 2)


@pytest.mark.parametrize("name", ["a", "b", "scale_a", "scale_b"])
def test_rejects_non_contiguous_inputs(kernel_module, name):
    """from_dlpack(origin_shape=...) rebuilds dense strides, so strided views
    must be rejected instead of being read as contiguous storage.
    """
    m, k, n = 2, 128, 128
    a, sa, b, sb = _inputs(m=m, k=k, n=n, in_fp4=0)
    if name == "a":
        a = torch.zeros((m, k * 2), dtype=torch.uint8)[:, ::2]
    elif name == "b":
        b = torch.zeros((k * 2, n), dtype=torch.uint8)[::2, :]
    elif name == "scale_a":
        groups = k // 64
        sa = torch.zeros((m, groups * 2, 2), dtype=torch.uint8)[:, ::2, :]
    else:
        groups = k // 64
        sb = torch.zeros((groups, n, 4), dtype=torch.uint8)[..., ::2]
    target = {"a": a, "b": b, "scale_a": sa, "scale_b": sb}[name]
    assert not target.is_contiguous()
    with pytest.raises(ValueError, match=f"{name} must be contiguous"):
        kernel_module.run(a, sa, b, sb, 0, 0)
