# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU unit tests for the fused transpose_dynamic_quant host path and kernels.

CI runs the UT suite without an NPU, so the device-only checks in ``tests/ops``
never contribute coverage. Here the argument checks and the axis/ tile dispatch
are exercised directly on CPU tensors, and the six Triton bodies are executed by
the strict CPU facade in ``UT.kernel_unit.triton_cpu`` against independent
PyTorch references (device rounding is covered by the NPU suite instead).
"""

# pylint: disable=redefined-outer-name  # pytest fixtures are requested by parameter name.

import importlib

import pytest
import torch

from UT.kernel_unit.triton_cpu import install

tdq = importlib.import_module("mindiesd.layers.quant.transpose_dynamic_quant")

pytestmark = pytest.mark.skipif(not tdq._HAS_TRITON, reason="triton is not installed")


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #
def test_validate_args_accepts_every_axis():
    x = torch.empty((2, 6, 5), dtype=torch.bfloat16)
    assert tdq._validate_args(x, None, 0) == (0, (6, 5))
    assert tdq._validate_args(x, None, 1) == (1, (2, 5))
    assert tdq._validate_args(x, None, -2) == (1, (2, 5))
    assert tdq._validate_args(x, None, 2) == (2, (2, 6))


def test_validate_args_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        tdq._validate_args(torch.empty((4, 8), dtype=torch.float32), None, 0)


def test_validate_args_rejects_rank_below_two():
    with pytest.raises(ValueError, match="at least 2 dims"):
        tdq._validate_args(torch.empty((8,), dtype=torch.bfloat16), None, 0)


def test_validate_args_rejects_int32_index_overflow():
    x = torch.empty((2**30, 2), dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="int32 index limit"):
        tdq._validate_args(x, None, 0)


@pytest.mark.parametrize("axis", [3, -4])
def test_validate_args_rejects_axis_out_of_range(axis):
    with pytest.raises(ValueError, match="out of range"):
        tdq._validate_args(torch.empty((2, 3, 4), dtype=torch.bfloat16), None, axis)


def test_validate_args_smooth_contract():
    x = torch.empty((2, 6, 5), dtype=torch.bfloat16)
    assert tdq._validate_args(x, torch.empty((6,), dtype=torch.bfloat16), 1) == (1, (2, 5))

    with pytest.raises(ValueError, match="device"):
        tdq._validate_args(x, torch.empty((6,), dtype=torch.bfloat16, device="meta"), 1)
    with pytest.raises(ValueError, match="dtype"):
        tdq._validate_args(x, torch.empty((6,), dtype=torch.float16), 1)
    with pytest.raises(ValueError, match="1-D with length"):
        tdq._validate_args(x, torch.empty((7,), dtype=torch.bfloat16), 1)
    with pytest.raises(ValueError, match="1-D with length"):
        tdq._validate_args(x, torch.empty((6, 1), dtype=torch.bfloat16), 1)


def test_validate_args_requires_float8_dtype(monkeypatch):
    monkeypatch.setattr(tdq, "_FP8_DTYPE", None)
    with pytest.raises(RuntimeError, match="float8_e4m3fn"):
        tdq._validate_args(torch.empty((2, 3), dtype=torch.bfloat16), None, 0)


# --------------------------------------------------------------------------- #
# dispatch: every tier must pick its kernel, tile and grid without a device
# --------------------------------------------------------------------------- #
class _Recorder:
    """Kernel stand-in that records (grid, args, constexprs) instead of launching."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


_KERNELS = (
    "_dynamic_quant_axis0_kernel",
    "_dynamic_quant_axis_last_kernel",
    "_dynamic_quant_axis_last_fullload_kernel",
    "_dynamic_quant_quant_single_kernel",
    "_dynamic_quant_max_partial_kernel",
    "_dynamic_quant_quant_same_kernel",
)


@pytest.fixture
def recorded(monkeypatch):
    recorders = {name: _Recorder() for name in _KERNELS}
    for name, recorder in recorders.items():
        monkeypatch.setattr(tdq, name, recorder, raising=False)
    return recorders


def test_dispatch_axis0_default_tile(recorded):
    x = torch.randn(128, 32, dtype=torch.bfloat16)
    y, scale = tdq._transpose_dynamic_quant_triton(x, None, 0)

    ((grid, args, kwargs),) = recorded["_dynamic_quant_axis0_kernel"].launches
    assert grid == (1,)
    assert args[0] is x
    assert (kwargs["BLOCK_M"], kwargs["BLOCK_K"]) == (64, 64)
    assert kwargs["HAS_SMOOTH"] is False
    assert kwargs["DTYPE_MAX"] == tdq._DTYPE_MAX
    assert tuple(y.shape) == (128, 32) and y.dtype == torch.float8_e4m3fn
    assert tuple(scale.shape) == (32,) and scale.dtype == torch.float32


def test_dispatch_axis0_thin_axis_tile(recorded):
    x = torch.randn(4, 2048, dtype=torch.bfloat16)
    tdq._transpose_dynamic_quant_triton(x, None, 0)

    _, _, kwargs = recorded["_dynamic_quant_axis0_kernel"].launches[0]
    assert (kwargs["BLOCK_M"], kwargs["BLOCK_K"]) == (4, 2048)


def test_dispatch_axis0_large_shape_tile(recorded):
    x = torch.randn(128, 40000, dtype=torch.bfloat16)
    tdq._transpose_dynamic_quant_triton(x, None, -2)

    _, _, kwargs = recorded["_dynamic_quant_axis0_kernel"].launches[0]
    assert (kwargs["BLOCK_M"], kwargs["BLOCK_K"]) == (128, 64)


def test_dispatch_axis_last_chunked_tile(recorded):
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    _, scale = tdq._transpose_dynamic_quant_triton(x, None, -1)

    _, _, kwargs = recorded["_dynamic_quant_axis_last_kernel"].launches[0]
    assert (kwargs["BLOCK_M"], kwargs["BLOCK_K"]) == (64, 64)
    assert tuple(scale.shape) == (16,)


def test_dispatch_axis_last_full_load_tier(recorded, monkeypatch):
    monkeypatch.setattr(tdq, "get_vectorcore_num", lambda: 40)
    x = torch.randn(2048, 4096, dtype=torch.bfloat16)
    _, scale = tdq._transpose_dynamic_quant_triton(x, None, 1)

    ((grid, _, kwargs),) = recorded["_dynamic_quant_axis_last_fullload_kernel"].launches
    assert grid == (40,)
    assert (kwargs["BLOCK_ROW"], kwargs["BLOCK_COL"]) == (2, 4096)
    assert tuple(scale.shape) == (2048,)


def test_dispatch_middle_axis_single_kernel(recorded):
    x = torch.randn(2, 100, 3, dtype=torch.bfloat16)
    y, scale = tdq._transpose_dynamic_quant_triton(x, None, 1)

    _, _, kwargs = recorded["_dynamic_quant_quant_single_kernel"].launches[0]
    assert (kwargs["BLOCK_M"], kwargs["BLOCK_K"]) == (128, tdq._SINGLE_KERNEL_BLOCK_K)
    assert tuple(y.shape) == (2, 100, 3)
    assert tuple(scale.shape) == (2, 3)
    assert not recorded["_dynamic_quant_max_partial_kernel"].launches


def test_dispatch_middle_axis_two_kernel(recorded):
    x = torch.randn(2, 10000, 4, dtype=torch.bfloat16)
    y, scale = tdq._transpose_dynamic_quant_triton(x, None, 1)

    ((grid1, args1, kwargs1),) = recorded["_dynamic_quant_max_partial_kernel"].launches
    ((grid2, args2, kwargs2),) = recorded["_dynamic_quant_quant_same_kernel"].launches
    assert grid1 == grid2
    assert (kwargs1["BLOCK_M"], kwargs1["BLOCK_K"]) == (tdq._TRANS_P1_BLOCK_M, tdq._TRANS_P1_BLOCK_K)
    assert kwargs2["M_GROUP"] == tdq._TRANS_P1_M_GROUP
    # Pass 1 accumulates into the scale output's int32 bit view; pass 2 reads it.
    assert args1[2].dtype == torch.int32
    assert args2[3] is scale
    assert torch.equal(scale, torch.zeros((2, 4), dtype=torch.float32))
    assert tuple(y.shape) == (2, 10000, 4)


def test_dispatch_middle_axis_thin_two_kernel(recorded):
    x = torch.randn(1, 32, 4096, dtype=torch.bfloat16)
    tdq._transpose_dynamic_quant_triton(x, None, 1)

    _, _, kwargs = recorded["_dynamic_quant_max_partial_kernel"].launches[0]
    assert (kwargs["BLOCK_M"], kwargs["BLOCK_K"]) == (32, 256)


def test_dispatch_empty_axis_returns_zero_scale(recorded):
    x = torch.empty((2, 0, 3), dtype=torch.bfloat16)
    y, scale = tdq._transpose_dynamic_quant_triton(x, None, 1)

    assert tuple(y.shape) == (2, 0, 3)
    assert torch.equal(scale, torch.zeros((2, 3), dtype=torch.float32))
    assert not any(recorder.launches for recorder in recorded.values())


def test_dispatch_smooth_and_noncontiguous_input(recorded):
    x = torch.randn(3, 4, 100, dtype=torch.bfloat16).transpose(1, 2)
    smooth = torch.randn(100, dtype=torch.bfloat16)
    _, scale = tdq._transpose_dynamic_quant_triton(x, smooth, 1)

    _, args, kwargs = recorded["_dynamic_quant_quant_single_kernel"].launches[0]
    assert kwargs["HAS_SMOOTH"] is True
    assert tuple(args[0].shape) == (3, 100, 4) and args[0].is_contiguous()
    assert tuple(args[1].shape) == (100,) and args[1].is_contiguous()
    assert tuple(scale.shape) == (3, 4)


# --------------------------------------------------------------------------- #
# public op boundary and fake metadata
# --------------------------------------------------------------------------- #
def test_op_requires_triton_ascend(monkeypatch):
    monkeypatch.setattr(tdq, "_TRITON_ON_ASCEND", False)
    with pytest.raises(RuntimeError, match="requires Triton-Ascend"):
        tdq.transpose_dynamic_quant(torch.empty((2, 3), dtype=torch.bfloat16), None, 0)


def test_op_requires_npu_device(monkeypatch):
    monkeypatch.setattr(tdq, "_TRITON_ON_ASCEND", True)
    with pytest.raises(ValueError, match="requires Ascend NPU inputs"):
        tdq.transpose_dynamic_quant(torch.empty((2, 3), dtype=torch.bfloat16), None, 0)


def test_fake_impl_reports_y_and_scale_metadata():
    with torch._subclasses.fake_tensor.FakeTensorMode():
        x = torch.empty((2, 3, 4), dtype=torch.bfloat16)
        y, scale = tdq.transpose_dynamic_quant(x, None, 1)

    assert tuple(y.shape) == (2, 3, 4) and y.dtype == torch.float8_e4m3fn
    assert tuple(scale.shape) == (2, 4) and scale.dtype == torch.float32


# --------------------------------------------------------------------------- #
# Triton bodies on the CPU facade
# --------------------------------------------------------------------------- #
class _GridInt(int):
    """int that still accepts the device-side ``.to(tl.int32)`` cast."""

    def to(self, dtype):
        return self


class _LangShim:
    """``tl`` delegate adding the few ops this operator needs beyond the facade."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def num_programs(self, axis=0):
        return _GridInt(self._base.num_programs(axis))

    @staticmethod
    def cdiv(left, right):
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return torch.div(torch.as_tensor(left), torch.as_tensor(right), rounding_mode="floor")
        return -(-int(left) // int(right))

    @staticmethod
    def atomic_max(pointer, value, mask=None):
        index = torch.as_tensor(pointer.offset, dtype=torch.long)
        index, value = torch.broadcast_tensors(index, value)
        mask = torch.ones_like(index, dtype=torch.bool) if mask is None else torch.broadcast_to(mask, index.shape)
        storage = pointer.storage()
        storage.scatter_reduce_(0, index[mask], value[mask].to(storage.dtype), reduce="amax", include_self=True)
        return value


@pytest.fixture
def kernels(monkeypatch):
    install(monkeypatch, tdq)
    # This operator uses three pieces of ``tl`` semantics the shared facade does
    # not implement (integer ceil-div, atomic_max, num_programs().to(int32)).
    # Shim them here instead of widening the public facade for one operator.
    monkeypatch.setattr(tdq, "tl", _LangShim(tdq.tl))
    return tdq


def _reference(x, smooth, axis):
    """Independent fp32 reference matching the documented aclnn semantics."""
    value = x.float()
    if smooth is not None:
        shape = [1] * value.dim()
        shape[axis] = smooth.numel()
        value = value * smooth.float().reshape(shape)
    scale = value.abs().amax(dim=axis) / tdq._DTYPE_MAX
    denom = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (value / denom.unsqueeze(axis)).clamp(-tdq._DTYPE_MAX, tdq._DTYPE_MAX)
    return quant.to(torch.float8_e4m3fn), scale


def _assert_matches(y, scale, expected_y, expected_scale):
    assert torch.equal(y.view(torch.uint8), expected_y.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize("has_smooth", [False, True])
def test_axis0_kernel_body(kernels, has_smooth):
    x = torch.randn(5, 7, dtype=torch.bfloat16)
    smooth = torch.randn(5, dtype=torch.bfloat16) if has_smooth else None
    y = torch.empty((5, 7), dtype=torch.float8_e4m3fn)
    scale = torch.empty((7,), dtype=torch.float32)

    tdq._dynamic_quant_axis0_kernel[(1,)](
        x,
        smooth,
        y,
        scale,
        5,
        7,
        HAS_SMOOTH=has_smooth,
        DTYPE_MAX=tdq._DTYPE_MAX,
        BLOCK_M=4,
        BLOCK_K=4,
    )

    _assert_matches(y, scale, *_reference(x, smooth, 0))


@pytest.mark.parametrize("has_smooth", [False, True])
def test_axis_last_kernel_body(kernels, has_smooth):
    x = torch.randn(5, 7, dtype=torch.bfloat16)
    smooth = torch.randn(7, dtype=torch.bfloat16) if has_smooth else None
    y = torch.empty((5, 7), dtype=torch.float8_e4m3fn)
    scale = torch.empty((5,), dtype=torch.float32)

    tdq._dynamic_quant_axis_last_kernel[(1,)](
        x,
        smooth,
        y,
        scale,
        5,
        7,
        HAS_SMOOTH=has_smooth,
        DTYPE_MAX=tdq._DTYPE_MAX,
        BLOCK_M=4,
        BLOCK_K=4,
    )

    _assert_matches(y, scale, *_reference(x, smooth, 1))


@pytest.mark.parametrize("has_smooth", [False, True])
def test_axis_last_full_load_kernel_body(kernels, has_smooth):
    x = torch.randn(3, 7, dtype=torch.bfloat16)
    smooth = torch.randn(7, dtype=torch.bfloat16) if has_smooth else None
    y = torch.empty((3, 7), dtype=torch.float8_e4m3fn)
    scale = torch.empty((3,), dtype=torch.float32)

    tdq._dynamic_quant_axis_last_fullload_kernel[(1,)](
        x,
        smooth,
        y,
        scale,
        3,
        7,
        HAS_SMOOTH=has_smooth,
        DTYPE_MAX=tdq._DTYPE_MAX,
        BLOCK_ROW=2,
        BLOCK_COL=8,
    )

    _assert_matches(y, scale, *_reference(x, smooth, 1))


@pytest.mark.parametrize("has_smooth", [False, True])
def test_middle_axis_single_kernel_body(kernels, has_smooth):
    x = torch.randn(2, 5, 3, dtype=torch.bfloat16)
    smooth = torch.randn(5, dtype=torch.bfloat16) if has_smooth else None
    y = torch.empty((2, 5, 3), dtype=torch.float8_e4m3fn)
    scale = torch.empty((2, 3), dtype=torch.float32)

    tdq._dynamic_quant_quant_single_kernel[(1,)](
        x,
        smooth,
        y,
        scale,
        5,
        3,
        2,
        DTYPE_MAX=tdq._DTYPE_MAX,
        HAS_SMOOTH=has_smooth,
        BLOCK_M=4,
        BLOCK_K=4,
    )

    _assert_matches(y, scale, *_reference(x, smooth, 1))


@pytest.mark.parametrize("has_smooth", [False, True])
def test_middle_axis_two_kernel_body(kernels, has_smooth):
    x = torch.randn(2, 5, 3, dtype=torch.bfloat16)
    smooth = torch.randn(5, dtype=torch.bfloat16) if has_smooth else None
    y = torch.empty((2, 5, 3), dtype=torch.float8_e4m3fn)
    scale = torch.zeros((2, 3), dtype=torch.float32)

    tdq._dynamic_quant_max_partial_kernel[(1,)](
        x,
        smooth,
        scale.view(torch.int32),
        5,
        3,
        2,
        HAS_SMOOTH=has_smooth,
        DTYPE_MAX=tdq._DTYPE_MAX,
        BLOCK_M=4,
        BLOCK_K=4,
        M_GROUP=1,
    )
    tdq._dynamic_quant_quant_same_kernel[(1,)](
        x,
        smooth,
        y,
        scale,
        5,
        3,
        2,
        DTYPE_MAX=tdq._DTYPE_MAX,
        HAS_SMOOTH=has_smooth,
        BLOCK_M=4,
        BLOCK_K=4,
        M_GROUP=1,
    )

    _assert_matches(y, scale, *_reference(x, smooth, 1))
