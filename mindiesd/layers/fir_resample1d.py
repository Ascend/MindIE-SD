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

"""Restricted 1D FIR resampling for inference workloads."""

import torch

from mindiesd.utils import ParametersInvalid

from .triton_utils import _HAS_TRITON

_BLOCK_SIZE = 1024
_FILTER_TAPS = 12
_PHASE_TAPS = _FILTER_TAPS // 2
_LEFT_OFFSET = 5
_MAX_OUTPUT_LENGTH = (1 << 31) - _FILTER_TAPS
_TRITON_ASCEND_DEVICES: set[torch.device] = set()
_VECTORCORE_COUNTS: dict[torch.device, int] = {}


def _is_triton_ascend_available(device: torch.device) -> bool:
    if not _HAS_TRITON or device.type != "npu" or not torch.npu.is_available():
        return False
    if device in _TRITON_ASCEND_DEVICES:
        return True
    try:
        from .triton_utils import triton as active_triton

        with torch.npu.device(device):
            properties = active_triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
        num_vectorcore = int(properties["num_vectorcore"])
        if num_vectorcore <= 0:
            return False
        _VECTORCORE_COUNTS[device] = num_vectorcore
        _TRITON_ASCEND_DEVICES.add(device)
        return True
    except Exception:
        return False


def _get_vectorcore_num(device: torch.device) -> int:
    if device not in _VECTORCORE_COUNTS and not _is_triton_ascend_available(device):
        raise RuntimeError(f"Triton-Ascend is unavailable for {device}")
    return _VECTORCORE_COUNTS[device]


if _HAS_TRITON:  # pragma: no cover - Imported only when Triton-Ascend is available.
    from .triton_utils import tl, triton

if _HAS_TRITON:  # pragma: no cover - Device kernel and launch path require an Ascend NPU.

    @triton.jit
    def fir_resample1d_kernel(
        x_ptr,
        filter_ptr,
        output_ptr,
        input_length,
        output_length,
        time_blocks,
        n_blocks,
        phase_taps: tl.constexpr,
        left_offset: tl.constexpr,
        block_size: tl.constexpr,
    ):
        program_id = tl.program_id(axis=0).to(tl.int64)
        num_programs = tl.num_programs(axis=0).to(tl.int64)
        lanes = tl.arange(0, block_size)
        for block_id in range(program_id, n_blocks, num_programs):
            row = block_id // time_blocks
            time_block = block_id - row * time_blocks
            output_base = time_block * block_size
            output_mask = lanes.to(tl.int64) < output_length - output_base
            output_index = time_block.to(tl.int32) * block_size + lanes
            tap_offset = (lanes + 1) % 2
            accumulator = tl.zeros((block_size,), dtype=tl.float32)
            for phase_tap in tl.static_range(phase_taps):
                tap = 2 * phase_tap + tap_offset
                source_position = output_index + tap - left_offset
                source_index = source_position // 2
                source_index = tl.maximum(0, tl.minimum(source_index, input_length - 1))
                input_offset = row * input_length + source_index.to(tl.int64)
                values = tl.load(
                    x_ptr + input_offset,
                    mask=output_mask,
                    other=0.0,
                ).to(tl.float32)
                coefficient = tl.load(filter_ptr + tap).to(tl.float32)
                accumulator += values * coefficient
            output_offset = row * output_length + output_base + lanes.to(tl.int64)
            tl.store(output_ptr + output_offset, accumulator, mask=output_mask)

    def _launch_fir_resample1d(x: torch.Tensor, fir_filter: torch.Tensor) -> torch.Tensor:
        output_length = 2 * x.shape[2]
        output = torch.empty(
            (x.shape[0], x.shape[1], output_length),
            dtype=x.dtype,
            device=x.device,
        )
        n_rows = x.shape[0] * x.shape[1]
        time_blocks = triton.cdiv(output_length, _BLOCK_SIZE)
        n_blocks = n_rows * time_blocks
        with torch.npu.device(x.device):
            num_programs = min(n_blocks, _get_vectorcore_num(x.device))
            fir_resample1d_kernel[(num_programs,)](
                x,
                fir_filter,
                output,
                x.shape[2],
                output_length,
                time_blocks,
                n_blocks,
                phase_taps=_PHASE_TAPS,
                left_offset=_LEFT_OFFSET,
                block_size=_BLOCK_SIZE,
            )
        return output

else:

    def _launch_fir_resample1d(  # pragma: no cover - Guarded by the backend capability check.
        x: torch.Tensor, fir_filter: torch.Tensor
    ) -> torch.Tensor:
        raise RuntimeError("FIR Resample1D requires Triton-Ascend on an available Ascend NPU")


def _validate_fir_resample1d_inputs(
    x: torch.Tensor,
    fir_filter: torch.Tensor,
    up: int,
    down: int,
    padding: str,
) -> None:
    if not isinstance(x, torch.Tensor) or not isinstance(fir_filter, torch.Tensor):
        raise ParametersInvalid("x and fir_filter must be torch.Tensor instances")
    if x.dim() != 3:
        raise ParametersInvalid(f"x must have shape [B, C, T], but got {tuple(x.shape)}")
    if all(isinstance(size, int) for size in x.shape):
        if any(size <= 0 for size in x.shape):
            raise ParametersInvalid(f"x dimensions must be positive, but got {tuple(x.shape)}")
        if 2 * x.shape[2] > _MAX_OUTPUT_LENGTH:
            raise ParametersInvalid(f"2*T must not exceed {_MAX_OUTPUT_LENGTH}, but got T={x.shape[2]}")
    if fir_filter.shape != (_FILTER_TAPS,):
        raise ParametersInvalid(
            f"fir_filter must contain exactly {_FILTER_TAPS} shared taps, but got {tuple(fir_filter.shape)}"
        )
    if x.dtype != torch.float32 or fir_filter.dtype != torch.float32:
        raise ParametersInvalid(f"x and fir_filter must use torch.float32; got x={x.dtype}, filter={fir_filter.dtype}")
    if x.device != fir_filter.device:
        raise ParametersInvalid(
            f"x and fir_filter must be on the same device; got x={x.device}, filter={fir_filter.device}"
        )
    if not x.is_contiguous() or not fir_filter.is_contiguous():
        raise ParametersInvalid("x and fir_filter must be contiguous")
    if isinstance(up, bool) or not isinstance(up, int) or up != 2:
        raise ParametersInvalid(f"up must be the integer 2, but got {up!r}")
    if isinstance(down, bool) or not isinstance(down, int) or down != 1:
        raise ParametersInvalid(f"down must be the integer 1, but got {down!r}")
    if padding != "replicate":
        raise ParametersInvalid(f"padding must be 'replicate', but got {padding!r}")
    if torch.is_grad_enabled() and (x.requires_grad or fir_filter.requires_grad):
        raise ParametersInvalid("fir_resample1d is inference-only and does not support autograd")


def _fir_resample1d_reference(x: torch.Tensor, fir_filter: torch.Tensor) -> torch.Tensor:
    input_length = x.shape[2]
    output_length = 2 * input_length
    output_index = torch.arange(output_length, device=x.device)
    output = torch.zeros(
        (x.shape[0], x.shape[1], output_length),
        dtype=torch.float32,
        device=x.device,
    )
    tap_offset = (output_index + 1).remainder(2)
    for phase_tap in range(_PHASE_TAPS):
        tap = 2 * phase_tap + tap_offset
        source_position = output_index + tap - _LEFT_OFFSET
        source_index = torch.div(source_position, 2, rounding_mode="floor").clamp(0, input_length - 1)
        values = x.index_select(2, source_index)
        coefficients = fir_filter.index_select(0, tap)
        output += values * coefficients
    return output


@torch.library.custom_op("mindiesd::_fir_resample1d_npu", mutates_args=())
def _fir_resample1d_npu(x: torch.Tensor, fir_filter: torch.Tensor) -> torch.Tensor:
    _validate_fir_resample1d_inputs(x, fir_filter, up=2, down=1, padding="replicate")
    if x.device.type != "npu":
        raise RuntimeError("_fir_resample1d_npu requires Ascend NPU tensors")
    if _is_triton_ascend_available(x.device):
        return _launch_fir_resample1d(x, fir_filter)  # pragma: no cover - NPU-only custom op.
    return _fir_resample1d_reference(x, fir_filter)


@_fir_resample1d_npu.register_fake
def _(x: torch.Tensor, fir_filter: torch.Tensor) -> torch.Tensor:
    _validate_fir_resample1d_inputs(x, fir_filter, up=2, down=1, padding="replicate")
    return torch.empty(
        (x.shape[0], x.shape[1], 2 * x.shape[2]),
        dtype=x.dtype,
        device=x.device,
    )


def is_fir_resample1d_supported(
    x: torch.Tensor,
    fir_filter: torch.Tensor,
    *,
    up: int = 2,
    down: int = 1,
    padding: str = "replicate",
) -> bool:
    """Return whether the optimized NPU backend supports this configuration."""
    try:
        _validate_fir_resample1d_inputs(x, fir_filter, up, down, padding)
    except ParametersInvalid:
        return False
    return _is_triton_ascend_available(x.device)


def fir_resample1d(
    x: torch.Tensor,
    fir_filter: torch.Tensor,
    *,
    up: int = 2,
    down: int = 1,
    padding: str = "replicate",
) -> torch.Tensor:
    r"""Upsample FP32 ``[B, C, T]`` input by two with a shared 12-tap FIR filter.

    Version 1 supports only ``up=2``, ``down=1`` and ``padding="replicate"``.
    It computes ``sum(h[k] * u[n + k - 5])`` where even positions of ``u``
    contain replicated input samples and odd positions are zero. Coefficients
    are not normalized or rescaled. The operation is inference-only.
    """
    _validate_fir_resample1d_inputs(x, fir_filter, up, down, padding)
    if x.device.type == "npu" and _HAS_TRITON:  # pragma: no cover - Covered by NPU tests.
        return _fir_resample1d_npu(x, fir_filter)
    return _fir_resample1d_reference(x, fir_filter)
