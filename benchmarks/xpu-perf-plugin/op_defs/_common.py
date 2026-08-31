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

"""Shared constants for Ascend core-op schemas.

Data-type byte sizes and MX (microscaling) quantization accounting follow the
RFC#299 conventions: fp8=1B, mxfp8=1B, mxfp4=0.5B per element plus per-block
e8m0 scale overhead.
"""

from common.metrics import util_metrics

DTYPE_BYTES = {
    "bf16": 2.0,
    "fp8": 1.0,
    "mxfp8": 1.0,
    "mxfp4": 0.5,
}

# Map our schema dtype names to xpu-perf get_torch_dtype keys.
XPU_PERF_DTYPE_MAP = {
    "bf16": "bfloat16",
    "fp8": "float8",
    "mxfp8": "float8",
    "mxfp4": "float8",
}

SUPPORTED_FA_DTYPES = ("bf16", "fp8", "mxfp8", "mxfp4")
SUPPORTED_MM_QUANT = ("NO_QUANT", "W8A8", "W8A8_MXFP8", "W4A4_MXFP4")
SUPPORTED_GMM_QUANT = ("NO_QUANT", "W8A8_DYNAMIC", "W8A8_MXFP8")

MX_BLOCK_SIZE = 32
MX_SCALE_BYTES = 1.0

# Per-element FLOPs charged for dynamic-MX / per-token quantization kernels.
# Convention: ~1 flop for the block/token scale reduction (amortized) + ~1 for
# the elementwise scaling, so 2 flops/element. Applied to every element that a
# quantized vendor path actually quantizes (fa q/k/v, mm x/w) so MFU reflects
# the full measured op (quantization runs inside the timed region).
QUANT_FLOPS_PER_ELEM = 2.0


def quant_flops(numel: int, per_elem: float = QUANT_FLOPS_PER_ELEM) -> float:
    """FLOPs for quantizing a tensor of numel elements."""
    if numel < 0:
        raise ValueError(f"numel must be >= 0, got {numel}")
    return numel * per_elem


def mx_scale_bytes(numel: int, block_size: int = MX_BLOCK_SIZE) -> float:
    """Bytes of per-block e8m0 scales for an MX-quantized tensor.

    Only mxfp8/mxfp4 carry block scales; bf16/fp8 have none.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    blocks = (numel + block_size - 1) // block_size
    return blocks * MX_SCALE_BYTES


def tensor_bytes(numel: int, dtype: str) -> float:
    """On-device bytes for a tensor of given dtype, incl. MX scale overhead."""
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype: {dtype}")
    data = numel * DTYPE_BYTES[dtype]
    if dtype in ("mxfp8", "mxfp4"):
        data += mx_scale_bytes(numel)
    return data


def schema_torch_dtype(dtype: str):
    """Resolve a schema dtype name to a torch dtype for OpTensorInfo.

    mxfp4 maps to torch_npu.float4_e2m1fn_x2 on NPU environments; on CPU-only
    environments (unit tests / simulation) it degrades to torch.float8_e4m3fn.
    """
    from xpu_perf.micro_perf.core.utils import get_torch_dtype

    if dtype == "mxfp4":
        try:
            import torch_npu

            return torch_npu.float4_e2m1fn_x2
        except ImportError:
            import torch

            return torch.float8_e4m3fn
    return get_torch_dtype(XPU_PERF_DTYPE_MAP[dtype])


def op_tensor_info(shape, dtype, device):
    """OpTensorInfo for a schema dtype; quantized dtypes create bf16 tensors.

    Quantized inputs are created in bf16 and quantized inside the vendor impl
    (npu_dynamic_mx_quant etc.); the schema dtype only drives the read/write
    byte accounting, not the on-device creation dtype.
    """
    from xpu_perf.micro_perf.core.utils import OpTensorInfo

    torch_dtype = schema_torch_dtype("bf16" if dtype in ("fp8", "mxfp8", "mxfp4") else dtype)
    return OpTensorInfo(shape=shape, dtype=torch_dtype, device=device)


def attention_valid_parts(q_len, kv_len, causal, sparsity):
    """Effective QK/PV element count after causal and sparsity discounts.

    Causal attention computes only the lower-triangular part when q_len ==
    kv_len; otherwise a q_len*kv_len/2 approximation is used. Sparsity keeps
    only the (1 - sparsity) fraction of blocks.
    """
    total_parts = q_len * kv_len
    if causal and q_len == kv_len:
        total_parts = q_len * (q_len + 1) / 2
    elif causal:
        total_parts = q_len * kv_len / 2
    return total_parts * (1 - sparsity)


class MfuMbuSummaryMixin:
    """Add MFU/MBU incremental fields to BasicOp.summary() output.

    peak_flops / peak_bw come from the case args (--config {"peak_flops":
    ...}); CPU simulation backends may leave them unset -> MFU/MBU are None.
    The MFU/MBU formula is shared with the offline report tool via
    common.metrics.util_metrics.

    Also injects ``executed_path`` when the vendor implementation records it
    (e.g. "bf16_fallback" when a quantized path degraded to bf16 on this
    platform), so reports can tell real quantized runs from fallbacks.
    """

    def summary(self, latency_us, kernel_mapping=None):
        target_dict = super().summary(latency_us, kernel_mapping or {})
        if not target_dict:
            return target_dict

        executed_path = getattr(self, "executed_path", None)
        if executed_path:
            target_dict["executed_path"] = executed_path

        # CUBE peaks come from the case args (--config {"peak_flops": ...});
        # MFU/MBU are None when the user did not provide them.
        peak_flops = (self.args_dict or {}).get("peak_flops")
        peak_bw = (self.args_dict or {}).get("peak_bw")
        mfu, mbu = util_metrics(
            target_dict.get("calc_flops_power(tflops)"),
            target_dict.get("mem_bw(GB/s)"),
            peak_flops,
            peak_bw,
        )
        if mfu is not None:
            target_dict["MFU"] = mfu
        if mbu is not None:
            target_dict["MBU"] = mbu
        return target_dict
