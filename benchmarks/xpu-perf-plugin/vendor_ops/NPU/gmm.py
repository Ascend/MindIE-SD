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

"""NPU grouped matmul vendor implementation for Ascend core ops.

The MoE-routing-specific npu_grouped_matmul per-group layout is not exercised
by the generic benchmark path; a dense matmul + swiglu fallback measures the
grouped shape's FLOPs/bytes on the underlying GEMM kernel. W8A8 uses
npu_weight_quant_batchmatmul; W8A8_MXFP8 requires DynamicMxQuant which is not
supported on all Ascend910_93 platforms (see RFC#299 risk list).
"""

from xpu_perf.micro_perf.core.op import ProviderRegistry


@ProviderRegistry.register_vendor_impl("gmm", "NPU")
class NPUGroupedMatMulOp:
    """NPU GMM: dense grouped-shape matmul + swiglu.

    NO_QUANT       -> torch.matmul (bf16 GEMM)
    W8A8_DYNAMIC   -> npu_weight_quant_batchmatmul
    W8A8_MXFP8     -> matmul fallback (DynamicMxQuant unsupported on 910_93)
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._per_col_scales = {}

    def vendor_impl_run(self, tensor_mapping):
        import torch

        x = tensor_mapping["x"]
        w13 = tensor_mapping["w13"]
        w2 = tensor_mapping["w2"]

        # Record what actually executed so reports can tell real quantized
        # runs from bf16 fallbacks (see MfuMbuSummaryMixin.executed_path).
        # W8A8_MXFP8 currently measures the dense bf16 kernel (DynamicMxQuant
        # grouped matmul unsupported), so it is tagged as a fallback.
        self.executed_path = self.quant_algo
        if self.quant_algo == "W8A8_MXFP8":
            self.executed_path = "bf16_fallback"

        if self.quant_algo == "W8A8_DYNAMIC":
            import torch_npu

            # npu_weight_quant_batchmatmul expects weight layout [B, K, N];
            # op_defs builds w13/w2 as [1, N, K] (the dense matmul layout), so
            # transpose before the call (verified by the x_k_dim/weight_k_dim
            # shape check this fixes). Per-column scale follows the N dim.
            w13_t = w13.transpose(-2, -1)
            gate_up = torch_npu.npu_weight_quant_batchmatmul(x, w13_t, self._per_col_scale(w13_t))
            act = _silu(gate_up)
            w2_t = w2.transpose(-2, -1)
            return torch_npu.npu_weight_quant_batchmatmul(act, w2_t, self._per_col_scale(w2_t))

        gate_up = torch.matmul(x, w13.transpose(-2, -1))
        act = _silu(gate_up)
        return torch.matmul(act, w2.transpose(-2, -1))

    def _per_col_scale(self, w):
        # Cache per weight shape; the scale is constant across benchmark iters.
        # For the [B, K, N] layout used by npu_weight_quant_batchmatmul the
        # per-column (N) scale is the trailing dim.
        key = tuple(w.shape)
        if key not in self._per_col_scales:
            import torch

            self._per_col_scales[key] = torch.ones(
                w.shape[-1], dtype=torch.float32, device=w.device
            )
        return self._per_col_scales[key]


def _silu(gate_up):
    import torch

    chunk = gate_up.shape[-1] // 2
    return torch.nn.functional.silu(gate_up[..., :chunk]) * gate_up[..., chunk:]
