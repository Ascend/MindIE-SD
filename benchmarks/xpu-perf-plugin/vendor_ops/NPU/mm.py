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

"""NPU matmul vendor implementation for Ascend core ops."""

import logging

from _quant import is_quant_unsupported
from xpu_perf.micro_perf.core.op import ProviderRegistry

logger = logging.getLogger(__name__)


@ProviderRegistry.register_vendor_impl("mm", "NPU")
class NPUMatMulOp:
    """NPU MM: dispatch by quant_algo following mindiesd/quantization/layer.py.

    NO_QUANT      -> torch.matmul
    W8A8          -> npu_dynamic_quant + npu_quant_matmul
    W8A8_MXFP8    -> npu_dynamic_mx_quant + npu_quant_matmul
    W4A4_MXFP4    -> npu_dynamic_mx_quant(float4) + npu_quant_matmul

    Quantized paths fall back to bf16 matmul when the platform lacks the
    required quant kernels (e.g. DynamicMxQuant on Ascend910_93); schema byte
    accounting still reflects the requested quant_algo.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        import torch

        x = tensor_mapping["x"]
        w = tensor_mapping["w"]
        # Record what actually executed so reports can tell real quantized
        # runs from bf16 fallbacks (see MfuMbuSummaryMixin.executed_path).
        self.executed_path = self.quant_algo
        if self.quant_algo == "NO_QUANT":
            return torch.matmul(x, w)

        try:
            if self.quant_algo == "W8A8":
                return self._w8a8(x, w)
            if self.quant_algo == "W8A8_MXFP8":
                return self._w8a8_mxfp8(x, w)
            return self._w4a4_mxfp4(x, w)
        except RuntimeError as exc:
            if not is_quant_unsupported(exc):
                raise
            logger.warning(
                "%s matmul falls back to bf16 (quant kernel unsupported): %s",
                self.quant_algo,
                exc,
            )
            self.executed_path = "bf16_fallback"
            return torch.matmul(x, w)

    @staticmethod
    def _w8a8(x, w):
        import torch
        import torch_npu

        x_int8, x_scale = torch_npu.npu_dynamic_quant(x)
        # The weight quant kernel runs to produce w_scale; the quantized
        # weights themselves are not needed (npu_quant_matmul takes bf16 w).
        _, w_scale = torch_npu.npu_dynamic_mx_quant(w, dst_type=torch.float8_e4m3fn)
        output = torch_npu.npu_quant_matmul(
            x_int8,
            w,
            w_scale.reshape(-1),
            pertoken_scale=x_scale,
            output_dtype=torch.bfloat16,
        )
        return output

    @staticmethod
    def _w8a8_mxfp8(x, w):
        import torch
        import torch_npu

        x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        w_fp8, w_scale = torch_npu.npu_dynamic_mx_quant(w, dst_type=torch.float8_e4m3fn)
        output = torch_npu.npu_quant_matmul(
            x_fp8,
            w_fp8,
            w_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32],
        )
        return output

    @staticmethod
    def _w4a4_mxfp4(x, w):
        import torch
        import torch_npu

        from mindiesd.quantization.layer import (
            MXFP4_GROUP_SIZES_W4A4,
            _dynamic_mx_quant,
        )

        x_fp4, x_scale = _dynamic_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2)
        w_fp4, w_scale = _dynamic_mx_quant(w, dst_type=torch_npu.float4_e2m1fn_x2)
        w_scale = w_scale.reshape(w_scale.shape[0], -1, 2)
        output = torch_npu.npu_quant_matmul(
            x_fp4,
            w_fp4,
            w_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            output_dtype=torch.bfloat16,
            group_sizes=MXFP4_GROUP_SIZES_W4A4,
        )
        return output
