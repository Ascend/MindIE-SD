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

"""Matmul / linear (MM)."""

from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry

from ._common import (
    SUPPORTED_MM_QUANT,
    MfuMbuSummaryMixin,
    op_tensor_info,
    quant_flops,
    tensor_bytes,
)


@ProviderRegistry.register_base_impl("mm", "ComputeEngine")
class MatMulOp(MfuMbuSummaryMixin, BasicOp):
    """Linear / FFN matmul with optional weight quantization.

    Args:
        M / K / N: matmul shape (M scanned over token seq length).
        dtype: bf16 / fp8 / mxfp8 / mxfp4 (activation dtype).
        quant_algo: NO_QUANT / W8A8 / W8A8_MXFP8 / W4A4_MXFP4.
        group_size / scale_alg: MX quantization knobs.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "default":
            raise ValueError(f"mm only supports arg_type=default, got {self.arg_type}")

        self.M = int(self.args_dict["M"])
        self.K = int(self.args_dict["K"])
        self.N = int(self.args_dict["N"])
        self.quant_algo = self.args_dict.get("quant_algo", "NO_QUANT")
        self.group_size = int(self.args_dict.get("group_size", 32))
        self.scale_alg = int(self.args_dict.get("scale_alg", 2))

        self._validate_args()
        self.flops_calc()

    def _validate_args(self):
        if self.quant_algo not in SUPPORTED_MM_QUANT:
            raise ValueError(f"mm quant_algo {self.quant_algo} not in {SUPPORTED_MM_QUANT}")
        if self.M <= 0 or self.K <= 0 or self.N <= 0:
            raise ValueError("M/K/N must be positive")

    @property
    def activation_dtype(self):
        if self.quant_algo in ("NO_QUANT",):
            return "bf16"
        if self.quant_algo == "W8A8":
            return "fp8"
        if self.quant_algo == "W8A8_MXFP8":
            return "mxfp8"
        return "mxfp4"

    @property
    def weight_dtype(self):
        if self.quant_algo in ("NO_QUANT",):
            return "bf16"
        if self.quant_algo in ("W8A8", "W8A8_MXFP8"):
            return "mxfp8"
        return "mxfp4"

    def flops_calc(self):
        self.calc_flops = 2 * self.M * self.N * self.K
        # Quantized paths quantize x and w inside the timed region (npu_dynamic_quant
        # / npu_dynamic_mx_quant); charge their elementwise work so MFU covers the
        # full measured op.
        if self.quant_algo != "NO_QUANT":
            self.calc_flops += quant_flops(self.M * self.K) + quant_flops(self.K * self.N)

    def vendor_parser(self):
        pass

    def vendor_impl(self):
        device = self.backend.get_torch_device_name()
        a_dtype = self.activation_dtype
        w_dtype = self.weight_dtype

        self.input_tensor_info = {
            "x": op_tensor_info([self.M, self.K], a_dtype, device),
            "w": op_tensor_info([self.K, self.N], w_dtype, device),
        }
        self.output_tensor_info = {
            "y": op_tensor_info([self.M, self.N], "bf16", device),
        }

        self.input_tensor_size = tensor_bytes(self.M * self.K, a_dtype) + tensor_bytes(self.K * self.N, w_dtype)
        self.output_tensor_size = tensor_bytes(self.M * self.N, "bf16")
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        raise NotImplementedError
