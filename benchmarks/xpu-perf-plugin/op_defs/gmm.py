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

"""Grouped matmul (GMM) for MoE models."""

from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry

from ._common import (
    SUPPORTED_GMM_QUANT,
    MfuMbuSummaryMixin,
    op_tensor_info,
    tensor_bytes,
)


@ProviderRegistry.register_base_impl("gmm", "ComputeEngine")
class GroupedMatMulOp(MfuMbuSummaryMixin, BasicOp):
    """Grouped expert matmul (MoE) for diffusion models.

    Args:
        num_tokens (M), hidden_size (C), moe_inter, experts, top_k.
        quant_algo: NO_QUANT / W8A8_DYNAMIC / W8A8_MXFP8.
        group_list: optional explicit per-token expert routing (len = M).
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "default":
            raise ValueError(f"gmm only supports arg_type=default, got {self.arg_type}")

        self.num_tokens = int(self.args_dict["num_tokens"])
        self.hidden_size = int(self.args_dict["hidden_size"])
        self.moe_inter = int(self.args_dict["moe_inter"])
        self.experts = int(self.args_dict["experts"])
        self.top_k = int(self.args_dict["top_k"])
        self.quant_algo = self.args_dict.get("quant_algo", "NO_QUANT")
        self.group_list = self.args_dict.get("group_list", None)

        self._validate_args()
        self.flops_calc()

    def _validate_args(self):
        if self.quant_algo not in SUPPORTED_GMM_QUANT:
            raise ValueError(f"gmm quant_algo {self.quant_algo} not in {SUPPORTED_GMM_QUANT}")
        if self.num_tokens <= 0 or self.hidden_size <= 0 or self.moe_inter <= 0:
            raise ValueError("num_tokens/hidden_size/moe_inter must be positive")
        if self.experts <= 0 or self.top_k <= 0 or self.top_k > self.experts:
            raise ValueError(f"top_k must be in (0, experts], got top_k={self.top_k}")
        if self.group_list is not None and len(self.group_list) != self.num_tokens:
            raise ValueError("group_list length must equal num_tokens")

    def flops_calc(self):
        # Dense grouped-shape MLP: gate_up 2*M*C*inter + w2 2*M*C*inter.
        # top_k is recorded for MoE routing context but the benchmark measures
        # the dense kernel, so FLOPs stay consistent with the measured op.
        self.calc_flops = 4 * self.num_tokens * self.hidden_size * self.moe_inter

    def vendor_parser(self):
        pass

    @property
    def weight_dtype(self):
        if self.quant_algo == "NO_QUANT":
            return "bf16"
        if self.quant_algo == "W8A8_DYNAMIC":
            return "fp8"
        return "mxfp8"

    def vendor_impl(self):
        device = self.backend.get_torch_device_name()
        m, c, inter, top_k = self.num_tokens, self.hidden_size, self.moe_inter, self.top_k
        w_dtype = self.weight_dtype

        self.input_tensor_info = {
            "x": op_tensor_info([1, m, c], "bf16", device),
            "w13": op_tensor_info([1, inter * 2, c], w_dtype, device),
            "w2": op_tensor_info([1, c, inter], w_dtype, device),
        }
        self.output_tensor_info = {
            "y": op_tensor_info([m, c], "bf16", device),
        }

        # Effective bytes: only top_k routed experts are touched.
        self.input_tensor_size = (
            tensor_bytes(m * c, "bf16")
            + tensor_bytes(inter * 2 * c, w_dtype) * top_k
            + tensor_bytes(c * inter, w_dtype) * top_k
        )
        self.output_tensor_size = tensor_bytes(m * c, "bf16")
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        raise NotImplementedError
