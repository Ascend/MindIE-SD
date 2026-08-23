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

"""Flash attention (BNSD layout, DiT-style)."""

from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry

from ._common import (
    SUPPORTED_FA_DTYPES,
    MfuMbuSummaryMixin,
    attention_valid_parts,
    op_tensor_info,
    quant_flops,
    tensor_bytes,
)


@ProviderRegistry.register_base_impl("fa", "ComputeEngine")
class FlashAttentionOp(MfuMbuSummaryMixin, BasicOp):
    """Flash attention for DiT diffusion latent shapes (BNSD layout).

    Args:
        batch_size (B), num_heads (N), head_dim (D), q_len / kv_len.
        causal (bool): when True, valid FLOPs use the lower-triangular part.
        dtype: bf16 / fp8 / mxfp8 / mxfp4.
        block_size / scale_alg: MX quantization knobs (aligned with layer.py).
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "default":
            raise ValueError(f"fa only supports arg_type=default, got {self.arg_type}")

        self.batch_size = int(self.args_dict["batch_size"])
        self.num_heads = int(self.args_dict["num_heads"])
        self.head_dim = int(self.args_dict["head_dim"])
        self.q_len = int(self.args_dict["q_len"])
        self.kv_len = int(self.args_dict.get("kv_len", self.q_len))
        self.causal = bool(self.args_dict.get("causal", False))
        self.sparsity = float(self.args_dict.get("sparsity", 0.0))

        self.dtype = self.args_dict.get("dtype", "bf16")
        self.block_size = int(self.args_dict.get("block_size", 32))
        self.scale_alg = int(self.args_dict.get("scale_alg", 2))
        self.softmax_scale = self.head_dim ** (-0.5)

        self._validate_args()
        self.flops_calc()

    def _validate_args(self):
        if self.dtype not in SUPPORTED_FA_DTYPES:
            raise ValueError(f"fa dtype {self.dtype} not in {SUPPORTED_FA_DTYPES}")
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads/head_dim must be positive")
        if self.q_len <= 0 or self.kv_len <= 0:
            raise ValueError("q_len/kv_len must be positive")
        if self.sparsity < 0 or self.sparsity >= 1:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")

    def flops_calc(self):
        # Per batch: QK^T -> B*N*S*S (2 flops/elem) and PV -> B*N*S*S (2 flops/elem).
        valid_parts = attention_valid_parts(self.q_len, self.kv_len, self.causal, self.sparsity)
        self.calc_flops = 2 * (self.num_heads * self.head_dim * valid_parts * 2)
        # Quantized paths quantize q/k/v inside the timed region (npu_dynamic_mx_quant
        # / _dynamic_mx_quant_fa); charge their elementwise work so MFU covers the
        # full measured op. Unpadded lengths are used (seqlen-scan configs are 2^k
        # so padded == unpadded for MXFP4_FA_SEQ_PAD_BASE=512).
        if self.dtype in ("fp8", "mxfp8", "mxfp4"):
            q_numel = self.batch_size * self.num_heads * self.q_len * self.head_dim
            kv_numel = self.batch_size * self.num_heads * self.kv_len * self.head_dim
            self.calc_flops += quant_flops(q_numel) + 2 * quant_flops(kv_numel)

    def vendor_parser(self):
        pass

    def vendor_impl(self):
        device = self.backend.get_torch_device_name()
        n, d, s_q, s_kv = self.num_heads, self.head_dim, self.q_len, self.kv_len

        self.input_tensor_info = {
            "q": op_tensor_info([self.batch_size, n, s_q, d], self.dtype, device),
            "k": op_tensor_info([self.batch_size, n, s_kv, d], self.dtype, device),
            "v": op_tensor_info([self.batch_size, n, s_kv, d], self.dtype, device),
        }
        self.output_tensor_info = {
            "out": op_tensor_info([self.batch_size, n, s_q, d], "bf16", device),
        }

        self.input_tensor_size = sum(
            tensor_bytes(int(self.batch_size * n * s * d), self.dtype) for s in (s_q, s_kv, s_kv)
        )
        self.output_tensor_size = tensor_bytes(int(self.batch_size * n * s_q * d), "bf16")
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        raise NotImplementedError
