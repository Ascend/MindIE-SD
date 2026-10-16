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

"""sparse_block_estimate (A5 / ascend950) block-score benchmark."""

from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry

from ._common import MfuMbuSummaryMixin, op_tensor_info, sbe_qk_flops, tensor_bytes

# A5 kernel contract. K above this is rejected by host tiling.
SBE_HEAD_DIM = 128
SBE_STRIDE = 8
SBE_SPARSE_SIZE = 128
SBE_MAX_K_SEQUENCE = 425984
SUPPORTED_SBE_DTYPES = ("bf16", "fp16")


@ProviderRegistry.register_base_impl("sbe", "ComputeEngine")
class SparseBlockEstimateOp(MfuMbuSummaryMixin, BasicOp):
    """Block-score estimate. seqlen scans kv_len; q_len stays fixed.

    The NPU path is Ascend 950 only: BNSD, head_dim 128, stride 8,
    sparse_size 128, non-causal, no actual_seq_lengths.
    """

    # BasicOp initializes these through prepare_args/vendor_impl lifecycle hooks.
    # pylint: disable=attribute-defined-outside-init

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "default":
            raise ValueError(f"sbe only supports arg_type=default, got {self.arg_type}")

        self.batch_size = int(self.args_dict["batch_size"])
        self.num_heads = int(self.args_dict["num_heads"])
        self.num_kv_heads = int(self.args_dict["num_kv_heads"])
        self.head_dim = int(self.args_dict["head_dim"])
        self.q_len = int(self.args_dict["q_len"])
        self.kv_len = int(self.args_dict["kv_len"])
        self.causal = bool(self.args_dict.get("causal", False))
        self.dtype = self.args_dict.get("dtype", "bf16")
        self.threshold = float(self.args_dict.get("threshold", 0.85))
        self.stride = int(self.args_dict.get("stride", SBE_STRIDE))
        self.sparse_size = int(self.args_dict.get("sparse_size", SBE_SPARSE_SIZE))
        self._validate_args()
        self.flops_calc()

    def _validate_args(self):
        if self.dtype not in SUPPORTED_SBE_DTYPES:
            raise ValueError(f"sbe dtype {self.dtype} not in {SUPPORTED_SBE_DTYPES}")
        if self.head_dim != SBE_HEAD_DIM or self.stride != SBE_STRIDE or self.sparse_size != SBE_SPARSE_SIZE:
            raise ValueError(
                f"sbe requires head_dim={SBE_HEAD_DIM}, stride={SBE_STRIDE}, sparse_size={SBE_SPARSE_SIZE}"
            )
        if self.causal:
            raise ValueError("sbe A5 path is non-causal")
        if self.batch_size <= 0 or self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("batch_size/num_heads/num_kv_heads must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.q_len <= 0 or self.kv_len <= 0 or self.kv_len > SBE_MAX_K_SEQUENCE:
            raise ValueError(f"kv_len must be in 1..{SBE_MAX_K_SEQUENCE}, q_len must be positive")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")

    def flops_calc(self):
        self.calc_flops = sbe_qk_flops(
            self.batch_size,
            self.num_heads,
            self.q_len,
            self.kv_len,
            self.head_dim,
            self.stride,
            self.sparse_size,
        )

    def vendor_parser(self):
        pass

    def vendor_impl(self):
        device = self.backend.get_torch_device_name()
        # fp16 and bf16 are both 2 bytes; tensor_bytes has no fp16 key.
        byte_dtype = "bf16"
        q_shape = [self.batch_size, self.num_heads, self.q_len, self.head_dim]
        k_shape = [self.batch_size, self.num_kv_heads, self.kv_len, self.head_dim]
        q_blocks = (self.q_len + self.sparse_size - 1) // self.sparse_size
        k_blocks = (self.kv_len + self.sparse_size - 1) // self.sparse_size
        k_blocks_aligned = (k_blocks + 31) // 32 * 32
        mask_shape = [self.batch_size, self.num_heads, q_blocks, k_blocks_aligned]
        count_shape = [self.batch_size, self.num_heads, q_blocks]

        self.input_tensor_info = {
            "q": op_tensor_info(q_shape, "bf16", device),
            "k": op_tensor_info(k_shape, "bf16", device),
        }
        if self.dtype == "fp16":
            import torch

            from xpu_perf.micro_perf.core.utils import OpTensorInfo

            self.input_tensor_info = {
                "q": OpTensorInfo(shape=q_shape, dtype=torch.float16, device=device),
                "k": OpTensorInfo(shape=k_shape, dtype=torch.float16, device=device),
            }
        self.output_tensor_info = {}

        q_numel = self.batch_size * self.num_heads * self.q_len * self.head_dim
        k_numel = self.batch_size * self.num_kv_heads * self.kv_len * self.head_dim
        mask_numel = self.batch_size * self.num_heads * q_blocks * k_blocks_aligned
        count_numel = self.batch_size * self.num_heads * q_blocks
        self.read_bytes = tensor_bytes(q_numel, byte_dtype) + tensor_bytes(k_numel, byte_dtype)
        self.write_bytes = mask_numel + count_numel * 4
        self.io_bytes = self.read_bytes + self.write_bytes
        self.input_tensor_size = self.read_bytes
        self.output_tensor_size = self.write_bytes
        self.tensor_size = self.io_bytes
        self._mask_shape = tuple(mask_shape)
        self._count_shape = tuple(count_shape)
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        raise NotImplementedError
