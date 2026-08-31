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

"""Block sparse attention (BSA)."""

from xpu_perf.micro_perf.core.op import ProviderRegistry

from ._common import attention_valid_parts
from .fa import FlashAttentionOp


@ProviderRegistry.register_base_impl("bsa", "ComputeEngine")
class BlockSparseAttentionOp(FlashAttentionOp):
    """Block sparse attention for video DiT long sequences.

    Extends fa schema with sparsity / block_size / mask_type. FLOPs are
    reduced by (1 - sparsity): the sparse kernel only touches the selected
    blocks, so effective FLOPs scale with the kept fraction. The sparsity scan
    (bsa 稀疏度扫描) then shows MFU/MBU vs sparsity.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        super().prepare_args()
        # Reproducibility: seed (via --config {"seed": 42}) fixes the RNG so
        # input tensors are identical across runs; the sparsity mask is
        # already deterministic (per-row uniform). Runs before create_tensors.
        seed = int(self.args_dict.get("seed", 42))
        import torch

        torch.manual_seed(seed)
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "manual_seed"):
            npu.manual_seed(seed)

    def flops_calc(self):
        # Sparse kernel only computes the kept (1 - sparsity) blocks.
        # ada_bsa receives a dense (all-ones) mask in the vendor impl, so its
        # FLOPs are not discounted; rf_v2/rf_v3 masks keep the sparsity ratio.
        effective_sparsity = self.sparsity if self.mask_type != "ada_bsa" else 0.0
        valid_parts = attention_valid_parts(
            self.q_len, self.kv_len, self.causal, effective_sparsity
        )
        self.calc_flops = 2 * (self.batch_size * self.num_heads * self.head_dim * valid_parts * 2)

    def _validate_args(self):
        super()._validate_args()
        self.mask_type = self.args_dict.get("mask_type", "rf_v3")
        if self.mask_type not in ("rf_v2", "rf_v3", "ada_bsa"):
            raise ValueError(f"mask_type {self.mask_type} not in rf_v2/rf_v3/ada_bsa")
        if self.sparsity <= 0:
            raise ValueError(f"bsa requires sparsity in (0, 1), got {self.sparsity}")
        # The NPU vendor routes to a bf16 kernel only; quantized dtypes would
        # be measured on bf16 while accounted at the quantized byte/FLOP rate,
        # producing wrong MFU/MBU. Reject them (case becomes a skip) instead of
        # emitting misleading data.
        if self.dtype != "bf16":
            raise ValueError(f"bsa vendor supports bf16 only, got {self.dtype}")
