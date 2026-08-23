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
    (workloads/bsa.json) then shows MFU/MBU vs sparsity.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def flops_calc(self):
        # Sparse kernel only computes the kept (1 - sparsity) blocks.
        valid_parts = attention_valid_parts(self.q_len, self.kv_len, self.causal, self.sparsity)
        self.calc_flops = 2 * (self.num_heads * self.head_dim * valid_parts * 2)

    def _validate_args(self):
        super()._validate_args()
        self.mask_type = self.args_dict.get("mask_type", "rf_v3")
        if self.mask_type not in ("rf_v2", "rf_v3", "ada_bsa"):
            raise ValueError(f"mask_type {self.mask_type} not in rf_v2/rf_v3/ada_bsa")
        if self.sparsity <= 0:
            raise ValueError(f"bsa requires sparsity in (0, 1), got {self.sparsity}")
