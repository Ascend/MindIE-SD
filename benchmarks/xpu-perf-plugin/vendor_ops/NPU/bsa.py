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

"""NPU block sparse attention vendor implementation for Ascend core ops."""

import math

from xpu_perf.micro_perf.core.op import ProviderRegistry


@ProviderRegistry.register_vendor_impl("bsa", "NPU")
class NPUBlockSparseAttentionOp:
    """NPU BSA: route to block_sparse_attention / ada_block_sparse_attention.

    mask_type rf_v2/rf_v3 -> torch.ops.mindiesd.block_sparse_attention
    mask_type ada_bsa     -> mindiesd ada_block_sparse_attention
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        import torch

        q = tensor_mapping["q"]
        k = tensor_mapping["k"]
        v = tensor_mapping["v"]
        n = self.num_heads
        d = self.head_dim
        scale = 1.0 / math.sqrt(d)

        if self.mask_type == "ada_bsa":
            from mindiesd.layers._custom_ops import ada_block_sparse_attention

            sparse_mask = torch.zeros(
                (self.batch_size, n, self.q_len, self.kv_len),
                dtype=torch.int8,
                device=q.device,
            )
            sparse_count_table = torch.zeros(
                (self.batch_size,),
                dtype=torch.int32,
                device=q.device,
            )
            out = ada_block_sparse_attention(
                q,
                k,
                v,
                sparse_mask=sparse_mask,
                sparse_count_table=sparse_count_table,
                input_layout="BNSD",
                num_heads=n,
                num_key_value_heads=n,
                scale_value=scale,
                causal=self.causal,
            )
            return out

        from mindiesd.layers._custom_ops import block_sparse_attention

        q_blocks = self.q_len // 128
        kv_blocks = self.kv_len // 128
        block_mask = torch.zeros(
            (self.batch_size, n, q_blocks, kv_blocks),
            dtype=torch.int8,
            device=q.device,
        )
        # Keep (1 - sparsity) of the blocks attending: 1=attend, 0=skip.
        keep = int(round(q_blocks * kv_blocks * (1 - self.sparsity)))
        if keep > 0:
            block_mask.view(-1)[:keep] = 1
        out, _ = block_sparse_attention(
            q,
            k,
            v,
            block_sparse_mask=block_mask,
            block_shape=[128, 128],
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=n,
            scale_value=scale,
            inner_precise=0,
        )
        return out
