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
        # Sparsity masks are input-constant across benchmark iterations;
        # building them inside the timed region would swamp the kernel time
        # with fixed allocation overhead, so they are built once and cached.
        self._sparse_mask_cache = {}
        self._output_validated = False

    def _sparse_mask(self, shape, sparsity):
        """Per-row uniform block mask: every query block keeps the same number
        of leading kv blocks, so no query row is left with zero attend blocks.

        The old "front-keep" construction (first N blocks set) left trailing
        query rows all-zero, which crashes block_sparse_attention (verified:
        EXC on front-keep vs OK on per-row-uniform; the operator UT passes).
        """
        key = (tuple(shape), sparsity)
        mask = self._sparse_mask_cache.get(key)
        if mask is None:
            import torch

            batch, n, q_blocks, kv_blocks = shape
            per_row = max(int(round(kv_blocks * (1 - sparsity))), 1)
            mask = torch.zeros(shape, dtype=torch.int8, device=self.backend.get_torch_device_name())
            mask[..., :per_row] = 1
            self._sparse_mask_cache[key] = mask
        return mask

    def _inner_precise(self):
        """950-series devices require inner_precise=4 (op vendor requirement);
        inner_precise=0 makes the kernel return all-zero output on 950PR."""
        import torch

        dev_name = torch.npu.get_device_properties(0).name
        return 4 if "950" in dev_name else 1

    def _check_output_valid(self, out, tensor_mapping):
        """Raise when the kernel returned no useful data (all-zero output).

        Kept out of the timed region: validated once on the first call
        (warmup), so a broken kernel marks the case invalid instead of
        emitting a fake latency row.
        """
        if self._output_validated:
            return
        self._output_validated = True
        import torch

        if not torch.is_floating_point(out) or int(torch.count_nonzero(out)) == 0:
            q_len = getattr(self, "q_len", tensor_mapping.get("q").shape[-2])
            raise RuntimeError(
                f"block_sparse_attention returned all-zero output "
                f"(kernel did not execute) for q_len={q_len}, "
                f"sparsity={getattr(self, 'sparsity', None)}"
            )

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

            sparse_mask = self._sparse_mask(
                (self.batch_size, n, self.q_len, self.kv_len), sparsity=0.0
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
            self._check_output_valid(out, tensor_mapping)
            return out

        from mindiesd.layers._custom_ops import block_sparse_attention

        q_blocks = self.q_len // 128
        kv_blocks = self.kv_len // 128
        block_mask = self._sparse_mask((self.batch_size, n, q_blocks, kv_blocks), self.sparsity)
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
            inner_precise=self._inner_precise(),
        )
        self._check_output_valid(out, tensor_mapping)
        return out
