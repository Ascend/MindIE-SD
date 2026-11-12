# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Reusable native BF16/RainFusion preparation and compute primitives."""

# pylint: disable=no-member,no-name-in-module  # fields and quant module arrive through prerequisite PRs

import inspect

import torch

from . import _layers
from . import distributed_quant as _q


class BF16Primitives:
    """Mixin that owns BF16 RF buffers, masks and native operator calls."""

    def _init_bf16(self):
        mapping = self.reorder_mapping
        _q.check_sendable(
            (mapping.bs, mapping.s_local, mapping.hc, mapping.hs),
            mapping.chunking,
            mapping.world,
        )
        if not self.optimize_attention_pipeline:
            return
        parameters = inspect.signature(_layers.get_multi_span_blockwise_mask).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        required = {"retained_idx"}
        if self.sparse_type == "rf_v3":
            required.update(("retained_mask", "pool_parts"))
        if not required.issubset(parameters) and not accepts_kwargs:
            raise RuntimeError(
                "BF16 parallel attention requires a matching mask helper accepting "
                + ", ".join(sorted(required))
                + "; load matching parallel and sparse RF modules"
            )
        self.retained_blocks["bf16_indices"] = torch.tensor(
            sorted(set(mapping.dense_blocks) | set(mapping.first_frame_blocks)),
            dtype=torch.long,
            device=mapping.reordered_position_to_token.device,
        )
        if self.sparse_type == "rf_v3":
            cells = (mapping.used_len + _q.POOL_BLOCK - 1) // _q.POOL_BLOCK
            protected = torch.zeros(
                (1, 1, cells, cells),
                dtype=torch.bool,
                device=mapping.reordered_position_to_token.device,
            )
            for block in set(mapping.dense_blocks) | set(mapping.first_frame_blocks):
                protected[:, :, block : block + 1, :] = True
                protected[:, :, :, block : block + 1] = True
            self.retained_blocks["bf16_protected_mask"] = protected

    def _prepare_bf16_head(self, rq, rk, rv):
        """Reorder/pool one ready head, retaining its RF tensors without head merging."""
        mapping = self.reorder_mapping
        tensors = [_q.from_send(received)[:, : mapping.used_len] for received in (rq, rk, rv)]
        return self._prepare_bf16_tensors(tensors)

    def _prepare_bf16_tensors(self, tensors):
        """Apply the same RF reorder/pool contract to serial-wide and ready-head inputs."""
        mapping = self.reorder_mapping
        if self.sparse_type == "rf_v3":
            return self._prepare_bf16_v3(tensors)
        tensors = [tensor.index_select(1, mapping.reordered_position_to_token) for tensor in tensors]
        pooled = _layers.avgpool(torch.cat(tensors, dim=0), _q.POOL_BLOCK, "BSND")
        return tensors, pooled

    def _prepare_bf16_v3(self, tensors):
        """Gather Q/K/V together and pool only the Q/K batches used for ranking."""
        mapping = self.reorder_mapping
        if not self.optimize_attention_pipeline:
            if not getattr(self, "pre_rearranged", False):
                tensors = [value.index_select(1, mapping.reordered_position_to_token) for value in tensors]
            return tensors, _layers.avgpool(torch.cat(tensors, dim=0), _q.POOL_BLOCK, "BSND")
        if getattr(self, "pre_rearranged", False):
            if tensors[0].device.type == "npu":
                pooled = _layers.pool_qk_separately(tensors[0], tensors[1], _q.POOL_BLOCK, "BSND")
                self.last_separate_qk_pool_calls += 1
            else:
                pooled = _layers.avgpool(torch.cat(tensors[:2], dim=0), _q.POOL_BLOCK, "BSND")
            return tensors, pooled
        reordered = torch.cat(tensors, dim=0).index_select(1, mapping.reordered_position_to_token)
        pooled = _layers.avgpool(reordered[: 2 * mapping.bs], _q.POOL_BLOCK, "BSND")
        return torch.chunk(reordered, 3, dim=0), pooled

    def _make_bf16_sparse_mask(self, pooled, *, scale, sparsity):
        """Build the selected backend's mask from the same RF ranking."""
        mapping = self.reorder_mapping
        if self.optimize_attention_pipeline:
            kwargs = {"retained_idx": self.retained_blocks["bf16_indices"]}
            if self.sparse_type == "rf_v3":
                kwargs.update(retained_mask=self.retained_blocks["bf16_protected_mask"], pool_parts=2)
        else:
            kwargs = {"async_index_transfer": False}
        return _layers.get_multi_span_blockwise_mask(
            pooled,
            sparsity,
            scale,
            mapping.dense_blocks,
            mapping.first_frame_blocks,
            "BSND",
            return_binary=self.sparse_type == "rf_v3",
            **kwargs,
        )

    def _slice_bf16_sparse_mask(self, mask, start, width):
        """Slice the common head dimension in either RF mask representation."""
        if self.sparse_type == "rf_v3":
            return mask[:, start : start + width].contiguous()
        indices, selected_counts = mask
        return (
            indices[:, start : start + width].contiguous(),
            selected_counts[:, start : start + width].contiguous(),
        )

    def _compute_bf16_head(self, tensors, mask, out_dst, *, scale, inner_precise):
        """Run BF16 and write the caller's sequence order into the reverse buffer."""
        mapping = self.reorder_mapping
        if self.sparse_type == "rf_v3":
            output = _layers.bf16_block_sparse_attention(
                bsa_backend=self.state.bsa_backend,
                query=tensors[0],
                key=tensors[1],
                value=tensors[2],
                block_sparse_mask=mask,
                scale_value=scale,
                inner_precise=inner_precise,
                actual_seq_lengths=[mapping.used_len] * mapping.bs,
                actual_seq_lengths_kv=[mapping.used_len] * mapping.bs,
            )
        else:
            indices, selected_counts = mask
            output = _layers.rain_fusion_attention(
                *(tensor.reshape(-1, mapping.w, mapping.hs) for tensor in tensors),
                scale=scale,
                head_num=mapping.w,
                input_layout="TND",
                select_idx=indices,
                select_num_idx=selected_counts,
                blockshape=_layers.DEVICE_OP_BLOCK_SHAPE,
                actual_seq_lengths=[mapping.used_len] * mapping.bs,
                actual_seq_lengths_kv=[mapping.used_len] * mapping.bs,
                inner_precise=inner_precise,
            ).reshape(mapping.bs, mapping.used_len, mapping.w, mapping.hs)
        if getattr(self, "pre_rearranged", False):
            out_dst.copy_(output)
        elif self.optimize_attention_pipeline:
            torch.index_select(output, 1, mapping.token_to_reordered_position, out=out_dst)
        else:
            out_dst.copy_(output.index_select(1, mapping.token_to_reordered_position))
        return out_dst


__all__ = ["BF16Primitives"]
