#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.

"""NPU Eagle quantized block sparse attention vendor implementation."""

from xpu_perf.micro_perf.core.op import ProviderRegistry

EQBSA_BLOCK_SHAPE = (128, 128)
EQBSA_Q_QUANT_BLOCK = 64


def _perblock_quant_int8(input_tensor, block_size=EQBSA_Q_QUANT_BLOCK):
    """Quantize Q/K per sequence block. Input/output layout is BNSD."""
    import torch
    import torch_npu
    from torch.nn import functional

    if input_tensor.dim() != 4:
        raise ValueError(f"per-block quant expects 4D BNSD, got {input_tensor.dim()} dims")

    batch, heads, seq_len, head_dim = input_tensor.shape
    padded = input_tensor
    if seq_len % block_size:
        pad_len = block_size - seq_len % block_size
        padded = functional.pad(input_tensor, (0, 0, 0, pad_len))

    blocks = (seq_len + block_size - 1) // block_size
    quant, scale = torch_npu.npu_dynamic_quant(padded.reshape(batch, heads, blocks, -1), dst_type=torch.int8)
    quant = quant.reshape(batch, heads, -1, head_dim)[:, :, :seq_len, :].contiguous()
    return quant, scale.contiguous()


def _perchannel_quant_fp8(value):
    """Quantize V per head channel and return BNSD FP8 with FP32 scales."""
    import torch_npu

    quant, scale = torch_npu.npu_dynamic_quant(value.transpose(-1, -2), dst_type=torch_npu.float8_e4m3fn)
    return quant.transpose(-1, -2).contiguous(), scale.contiguous()


@ProviderRegistry.register_vendor_impl("eqbsa", "NPU")
class NPUEagleQuantBlockSparseAttentionOp:
    """NPU EQBSA benchmark with Q/K INT8 and FP8 E4M3 V."""

    # These members come from the op_defs half of the runtime class
    # (prepare_args / vendor_impl lifecycle).
    # pylint: disable=no-member

    def _quantize(self, query, key, value):
        """Build the quantized QKV pair accepted by the fused kernel."""
        query_quant, query_scale = _perblock_quant_int8(query)
        key_quant, key_scale = _perblock_quant_int8(key)
        value_quant, value_scale = _perchannel_quant_fp8(value)
        return query_quant, key_quant, value_quant, query_scale, key_scale, value_scale

    def vendor_impl_run(self, tensor_mapping):
        import torch
        import mindiesd  # noqa: F401  # registers torch.ops.mindiesd

        assert "query_scale" in tensor_mapping, "pre-quantizes should outside the timed region"
        query = tensor_mapping["q"]
        key = tensor_mapping["k"]
        value = tensor_mapping["v"]
        query_scale = tensor_mapping["query_scale"]
        key_scale = tensor_mapping["key_scale"]
        value_scale = tensor_mapping["value_scale"]
        mask = tensor_mapping["block_sparse_mask"]
        out, _ = torch.ops.mindiesd.eagle_quant_block_sparse_attention(
            query=query,
            key=key,
            value=value.view(torch.int8),
            block_sparse_mask=mask.view(torch.int8),
            block_shape=list(EQBSA_BLOCK_SHAPE),
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=self.num_kv_heads,
            scale_value=self.softmax_scale,
            inner_precise=4,
            actual_seq_lengths=self.actual_seq_lengths,
            actual_seq_lengths_kv=self.actual_seq_lengths_kv,
            softmax_lse_flag=0,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            query_dtype=torch.int8,
            key_dtype=torch.int8,
            value_dtype=self._fp8_dtype(),
            output_dtype=torch.bfloat16,
        )
        return out

    @staticmethod
    def _fp8_dtype():
        import torch_npu

        return torch_npu.float8_e4m3fn
