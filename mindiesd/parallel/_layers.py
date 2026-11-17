# SPDX-License-Identifier: Apache-2.0
"""The package's only boundary to stock MindIE-SD layers and device operators."""

from __future__ import annotations

import torch
import torch_npu

from mindiesd.layers.flash_attn.sparse_flash_attn import A5_V3_INNER_PRECISE as A5_INNER_PRECISE

from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (  # noqa: F401
    avgpool,
    pool_qk_separately,
    do_multi_span_tensor_rearrange_pooling,
    get_multi_span_blockwise_mask,
    is_a5_device,
    rain_fusion_attention,
)
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import rain_fusion_attention_v3

__all__ = [
    "avgpool",
    "do_multi_span_tensor_rearrange_pooling",
    "get_multi_span_blockwise_mask",
    "eagle_quant_block_sparse_attention",
    "bf16_block_sparse_attention",
    "rain_fusion_attention",
    "DEVICE_OP_BLOCK_SHAPE",
    "A5_INNER_PRECISE",
]

# The device operator's fixed arguments -- named so they cannot be confused with the
# mask/pool blocks of the parallel configuration (`config.MASK_BLOCK` / `POOL_BLOCK`).
DEVICE_OP_BLOCK_SHAPE = [128, 128]


_DEVICE_OP_DEFAULTS = {
    "block_shape": DEVICE_OP_BLOCK_SHAPE,
    "q_input_layout": "BNSD",
    "kv_input_layout": "BNSD",
    "num_key_value_heads": 1,
    "query_dtype": torch.int8,
    "key_dtype": torch.int8,
    "value_dtype": torch_npu.float8_e4m3fn,
    "output_dtype": torch.bfloat16,
    "softmax_lse_flag": 0,
}


def bf16_block_sparse_attention(
    *,
    query,
    key,
    value,
    block_sparse_mask,
    scale_value,
    inner_precise,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    bsa_backend="cann",
):
    """Run the stock RF-v3 BF16 BSA consumer while preserving BSND layout."""
    return rain_fusion_attention_v3(
        query,
        key,
        value,
        block_sparse_mask=block_sparse_mask,
        scale=scale_value,
        head_num=query.shape[2],
        num_key_value_heads=key.shape[2],
        input_layout="BSND",
        actual_seq_lengths=list(actual_seq_lengths),
        actual_seq_lengths_kv=list(actual_seq_lengths_kv),
        block_size_q=DEVICE_OP_BLOCK_SHAPE[0],
        block_size_kv=DEVICE_OP_BLOCK_SHAPE[1],
        inner_precise=int(inner_precise),
        bsa_backend=bsa_backend,
    )


def eagle_quant_block_sparse_attention(
    *,
    query,
    key,
    value,
    block_sparse_mask,
    scale_value,
    inner_precise,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    query_scale,
    key_scale,
    value_scale,
):
    """The quantized BSA operator, with its fixed arguments bound and the caller's passed through."""
    if key.ndim != 4 or key.shape[1] != 1:
        raise ValueError("quantized parallel attention requires one KV head per packet row")
    out, _ = torch.ops.mindiesd.eagle_quant_block_sparse_attention(
        query=query,
        key=key,
        value=value.view(torch.int8),
        block_sparse_mask=block_sparse_mask.view(torch.int8),
        scale_value=scale_value,
        inner_precise=int(inner_precise),
        actual_seq_lengths=list(actual_seq_lengths),
        actual_seq_lengths_kv=list(actual_seq_lengths_kv),
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        **_DEVICE_OP_DEFAULTS,
    )
    return out
