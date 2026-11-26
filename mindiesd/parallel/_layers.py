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
    get_mask_index,
    is_a5_device,
    rain_fusion_attention,
)
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import (
    rain_fusion_attention_v3,
    _eagle_qbsa_quant_qkv as quantize_qbsa,
)

from .config import QUANT_BLOCK

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
    """Run the stock RF-v3 BF16/FP16 BSA consumer while preserving BSND layout."""
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


def is_a2_or_a3_device():
    from mindiesd.utils.get_platform import NPUDevice, get_npu_device

    return get_npu_device() in (NPUDevice.A2, NPUDevice.A3)


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


def run_bf16_bsa(precision, tensors, options):
    """Quantize received RF-ordered BF16 rows at the BSA consumer boundary."""
    query, key, value = tensors
    if precision == "mix":
        query, key, value, qs, ks, vs = quantize_qbsa(query, key, value, block_size_q=QUANT_BLOCK, layout="BSND")
        return eagle_quant_block_sparse_attention(
            query=query,
            key=key,
            value=value,
            query_scale=qs,
            key_scale=ks,
            value_scale=vs,
            **options,
        ).transpose(1, 2)

    from ._native_mxfp8 import bf16_mxfp8_bsa  # pylint: disable=import-error,no-name-in-module

    return bf16_mxfp8_bsa(tensors, options)


def cp_mix_wire(chain, query, key, value, scales, geometry, local):
    if not chain.optimize_attention_pipeline or not chain.pre_rearranged or query.device.type != "npu":
        return None
    from ._quantized_payload_kernels import cp_mix_wire as write

    return write(chain, query, key, value, scales, geometry, local, is_a5=is_a5_device)


def rf_v2_block_sparse_attention(
    *,
    query,
    key,
    value,
    block_sparse_mask,
    scale_value,
    inner_precise,
    actual_seq_lengths,
    actual_seq_lengths_kv,
):
    """Pass one RF query window and global KV to the indexed RF-v2 operator."""
    keep = block_sparse_mask.bool()
    return rain_fusion_attention(
        query.flatten(0, 1),
        key.flatten(0, 1),
        value.flatten(0, 1),
        scale=scale_value,
        head_num=query.shape[2],
        input_layout="TND",
        select_idx=get_mask_index(keep)[0].transpose(0, 1).contiguous(),
        select_num_idx=keep[0].transpose(0, 1).sum(dim=-1).contiguous(),
        blockshape=DEVICE_OP_BLOCK_SHAPE,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        inner_precise=inner_precise,
    ).reshape(1, query.shape[1], query.shape[2], query.shape[3])
