# SPDX-License-Identifier: Apache-2.0

import torch
import torch_npu
from packaging.version import InvalidVersion, Version

from mindiesd.utils.get_platform import get_cann_version

BLOCK = 128


def require_native_mxfp8_runtime():
    version = get_cann_version()
    try:
        supported = Version(version or "") >= Version("9.2.0b2")
    except InvalidVersion:
        supported = False
    if not supported:
        raise RuntimeError(f"native MXFP8 BSA requires CANN >= 9.2.0 beta 2, found {version or 'unknown'}")
    ops = torch.ops.custom
    if not hasattr(ops, "npu_quant_block_sparse_attn") or not hasattr(ops, "npu_quant_block_sparse_attn_metadata"):
        raise RuntimeError("native MXFP8 BSA requires the CANN attention and metadata extension on every rank")


def indexed_mask(mask):
    keep = mask.bool()
    width = keep.shape[-1]
    ids = torch.arange(width, dtype=torch.int32, device=mask.device)
    indices = torch.where(keep, ids, width).sort(dim=-1).values
    return indices.masked_fill_(indices == width, -1), keep.sum(dim=-1, dtype=torch.int32)


def paged_kv(tensor, length):
    pages = (length + BLOCK - 1) // BLOCK
    _, _, heads, width = tensor.shape
    padded = torch.nn.functional.pad(tensor[:, :length], (0, 0, 0, 0, 0, pages * BLOCK - length))
    if heads == 1:
        return padded.reshape(pages, BLOCK, width).contiguous().unsqueeze(1)
    return padded.reshape(pages, BLOCK, heads, width).transpose(1, 2).contiguous()


def e8m0_scales(scale):
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)
    if scale.dtype != torch.float8_e8m0fnu:
        raise TypeError("native MXFP8 requires E8M0 scale")
    return scale.contiguous()


def quantize_mx(tensor, *, axis):
    payload, scale = torch_npu.npu_dynamic_mx_quant(tensor, dst_type=torch.float8_e4m3fn, axis=axis)
    return payload, e8m0_scales(scale)


def prequantized_mxfp8_bsa(
    *,
    query,
    key,
    value,
    query_scale,
    key_scale,
    value_scale,
    block_sparse_mask,
    scale_value,
    actual_seq_lengths,
    actual_seq_lengths_kv,
):
    q_len, kv_len = actual_seq_lengths[0], actual_seq_lengths_kv[0]
    kv_blocks = (kv_len + BLOCK - 1) // BLOCK
    if block_sparse_mask.shape[-2:] != ((q_len + BLOCK - 1) // BLOCK, kv_blocks):
        raise ValueError("RainFusion mask does not match the Q/KV windows")
    qs, ks, vs = (e8m0_scales(scale) for scale in (query_scale, key_scale, value_scale))
    if query.ndim != 4 or query.shape[0] != 1 or qs.ndim != 5 or qs.shape[0] != 1:
        raise ValueError("native MXFP8 bridge requires B1 BSND Q and B1 Q scales")
    query_tnd = query.squeeze(0).contiguous()
    qs_tnd = qs.squeeze(0).contiguous()
    indices, counts = indexed_mask(block_sparse_mask)
    lengths = torch.tensor([kv_len], dtype=torch.int32, device=query.device)
    table = torch.arange(kv_blocks, dtype=torch.int32, device=query.device).reshape(1, -1)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=query.device)
    options = dict(
        layout_q="TND",
        layout_kv="PA_BNBD",
        quant_mode=2,
        mask_mode=0,
        cu_seqlens_q=cu_q,
        seqused_kv=lengths,
        cu_seqlens_kv=None,
        seqused_q=None,
    )
    ops = torch.ops.custom
    metadata = ops.npu_quant_block_sparse_attn_metadata(
        counts,
        query.shape[2],
        key.shape[1],
        query.shape[-1],
        batch_size=1,
        **options,
    )
    output, _ = ops.npu_quant_block_sparse_attn(
        query_tnd,
        key,
        value,
        qs_tnd,
        ks,
        vs,
        None,
        indices,
        counts,
        None,
        scale_value,
        BLOCK,
        BLOCK,
        block_table=table,
        metadata=metadata,
        layout_out="TND",
        **options,
    )
    return output.unsqueeze(0)


def bf16_mxfp8_bsa(tensors, options):
    query, key, value = tensors
    kv_len = options["actual_seq_lengths_kv"][0]
    query, qs = quantize_mx(query.contiguous(), axis=-1)
    key, ks = quantize_mx(paged_kv(key, kv_len), axis=-1)
    value, vs = quantize_mx(paged_kv(value, kv_len), axis=2)
    return prequantized_mxfp8_bsa(
        query=query,
        key=key,
        value=value,
        query_scale=qs,
        key_scale=ks,
        value_scale=vs,
        block_sparse_mask=options["block_sparse_mask"],
        scale_value=options["scale_value"],
        actual_seq_lengths=options["actual_seq_lengths"],
        actual_seq_lengths_kv=options["actual_seq_lengths_kv"],
    )
