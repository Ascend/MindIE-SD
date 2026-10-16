#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unified sequence-parallel attention.

The public boundary deliberately contains only MindIE-SD owned tensor/scalar
semantics. Callers pass normalized primitives; this module does not import or
introspect caller-owned objects.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

from .flash_attn.attention_forward import attention_forward
from .flash_attn.fused_infer_attention_score import fused_infer_attention_score_v2

_LAYOUTS = ("BSND", "BNSD")
_COMM_DTYPES = ("none", "fp8_e4m3", "mxfp8")
_COMM_TENSORS = frozenset(("q", "k", "v", "out"))
_COMM_QUANT_SCOPES = ("exposed", "all")
_BACKENDS = ("auto", "npu_fa", "quant_fa")
_FP8_MAX = 448.0


@dataclass
class _QuantizedFAInput:
    payload: torch.Tensor
    scale: torch.Tensor


class USPError(RuntimeError):
    """Base class for structured USP execution errors."""

    error_code = "MIE-USP-000"

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"[{self.error_code}] {reason}")


class USPNotSupported(USPError):
    """The requested, otherwise well-formed execution mode is unavailable."""

    error_code = "MIE-USP-001"


class USPTopologyError(USPError):
    """The process groups and tensor partition are inconsistent."""

    error_code = "MIE-USP-002"


class USPShapeError(USPError):
    """The tensor shape, dtype, layout, or semantic lengths are invalid."""

    error_code = "MIE-USP-003"


class USPWorkspaceError(USPError):
    """A caller-owned output or workspace buffer is invalid."""

    error_code = "MIE-USP-004"


def _world_size(group) -> int:
    return 1 if group is None else dist.get_world_size(group)


def _rank(group) -> int:
    return 0 if group is None else dist.get_rank(group)


def _to_bsnd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor if layout == "BSND" else tensor.transpose(1, 2)


def _from_bsnd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor if layout == "BSND" else tensor.transpose(1, 2)


def _validate_tensor(name: str, tensor: torch.Tensor, reference: torch.Tensor | None = None) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise USPShapeError(f"{name} must be a torch.Tensor, but got {type(tensor)}.")
    if tensor.dim() != 4:
        raise USPShapeError(f"{name} must be rank 4, but got rank {tensor.dim()}.")
    if reference is not None:
        if tensor.device != reference.device:
            raise USPShapeError(f"{name} must be on {reference.device}, but got {tensor.device}.")
        if tensor.dtype != reference.dtype:
            raise USPShapeError(f"{name} must have dtype {reference.dtype}, but got {tensor.dtype}.")


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str) -> None:
    _validate_tensor("q", q)
    _validate_tensor("k", k, q)
    _validate_tensor("v", v, q)
    q_bsnd, k_bsnd, v_bsnd = (_to_bsnd(tensor, layout) for tensor in (q, k, v))
    if q_bsnd.shape[0] != k_bsnd.shape[0] or k_bsnd.shape[0] != v_bsnd.shape[0]:
        raise USPShapeError("q, k, and v must have the same batch size.")
    if k_bsnd.shape[1] != v_bsnd.shape[1]:
        raise USPShapeError("k and v must have the same sequence length.")
    if k_bsnd.shape[2:] != v_bsnd.shape[2:]:
        raise USPShapeError("k and v must have the same head count and head dimension.")
    if q_bsnd.shape[-1] != k_bsnd.shape[-1]:
        raise USPShapeError("q and k must have the same head dimension.")
    if q_bsnd.shape[2] % k_bsnd.shape[2] != 0:
        raise USPShapeError("the number of q heads must be divisible by the number of kv heads.")


def _validate_joint(
    joint_k: torch.Tensor | None,
    joint_v: torch.Tensor | None,
    key: torch.Tensor,
    layout: str,
) -> None:
    if (joint_k is None) != (joint_v is None):
        raise USPShapeError("joint_k and joint_v must be provided together.")
    if joint_k is None:
        return
    _validate_tensor("joint_k", joint_k, key)
    _validate_tensor("joint_v", joint_v, key)
    key_bsnd = _to_bsnd(key, layout)
    joint_k_bsnd = _to_bsnd(joint_k, layout)
    joint_v_bsnd = _to_bsnd(joint_v, layout)
    if joint_k_bsnd.shape != joint_v_bsnd.shape:
        raise USPShapeError("joint_k and joint_v must have the same shape.")
    if joint_k_bsnd.shape[0] != key_bsnd.shape[0] or joint_k_bsnd.shape[2:] != key_bsnd.shape[2:]:
        raise USPShapeError("joint KV must match k batch, head count, and head dimension.")


def _validate_workspace(workspace: torch.Tensor | None, output: torch.Tensor | None, q: torch.Tensor) -> None:
    if workspace is not None:
        if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8:
            raise USPWorkspaceError("workspace must be a torch.uint8 tensor.")
        if workspace.device != q.device or not workspace.is_contiguous():
            raise USPWorkspaceError("workspace must be contiguous and on the same device as q.")
    if output is not None:
        if not isinstance(output, torch.Tensor):
            raise USPWorkspaceError("out must be a torch.Tensor.")
        if output.device != q.device or not output.is_contiguous():
            raise USPWorkspaceError("out must be contiguous and on the same device as q.")


def _validate_lengths(seq_lens: torch.Tensor | None, batch: int, max_length: int) -> None:
    if seq_lens is None:
        return
    if not isinstance(seq_lens, torch.Tensor) or seq_lens.dtype not in (torch.int32, torch.int64):
        raise USPShapeError("seq_lens must be an int32 or int64 tensor.")
    if seq_lens.dim() != 1 or seq_lens.numel() != batch:
        raise USPShapeError(f"seq_lens must have shape [{batch}].")
    if torch.any(seq_lens <= 0) or torch.any(seq_lens > max_length):
        raise USPShapeError(f"seq_lens values must be in [1, {max_length}].")


def _validate_topology(q: torch.Tensor, k: torch.Tensor, layout: str, ulysses_group) -> None:
    world_size = _world_size(ulysses_group)
    if world_size == 1:
        return
    q_heads = _to_bsnd(q, layout).shape[2]
    kv_heads = _to_bsnd(k, layout).shape[2]
    if q_heads % world_size != 0 or kv_heads % world_size != 0:
        raise USPTopologyError(
            f"q/kv heads must be divisible by Ulysses world size {world_size}, but got {q_heads}/{kv_heads}."
        )


def _get_fp8_dtype() -> torch.dtype:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise USPNotSupported("FP8 communication requires torch.float8_e4m3fn support.")
    return fp8_dtype


def _block_quantize(
    tensor: torch.Tensor,
    block_size: int,
    comm_dtype: str,
    scale_override: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize each leading communication chunk independently."""
    chunks = tensor.shape[0]
    flat = tensor.float().reshape(chunks, -1)
    original_width = flat.shape[1]
    padded_width = math.ceil(original_width / block_size) * block_size
    if padded_width != original_width:
        flat = torch.nn.functional.pad(flat, (0, padded_width - original_width))
    blocks = flat.reshape(chunks, -1, block_size)
    scales = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny) / _FP8_MAX
    if comm_dtype == "mxfp8":
        scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    if scale_override is not None:
        if not isinstance(scale_override, torch.Tensor):
            raise USPShapeError("communication scale overrides must be torch.Tensor values.")
        if scale_override.numel() == 1:
            scales = scale_override.to(device=tensor.device, dtype=torch.float32).expand_as(scales)
        elif tuple(scale_override.shape) == tuple(scales.shape):
            scales = scale_override.to(device=tensor.device, dtype=torch.float32)
        else:
            raise USPShapeError(
                f"scale override must be scalar or have shape {tuple(scales.shape)}, "
                f"but got {tuple(scale_override.shape)}."
            )
    quantized = (blocks / scales.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(_get_fp8_dtype())
    quantized = quantized.reshape(chunks, padded_width)
    return quantized, scales.contiguous(), original_width


def _block_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    original_width: int,
    shape: torch.Size,
    dtype: torch.dtype,
) -> torch.Tensor:
    block_size = quantized.shape[1] // scales.shape[1]
    output = quantized.float().reshape(scales.shape[0], -1, block_size) * scales.unsqueeze(-1)
    return output.reshape(scales.shape[0], -1)[:, :original_width].reshape(shape).to(dtype)


def _all_to_all(
    packed: torch.Tensor,
    group,
    comm_dtype: str,
    block_size: int,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if comm_dtype == "none":
        output = torch.empty_like(packed)
        dist.all_to_all_single(output, packed.contiguous(), group=group, async_op=False)
        return output

    quantized, scales, original_width = _block_quantize(packed, block_size, comm_dtype, scale)
    recv_quantized = torch.empty_like(quantized)
    recv_scales = torch.empty_like(scales)
    dist.all_to_all_single(recv_quantized, quantized, group=group, async_op=False)
    dist.all_to_all_single(recv_scales, scales, group=group, async_op=False)
    return _block_dequantize(recv_quantized, recv_scales, original_width, packed.shape, packed.dtype)


def _ulysses_forward(
    tensor: torch.Tensor,
    group,
    comm_dtype: str,
    block_size: int,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    world_size = _world_size(group)
    if world_size == 1:
        return tensor
    batch, sequence, heads, head_dim = tensor.shape
    heads_per_rank = heads // world_size
    packed = tensor.reshape(batch, sequence, world_size, heads_per_rank, head_dim)
    packed = packed.permute(2, 0, 1, 3, 4).contiguous()
    received = _all_to_all(packed, group, comm_dtype, block_size, scale)
    return received.permute(1, 0, 2, 3, 4).reshape(batch, world_size * sequence, heads_per_rank, head_dim)


def _ulysses_reverse(
    tensor: torch.Tensor,
    group,
    comm_dtype: str = "none",
    block_size: int = 1,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    world_size = _world_size(group)
    if world_size == 1:
        return tensor
    batch, global_sequence, heads_per_rank, head_dim = tensor.shape
    if global_sequence % world_size != 0:
        raise USPTopologyError(
            f"FA output sequence {global_sequence} is not divisible by Ulysses world size {world_size}."
        )
    local_sequence = global_sequence // world_size
    packed = tensor.reshape(batch, world_size, local_sequence, heads_per_rank, head_dim)
    packed = packed.permute(1, 0, 2, 3, 4).contiguous()
    received = _all_to_all(packed, group, comm_dtype, block_size, scale)
    return received.permute(1, 2, 0, 3, 4).reshape(batch, local_sequence, world_size * heads_per_rank, head_dim)


def _all_gather_sequence(
    tensor: torch.Tensor,
    group,
    comm_dtype: str,
    block_size: int,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    world_size = _world_size(group)
    if world_size == 1:
        return tensor
    original_shape = tensor.shape
    packed = tensor.unsqueeze(0)
    if comm_dtype == "none":
        gathered_flat = torch.empty(
            (world_size * original_shape[0],) + original_shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(gathered_flat, tensor.contiguous(), group=group)
        gathered = gathered_flat.reshape((world_size,) + original_shape)
    else:
        quantized, scales, original_width = _block_quantize(packed, block_size, comm_dtype, scale)
        gathered_quantized = torch.empty(
            (world_size,) + quantized.shape[1:], dtype=quantized.dtype, device=quantized.device
        )
        gathered_scales = torch.empty((world_size,) + scales.shape[1:], dtype=scales.dtype, device=scales.device)
        dist.all_gather_into_tensor(gathered_quantized, quantized.contiguous(), group=group)
        dist.all_gather_into_tensor(gathered_scales, scales.contiguous(), group=group)
        gathered = _block_dequantize(
            gathered_quantized,
            gathered_scales,
            original_width,
            torch.Size((world_size,) + original_shape),
            tensor.dtype,
        )
    return gathered.permute(1, 0, 2, 3, 4).reshape(
        original_shape[0], world_size * original_shape[1], original_shape[2], original_shape[3]
    )


def _slice_joint(tensor: torch.Tensor | None, group) -> torch.Tensor | None:
    if tensor is None or _world_size(group) == 1:
        return tensor
    world_size = _world_size(group)
    rank = _rank(group)
    heads_per_rank = tensor.shape[2] // world_size
    return tensor[:, :, rank * heads_per_rank : (rank + 1) * heads_per_rank, :]


def _fa_block_quantize(tensor: torch.Tensor, block_size: int) -> _QuantizedFAInput:
    try:
        import torch_npu

        from .quant.block_quant import fa_block_quant_preprocess
    except ImportError as error:
        raise USPNotSupported("quant_fa requires torch_npu and the MindIE-SD block quant operator.") from error
    payload, scale = fa_block_quant_preprocess(
        tensor, block_size=block_size, dst_type=torch_npu.float8_e4m3fn, layout="BSND"
    )
    return _QuantizedFAInput(payload, scale)


def _all_gather_quant_fa(tensor: _QuantizedFAInput, group) -> _QuantizedFAInput:
    """Gather BNSD FA payload and its sequence-block scale without dequantizing."""
    world_size = _world_size(group)
    if world_size == 1:
        return tensor

    def gather(value: torch.Tensor) -> torch.Tensor:
        gathered = torch.empty((world_size,) + tuple(value.shape), dtype=value.dtype, device=value.device)
        dist.all_gather_into_tensor(gathered, value.contiguous(), group=group)
        return gathered.permute(1, 2, 0, 3, 4).reshape(
            value.shape[0], value.shape[1], world_size * value.shape[2], value.shape[3]
        )

    return _QuantizedFAInput(gather(tensor.payload), gather(tensor.scale))


def _concat_quant_fa(
    tensor: _QuantizedFAInput,
    joint: torch.Tensor | None,
    block_size: int,
) -> _QuantizedFAInput:
    if joint is None:
        return tensor
    joint_quant = _fa_block_quantize(joint, block_size)
    return _QuantizedFAInput(
        torch.cat((tensor.payload, joint_quant.payload), dim=2),
        torch.cat((tensor.scale, joint_quant.scale), dim=2),
    )


def _run_quant_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    q_block_size: int,
    kv_block_size: int,
    out_dtype: torch.dtype,
    prepared_key: _QuantizedFAInput | None = None,
    prepared_value: _QuantizedFAInput | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_bnsd = query.transpose(1, 2)
    q_quant = _fa_block_quantize(query, q_block_size)
    k_quant = prepared_key or _fa_block_quantize(key, kv_block_size)
    v_quant = prepared_value or _fa_block_quantize(value, kv_block_size)
    output, lse = fused_infer_attention_score_v2(
        q_quant.payload,
        k_quant.payload,
        v_quant.payload,
        input_layout="BNSD",
        num_query_heads=query_bnsd.shape[1],
        num_key_value_heads=k_quant.payload.shape[1],
        softmax_scale=1.0 / math.sqrt(query_bnsd.shape[-1]),
        atten_mask=attn_mask,
        query_quant_mode=7,
        key_quant_mode=7,
        value_quant_mode=7,
        dequant_scale_query=q_quant.scale,
        dequant_scale_key=k_quant.scale,
        dequant_scale_value=v_quant.scale,
        out_dtype=out_dtype,
    )
    return output.transpose(1, 2), lse


def _run_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    backend: str,
    q_block_size: int,
    kv_block_size: int,
    out_dtype: torch.dtype,
    prepared_key: _QuantizedFAInput | None = None,
    prepared_value: _QuantizedFAInput | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if backend == "quant_fa":
        return _run_quant_fa(
            query,
            key,
            value,
            attn_mask,
            q_block_size,
            kv_block_size,
            out_dtype,
            prepared_key,
            prepared_value,
        )
    kwargs = {"opt_mode": "runtime"}
    if backend == "npu_fa":
        kwargs = {"opt_mode": "manual", "op_type": "fused_attn_score", "layout": "BSND"}
    output = attention_forward(query, key, value, attn_mask=attn_mask, fused=True, head_first=False, **kwargs)
    return output, None


def _get_npu_runtime():
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return None
    return npu


def _chunk_comm_dtype(
    name: str,
    comm_dtype: str,
    comm_tensors: tuple[str, ...],
    comm_quant_scope: str,
    overlap: bool,
    chunk_index: int,
    chunk_count: int,
    reverse: bool = False,
) -> str:
    if comm_dtype == "none" or name not in comm_tensors:
        return "none"
    if not overlap or comm_quant_scope == "all":
        return comm_dtype
    boundary = chunk_count - 1 if reverse else 0
    return comm_dtype if chunk_index == boundary else "none"


def _split_heads(tensor: torch.Tensor, chunk_size: int) -> list[torch.Tensor]:
    return list(torch.split(tensor, chunk_size, dim=2))


def _record_stream(tensors, stream) -> None:
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "npu":
            tensor.record_stream(stream)


def _forward_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    joint_key: torch.Tensor | None,
    joint_value: torch.Tensor | None,
    ulysses_group,
    kv_gather_group,
    comm_types: dict[str, str],
    block_sizes: dict[str, int],
    scales: dict[str, torch.Tensor | None],
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _QuantizedFAInput | None, _QuantizedFAInput | None]:
    query = _ulysses_forward(query, ulysses_group, comm_types["q"], block_sizes["q"], scales["q"])
    key = _ulysses_forward(key, ulysses_group, comm_types["k"], block_sizes["k"], scales["k"])
    value = _ulysses_forward(value, ulysses_group, comm_types["v"], block_sizes["v"], scales["v"])

    # This is the model-repository quantized KV AllGather path: FA block
    # quantization precedes communication, and payload+scale remain quantized
    # until fused quant FA consumes them.
    reuse_fa_quant = (
        backend == "quant_fa"
        and comm_types["k"] == "fp8_e4m3"
        and comm_types["v"] == "fp8_e4m3"
        and key.shape[1] % block_sizes["k"] == 0
        and value.shape[1] % block_sizes["v"] == 0
    )
    prepared_key = prepared_value = None
    if reuse_fa_quant:
        prepared_key = _all_gather_quant_fa(_fa_block_quantize(key, block_sizes["k"]), kv_gather_group)
        prepared_value = _all_gather_quant_fa(_fa_block_quantize(value, block_sizes["v"]), kv_gather_group)
        prepared_key = _concat_quant_fa(prepared_key, joint_key, block_sizes["k"])
        prepared_value = _concat_quant_fa(prepared_value, joint_value, block_sizes["v"])
    else:
        key = _all_gather_sequence(key, kv_gather_group, comm_types["k"], block_sizes["k"], scales["k"])
        value = _all_gather_sequence(value, kv_gather_group, comm_types["v"], block_sizes["v"], scales["v"])
        if joint_key is not None:
            key = torch.cat((key, joint_key), dim=1)
            value = torch.cat((value, joint_value), dim=1)
    return query, key, value, prepared_key, prepared_value


def _run_head_chunks(
    q_chunks: list[torch.Tensor],
    k_chunks: list[torch.Tensor],
    v_chunks: list[torch.Tensor],
    joint_k_chunks: list[torch.Tensor | None],
    joint_v_chunks: list[torch.Tensor | None],
    *,
    ulysses_group,
    kv_gather_group,
    comm_dtype: str,
    comm_tensors: tuple[str, ...],
    comm_quant_scope: str,
    overlap: bool,
    block_sizes: dict[str, int],
    scales: dict[str, torch.Tensor | None],
    backend: str,
    attn_mask: torch.Tensor | None,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    chunk_count = len(q_chunks)
    runtime = _get_npu_runtime() if overlap else None
    if overlap and runtime is None:
        raise USPNotSupported("overlap=True requires an available torch.npu stream runtime.")
    current_stream = runtime.current_stream() if runtime is not None else None
    comm_stream = runtime.Stream() if runtime is not None else None
    ready_events = []
    forwarded = []

    for index, tensors in enumerate(zip(q_chunks, k_chunks, v_chunks)):
        comm_types = {
            name: _chunk_comm_dtype(name, comm_dtype, comm_tensors, comm_quant_scope, overlap, index, chunk_count)
            for name in ("q", "k", "v")
        }
        if runtime is None:
            forwarded.append(
                _forward_chunk(
                    *tensors,
                    joint_k_chunks[index],
                    joint_v_chunks[index],
                    ulysses_group,
                    kv_gather_group,
                    comm_types,
                    block_sizes,
                    scales,
                    backend,
                )
            )
            ready_events.append(None)
            continue
        input_ready = runtime.Event()
        input_ready.record(current_stream)
        ready = runtime.Event()
        with runtime.stream(comm_stream):
            comm_stream.wait_event(input_ready)
            _record_stream(tensors, comm_stream)
            forwarded.append(
                _forward_chunk(
                    *tensors,
                    joint_k_chunks[index],
                    joint_v_chunks[index],
                    ulysses_group,
                    kv_gather_group,
                    comm_types,
                    block_sizes,
                    scales,
                    backend,
                )
            )
            ready.record(comm_stream)
        ready_events.append(ready)

    outputs = []
    lse_chunks = []
    reverse_events = []
    for index, (query, key, value, prepared_key, prepared_value) in enumerate(forwarded):
        if current_stream is not None:
            current_stream.wait_event(ready_events[index])
            _record_stream((query, key, value), current_stream)
            if prepared_key is not None:
                _record_stream(
                    (
                        prepared_key.payload,
                        prepared_key.scale,
                        prepared_value.payload,
                        prepared_value.scale,
                    ),
                    current_stream,
                )
        output, lse = _run_fa(
            query,
            key,
            value,
            attn_mask,
            backend,
            block_sizes["q"],
            block_sizes["k"],
            target_dtype,
            prepared_key,
            prepared_value,
        )
        reverse_dtype = _chunk_comm_dtype(
            "out",
            comm_dtype,
            comm_tensors,
            comm_quant_scope,
            overlap,
            index,
            chunk_count,
            reverse=True,
        )
        if runtime is None:
            output = _ulysses_reverse(output, ulysses_group, reverse_dtype, block_sizes["q"], None)
            reverse_events.append(None)
        else:
            fa_done = runtime.Event()
            fa_done.record(current_stream)
            reverse_done = runtime.Event()
            with runtime.stream(comm_stream):
                comm_stream.wait_event(fa_done)
                _record_stream((output,), comm_stream)
                output = _ulysses_reverse(output, ulysses_group, reverse_dtype, block_sizes["q"], None)
                reverse_done.record(comm_stream)
            reverse_events.append(reverse_done)
        outputs.append(output)
        if lse is not None:
            lse_chunks.append(lse)
    if current_stream is not None:
        for output, reverse_done in zip(outputs, reverse_events):
            current_stream.wait_event(reverse_done)
            _record_stream((output,), current_stream)
    return torch.cat(outputs, dim=2), torch.cat(lse_chunks, dim=1) if lse_chunks else None


def _copy_to_out(result: torch.Tensor, output: torch.Tensor | None) -> torch.Tensor:
    if output is None:
        return result
    if output.shape != result.shape or output.dtype != result.dtype:
        raise USPWorkspaceError(
            f"out must have shape {tuple(result.shape)} and dtype {result.dtype}, "
            f"but got {tuple(output.shape)} and {output.dtype}."
        )
    output.copy_(result)
    return output


# The explicit ABI is intentional: unlike an opaque metadata object, every
# accepted primitive has stable MindIE-SD semantics and is visible to type checkers.
def usp_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    ulysses_group=None,
    kv_gather_group=None,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    seq_lens: torch.Tensor | None = None,
    chunk_size: int | None = None,
    head_chunk_size: int | None = None,
    layout: Literal["BSND", "BNSD"] = "BSND",
    joint_k: torch.Tensor | None = None,
    joint_v: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    comm_dtype: Literal["none", "fp8_e4m3", "mxfp8"] = "none",
    comm_tensors: tuple[str, ...] = ("k", "v", "out"),
    comm_quant_scope: Literal["exposed", "all"] = "exposed",
    q_block_size: int = 128,
    kv_block_size: int = 256,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    overlap: bool = False,
    backend: Literal["auto", "npu_fa", "quant_fa"] = "auto",
    out_dtype: torch.dtype | None = None,
    workspace: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute Ulysses/KV-AllGather sequence-parallel attention in one call.

    ``q``, ``k`` and ``v`` are local sequence shards. Ulysses exchanges the
    head dimension for a full sequence, KV-AllGather collects KV sequence
    shards, and the reverse Ulysses exchange restores the caller's original partition.
    Joint KV is interpreted as replicated data and is head-sliced, not gathered.
    Communication quantization applies only to names in ``comm_tensors``.
    It is disabled when ``comm_dtype='none'``. In an overlap pipeline,
    ``comm_quant_scope='exposed'`` applies it only to the first forward chunk
    and last reverse-output chunk; ``'all'`` applies it to every chunk.

    KV-AllGather is not Ring Attention: it materializes global K/V before one
    local FA call and does not circulate KV blocks or merge partial FA results.

    This function does not infer or create process groups and never silently
    changes the requested execution strategy.
    """
    if layout not in _LAYOUTS:
        raise USPShapeError(f"layout must be one of {_LAYOUTS}, but got {layout}.")
    if comm_dtype not in _COMM_DTYPES:
        raise USPNotSupported(f"comm_dtype must be one of {_COMM_DTYPES}, but got {comm_dtype}.")
    if backend not in _BACKENDS:
        raise USPNotSupported(f"backend must be one of {_BACKENDS}, but got {backend}.")
    if comm_quant_scope not in _COMM_QUANT_SCOPES:
        raise USPNotSupported(f"comm_quant_scope must be one of {_COMM_QUANT_SCOPES}.")
    if scatter_dim != 2 or gather_dim != 1:
        raise USPNotSupported("v1 supports only scatter_dim=2 and gather_dim=1 in normalized BSND layout.")
    if set(comm_tensors) - _COMM_TENSORS:
        raise USPShapeError(f"comm_tensors may contain only {_COMM_TENSORS}.")
    if comm_dtype == "none" and any(scale is not None for scale in (q_scale, k_scale, v_scale)):
        raise USPShapeError("q_scale/k_scale/v_scale require a quantized comm_dtype.")
    if q_block_size <= 0 or kv_block_size <= 0:
        raise USPShapeError("q_block_size and kv_block_size must be positive.")
    if chunk_size is not None and chunk_size <= 0:
        raise USPShapeError("chunk_size must be positive when provided.")
    if head_chunk_size is not None and head_chunk_size <= 0:
        raise USPShapeError("head_chunk_size must be positive when provided.")
    if chunk_size is not None and (head_chunk_size is not None or overlap):
        raise USPNotSupported("chunk_size cannot be combined with head-cut or overlap execution.")
    if return_lse and backend != "quant_fa":
        raise USPNotSupported("return_lse is currently supported only by backend='quant_fa'.")

    _validate_qkv(q, k, v, layout)
    _validate_joint(joint_k, joint_v, k, layout)
    _validate_workspace(workspace, out, q)
    _validate_topology(q, k, layout, ulysses_group)

    q_bsnd, k_bsnd, v_bsnd = (_to_bsnd(tensor, layout) for tensor in (q, k, v))
    joint_k_bsnd = None if joint_k is None else _to_bsnd(joint_k, layout)
    joint_v_bsnd = None if joint_v is None else _to_bsnd(joint_v, layout)
    global_kv_length = k_bsnd.shape[1] * _world_size(ulysses_group) * _world_size(kv_gather_group)
    if joint_k_bsnd is not None:
        global_kv_length += joint_k_bsnd.shape[1]
    _validate_lengths(seq_lens, q_bsnd.shape[0], global_kv_length)
    scales = {"q": q_scale, "k": k_scale, "v": v_scale}
    block_sizes = {"q": q_block_size, "k": kv_block_size, "v": kv_block_size}

    world_size = _world_size(ulysses_group)
    if overlap and head_chunk_size is None:
        head_chunk_size = world_size
    if head_chunk_size is not None:
        q_heads = q_bsnd.shape[2]
        kv_heads = k_bsnd.shape[2]
        if head_chunk_size % world_size != 0 or q_heads % head_chunk_size != 0:
            raise USPTopologyError("head_chunk_size must be divisible by Ulysses world size and evenly divide q heads.")
        if head_chunk_size * kv_heads % q_heads != 0:
            raise USPTopologyError("head_chunk_size does not map to an integral KV head chunk.")
        kv_head_chunk_size = head_chunk_size * kv_heads // q_heads
        if kv_head_chunk_size % world_size != 0 or kv_heads % kv_head_chunk_size != 0:
            raise USPTopologyError("the derived KV head chunk is incompatible with Ulysses topology.")

        sliced_joint_k = _slice_joint(joint_k_bsnd, ulysses_group)
        sliced_joint_v = _slice_joint(joint_v_bsnd, ulysses_group)
        q_chunks = _split_heads(q_bsnd, head_chunk_size)
        k_chunks = _split_heads(k_bsnd, kv_head_chunk_size)
        v_chunks = _split_heads(v_bsnd, kv_head_chunk_size)
        joint_chunk_size = kv_head_chunk_size // world_size
        if sliced_joint_k is None:
            joint_k_chunks = [None] * len(q_chunks)
            joint_v_chunks = [None] * len(q_chunks)
        else:
            joint_k_chunks = _split_heads(sliced_joint_k, joint_chunk_size)
            joint_v_chunks = _split_heads(sliced_joint_v, joint_chunk_size)
        if not (len(q_chunks) == len(k_chunks) == len(v_chunks) == len(joint_k_chunks)):
            raise USPTopologyError("Q, KV, and joint KV produce different head chunk counts.")
        result_bsnd, lse = _run_head_chunks(
            q_chunks,
            k_chunks,
            v_chunks,
            joint_k_chunks,
            joint_v_chunks,
            ulysses_group=ulysses_group,
            kv_gather_group=kv_gather_group,
            comm_dtype=comm_dtype,
            comm_tensors=comm_tensors,
            comm_quant_scope=comm_quant_scope,
            overlap=overlap,
            block_sizes=block_sizes,
            scales=scales,
            backend=backend,
            attn_mask=attn_mask,
            target_dtype=q.dtype if out_dtype is None else out_dtype,
        )
        result = _from_bsnd(result_bsnd, layout).to(q.dtype if out_dtype is None else out_dtype)
        result = _copy_to_out(result, out)
        if return_lse:
            if lse is None:
                raise USPNotSupported("the selected FA kernel did not return softmax LSE.")
            return result, lse
        return result

    transformed = []
    for name, tensor in (("q", q_bsnd), ("k", k_bsnd), ("v", v_bsnd)):
        tensor_comm_dtype = comm_dtype if name in comm_tensors else "none"
        transformed.append(_ulysses_forward(tensor, ulysses_group, tensor_comm_dtype, block_sizes[name], scales[name]))
    q_bsnd, k_bsnd, v_bsnd = transformed  # pylint: disable=unbalanced-tuple-unpacking

    k_comm_dtype = comm_dtype if "k" in comm_tensors else "none"
    v_comm_dtype = comm_dtype if "v" in comm_tensors else "none"
    prepared_key = prepared_value = None
    reuse_fa_quant = (
        backend == "quant_fa"
        and k_comm_dtype == v_comm_dtype == "fp8_e4m3"
        and k_bsnd.shape[1] % kv_block_size == 0
        and v_bsnd.shape[1] % kv_block_size == 0
    )
    if reuse_fa_quant:
        prepared_key = _all_gather_quant_fa(_fa_block_quantize(k_bsnd, kv_block_size), kv_gather_group)
        prepared_value = _all_gather_quant_fa(_fa_block_quantize(v_bsnd, kv_block_size), kv_gather_group)
    else:
        k_bsnd = _all_gather_sequence(k_bsnd, kv_gather_group, k_comm_dtype, kv_block_size, k_scale)
        v_bsnd = _all_gather_sequence(v_bsnd, kv_gather_group, v_comm_dtype, kv_block_size, v_scale)

    joint_k_bsnd = _slice_joint(joint_k_bsnd, ulysses_group)
    joint_v_bsnd = _slice_joint(joint_v_bsnd, ulysses_group)
    if joint_k_bsnd is not None:
        if reuse_fa_quant:
            prepared_key = _concat_quant_fa(prepared_key, joint_k_bsnd, kv_block_size)
            prepared_value = _concat_quant_fa(prepared_value, joint_v_bsnd, kv_block_size)
        else:
            k_bsnd = torch.cat((k_bsnd, joint_k_bsnd), dim=1)
            v_bsnd = torch.cat((v_bsnd, joint_v_bsnd), dim=1)

    target_dtype = q.dtype if out_dtype is None else out_dtype
    if chunk_size is not None and chunk_size < q_bsnd.shape[1]:
        if attn_mask is not None:
            raise USPNotSupported("chunk_size with attn_mask requires a pre-sliced mask.")
        chunks = []
        lse_chunks = []
        for start in range(0, q_bsnd.shape[1], chunk_size):
            chunk_output, chunk_lse = _run_fa(
                q_bsnd[:, start : start + chunk_size],
                k_bsnd,
                v_bsnd,
                None,
                backend,
                q_block_size,
                kv_block_size,
                target_dtype,
                prepared_key,
                prepared_value,
            )
            chunks.append(chunk_output)
            if chunk_lse is not None:
                lse_chunks.append(chunk_lse)
        result_bsnd = torch.cat(chunks, dim=1)
        lse = torch.cat(lse_chunks, dim=-1) if lse_chunks else None
    else:
        result_bsnd, lse = _run_fa(
            q_bsnd,
            k_bsnd,
            v_bsnd,
            attn_mask,
            backend,
            q_block_size,
            kv_block_size,
            target_dtype,
            prepared_key,
            prepared_value,
        )

    reverse_dtype = comm_dtype if "out" in comm_tensors else "none"
    result_bsnd = _ulysses_reverse(result_bsnd, ulysses_group, reverse_dtype, q_block_size, None)
    result = _from_bsnd(result_bsnd, layout).to(target_dtype)
    result = _copy_to_out(result, out)
    if return_lse:
        if lse is None:
            raise USPNotSupported("the selected FA kernel did not return softmax LSE.")
        return result, lse
    return result


__all__ = [
    "USPError",
    "USPNotSupported",
    "USPShapeError",
    "USPTopologyError",
    "USPWorkspaceError",
    "usp_attention",
]
