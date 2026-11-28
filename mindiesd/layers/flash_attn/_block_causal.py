"""Local self-prefill from host-side block-causal geometry."""

from collections.abc import Sequence
from functools import lru_cache
import math
import warnings

import torch
import torch_npu

from ...utils.exception import ParametersInvalid
from ...utils.get_platform import is_a5_device
from ._prebuilt import PrebuiltUnavailable, packaged_execution
from .attention_forward import _attention_dynamo_disable_enabled, check_input_params
from .attention_func import attention_math


def _normalize_spans(spans, length):
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        raise ParametersInvalid("spans must be a host sequence of (start, end) pairs.")
    intervals = []
    previous = 0
    for pair in spans:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ParametersInvalid("Each span must be a (start, end) pair.")
        start, end = pair
        # Only plain host integers are valid token offsets, including for cache keys.
        # pylint: disable-next=unidiomatic-typecheck
        if type(start) is not int or type(end) is not int or not previous <= start < end <= length:
            raise ParametersInvalid("spans must be sorted, disjoint integer intervals within the sequence.")
        intervals.append((start, end))
        previous = end
    return tuple(intervals)


def _segments(spans, length):
    cursor = 0
    for start, end in spans:
        if cursor < start:
            yield cursor, start, True
        yield start, end, False
        cursor = end
    if cursor < length:
        yield cursor, length, True


def _candidate_geometry(shape, spans):
    length = shape[1]  # Normalized BSND layout.
    return length >= 28_000 and sum(1 for _ in _segments(spans, length)) >= 8


class _FlexImportError(ImportError):
    """FA4 could not be loaded before execution."""


def _run_flex(query, key, value, spans, scale):
    try:
        from ._block_causal_fa4 import run
    except (ImportError, OSError, RuntimeError) as exc:
        raise _FlexImportError(f"Could not load the FA4 backend for block_causal_attention: {exc}") from exc
    return run(query, key, value, spans, scale)


def _new_compressed_mask(device):
    with torch.inference_mode(False):
        return torch.ones((2048, 2048), dtype=torch.bool, device=device).triu_(1)


@lru_cache(maxsize=4)
def _compressed_mask(device, stream):
    return _new_compressed_mask(device)


def _piecewise(query, key, value, spans, scale, fused):
    outputs = []
    causal_mask = None
    for start, end, causal in _segments(spans, query.size(1)):
        q_part, k_part, v_part = query[:, start:end], key[:, :end], value[:, :end]
        if fused:
            if causal and causal_mask is None:
                if torch.npu.is_current_stream_capturing():
                    causal_mask = _new_compressed_mask(query.device)
                else:
                    causal_mask = _compressed_mask(query.device, torch.npu.current_stream(query.device))
            output = torch_npu.npu_fusion_attention(
                q_part.contiguous(),
                k_part.contiguous(),
                v_part.contiguous(),
                head_num=query.size(2),
                input_layout="BSND",
                atten_mask=causal_mask if causal else None,
                scale=scale,
                keep_prob=1.0,
                sparse_mode=3 if causal else 0,
            )[0]
        else:
            mask = None
            if causal:
                rows = torch.arange(start, end, device=query.device)
                mask = rows[:, None] >= torch.arange(end, device=query.device)
            output = attention_math(q_part, k_part, v_part, mask, scale)
        outputs.append(output)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


def _flex_unavailable_reason(query):
    if not is_a5_device():
        return "Flex spans require an Ascend A5 device."
    if query.dtype not in (torch.float16, torch.bfloat16) or query.shape[-1] != 128:
        return f"Flex spans require BF16 or FP16 and head dimension 128, got {query.dtype} and {query.shape[-1]}."
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.npu.is_current_stream_capturing():
        return "Flex spans require eager inference outside graph capture."
    return None


def _dispatch_npu(query, key, value, spans, scale, backend):
    forced = backend == "flex"
    # Only an explicit backend choice can enter the packaged Flex path.
    if forced or (backend == "auto" and _candidate_geometry(query.shape, spans)):
        reason = _flex_unavailable_reason(query)
        if reason is None:
            try:
                with packaged_execution(query.dtype):
                    return _run_flex(query, key, value, spans, scale)
            except (PrebuiltUnavailable, _FlexImportError) as exc:
                if forced:
                    raise
                warnings.warn(
                    "block_causal_attention(backend='auto') is falling back to piecewise, which may be slower. "
                    f"Flex is unavailable: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
        elif forced:
            raise ParametersInvalid(reason)
    return _piecewise(query, key, value, spans, scale, True)


def block_causal_attention(query, key, value, *, spans, backend="piecewise", scale=None, fused=True, head_first=False):
    """Compute unpadded self-attention with bidirectional spans in an otherwise causal sequence.

    Q, K and V share the same shape, dtype and device. The default layout is BSND.
    Set head_first=True for BNSD. The output has the same layout as Q.
    spans is a sorted sequence of disjoint [start, end) token ranges shared across
    the batch. An empty sequence gives causal attention including the diagonal.
    scale defaults to head_dim ** -0.5.

    The default piecewise backend uses segmented FA Score on NPU. With fused=False
    it uses PyTorch operations. The auto backend can select a packaged Flex kernel
    for eligible inputs. If the selected Flex backend or its packaged executor
    is unavailable, auto warns and falls back to piecewise. The flex backend
    raises in that case. Errors during kernel execution propagate to the caller.
    """
    check_input_params((query, key, value, None, scale, fused))
    if backend not in ("piecewise", "auto", "flex"):
        raise ParametersInvalid(f"Unsupported block-causal backend {backend!r}. Choose 'piecewise', 'auto' or 'flex'.")
    if not fused and backend != "piecewise":
        raise ParametersInvalid(f"backend={backend!r} requires fused=True.")
    if query.shape != key.shape or query.shape != value.shape or min(query.shape) <= 0:
        raise ParametersInvalid("spans requires equal, nonempty Q/K/V shapes for local self-prefill.")
    if any(tensor.device != query.device or tensor.dtype != query.dtype for tensor in (key, value)):
        raise ParametersInvalid("spans requires Q/K/V on the same device with the same dtype.")
    if not query.is_floating_point() or (scale is not None and not math.isfinite(scale)):
        raise ParametersInvalid("spans requires floating-point inputs and a finite scale.")
    if head_first:
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    spans = _normalize_spans(spans, query.size(1))
    scale = query.size(-1) ** -0.5 if scale is None else scale
    if fused:
        if query.device.type != "npu":
            raise ParametersInvalid("Fused block-causal attention requires NPU tensors.")
        with torch.npu.device(query.device):
            output = _dispatch_npu(query, key, value, spans, scale, backend)
    else:
        output = _piecewise(query, key, value, spans, scale, False)
    return output.transpose(1, 2) if head_first else output


if _attention_dynamo_disable_enabled():
    block_causal_attention = torch._dynamo.disable(block_causal_attention)
