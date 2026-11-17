# SPDX-License-Identifier: Apache-2.0
"""Stable keyword facades for the existing fused and indexed SDK operators.

All validation, arithmetic, allocation and execution remain with the original
implementations. These wrappers only translate public keyword names.
"""

import torch

_QKV_KEYWORDS = {
    "q_weight": "norm_query_weight",
    "k_weight": "norm_key_weight",
    "split_sizes": "head_counts",
    "rope_mode": "rope_type",
    "epsilon": "eps",
}
_MODULATION_KEYWORDS = {"epsilon": "eps", "out_dtype": "output_dtype"}
_SWIGLU_KEYWORDS = {"x": "x1", "weight": "x2", "transpose_x": "transpose_x1", "transpose_weight": "transpose_x2"}


def _call_native(name, arguments, aliases=None):
    """Load only the requested native symbol and rename its bound keywords.

    Explicit imports retain the original missing-symbol ImportError behavior.
    Iterating the bound argument dictionary preserves keyword order and objects.
    """
    if name == "eagle_split_norm_rope_concat":
        from .layers._custom_ops import eagle_split_norm_rope_concat as native
    elif name == "eagle_indexed_gate_norm_modulate":
        from .layers._custom_ops import eagle_indexed_gate_norm_modulate as native
    elif name == "eagle_quant_batch_matmul_swiglu":
        from .layers._custom_ops import eagle_quant_batch_matmul_swiglu as native
    elif name == "rms_norm_indexed_scale_shift":
        from .layers.index_modulation import rms_norm_indexed_scale_shift as native
    else:
        from .layers.index_modulation import indexed_gate as native
    aliases = aliases or {}
    return native(**{aliases.get(key, key): value for key, value in arguments.items()})


def eagle_split_norm_rope_concat(
    qkv,
    q_weight,
    k_weight,
    rope_cos,
    rope_sin,
    split_sizes,
    norm_type,
    rope_mode,
    rotary_dim=None,
    epsilon=1e-5,
    output_bnsd=True,
):
    """Forward packed QKV norm/RoPE with the native output ownership contract."""
    return _call_native("eagle_split_norm_rope_concat", locals(), _QKV_KEYWORDS)


def eagle_indexed_gate_norm_modulate(
    residual,
    gate,
    branch,
    shift,
    scale,
    indices,
    weight=None,
    bias=None,
    norm_type=0,
    epsilon=1e-6,
    out_dtype=None,
):
    """Forward the existing fused gate, RMSNorm and indexed modulation operator."""
    return _call_native("eagle_indexed_gate_norm_modulate", locals(), _MODULATION_KEYWORDS)


def eagle_quant_batch_matmul_swiglu(
    x,
    weight,
    scale,
    offset=None,
    bias=None,
    pertoken_scale=None,
    output_dtype=torch.bfloat16,
    transpose_x=False,
    transpose_weight=False,
    group_size=0,
):
    """Forward the existing INT8 QBMM/SwiGLU without extra casts or arithmetic."""
    return _call_native("eagle_quant_batch_matmul_swiglu", locals(), _SWIGLU_KEYWORDS)


def rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps):
    """Forward the original indexed RMSNorm, including its final BF16 store."""
    return _call_native("rms_norm_indexed_scale_shift", locals())


def indexed_gate(output, x, gate, other, indices):
    """Forward the original caller-owned output operator and alias checks."""
    return _call_native("indexed_gate", locals())


__all__ = [
    "eagle_split_norm_rope_concat",
    "eagle_indexed_gate_norm_modulate",
    "eagle_quant_batch_matmul_swiglu",
    "rms_norm_indexed_scale_shift",
    "indexed_gate",
]
