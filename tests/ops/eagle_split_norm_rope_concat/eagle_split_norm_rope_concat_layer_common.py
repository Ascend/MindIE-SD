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

"""Production paths for the 910B packed-QKV Slice/RmsNorm/RoPE/Concat chain.

No device initialization or business-package imports at module load time.
"""

from eagle_split_norm_rope_concat_cases import PROD_HEADS, PROD_SHAPE, case_attrs, is_compact, output_shapes

PATHS = ("composed",)
PROTOCOL = "eagle_snrc_layer_event_v6"


def scenario_name(path, case_id):
    attrs = case_attrs(case_id)
    rope = "HALF" if attrs["rope_type"] == 1 else "INTERLEAVE"
    suffix = "+VSlice"
    layout = "+TransposeContiguous(BNSD)" if attrs["output_bnsd"] else "(BSND)"
    boundary = "QKV-out"
    rotary = attrs["rotary_dim"]
    cache = f"+ExpandCache{rotary // 2}to{rotary}" if is_compact(case_id) else f"+Cache{rotary}"
    suffix_compute = "+Slice32+Cat" if rotary == 96 else ""
    prefix_slice = "+Slice96" if rotary == 96 else ""
    return f"Slice{cache}+RmsNorm(BF16){prefix_slice}+{rope}-RoPE(BF16){suffix_compute}{suffix}{layout}({boundary})"


def input_config(case_id):
    rotary = case_attrs(case_id)["rotary_dim"]
    width = rotary // 2 if is_compact(case_id) else rotary
    return {
        "case_id": case_id,
        "qkv_shape": list(PROD_SHAPE),
        "qkv_dtype": "bfloat16",
        "head_counts": list(PROD_HEADS),
        "head_counts_type": "int[]",
        "weight_shapes": [[128], [128]],
        "weight_dtype": "bfloat16",
        "rope_shapes": [[1, 5672, 1, width], [1, 5672, 1, width]],
        "rope_dtype": "bfloat16",
        "layout": "BSND",
        **case_attrs(case_id),
        "output_shape": list(output_shapes(case_id)[0]),
        "output_dtype": "bfloat16",
        "compute_dtype": "native_operator_internal",
        "intermediate_dtype": "bfloat16",
        "norm_output_rounding": True,
        "rms_op": "torch_npu.npu_rms_norm",
        "rope_op": "torch_npu.npu_rotary_mul",
        "cache_policy": "reuse_same_inputs_hot_cache",
        "jit_compile": False,
    }


def tensor_state(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
    }


def run_path(path, inputs, case_id, trace=None):
    import torch
    import torch_npu

    packed, qw, kw, cos, sin, heads = inputs
    if path not in PATHS:
        raise ValueError(path)
    attrs = case_attrs(case_id)
    rotary = attrs["rotary_dim"]
    rotary_mode = "half" if attrs["rope_type"] == 1 else "interleave"
    cos_full, sin_full = cos, sin
    if is_compact(case_id):
        if attrs["rope_type"] == 1:
            cos_full = torch.cat((cos, cos), dim=-1)
            sin_full = torch.cat((sin, sin), dim=-1)
        else:
            cos_full = cos.repeat_interleave(2, dim=-1)
            sin_full = sin.repeat_interleave(2, dim=-1)
    results = []
    # Process Q then K; npu_slice materializes each slice.
    # Offsets and sizes specify the batch, token, head and feature dimensions.
    for name, offset, weight in (("q", 0, qw), ("k", 56, kw)):
        sliced = torch_npu.npu_slice(packed, [0, 0, offset, 0], [1, 5672, 56, 128])
        norm, rstd = torch_npu.npu_rms_norm(sliced, weight, epsilon=attrs["eps"])
        rope_input = torch_npu.npu_slice(norm, [0, 0, 0, 0], [1, 5672, 56, rotary]) if rotary < 128 else norm
        rotated = torch_npu.npu_rotary_mul(rope_input, cos_full, sin_full, rotary_mode)
        tail = None
        if rotary < 128:
            tail = torch_npu.npu_slice(norm, [0, 0, 0, rotary], [1, 5672, 56, 128 - rotary])
        output = torch.cat((rotated, tail), dim=-1) if tail is not None else rotated
        if trace is not None:
            trace.append(
                {
                    "branch": name,
                    "slice_offsets": [0, 0, offset, 0],
                    "slice_sizes": [1, 5672, 56, 128],
                    "slice_metadata_dtype": "int64",
                    "rms_op": "torch_npu.npu_rms_norm",
                    "rms_input": tensor_state(sliced),
                    "weight_input": tensor_state(weight),
                    "rms_output": tensor_state(norm),
                    "rstd": tensor_state(rstd),
                    "rope_input": tensor_state(rope_input),
                    "rope_op": "torch_npu.npu_rotary_mul",
                    "cos": tensor_state(cos_full),
                    "sin": tensor_state(sin_full),
                    "rotary_mode": rotary_mode,
                    "tail": tensor_state(tail) if tail is not None else None,
                    "concat_output": tensor_state(output),
                }
            )
        if attrs["output_bnsd"]:
            output = output.transpose(1, 2).contiguous()
        if trace is not None:
            trace[-1]["output"] = tensor_state(output)
        results.append(output)
    if path == "composed":
        # Materialize V to match the fused operator's three-output contract.
        value = torch_npu.npu_slice(packed, [0, 0, 112, 0], [1, 5672, 56, 128])
        if attrs["output_bnsd"]:
            value = value.transpose(1, 2).contiguous()
        if trace is not None:
            trace.append(
                {
                    "branch": "v",
                    "operation": "slice+transpose_contiguous" if attrs["output_bnsd"] else "slice",
                    "output": tensor_state(value),
                }
            )
        results.append(value)
    return tuple(results)
