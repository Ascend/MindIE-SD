#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# pylint: disable=too-many-lines,duplicate-code

from abc import ABC, abstractmethod
import math
import torch
import torch.nn.functional as F
import torch_npu
from torch import nn

from ..layers.flash_attn.common import AttentionParam, lru_cache_by_attn_param
from .config import QuantConfig, TimestepPolicyConfig
from .utils import get_mxfp4_quant_kwargs, get_quant_weight, TimestepManager


MXFP4_Q_QUANT_MODE = 3
MXFP4_K_QUANT_MODE = 3
MXFP4_V_QUANT_MODE = 3
MXFP4_FA_SEQ_PAD_BASE = 512
MXFP4_FA_SEQ_CACHE_MAX_SIZE = 512
MXFP4_GROUP_SIZES_W4A4 = [1, 1, 32]
MXFP4_GROUP_SIZES_W4A8 = [0, 0, 32]
MXFP4_SCALE_ALG_C7 = 2
MXFP4_DST_TYPE_MAX_C7 = 7.25


def _prepare_mxfp4_weight(weight, use_nz=False):
    fp8_dtype = getattr(torch, 'float8_e4m3fn', None)
    if fp8_dtype is not None and weight.dtype == fp8_dtype:
        weight = torch_npu.npu_dtype_cast(weight.npu(), torch_npu.float4_e2m1fn_x2)
    elif weight.dtype != torch.uint8:
        raise TypeError(f"W4A4 MXFP4 weight must be torch.float8_e4m3fn or torch.uint8, but got {weight.dtype}.")
    if not use_nz:
        return weight
    return torch_npu.npu_format_cast(weight.view(torch.int8).npu(), 29, customize_dtype=torch.int8)


def _get_quant_config(kwargs):
    quant_config = kwargs.get('quant_config', None)
    if quant_config is None:
        quant_config = QuantConfig.from_kwargs(kwargs)
    return quant_config


def _has_quant_weight(weights, key):
    return weights is not None and key in weights.keys()


def _dynamic_mx_quant(input_tensor, dst_type, quant_config=None, **kwargs):
    quant_kwargs = {'dst_type': dst_type}
    if dst_type == torch_npu.float4_e2m1fn_x2:
        quant_kwargs.update(get_mxfp4_quant_kwargs(quant_config))
    quant_kwargs.update(kwargs)

    try:
        result = torch_npu.npu_dynamic_mx_quant(input_tensor, **quant_kwargs)
    except TypeError as exc:
        if any(name in quant_kwargs for name in ('axis', 'scale_alg', 'dst_type_max')):
            raise RuntimeError(
                "npu_dynamic_mx_quant must support axis, scale_alg and dst_type_max for MXFP4 quantization."
            ) from exc
        try:
            result = torch_npu.npu_dynamic_mx_quant(input_tensor, dst_type=dst_type)
        except TypeError:
            result = torch_npu.npu_dynamic_mx_quant(input_tensor)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("npu_dynamic_mx_quant must return at least quantized tensor and scale.")
    return result[0], result[1]


def _dynamic_mx_quant_fa(input_tensor, axis, quant_config=None):
    return _dynamic_mx_quant(
        input_tensor,
        dst_type=torch_npu.float4_e2m1fn_x2,
        quant_config=quant_config,
        axis=axis,
    )


def _get_fa_shape(query, layout):
    if layout == "BNSD":
        _, n, s, d = query.shape
    elif layout == "BSND":
        _, s, n, d = query.shape
    else:
        raise ValueError(f"Unsupported layout: {layout}, expected 'BNSD' or 'BSND'.")
    return n, s, d


def _pad_fa_seq_before_quant(input_tensor, base, layout):
    if layout == "BNSD":
        _, _, s, _ = input_tensor.shape
        padding_length = (base - s % base) % base
        pad = (0, 0, 0, padding_length)
    elif layout == "BSND":
        _, s, _, _ = input_tensor.shape
        padding_length = (base - s % base) % base
        pad = (0, 0, 0, 0, 0, padding_length)
    else:
        raise ValueError(f"Unsupported layout: {layout}, expected 'BNSD' or 'BSND'.")

    if padding_length != 0:
        input_tensor = F.pad(input_tensor, pad)
    return input_tensor, s, s + padding_length


def _get_fa_seq_axis(layout):
    if layout == "BNSD":
        return 2
    if layout == "BSND":
        return 1
    raise ValueError(f"Unsupported layout: {layout}, expected 'BNSD' or 'BSND'.")


def _reshape_mxfp4_v_scale_for_fa(v_scale, layout):
    if layout == "BNSD":
        scale_blocks = v_scale.shape[2]
        if v_scale.dim() == 5:
            return v_scale
        if scale_blocks % 2 != 0:
            raise ValueError(f"V scale S blocks must be even for layout BNSD, got {scale_blocks}.")
        return (
            v_scale.reshape(v_scale.shape[0], v_scale.shape[1], scale_blocks // 2, 2, v_scale.shape[3])
            .transpose(-1, -2)
            .contiguous()
        )
    if layout == "BSND":
        scale_blocks = v_scale.shape[1]
        if v_scale.dim() == 5:
            return v_scale
        if scale_blocks % 2 != 0:
            raise ValueError(f"V scale S blocks must be even for layout BSND, got {scale_blocks}.")
        return (
            v_scale.reshape(v_scale.shape[0], scale_blocks // 2, 2, v_scale.shape[2], v_scale.shape[3])
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
    raise ValueError(f"Unsupported layout: {layout}, expected 'BNSD' or 'BSND'.")


@lru_cache_by_attn_param(maxsize=MXFP4_FA_SEQ_CACHE_MAX_SIZE)
def _get_qfa_seqused(param):
    device = torch.device(param.head_first)
    seqused_q = torch.full((param.batch_size,), param.q_seqlen, dtype=torch.int32, device=device)
    seqused_kv = torch.full((param.batch_size,), param.kv_seqlen, dtype=torch.int32, device=device)
    return seqused_q, seqused_kv


def _crop_fa_output(output, seq_len, layout):
    if layout == "BNSD":
        if output.shape[2] != seq_len:
            output = output[:, :, :seq_len, :]
    elif layout == "BSND":
        if output.shape[1] != seq_len:
            output = output[:, :seq_len, :, :]
    return output


class WeightQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__()
        # 根据入参作为可选属性
        self.prefix = prefix
        self.dtype = kwargs.get('dtype', torch.bfloat16)
        self.input_feature = in_features
        self.output_feature = out_features

        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale').T.to(self.dtype)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

        weight = get_quant_weight(weights, f'{prefix}.weight')
        if kwargs.get('use_nz', False):
            weight = torch_npu.npu_format_cast(weight, 29).T
            if kwargs.get('is_w4', False):
                weight = torch_npu.npu_convert_weight_to_int4pack(weight.npu().to(torch.int32))
        else:
            weight = weight.T
            if kwargs.get('is_w4', False):
                weight = torch_npu.npu_convert_weight_to_int4pack(weight.npu().to(torch.int32))
        self.register_buffer("weight", weight, persistent=False)

        if bias:
            bias = get_quant_weight(weights, f'{prefix}.bias')
            if self.dtype == torch.bfloat16:
                bias = bias.to(torch.float32)
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        output = torch_npu.npu_weight_quant_batchmatmul(x, self.weight, self.weight_scale, bias=self.bias)
        return output

    def forward(self, x):
        # dynamic场景算子虽然也支持3维，但性能会劣化，这里展平做运算
        if x.ndim >= 3:
            return self._flatten_linear(x)
        output = self.quant_matmul(x)
        return output

    def _flatten_linear(self, x):
        x_reshpe = x.reshape(x.shape[:-1].numel(), -1)
        output = self.quant_matmul(x_reshpe)
        new_size = list(x.shape)[:-1]
        new_size.append(output.shape[1])
        return output.view(*new_size)


class W8A8QuantBaseLinear(ABC, nn.Module):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__()
        # 根据入参作为可选属性
        self.dtype = kwargs.get('dtype', torch.bfloat16)
        self.input_feature = in_features
        self.output_feature = out_features

        if bias:
            bias = get_quant_weight(weights, f'{prefix}.bias')
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None
        mul_scale = kwargs.get('mul_scale', None)
        if mul_scale is not None:
            mul_scale = mul_scale.to(self.dtype)
            self.register_buffer("mul_scale", mul_scale, persistent=False)
        else:
            self.mul_scale = None

    def pack_weight(self, weight, **kwargs):
        return weight

    @abstractmethod
    def quant_matmul(self, x):
        pass

    def forward(self, x):
        # dynamic场景算子虽然也支持3维，但性能会劣化，这里展平做运算
        if x.ndim >= 3 or (x.ndim == 3 and self.is_dynamic):
            return self._flatten_linear(x)
        output = self.quant_matmul(x)
        return output

    def _flatten_linear(self, x):
        x_reshpe = x.reshape(x.shape[:-1].numel(), -1)
        output = self.quant_matmul(x_reshpe)
        new_size = list(x.shape)[:-1]
        new_size.append(output.shape[1])
        return output.view(*new_size)

    def _init_static_quant_param(self, prefix=None, weights=None, **kwargs):
        quant_bias = get_quant_weight(weights, f'{prefix}.quant_bias')
        self.register_buffer("quant_bias", quant_bias, persistent=False)
        deq_scale = get_quant_weight(weights, f'{prefix}.deq_scale')
        self.register_buffer("deq_scale", deq_scale, persistent=False)
        weight = get_quant_weight(weights, f'{prefix}.weight')
        self.register_buffer("weight", weight, persistent=False)
        if self.dtype == torch.float16:
            input_scale = get_quant_weight(weights, f'{prefix}.input_scale').to(torch.float32)
        else:
            input_scale = get_quant_weight(weights, f'{prefix}.input_scale').to(self.dtype)
        if input_scale.dim() == 1:
            input_scale = input_scale.repeat(weight.data.shape[1])
        else:
            input_scale = input_scale.repeat(1, weight.data.shape[1])

        self.register_buffer("input_scale", input_scale, persistent=False)

        if self.dtype == torch.bfloat16:
            input_offset = get_quant_weight(weights, f'{prefix}.input_offset').to(self.dtype)
        else:
            input_offset = get_quant_weight(weights, f'{prefix}.input_offset').to(torch.int8)
        if input_offset.dim() == 1:
            input_offset = input_offset.repeat(weight.data.shape[1])
        else:
            input_offset = input_offset.repeat(1, weight.data.shape[1])

        self.register_buffer("input_offset", input_offset, persistent=False)

    def _init_dynamic_quant_param(self, prefix=None, weights=None, **kwargs):
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale').squeeze().to(self.dtype)
        if self.dtype == torch.float16:
            weight_scale = weight_scale.to(torch.float32)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        if self.bias is not None:
            self.bias = self.bias.to(self.dtype)
        weight = get_quant_weight(weights, f'{prefix}.weight')
        weight = self.pack_weight(weight, **kwargs)
        if kwargs.get('use_nz', False):
            weight = torch_npu.npu_format_cast(weight.npu(), 29).T
        else:
            weight = weight.T
        self.register_buffer("weight", weight, persistent=False)


class W8A8QuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.is_dynamic = kwargs.get('is_dynamic', False)

        if not self.is_dynamic:
            self._init_static_quant_param(prefix, weights, **kwargs)
        else:
            self._init_dynamic_quant_param(prefix, weights, **kwargs)

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if not self.is_dynamic:
            if self.mul_scale is not None:
                x_scaled = x * self.mul_scale
                x_int8 = torch_npu.npu_quantize(
                    x_scaled, scales=self.input_scale, zero_points=self.input_offset, dtype=torch.qint8, axis=-1
                )
            else:
                x_int8 = torch_npu.npu_quantize(
                    x, scales=self.input_scale, zero_points=self.input_offset, dtype=torch.qint8, axis=-1
                )

            output = torch_npu.npu_quant_matmul(
                x_int8, self.weight.T, self.deq_scale, bias=self.quant_bias, output_dtype=self.dtype
            )
        else:
            if self.mul_scale is not None:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x * self.mul_scale)
            else:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x)

            output = torch_npu.npu_quant_matmul(
                x_int8,
                self.weight,
                self.weight_scale,
                pertoken_scale=input_scale,
                output_dtype=self.dtype,
                bias=self.bias,
            )
        return output


class W4A4QuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)
        self.is_dynamic = True
        self._init_dynamic_quant_param(prefix, weights, **kwargs)

    def pack_weight(self, weight, **kwargs):
        weight.data = torch_npu.npu_convert_weight_to_int4pack(weight.data.to(torch.int32).npu())
        return weight

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
        pertoken_scale = pertoken_scale.reshape(-1, 1)
        pertoken_scale = pertoken_scale.squeeze(-1)
        output = torch_npu.npu_quant_matmul(
            x,
            self.weight,
            self.weight_scale.data.view(-1),
            pertoken_scale=pertoken_scale,
            bias=None,
            output_dtype=self.dtype,
        )
        return output


class W8A8TimeStepQuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.timestep_config = _get_quant_config(kwargs).timestep_config or TimestepPolicyConfig()

        self.is_dynamic = True

        self._init_dynamic_quant_param(prefix, weights, **kwargs)
        # 最后使用的是n k的权重
        self._init_static_quant_param(prefix, weights, **kwargs)

        TimestepManager.set_timestep_idx_max(self.input_scale.shape[0])

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        # 判断时间步状态
        t_idx = TimestepManager.get_timestep_idx()
        strategy = self.timestep_config.get_strategy(t_idx, target="w8a8_static_linear")

        if strategy == "static":
            self.is_dynamic = False
        else:
            self.is_dynamic = True

        if not self.is_dynamic:
            if self.mul_scale is not None:
                x_scaled = x * self.mul_scale
                x_int8 = torch_npu.npu_quantize(
                    x_scaled,
                    scales=self.input_scale[t_idx],
                    zero_points=self.input_offset[t_idx],
                    dtype=torch.qint8,
                    axis=-1,
                )
            else:
                x_int8 = torch_npu.npu_quantize(
                    x, scales=self.input_scale[t_idx], zero_points=self.input_offset[t_idx], dtype=torch.qint8, axis=-1
                )

            output = torch_npu.npu_quant_matmul(
                x_int8, self.weight.T, self.deq_scale[t_idx], bias=self.quant_bias[t_idx], output_dtype=self.dtype
            )
        else:
            if self.mul_scale is not None:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x * self.mul_scale)
            else:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x)

            output = torch_npu.npu_quant_matmul(
                x_int8,
                self.weight.T,
                self.weight_scale,
                pertoken_scale=input_scale,
                output_dtype=self.dtype,
                bias=self.bias,
            )
        return output


class FP8RotateQuantFA(nn.Module):
    def __init__(self, prefix=None, weights=None):
        super().__init__()

        q_rot = get_quant_weight(weights, f'{prefix}.q_rot')
        self.register_buffer("q_rot", q_rot, persistent=False)
        k_rot = get_quant_weight(weights, f'{prefix}.k_rot')
        self.register_buffer("k_rot", k_rot, persistent=False)

    def forward(self, query, key, value, **kwargs):
        query = torch.matmul(query, self.q_rot)
        key = torch.matmul(key, self.k_rot)

        layout = kwargs.get("layout", "BNSD")
        n, s, d = _get_fa_shape(query, layout)

        from ..layers.quant.block_quant import fa_block_quant_preprocess

        q, q_scale = fa_block_quant_preprocess(query, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout=layout)
        k, k_scale = fa_block_quant_preprocess(key, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)
        v, v_scale = fa_block_quant_preprocess(value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)

        x = torch_npu.npu_fused_infer_attention_score_v2(
            q,
            k,
            v,
            input_layout="BNSD",
            num_query_heads=n,
            softmax_scale=1.0 / math.sqrt(d),
            pre_tokens=2147483647,
            next_tokens=2147483647,
            query_quant_mode=7,
            key_quant_mode=7,
            value_quant_mode=7,
            dequant_scale_query=q_scale,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            out_dtype=query.dtype,
        )[0]

        x = _crop_fa_output(x, s, "BNSD")
        if layout == "BSND":
            x = x.transpose(1, 2)

        return x


class MXFP8RotateQuantFA(nn.Module):
    def __init__(self, prefix=None, weights=None):
        super().__init__()

        q_rot = get_quant_weight(weights, f'{prefix}.q_rot')
        self.register_buffer("q_rot", q_rot, persistent=False)
        k_rot = get_quant_weight(weights, f'{prefix}.k_rot')
        self.register_buffer("k_rot", k_rot, persistent=False)

    def forward(self, query, key, value, **kwargs):
        query = torch.matmul(query, self.q_rot)
        key = torch.matmul(key, self.k_rot)

        layout = kwargs.get("layout", "BNSD")
        if layout == "BNSD":
            b, n, s, d = query.shape
            query = query.permute(0, 2, 1, 3).reshape(b * s, n, d)
            key = key.permute(0, 2, 1, 3).reshape(b * s, n, d)
            value = value.permute(0, 2, 1, 3).reshape(b * s, n, d)
        elif layout == "BSND":
            b, s, n, d = query.shape
            query = query.reshape(b * s, n, d)
            key = key.reshape(b * s, n, d)
            value = value.reshape(b * s, n, d)
        else:
            raise ValueError(f"Unsupported layout: {layout}, expected 'BNSD' or 'BSND'.")

        actual_seq_qlen = torch.arange(s, s * (b + 1), s, dtype=torch.int64, device=query.device)
        actual_seq_kvlen = torch.arange(s, s * (b + 1), s, dtype=torch.int64, device=key.device)

        q, q_scale = torch_npu.npu_dynamic_mx_quant(query, dst_type=torch.float8_e4m3fn, axis=-1)
        k, k_scale = torch_npu.npu_dynamic_mx_quant(key, dst_type=torch.float8_e4m3fn, axis=-1)
        v, v_scale = torch_npu.npu_dynamic_mx_quant(value, dst_type=torch.float8_e4m3fn, axis=0)

        x = torch_npu.npu_fused_infer_attention_score_v2(
            q,
            k,
            v,
            input_layout="TND",
            num_query_heads=n,
            num_key_value_heads=n,
            softmax_scale=1.0 / math.sqrt(d),
            dequant_scale_query=q_scale,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            sparse_mode=0,  # could be 0/3, atten_mask is needed if set 3
            query_quant_mode=6,
            key_quant_mode=6,
            value_quant_mode=8,
            query_dtype=torch.float8_e4m3fn,
            key_dtype=torch.float8_e4m3fn,
            value_dtype=torch.float8_e4m3fn,
            dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
            dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
            dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
            out_dtype=query.dtype,
        )[0]

        if layout == "BNSD":
            # [B*S, N, D] -> [B, S, N, D] -> [B, N, S, D]
            x = x.reshape(b, s, n, d).permute(0, 2, 1, 3)
        elif layout == "BSND":
            # [B*S, N, D] -> [B, S, N, D]
            x = x.reshape(b, s, n, d)

        return x


class MXFP4QuantFA(nn.Module):
    def __init__(self, prefix=None, weights=None, **kwargs):
        super().__init__()
        self.prefix = prefix
        self.quant_config = _get_quant_config(kwargs)
        self.timestep_config = self.quant_config.timestep_config or TimestepPolicyConfig()

        if _has_quant_weight(weights, f'{prefix}.q_rot'):
            self.register_buffer("q_rot", get_quant_weight(weights, f'{prefix}.q_rot'), persistent=False)
        else:
            self.q_rot = None
        if _has_quant_weight(weights, f'{prefix}.k_rot'):
            self.register_buffer("k_rot", get_quant_weight(weights, f'{prefix}.k_rot'), persistent=False)
        else:
            self.k_rot = None

    def _apply_rotate(self, query, key):
        if self.q_rot is not None:
            query = torch.matmul(query, self.q_rot)
        if self.k_rot is not None:
            key = torch.matmul(key, self.k_rot)
        return query, key

    def _forward_float(self, query, key, value, **kwargs):
        layout = kwargs.get("layout", "BNSD")
        n, s, d = _get_fa_shape(query, layout)
        output = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key,
            value,
            input_layout=layout,
            num_query_heads=n,
            softmax_scale=kwargs.get("softmax_scale", 1.0 / math.sqrt(d)),
            pre_tokens=kwargs.get("pre_tokens", 2147483647),
            next_tokens=kwargs.get("next_tokens", 2147483647),
            out_dtype=query.dtype,
        )[0]
        return _crop_fa_output(output, s, layout)

    def _forward_fp8(self, query, key, value, **kwargs):
        query, key = self._apply_rotate(query, key)
        layout = kwargs.get("layout", "BNSD")
        n, s, d = _get_fa_shape(query, layout)

        from ..layers.quant.block_quant import fa_block_quant_preprocess

        q, q_scale = fa_block_quant_preprocess(query, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout=layout)
        k, k_scale = fa_block_quant_preprocess(key, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)
        v, v_scale = fa_block_quant_preprocess(value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)

        output = torch_npu.npu_fused_infer_attention_score_v2(
            q,
            k,
            v,
            input_layout="BNSD",
            num_query_heads=n,
            softmax_scale=kwargs.get("softmax_scale", 1.0 / math.sqrt(d)),
            pre_tokens=kwargs.get("pre_tokens", 2147483647),
            next_tokens=kwargs.get("next_tokens", 2147483647),
            query_quant_mode=7,
            key_quant_mode=7,
            value_quant_mode=7,
            dequant_scale_query=q_scale,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            out_dtype=query.dtype,
        )[0]
        output = _crop_fa_output(output, s, "BNSD")
        if layout == "BSND":
            output = output.transpose(1, 2)
        return output

    def _forward_mxfp4(self, query, key, value, **kwargs):
        query, key = self._apply_rotate(query, key)
        layout = kwargs.get("layout", "BNSD")
        layout_kv = kwargs.get("layout_kv", layout)
        layout_out = kwargs.get("layout_out", layout)
        n, s, d = _get_fa_shape(query, layout)
        n_kv, kv_s, _ = _get_fa_shape(key, layout_kv)

        query, s, padded_s = _pad_fa_seq_before_quant(query, MXFP4_FA_SEQ_PAD_BASE, layout)
        key, kv_s, padded_kv_s = _pad_fa_seq_before_quant(key, MXFP4_FA_SEQ_PAD_BASE, layout_kv)
        value, _, _ = _pad_fa_seq_before_quant(value, MXFP4_FA_SEQ_PAD_BASE, layout_kv)
        batch_size = query.shape[0]
        seq_param = AttentionParam(batch_size, n, d, padded_s, padded_kv_s, torch.int32, str(query.device))
        seqused_q, seqused_kv = _get_qfa_seqused(seq_param)

        v_seq_axis = _get_fa_seq_axis(layout_kv)
        q, q_scale = _dynamic_mx_quant_fa(query, axis=-1, quant_config=self.quant_config)
        k, k_scale = _dynamic_mx_quant_fa(key, axis=-1, quant_config=self.quant_config)
        v, v_scale = _dynamic_mx_quant_fa(value, axis=v_seq_axis, quant_config=self.quant_config)
        v_scale = _reshape_mxfp4_v_scale_for_fa(v_scale, layout_kv)

        qfa_metadata = kwargs.get("metadata", None)
        if qfa_metadata is None:
            qfa_metadata = torch.ops.mindiesd.quant_flash_attn_metadata(
                num_heads_q=n,
                num_heads_kv=kwargs.get("num_key_value_heads", n_kv),
                head_dim=d,
                q_quant_mode=MXFP4_Q_QUANT_MODE,
                k_quant_mode=MXFP4_K_QUANT_MODE,
                v_quant_mode=MXFP4_V_QUANT_MODE,
                cu_seqlens_q=None,
                cu_seqlens_kv=None,
                seqused_q=seqused_q,
                seqused_kv=seqused_kv,
                batch_size=query.shape[0],
                max_seqlen_q=kwargs.get("max_seqlen_q", -1),
                max_seqlen_kv=kwargs.get("max_seqlen_kv", -1),
                q_dtype=torch_npu.float4_e2m1fn_x2,
                k_dtype=torch_npu.float4_e2m1fn_x2,
                v_dtype=torch_npu.float4_e2m1fn_x2,
                mask_mode=kwargs.get("mask_mode", 0),
                win_left=kwargs.get("win_left", kwargs.get("pre_tokens", 2147483647)),
                win_right=kwargs.get("win_right", kwargs.get("next_tokens", 2147483647)),
                layout_q=layout,
                layout_kv=layout_kv,
                layout_out=layout_out,
            )

        output, _ = torch.ops.mindiesd.quant_flash_attn(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            q_quant_mode=MXFP4_Q_QUANT_MODE,
            k_quant_mode=MXFP4_K_QUANT_MODE,
            v_quant_mode=MXFP4_V_QUANT_MODE,
            block_table=kwargs.get("block_table", None),
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=kwargs.get("sinks", None),
            attn_mask=kwargs.get("attn_mask", None),
            metadata=qfa_metadata,
            q_dtype=torch_npu.float4_e2m1fn_x2,
            k_dtype=torch_npu.float4_e2m1fn_x2,
            v_dtype=torch_npu.float4_e2m1fn_x2,
            q_descale_dtype=torch_npu.float8_e8m0fnu,
            k_descale_dtype=torch_npu.float8_e8m0fnu,
            v_descale_dtype=torch_npu.float8_e8m0fnu,
            softmax_scale=kwargs.get("softmax_scale", 1.0 / math.sqrt(d)),
            mask_mode=kwargs.get("mask_mode", 0),
            win_left=kwargs.get("win_left", kwargs.get("pre_tokens", 2147483647)),
            win_right=kwargs.get("win_right", kwargs.get("next_tokens", 2147483647)),
            max_seqlen_q=kwargs.get("max_seqlen_q", -1),
            max_seqlen_kv=kwargs.get("max_seqlen_kv", -1),
            layout_q=layout,
            layout_kv=layout_kv,
            layout_out=layout_out,
            return_softmax_lse=kwargs.get("return_softmax_lse", 0),
        )
        return _crop_fa_output(output, s, layout_out)

    def forward(self, query, key, value, **kwargs):
        t_idx = TimestepManager.get_timestep_idx()
        strategy = self.timestep_config.get_strategy(t_idx, target="fa")
        if strategy == "FLOAT":
            return self._forward_float(query, key, value, **kwargs)
        if strategy == "FP8":
            return self._forward_fp8(query, key, value, **kwargs)
        return self._forward_mxfp4(query, key, value, **kwargs)


class W8A8MXFP8QuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.is_dynamic = True
        self._init_dynamic_quant_param(prefix, weights, **kwargs)

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if self.mul_scale is not None:
            x1, input_scale = torch_npu.npu_dynamic_mx_quant(x * self.mul_scale, dst_type=torch_npu.float8_e4m3fn)
        else:
            x1, input_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)

        if self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32)

        x2 = self.weight
        if x2.dtype != torch.float8_e4m3fn:
            x2 = torch_npu.npu_dtype_cast(x2, torch_npu.float8_e4m3fn)
        x2 = x2.transpose(0, 1)

        output = torch_npu.npu_quant_matmul(
            x1,
            x2,
            self.weight_scale.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=self.bias,
            output_dtype=self.dtype,
            group_sizes=[1, 1, 32],
        )
        return output

    def _init_dynamic_quant_param(self, prefix=None, weights=None, **kwargs):
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale')
        if weight_scale.shape[1] % 2 != 0:
            weight_scale = F.pad(weight_scale, pad=(0, 1))
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

        weight = get_quant_weight(weights, f'{prefix}.weight')
        if kwargs.get('use_nz', False):
            weight = torch_npu.npu_format_cast(weight.npu(), 29)
        self.register_buffer("weight", weight, persistent=False)


class W4A4MXFP4DualQuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.is_dynamic = kwargs.get('is_dynamic', True)
        self._init_dynamic_quant_param(prefix, weights, **kwargs)

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        x1, l0_scale, l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=self.mul_scale)
        if self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32)

        output = torch_npu.npu_dual_level_quant_matmul(
            x1,
            self.weight,
            l0_scale,
            self.weight_dual_scale,
            l1_scale,
            self.weight_scale,
            bias=self.bias,
            output_dtype=self.dtype,
        )
        return output

    def _init_dynamic_quant_param(self, prefix=None, weights=None, **kwargs):
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale')
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

        weight_dual_scale = get_quant_weight(weights, f'{prefix}.weight_dual_scale')
        weight_dual_scale = weight_dual_scale.squeeze(-1).transpose(0, 1).contiguous()
        self.register_buffer("weight_dual_scale", weight_dual_scale, persistent=False)

        weight = get_quant_weight(weights, f'{prefix}.weight')
        weight = _prepare_mxfp4_weight(weight, kwargs.get('use_nz', False))
        self.register_buffer("weight", weight, persistent=False)


class W4A4MXFP4QuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.is_dynamic = True
        self.prefix = f'{prefix}'
        self.quant_config = _get_quant_config(kwargs)
        self.timestep_config = self.quant_config.timestep_config or TimestepPolicyConfig()
        self._init_dynamic_quant_param(prefix, weights, **kwargs)

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        t_idx = TimestepManager.get_timestep_idx()
        strategy = self.timestep_config.get_strategy(t_idx, target="w4a4_linear")
        if strategy == "W4A8":
            return self._quant_matmul_w4a8(x)
        return self._quant_matmul_w4a4(x)

    def _get_weight_for_matmul(self):
        x2 = self.weight
        x2 = x2.transpose(0, 1)
        weight_scale = self.weight_scale.transpose(0, 1)
        if 'fc2' in self.prefix:
            x2 = x2.contiguous()
            weight_scale = weight_scale.contiguous()
        return x2, weight_scale

    def _get_bias_for_matmul(self, target_dtype, squeeze=False, unsqueeze=False):
        if self.bias is None:
            return None
        bias = self.bias
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
        if squeeze and len(bias.shape) == 2 and bias.shape[0] == 1:
            bias = bias.squeeze(0)
        if unsqueeze and len(bias.shape) == 1:
            bias = bias.unsqueeze(0)
        return bias

    def _quant_matmul_w4a4(self, x):
        if self.mul_scale is not None:
            x1, input_scale = _dynamic_mx_quant(
                x * self.mul_scale, dst_type=torch_npu.float4_e2m1fn_x2, quant_config=self.quant_config
            )
        else:
            x1, input_scale = _dynamic_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2, quant_config=self.quant_config)

        x2, weight_scale = self._get_weight_for_matmul()
        bias = self._get_bias_for_matmul(torch.float32, squeeze=True)
        output = torch_npu.npu_quant_matmul(
            x1,
            x2,
            weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=self.dtype,
            group_sizes=MXFP4_GROUP_SIZES_W4A4,
        )
        return output

    def _quant_matmul_w4a8(self, x):
        if self.mul_scale is not None:
            x1, input_scale = _dynamic_mx_quant(x * self.mul_scale, dst_type=torch_npu.float8_e4m3fn)
        else:
            x1, input_scale = _dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)

        x2, weight_scale = self._get_weight_for_matmul()
        bias = self._get_bias_for_matmul(torch.bfloat16, unsqueeze=True)
        output = torch_npu.npu_quant_matmul(
            x1,
            x2,
            weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=self.dtype,
            group_sizes=MXFP4_GROUP_SIZES_W4A8,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
        )
        return output

    def _init_dynamic_quant_param(self, prefix=None, weights=None, **kwargs):
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale')
        if weight_scale.shape[1] % 2 != 0:
            weight_scale = torch.nn.functional.pad(weight_scale, pad=(0, 1))
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

        weight = get_quant_weight(weights, f'{prefix}.weight')
        weight = _prepare_mxfp4_weight(weight, kwargs.get('use_nz', False))
        self.register_buffer("weight", weight, persistent=False)


class _OnlineQuantLinearBase(nn.Module):
    def __init__(self, original_linear, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.input_feature = original_linear.in_features
        self.output_feature = original_linear.out_features
        self.is_dynamic = True
        self.mul_scale = None
        if original_linear.bias is not None:
            bias = original_linear.bias.data.to(dtype)
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    @abstractmethod
    def quant_matmul(self, x):
        pass

    def forward(self, x):
        if x.ndim >= 3:
            return self._flatten_linear(x)
        output = self.quant_matmul(x)
        return output

    def _flatten_linear(self, x):
        x_reshpe = x.reshape(x.shape[:-1].numel(), -1)
        output = self.quant_matmul(x_reshpe)
        new_size = list(x.shape)[:-1]
        new_size.append(output.shape[1])
        return output.view(*new_size)


class W8A8OnlineQuantLinear(_OnlineQuantLinearBase):
    def __init__(self, original_linear, dtype=torch.bfloat16):
        super().__init__(original_linear, dtype)
        weight = original_linear.weight.data.npu().to(dtype)
        qweight, weight_scale = torch_npu.npu_dynamic_quant(weight)
        qweight = qweight.t().contiguous()
        weight_scale = weight_scale.squeeze().to(dtype)
        if dtype == torch.float16:
            weight_scale = weight_scale.to(torch.float32)
        self.register_buffer("weight", qweight, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        x_int8, input_scale = torch_npu.npu_dynamic_quant(x)
        output = torch_npu.npu_quant_matmul(
            x_int8, self.weight, self.weight_scale, pertoken_scale=input_scale, output_dtype=self.dtype, bias=self.bias
        )
        return output


class W8A8MXFP8OnlineQuantLinear(_OnlineQuantLinearBase):
    def __init__(self, original_linear, dtype=torch.bfloat16):
        super().__init__(original_linear, dtype)
        weight = original_linear.weight.data.npu().to(dtype)
        weight_fp8, weight_scale_raw = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch_npu.float8_e4m3fn)
        weight_scale = weight_scale_raw.reshape(weight_scale_raw.shape[0], -1, 2)
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        x1, input_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)
        if self.bias is not None and self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32)
        x2 = self.weight
        if x2.dtype != torch.float8_e4m3fn:
            x2 = torch_npu.npu_dtype_cast(x2, torch_npu.float8_e4m3fn)
        x2 = x2.transpose(0, 1)
        output = torch_npu.npu_quant_matmul(
            x1,
            x2,
            self.weight_scale.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=self.bias,
            output_dtype=self.dtype,
            group_sizes=[1, 1, 32],
        )
        return output


class W4A4MXFP4OnlineQuantLinear(_OnlineQuantLinearBase):
    def __init__(self, original_linear, dtype=torch.bfloat16, quant_config=None):
        super().__init__(original_linear, dtype)
        self.quant_config = quant_config or QuantConfig()
        self.timestep_config = self.quant_config.timestep_config or TimestepPolicyConfig()
        weight = original_linear.weight.data.npu().to(dtype)
        weight_fp4, weight_scale_raw = _dynamic_mx_quant(
            weight, dst_type=torch_npu.float4_e2m1fn_x2, quant_config=self.quant_config
        )
        weight_scale = weight_scale_raw.reshape(weight_scale_raw.shape[0], -1, 2)
        self.register_buffer("weight", weight_fp4, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

    def _w4a4_matmul(self, x):
        x1, input_scale = _dynamic_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2, quant_config=self.quant_config)
        if self.bias is not None and self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32)
        x2 = self.weight.transpose(0, 1)
        output = torch_npu.npu_quant_matmul(
            x1,
            x2,
            self.weight_scale.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=self.bias,
            output_dtype=self.dtype,
            group_sizes=MXFP4_GROUP_SIZES_W4A4,
        )
        return output

    def _w4a8_matmul(self, x):
        x1, input_scale = _dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)
        bias = self.bias
        if bias is not None:
            bias = bias.to(torch.bfloat16)
            if len(bias.shape) == 1:
                bias = bias.unsqueeze(0)
        x2 = self.weight.transpose(0, 1)
        output = torch_npu.npu_quant_matmul(
            x1,
            x2,
            self.weight_scale.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=self.dtype,
            group_sizes=MXFP4_GROUP_SIZES_W4A8,
        )
        return output

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        t_idx = TimestepManager.get_timestep_idx()
        strategy = self.timestep_config.get_strategy(t_idx, target="w4a4_linear")
        if strategy == "W4A8":
            return self._w4a8_matmul(x)
        return self._w4a4_matmul(x)


class W4A4MXFP4DualOnlineQuantLinear(_OnlineQuantLinearBase):
    def __init__(self, original_linear, dtype=torch.bfloat16, quant_config=None):
        super().__init__(original_linear, dtype)
        self.quant_config = quant_config or QuantConfig()
        self.timestep_config = self.quant_config.timestep_config or TimestepPolicyConfig()
        weight = original_linear.weight.data.npu().to(dtype)
        weight_fp4, w_l0_scale, w_l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(weight, smooth_scale=None)
        w = torch_npu.npu_format_cast(weight_fp4.view(torch.int8), 29, customize_dtype=torch.int8)
        s = w_l1_scale.reshape(w_l1_scale.shape[0], -1, 2).contiguous()
        ds = w_l0_scale.reshape(w_l0_scale.shape[0], -1).transpose(0, 1).contiguous()
        ms = torch.ones(self.input_feature, dtype=torch.bfloat16, device=weight.device)
        self.register_buffer("weight", w, persistent=False)
        self.register_buffer("weight_scale", s, persistent=False)
        self.register_buffer("weight_dual_scale", ds, persistent=False)
        self.mul_scale = ms

    def _w4a4_matmul(self, x):
        x1, l0_scale, l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=self.mul_scale)
        if self.bias is not None and self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32)
        output = torch_npu.npu_dual_level_quant_matmul(
            x1,
            self.weight,
            l0_scale,
            self.weight_dual_scale,
            l1_scale,
            self.weight_scale,
            bias=self.bias,
            output_dtype=self.dtype,
        )
        return output

    def _w4a8_matmul(self, x):
        x1, input_scale = _dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)
        bias = self.bias
        if bias is not None:
            bias = bias.to(torch.bfloat16)
            if len(bias.shape) == 1:
                bias = bias.unsqueeze(0)
        x2 = self.weight.transpose(0, 1)
        output = torch_npu.npu_quant_matmul(
            x1,
            x2,
            self.weight_scale.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=self.dtype,
            group_sizes=MXFP4_GROUP_SIZES_W4A8,
        )
        return output

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        t_idx = TimestepManager.get_timestep_idx()
        strategy = self.timestep_config.get_strategy(t_idx, target="w4a4_linear")
        if strategy == "W4A8":
            return self._w4a8_matmul(x)
        return self._w4a4_matmul(x)
