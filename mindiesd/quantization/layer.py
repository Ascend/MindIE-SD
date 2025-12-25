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

from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch_npu

from .config import TimestepPolicyConfig
from .utils import get_quant_weight, TimestepManager
from ..utils.logs.logging import logger
from ..utils.exception import ModelInitError, ParametersInvalid


def import_atb():
    try:
        import torch_atb
    except ImportError as e:
        raise ModelInitError("Failed to find the torch_atb. Please use\
            ./Ascend-cann-nnal_{version}_linux-{arch}.run --install --torch_atb to install torch_atb\
            and source /usr/local/Ascend/nnal/atb/set_env.sh") from e
            
    return torch_atb
    

class WeightQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super(WeightQuantLinear, self).__init__()
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

        output = torch_npu.npu_weight_quant_batchmatmul(x, self.weight, self.weight_scale,
            bias=self.bias)
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
        torch_atb = import_atb()

        input_scale = get_quant_weight(weights, f'{prefix}.input_scale').to(self.dtype)
        self.register_buffer("input_scale", input_scale, persistent=False)

        input_offset = get_quant_weight(weights, f'{prefix}.input_offset').to(torch.int8)
        self.register_buffer("input_offset", input_offset, persistent=False)
        elewise_param = torch_atb.ElewiseParam()
        elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_QUANT_PER_CHANNEL
        elewise_param.out_tensor_type = torch_atb.ACL_INT8
        self.quant = torch_atb.Operation(elewise_param)

        quant_bias = get_quant_weight(weights, f'{prefix}.quant_bias')
        self.register_buffer("quant_bias", quant_bias, persistent=False)
        deq_scale = get_quant_weight(weights, f'{prefix}.deq_scale')
        self.register_buffer("deq_scale", deq_scale, persistent=False)
        weight = get_quant_weight(weights, f'{prefix}.weight')
        self.register_buffer("weight", weight, persistent=False)

        linear_param = torch_atb.LinearParam()
        linear_param.has_bias = True
        linear_param.out_data_type = torch_atb.ACL_FLOAT16 \
            if self.dtype == torch.float16 else torch_atb.ACL_BF16
        linear_param.transpose_a = False
        linear_param.transpose_b = True
        self.linear = torch_atb.Operation(linear_param)
        
    def _init_dynamic_quant_param(self, prefix=None, weights=None, **kwargs):
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale').squeeze().to(self.dtype)
        if self.dtype == torch.float16:
            weight_scale = weight_scale.to(torch.float32)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        if self.bias is not None:
            self.bias = self.bias.to(self.dtype)
        weight = get_quant_weight(weights, f'{prefix}.weight')
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
                x_int8 = \
                    self.quant.forward([x * self.mul_scale, self.input_scale, self.input_offset])[0]
            else:
                x_int8 = self.quant.forward([x, self.input_scale, self.input_offset])[0]

            output = self.linear.forward([x_int8, self.weight, self.quant_bias, self.deq_scale])[0]
        else:
            if self.mul_scale is not None:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x * self.mul_scale)
            else:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x)

            output = torch_npu.npu_quant_matmul(x_int8, self.weight, self.weight_scale,
                                                pertoken_scale=input_scale, output_dtype=self.dtype,
                                                bias=self.bias)
        return output


class W8A8TimeStepQuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.timestep_config = kwargs.get('timestep_config', TimestepPolicyConfig())

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
        strategy = self.timestep_config.get_strategy(t_idx)

        if strategy == "static":
            self.is_dynamic = False
        else:
            self.is_dynamic = True

        if not self.is_dynamic:
            if self.mul_scale is not None:
                x_int8 = \
                    self.quant.forward([x * self.mul_scale, self.input_scale[t_idx], self.input_offset[t_idx]])[0]
            else:
                x_int8 = self.quant.forward([x, self.input_scale[t_idx], self.input_offset[t_idx]])[0]

            output = self.linear.forward([x_int8, self.weight, self.quant_bias[t_idx], self.deq_scale[t_idx]])[0]
        else:
            if self.mul_scale is not None:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x * self.mul_scale)
            else:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x)

            output = torch_npu.npu_quant_matmul(x_int8, self.weight.T, self.weight_scale,
                                                pertoken_scale=input_scale, output_dtype=self.dtype,
                                                bias=self.bias)
        return output


class QuantFA(nn.Module):
    def __init__(self, ori_head_num, ori_inner_dim, prefix, quant_weights=None, dtype=torch.bfloat16):
        r"""Initializes the QuantFA with the given parameters.

        Args:
            ori_head_num (int): The original number of attention heads.
            ori_inner_dim (int): The original inner dimension size.
            prefix (str): A prefix string used for naming components.
            quant_weights (Optional[Tensor]): Quantized weights to initialize the module with.
            dtype (torch.dtype): The desired data type for the output.The supported dytpe: float16/bfloat16.
                Defaults to torch.bfloat16.
        """
        super().__init__()
        torch_atb = import_atb()

        if not isinstance(ori_head_num, int) or ori_head_num <= 0:
            raise ParametersInvalid(f"ori_head_num must be a positive integer,\
                got {type(ori_head_num)} with value {ori_head_num}")
    
        # Check ori_inner_dim is integer
        if not isinstance(ori_inner_dim, int) or ori_inner_dim <= 0:
            raise ParametersInvalid(f"ori_inner_dim must be a positive integer,\
                got {type(ori_inner_dim)} with value {ori_inner_dim}")
        
        # Check prefix is string
        if not isinstance(prefix, str):
            raise ParametersInvalid(f"prefix must be a string, got {type(prefix)}")
        
        # Check quant_weights is either None or torch.Tensor
        if quant_weights is None:
            raise ParametersInvalid(f"quant_weights must can't be None")
        
        # Check dtype is supported
        supported_dtypes = [torch.float16, torch.bfloat16]
        if dtype not in supported_dtypes:
            raise ParametersInvalid(f"dtype must be float16 or bfloat16, got {dtype}")

        q_scale = get_quant_weight(quant_weights, f'{prefix}.fa_q.scale')
        self.register_buffer("q_scale", q_scale, persistent=False)
        k_scale = get_quant_weight(quant_weights, f'{prefix}.fa_k.scale')
        self.register_buffer("k_scale", k_scale, persistent=False)
        v_scale = get_quant_weight(quant_weights, f'{prefix}.fa_v.scale')
        self.register_buffer("v_scale", v_scale, persistent=False)
        q_offset = get_quant_weight(quant_weights, f'{prefix}.fa_q.offset')
        self.register_buffer("q_offset", q_offset, persistent=False)
        k_offset = get_quant_weight(quant_weights, f'{prefix}.fa_k.offset')
        self.register_buffer("k_offset", k_offset, persistent=False)
        v_offset = get_quant_weight(quant_weights, f'{prefix}.fa_v.offset')
        self.register_buffer("v_offset", v_offset, persistent=False)

        self.dtype = dtype
        head_num = int(self.q_scale.shape[0])
        parallel_count = ori_head_num // head_num
        head_dim = ori_inner_dim // parallel_count // head_num
        gqa_size = self.q_scale.shape[0] // self.k_scale.shape[0]

        q_scale2 = self.q_scale.repeat(1, head_dim)
        self.register_buffer("q_scale2", q_scale2, persistent=False)
        k_scale2 = self.k_scale.repeat(1, head_dim)
        self.register_buffer("k_scale2", k_scale2, persistent=False)
        v_scale2 = self.v_scale.repeat(1, head_dim)
        self.register_buffer("v_scale2", v_scale2, persistent=False)
        q_offset2 = self.q_offset.repeat(1, head_dim).to(torch.int8)
        self.register_buffer("q_offset2", q_offset2, persistent=False)
        kv_offset2 = self.k_offset.repeat(1, head_dim).to(torch.int8)
        self.register_buffer("kv_offset2", kv_offset2, persistent=False)

        fa3_k_scale, fa3_v_scale = self.k_scale.repeat(1, gqa_size).view(-1, 1), self.v_scale.repeat(1, gqa_size).view(
            -1, 1)

        qk_scale = torch.squeeze(self.q_scale * fa3_k_scale).to(torch.float32)
        self.register_buffer("qk_scale", qk_scale, persistent=False)
        fa3_v_scale = torch.squeeze(fa3_v_scale).contiguous().to(torch.float32)
        self.register_buffer("fa3_v_scale", fa3_v_scale, persistent=False)
        fa3_offset = torch.zeros(self.q_scale.shape[0], dtype=torch.int32)
        self.register_buffer("fa3_offset", fa3_offset, persistent=False)


        elewise_param = torch_atb.ElewiseParam()
        elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_QUANT_PER_CHANNEL
        elewise_param.out_tensor_type = torch_atb.ACL_INT8
        self.quant1 = torch_atb.Operation(elewise_param)

        elewise_param = torch_atb.ElewiseParam()
        elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_QUANT_PER_CHANNEL
        elewise_param.out_tensor_type = torch_atb.ACL_INT8
        self.quant2 = torch_atb.Operation(elewise_param)

        elewise_param = torch_atb.ElewiseParam()
        elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_QUANT_PER_CHANNEL
        elewise_param.out_tensor_type = torch_atb.ACL_INT8
        self.quant3 = torch_atb.Operation(elewise_param)

        self_attention_param = torch_atb.SelfAttentionParam()
        self_attention_param.head_num = head_num
        self_attention_param.kv_head_num = head_num
        self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
        self_attention_param.quant_type = torch_atb.SelfAttentionParam.QuantType.TYPE_QUANT_QKV_ONLINE
        self_attention_param.out_data_type = torch_atb.ACL_FLOAT16 \
            if self.dtype == torch.float16 else torch_atb.ACL_BF16
        self_attention_param.qk_scale = 1.0 / math.sqrt(1.0 * head_dim)

        self.quant_attn = torch_atb.Operation(self_attention_param)

        self.attn_param_cache = {}
        self.maxsize = 512

    def forward(self, query, key, value, seq_len_list):
        """
        Apply the QuantFA with the given parameters.

        Args:
            query (torch.Tensor):
                The input query of attention calculation formula.
                The supported layout: [T,N,D].The supported dytpe: float16/bfloat16.
            key (torch.Tensor):
                The input key of attention calculation formula.
                The supported layout: [T,N,D].The supported dytpe: float16/bfloat16.
            value (torch.Tensor):
                The input value of attention calculation formula.
                The supported layout: [T,N,D].The supported dytpe: float16/bfloat16.
            seq_len_list (list):
                The sum of tokens on each batch, shape is [batch]

        Returns:
            (torch.Tensor): QuantFa results derived from dequant.
        """
        if not isinstance(query, torch.Tensor):
            raise ParametersInvalid(f"query must be torch.Tensor, got {type(query)}")
        if not isinstance(key, torch.Tensor):
            raise ParametersInvalid(f"key must be torch.Tensor, got {type(key)}")
        if not isinstance(value, torch.Tensor):
            raise ParametersInvalid(f"value must be torch.Tensor, got {type(value)}")
        
        # Check dtypes
        supported_dtypes = [torch.float16, torch.bfloat16]
        if query.dtype not in supported_dtypes:
            raise ParametersInvalid(f"query dtype must be float16 or bfloat16, got {query.dtype}")
        if key.dtype not in supported_dtypes:
            raise ParametersInvalid(f"key dtype must be float16 or bfloat16, got {key.dtype}")
        if value.dtype not in supported_dtypes:
            raise ParametersInvalid(f"value dtype must be float16 or bfloat16, got {value.dtype}")
        
        # Check shapes
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ParametersInvalid(f"query, key and value must be 3D tensors with shape [T,N,D], "
                        f"got shapes {query.shape}, {key.shape}, {value.shape}")
        
        # Check all tensors have same shape
        if query.shape != key.shape or key.shape != value.shape:
            raise ParametersInvalid(f"query, key and value must have same shape, "
                        f"got {query.shape}, {key.shape}, {value.shape}")
        
        # Check seq_len_list is list
        if not isinstance(seq_len_list, list):
            raise ParametersInvalid(f"seq_len_list must be list, got {type(seq_len_list)}")
        
        # Check all elements in seq_len_list are integers
        if not all(isinstance(x, int) for x in seq_len_list):
            raise ParametersInvalid("All elements in seq_len_list must be integers")

        cache_key = hash(tuple(seq_len_list))
        if cache_key in self.attn_param_cache:
            seq_len = self.attn_param_cache[cache_key]
        else:
            seq_len = torch.tensor(seq_len_list, dtype=torch.int32)
            self.attn_param_cache[cache_key] = seq_len
        if self.maxsize is not None and len(self.attn_param_cache) > self.maxsize:
            self.attn_param_cache.popitem(last=False)
        query = self.quant1.forward([query, self.q_scale2, self.q_offset2])[0]
        key = self.quant2.forward([key, self.k_scale2, self.kv_offset2])[0]
        value = self.quant3.forward([value, self.v_scale2, self.kv_offset2])[0]
        return self.quant_attn.forward(
            [query, key, value, seq_len, self.qk_scale, self.fa3_offset, self.fa3_v_scale, self.fa3_offset])[0]


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
        if x2.dtype != torch_npu.float8_e4m3fn:
            x2 = torch_npu.npu_dtype_cast(x2, torch_npu.float8_e4m3fn)
        x2 = x2.transpose(0, 1)

        output = torch_npu.npu_quant_matmul(
                            x1,
                            x2,
                            self.weight_scale,
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
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

        weight = get_quant_weight(weights, f'{prefix}.weight')
        if kwargs.get('use_nz', False):
            weight = torch_npu.npu_format_cast(weight.npu(), 29)
        self.register_buffer("weight", weight, persistent=False)