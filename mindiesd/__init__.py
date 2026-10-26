#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

__all__ = [
    'CacheAgent',
    'CacheConfig',
    'FP8FAMode',
    'OnlineQuantConfig',
    'QuantConfig',
    'RMSNorm',
    'TimestepManager',
    'TimestepPolicyConfig',
    'add_layer_norm',
    'add_rms_norm',
    'apply_rotary_pos_emb',
    'attention_forward',
    'attention_forward_varlen',
    'fast_layernorm',
    'frequency_regulator',
    'fused_moe',
    'get_activation_layer',
    'group_norm_silu',
    'layernorm_scale_shift',
    'norm_rope_concat',
    'quant_attention',
    'quantize',
    'rotary_position_embedding',
    'rms_norm_indexed_scale_shift',
    'sparse_attention',
]


from .env import set_environment_variables

set_environment_variables()

from .cache_agent import CacheAgent, CacheConfig  # noqa: E402
from .layers import (  # noqa: E402
    RMSNorm,
    add_layer_norm,
    add_rms_norm,
    apply_rotary_pos_emb,
    attention_forward,
    attention_forward_varlen,
    fast_layernorm,
    frequency_regulator,
    fused_moe,
    get_activation_layer,
    group_norm_silu,
    layernorm_scale_shift,
    quant_attention,
    rotary_position_embedding,
    sparse_attention,
)
from .layers.norm_rope_concat import norm_rope_concat  # noqa: E402
from .layers.index_modulation import rms_norm_indexed_scale_shift  # noqa: E402
from .quantization import (  # noqa: E402
    FP8FAMode,
    OnlineQuantConfig,
    QuantConfig,
    TimestepManager,
    TimestepPolicyConfig,
    quantize,
)
