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


__all__ = [
    'RMSNormPatternGroup',
    'RopePatternGroup',
    'QwenRopePatternGroup',
    'AdaLayerNormPatternGroup',
    'GELUPatternGroup',
    'MulAddPatternGroup',
    'NormOutAdaLayerNormPatternGroup',
    'MiniMaxH3AdaLnPatternGroup',
    'MiniMaxH3SwigluPatternGroup',
    'MiniMaxH3GatePatternGroup',
    'MiniMaxH3RmsNormPatternGroup',
    'MiniMaxH3RopePatternGroup',
    'WanAdaLayerNormPatternGroup',
    'WanResidualGatePatternGroup',
    'WanRopePatternGroup',
    'WanRmsNormPatternGroup',
]

from .adalayernorm_pattern import AdaLayerNormPatternGroup
from .gelu_pattern import GELUPatternGroup
from .minimax_h3_adaln_pattern import MiniMaxH3AdaLnPatternGroup
from .minimax_h3_gate_pattern import MiniMaxH3GatePatternGroup
from .minimax_h3_swiglu_pattern import MiniMaxH3SwigluPatternGroup
from .minimax_h3_rmsnorm_pattern import MiniMaxH3RmsNormPatternGroup
from .minimax_h3_rope_pattern import MiniMaxH3RopePatternGroup
from .mul_add_pattern import MulAddPatternGroup
from .norm_out_adalayernorm_pattern import NormOutAdaLayerNormPatternGroup
from .qwen_rope_pattern import QwenRopePatternGroup
from .rms_norm_pattern import RMSNormPatternGroup
from .rope_pattern import RopePatternGroup
from .wan_adalayernorm_pattern import WanAdaLayerNormPatternGroup
from .wan_residual_gate_pattern import WanResidualGatePatternGroup
from .wan_rmsnorm_pattern import WanRmsNormPatternGroup
from .wan_rope_pattern import WanRopePatternGroup
