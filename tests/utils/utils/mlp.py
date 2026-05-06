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

import collections.abc
from itertools import repeat
from functools import partial

import torch.nn as nn
from mindiesd.layers.activation import get_activation_layer


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            features_in,
            features_hidden=None,
            features_out=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            use_conv=False,
    ):
        super().__init__()
        features_out = features_out or features_in
        features_hidden = features_hidden or features_in
        to_2tuple = self._ntuple(2)
        bias = to_2tuple(bias)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(features_in, features_hidden, bias=bias[0])
        self.act = act_layer() if not isinstance(act_layer, str) else get_activation_layer(act_layer)
        self.norm = norm_layer(features_hidden) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(features_hidden, features_out, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

    def _ntuple(self, n):
        def parse(x):
            if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
                return tuple(x)
            return tuple(repeat(x, n))
        return parse