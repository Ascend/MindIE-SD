#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest

import torch

from mindiesd.layers import apply_rotary_pos_emb
from mindiesd.utils import ParametersInvalid


def rotate(x: torch.Tensor, rotary_mode: str) -> torch.Tensor:
    if rotary_mode == "half":
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    if rotary_mode == "interleave":
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    if rotary_mode == "quarter":
        x1, x2, x3, x4 = x.chunk(4, dim=-1)
        return torch.cat((-x2, x1, -x4, x3), dim=-1)
    raise ValueError(f"unsupported rotary mode: {rotary_mode}")


def make_inputs(layout: str, dtype: torch.dtype, device: torch.device):
    shapes = {
        "BSND": ((2, 8, 6, 64), (2, 8, 2, 64), (2, 8, 1, 64)),
        "SBND": ((8, 2, 6, 64), (8, 2, 2, 64), (8, 2, 1, 64)),
        "BNSD": ((2, 6, 8, 64), (2, 2, 8, 64), (2, 1, 8, 64)),
        "TND": ((16, 6, 64), (16, 2, 64), (16, 1, 64)),
    }
    query_shape, key_shape, angle_shape = shapes[layout]
    query = torch.randn(query_shape, device=device, dtype=dtype)
    key = torch.randn(key_shape, device=device, dtype=dtype)
    angles = torch.randn(angle_shape, device=device, dtype=dtype)
    return query, key, angles.cos(), angles.sin()


class TestApplyRotaryPosEmb(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_invalid_layout(self):
        query, key, cos, sin = make_inputs("BSND", torch.float16, self.device)
        with self.assertRaisesRegex(ParametersInvalid, "layout must be"):
            apply_rotary_pos_emb(query, key, cos, sin, layout="INVALID", rotary_mode="half")

    def test_invalid_rotary_mode(self):
        query, key, cos, sin = make_inputs("BSND", torch.float16, self.device)
        with self.assertRaisesRegex(ParametersInvalid, "rotary_mode must be"):
            apply_rotary_pos_emb(query, key, cos, sin, layout="BSND", rotary_mode="invalid")


if __name__ == "__main__":
    unittest.main()
