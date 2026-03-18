#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import unittest
import torch

from mindiesd.layers.rope import rotary_position_embedding
from mindiesd.utils import ParametersInvalid
from utils.utils.embedding import RotaryPositionEmbedding
from utils.utils.precision_compare import data_compare


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestRope(unittest.TestCase):

    def test_x_type(self):
        device = "npu"
        x = [2, 64, 8, 16]
        cos = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated_half"
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_x_dim(self):
        device = "npu"
        x = torch.randn([2, 64, 128], dtype=torch.float16).to(device)
        cos = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated_half"
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_cos_type(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = [1, 64, 1, 16]
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated_half"
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_sin_type(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        sin = [1, 64, 1, 16]
        rotated_mode = "rotated_half"
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_cos_dim(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = torch.randn([1, 64, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated_half"
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_sin_dim(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated_half"
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_cos_sin_dim_equal(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = torch.randn([64, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated_half"
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_rotated_mode_type(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = 1
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_rotated_mode(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated"
        head_first = False
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_head_first_type(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated_half"
        head_first = "False"
        fused = True

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_fused_type(self):
        device = "npu"
        x = torch.randn([2, 64, 8, 16], dtype=torch.float16).to(device)
        cos = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        sin = torch.randn([1, 64, 1, 16], dtype=torch.float32).to(device)
        rotated_mode = "rotated_half"
        head_first = False
        fused = "True"

        with self.assertRaises(ParametersInvalid):
            rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)

    def test_rope_rotated_half_4d(self):
        device = "npu"
        shapes = [(2, 16, 88), (1, 24, 128), (4, 8, 64)]
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        grid_sizes = [(64, 64), (80, 48), (72, 54)]
        base_size = 32
        for shape in shapes:
            batch, num_heads, dim = shape
            for dtype in dtypes:
                for grid_height, grid_width in grid_sizes:
                    seqlen = grid_height * grid_width
                    shape = (batch, num_heads, seqlen, dim)
                    hidden_states = torch.randn(shape, dtype=dtype).to(device)

                    rope = RotaryPositionEmbedding(embed_dim=dim)
                    rotary_pos_emb = rope.get_2d_rotary_pos_embed(grid_height, grid_width, base_size)
                    cos, sin = rotary_pos_emb
                    cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)
                    cos, sin = rope.reshape_for_broadcast(hidden_states, cos, sin, head_first=True)

                    rope_rotated_half = rotary_position_embedding(hidden_states, cos, sin,
                        rotated_mode="rotated_half", head_first=True, fused=False)
                    rope_rotated_half_fused = rotary_position_embedding(hidden_states, cos, sin,
                        rotated_mode="rotated_half", head_first=True, fused=True)
                    self.assertEqual(rope_rotated_half.shape, rope_rotated_half_fused.shape)

                    result, _, max_err = data_compare(rope_rotated_half.cpu(), rope_rotated_half_fused.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    def test_rope_rotated_interleaved_4d(self):
        device = "npu"
        shapes = [(2, 16, 88), (1, 24, 128), (4, 8, 64)]
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        grid_sizes = [(64, 64), (80, 48), (72, 54)]
        base_size = 32
        for shape in shapes:
            batch, num_heads, dim = shape
            for dtype in dtypes:
                for grid_height, grid_width in grid_sizes:
                    seqlen = grid_height * grid_width
                    shape = (batch, seqlen, num_heads, dim)
                    hidden_states = torch.randn(shape, dtype=dtype).to(device)

                    rope = RotaryPositionEmbedding(embed_dim=dim)
                    rotary_pos_emb = rope.get_2d_rotary_pos_embed(grid_height, grid_width, base_size)
                    cos, sin = rotary_pos_emb
                    cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)
                    cos, sin = rope.reshape_for_broadcast(hidden_states, cos, sin, head_first=False)

                    rope_rotated_half = rotary_position_embedding(hidden_states, cos, sin,
                        rotated_mode="rotated_interleaved", head_first=False, fused=False)
                    rope_rotated_half_fused = rotary_position_embedding(hidden_states, cos, sin,
                        rotated_mode="rotated_interleaved", head_first=False, fused=True)
                    self.assertEqual(rope_rotated_half.shape, rope_rotated_half_fused.shape)

                    result, _, max_err = data_compare(rope_rotated_half.cpu(), rope_rotated_half_fused.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    def test_rope_rotated_half_2d(self):
        device = "npu"
        shapes = [(2, 16, 88), (1, 24, 128), (4, 8, 64)]
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        grid_sizes = [(64, 64), (80, 48), (72, 54)]
        base_size = 32
        for shape in shapes:
            batch, num_heads, dim = shape
            for dtype in dtypes:
                for grid_height, grid_width in grid_sizes:
                    seqlen = grid_height * grid_width
                    shape = (batch, num_heads, seqlen, dim)
                    hidden_states = torch.randn(shape, dtype=dtype).to(device)

                    rope = RotaryPositionEmbedding(embed_dim=dim)
                    rotary_pos_emb = rope.get_2d_rotary_pos_embed(grid_height, grid_width, base_size)
                    cos, sin = rotary_pos_emb
                    cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)

                    rope_rotated_half = rotary_position_embedding(hidden_states, cos, sin,
                        rotated_mode="rotated_half", head_first=True, fused=False)
                    rope_rotated_half_fused = rotary_position_embedding(hidden_states, cos, sin,
                        rotated_mode="rotated_half", head_first=True, fused=True)
                    self.assertEqual(rope_rotated_half.shape, rope_rotated_half_fused.shape)

                    result, _, max_err = data_compare(rope_rotated_half.cpu(), rope_rotated_half_fused.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    def test_rope_rotated_interleaved_2d(self):
        device = "npu"
        shapes = [(2, 16, 88), (1, 24, 128), (4, 8, 64)]
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        grid_sizes = [(64, 64), (80, 48), (72, 54)]
        base_size = 32
        for shape in shapes:
            batch, num_heads, dim = shape
            for dtype in dtypes:
                for grid_height, grid_width in grid_sizes:
                    seqlen = grid_height * grid_width
                    shape = (batch, seqlen, num_heads, dim)
                    hidden_states = torch.randn(shape, dtype=dtype).to(device)

                    rope = RotaryPositionEmbedding(embed_dim=dim)
                    rotary_pos_emb = rope.get_2d_rotary_pos_embed(grid_height, grid_width, base_size)
                    cos, sin = rotary_pos_emb
                    cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)

                    rope_rotated_half = rotary_position_embedding(hidden_states, cos, sin,
                        rotated_mode="rotated_interleaved", head_first=False, fused=False)
                    rope_rotated_half_fused = rotary_position_embedding(hidden_states, cos, sin,
                        rotated_mode="rotated_interleaved", head_first=False, fused=True)
                    self.assertEqual(rope_rotated_half.shape, rope_rotated_half_fused.shape)

                    result, _, max_err = data_compare(rope_rotated_half.cpu(), rope_rotated_half_fused.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")


if __name__ == '__main__':
    unittest.main()
