# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.


import os
import unittest
import numpy as np
import torch
from torch import nn
from mindiesd.utils import ParametersInvalid
from utils.utils.embedding import RotaryPositionEmbedding
from utils.utils.precision_compare import data_compare


class HunyuanDiTRoPE(nn.Module):
    def __init__(self, embed_dim: int, use_real: bool = True):
        super().__init__()

        if embed_dim % 4 != 0 or embed_dim <= 2:
            raise ParametersInvalid(
                f"The value of input embed_dim must be divisible by 4 and > 2, but got {embed_dim}."
            )
        self.embed_dim = embed_dim
        self.use_real = use_real

    def get_fill_resize_and_crop(self, grid_size, base_size):
        h, w = grid_size
        r = h / w  # target resolution
        # resize
        if r > 1:
            resize_height = base_size
            resize_width = int(round(base_size / h * w))
        else:
            resize_width = base_size
            resize_height = int(round(base_size / w * h))
        crop_top = int(round((base_size - resize_height) / 2.0))
        crop_left = int(round((base_size - resize_width) / 2.0))
        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def get_meshgrid(self, start, stop, grid_size):
        grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
        grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)  # [2, W, H]
        return grid

    def get_1d_rotary_pos_embed(self, pos, theta: float = 10000.0):
        half_of_dim = self.embed_dim // 2
        if isinstance(pos, int):
            pos = np.arange(pos)
        freqs = 1.0 / (theta ** (torch.arange(0, half_of_dim, 2)[: (half_of_dim // 2)].float() / half_of_dim))  # [D/2]
        t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
        freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
        if self.use_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
            return freqs_cos, freqs_sin
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
            return freqs_cis

    def get_2d_rotary_pos_embed_from_grid(self, grid):
        emb_h = self.get_1d_rotary_pos_embed(grid[0].reshape(-1))  # (H*W, D/4)
        emb_w = self.get_1d_rotary_pos_embed(grid[1].reshape(-1))  # (H*W, D/4)

        if self.use_real:
            cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D/2)
            sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D/2)
            return cos, sin
        else:
            emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
            return emb

    def get_2d_rotary_pos_embed(self, grid_height, grid_width, base_size):
        grid_size = (grid_height, grid_width)
        start, stop = self.get_fill_resize_and_crop(grid_size, base_size)
        grid = self.get_meshgrid(start, stop, grid_size)  # [2, H, w]
        grid = grid.reshape([2, 1, *grid.shape[1:]])
        pos_embed = self.get_2d_rotary_pos_embed_from_grid(grid)
        return pos_embed

    def reshape_for_broadcast(self, freqs_cis, x):
        ndim = x.ndim
        if isinstance(freqs_cis, tuple):
            shape = [d if i in (ndim - 2, ndim - 1) else 1 for i, d in enumerate(x.shape)]
            return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
        else:
            shape = [d if i in (ndim - 2, ndim - 1) else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

    def rotate_half(self, x):
        x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    def forward(self, xq, freqs_cis):
        if isinstance(freqs_cis, tuple):
            cos, sin = self.reshape_for_broadcast(freqs_cis, xq)  # [S, D]
            cos, sin = cos.to(xq.device), sin.to(xq.device)
            xq_out = (xq.float() * cos + self.rotate_half(xq.float()) * sin).type_as(xq)
        else:
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
            freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_).to(xq.device)  # [S, D//2] --> [1, S, 1, D//2]
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        return xq_out


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU."
)
class TestEmbedding(unittest.TestCase):
    @torch.no_grad()
    def test_get_2d_rotary_pos_embed(self):
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
                    rope_test = HunyuanDiTRoPE(embed_dim=dim)
                    rotary_pos_emb_test = rope_test.get_2d_rotary_pos_embed(grid_height, grid_width, base_size)
                    rotary_pos_emb_test = (
                        rotary_pos_emb_test[0].to(dtype).to(device),
                        rotary_pos_emb_test[1].to(dtype).to(device),
                    )

                    rope = RotaryPositionEmbedding(embed_dim=dim)
                    rotary_pos_emb = rope.get_2d_rotary_pos_embed(grid_height, grid_width, base_size)
                    rotary_pos_emb = (rotary_pos_emb[0].to(dtype).to(device), rotary_pos_emb[1].to(dtype).to(device))

                    embedding_test = rope_test(hidden_states, rotary_pos_emb_test)
                    embedding = rope(
                        hidden_states, rotary_pos_emb, rotated_mode="rotated_interleaved", head_first=True, fused=True
                    )
                    self.assertEqual(embedding.shape, embedding_test.shape)

                    embedding_test = embedding_test.reshape(1, -1).to(torch.float32)
                    embedding = embedding.reshape(1, -1).to(torch.float32)
                    result, _, max_err = data_compare(embedding.cpu(), embedding_test.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")
