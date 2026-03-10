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

import os
import functools
import unittest
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from einops import repeat, rearrange
import torch.nn.functional as F

import torch_npu
from device import DEVICE_ID
torch_npu.npu.set_device(DEVICE_ID)

from mindiesd.utils import ModelInitError, ParametersInvalid
from utils.utils.embedding import RotaryEmbedding, TimestepEmbedder, SizeEmbedder, \
    CombinedTimestepTextProjEmbeddings, PositionEmbedding2D, PatchEmbed, RotaryPositionEmbedding
from utils.utils.precision_compare import data_compare


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def hunyuandit_timestep_embedding(t, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)   # size: [dim/2], 一个指数衰减的曲线
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(t, "b -> b d", d=dim)
    return embedding


def get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    scale: float = 1,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class HunyuanDitTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = hunyuandit_timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class OpenSoraTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class OpensoraSizeEmbedder(OpenSoraTimestepEmbedder):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            scale=self.scale,
        )
        return t_emb


class FP32SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class SD3CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning


class OpenSoraPositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, h: int, w: int, scale: Optional[float] = 1.0):
        s_hw = h * w
        base_size = round(s_hw ** 0.5)
        grid_size = (h, w)
        return self._get_cached_emb(x, grid_size, base_size, scale)
    
    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(self, x, grid_size, base_size: Optional[int] = None, scale=1.0):
        device = x.device
        dtype = x.dtype
        grid_h = torch.arange(grid_size[0], device=device) / scale
        grid_w = torch.arange(grid_size[1], device=device) / scale
        if base_size is not None:
            grid_h *= base_size / grid_size[0]
            grid_w *= base_size / grid_size[1]
        grid_h, grid_w = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)
    
    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)


class HunyuanDitPatchEmbed(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            img_size = tuple(img_size)
        else:
            raise ValueError(f"The data type of img_size must be int or tuple/list of length 2, but got {img_size}.")
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def forward(self, x):
        x_dtype = x.dtype
        x = self.proj(x.to(self.dtype))
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x.to(x_dtype)


class SD3PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with support for SD3 cropping."""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height:({height}) cannot be > `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width:({width}) cannot be > `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top: top + height, left: left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed
    
    def forward(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent_dtype = latent.dtype
        latent = self.proj(latent.to(self.dtype))
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent_dtype)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent_dtype)


class HunyuanDiTRoPE(nn.Module):

    def __init__(self, embed_dim: int, use_real: bool = True):
        super().__init__()

        if embed_dim % 4 != 0 or embed_dim <= 2:
            raise ParametersInvalid(f"The value of input embed_dim must be divisible by 4 and > 2, "
                                    f"but got {embed_dim}.")
        self.embed_dim = embed_dim
        self.use_real = use_real

    def get_fill_resize_and_crop(self, grid_size, base_size):
        h, w = grid_size
        r = h / w           # target resolution
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
        grid = np.stack(grid, axis=0)   # [2, W, H]
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
            cos = torch.cat([emb_h[0], emb_w[0]], dim=1)    # (H*W, D/2)
            sin = torch.cat([emb_h[1], emb_w[1]], dim=1)    # (H*W, D/2)
            return cos, sin
        else:
            emb = torch.cat([emb_h, emb_w], dim=1)    # (H*W, D/2)
            return emb

    def get_2d_rotary_pos_embed(self, grid_height, grid_width, base_size):
        grid_size = (grid_height, grid_width)
        start, stop = self.get_fill_resize_and_crop(grid_size, base_size)
        grid = self.get_meshgrid(start, stop, grid_size)   # [2, H, w]
        grid = grid.reshape([2, 1, *grid.shape[1:]])
        pos_embed = self.get_2d_rotary_pos_embed_from_grid(grid)
        return pos_embed

    def reshape_for_broadcast(self, freqs_cis, x):
        ndim = x.ndim
        if isinstance(freqs_cis, tuple):
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
        else:
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

    def rotate_half(self, x):
        x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    def forward(self, xq, freqs_cis):
        if isinstance(freqs_cis, tuple):
            cos, sin = self.reshape_for_broadcast(freqs_cis, xq)    # [S, D]
            cos, sin = cos.to(xq.device), sin.to(xq.device)
            xq_out = (xq.float() * cos + self.rotate_half(xq.float()) * sin).type_as(xq)
        else:
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
            freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_).to(xq.device)   # [S, D//2] --> [1, S, 1, D//2]
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        return xq_out


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU", "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.")
class TestEmbedding(unittest.TestCase):
    def test_rotary_embedding(self):
        """
        测试遇到不支持的freqs_for时的报错情况
        """

        flag = False
        try:
            RotaryEmbedding(dim=256, freqs_for="unsupported_freqs")
        except ModelInitError:
            flag = True
        
        self.assertTrue(flag)
    
    @torch.no_grad()
    def test_time_step_embedder(self):
        devices = ["npu", "cpu"]
        sizes = [32, 64, 128, 256]
        max_values = [32, 64, 128, 256]
        for device in devices:
            for size in sizes:
                for max_value in max_values:
                    t = torch.randint(0, max_value + 1, (256,)).to(device)
                    
                    test1_timestepembedder = HunyuanDitTimestepEmbedder(hidden_size=256).to(device)
                    test2_timestepembedder = OpenSoraTimestepEmbedder(hidden_size=256).to(device)
                    timestepembedder = TimestepEmbedder(hidden_size=256, size=size).to(device)

                    para_dict = test1_timestepembedder.state_dict()
                    test2_timestepembedder.load_state_dict(para_dict)
                    timestepembedder.load_state_dict(para_dict)

                    embedding_test1 = test1_timestepembedder(t)
                    embedding_test2 = test2_timestepembedder(t, test2_timestepembedder.mlp[0].weight.dtype)
                    embedding = timestepembedder(t, test2_timestepembedder.mlp[0].weight.dtype)
                    self.assertEqual(embedding.shape, embedding_test1.shape)
                    self.assertEqual(embedding.shape, embedding_test2.shape)

                    embedding_test1 = embedding_test1.reshape(1, -1).to(torch.float32)
                    embedding_test2 = embedding_test2.reshape(1, -1).to(torch.float32)
                    embedding = embedding.reshape(1, -1).to(torch.float32)
                    result1, _, max_err1 = data_compare(embedding.cpu(), embedding_test1.cpu())
                    result2, _, max_err2 = data_compare(embedding.cpu(), embedding_test2.cpu())
                    self.assertEqual(result1, "success", msg=f"Data compare failed. Max error is: {max_err1}")
                    self.assertEqual(result2, "success", msg=f"Data compare failed. Max error is: {max_err2}")

    @torch.no_grad()
    def test_size_embedder(self):
        devices = ["npu", "cpu"]
        sizes = [32, 64, 128, 256]
        max_values = [32, 64, 128, 256]
        for device in devices:
            for size in sizes:
                for max_value in max_values:
                    s = torch.randint(0, max_value + 1, (256,)).to(device)
                    
                    test_size_embedder = OpensoraSizeEmbedder(hidden_size=256).to(device)
                    size_embedder = SizeEmbedder(hidden_size=256, size=size).to(device)

                    para_dict = test_size_embedder.state_dict()
                    size_embedder.load_state_dict(para_dict)

                    embedding_test = test_size_embedder(s, len(s))
                    embedding = size_embedder(s, len(s))
                    self.assertEqual(embedding.shape, embedding_test.shape)

                    embedding_test = embedding_test.reshape(1, -1).to(torch.float32)
                    embedding = embedding.reshape(1, -1).to(torch.float32)
                    result, _, max_err = data_compare(embedding.cpu(), embedding_test.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")
    
    @torch.no_grad()
    def test_combined_timestep_text_porj_embedder(self):
        devices = ["npu", "cpu"]
        sizes = [32, 64, 128, 256]
        max_values = [32, 64, 128, 256]
        for device in devices:
            for size in sizes:
                for max_value in max_values:
                    t = torch.randint(0, max_value + 1, (256,)).to(device)
                    pooled_projection = torch.rand([256, 256]).to(device)

                    test_embedder = SD3CombinedTimestepTextProjEmbeddings(256, 256).to(device)
                    embedder = CombinedTimestepTextProjEmbeddings(256, 256, size=size).to(device)

                    para_dict_test = list(test_embedder.state_dict().items())
                    para_dict = list(embedder.state_dict().items())
                    para_dict_ex = {}
                    for i, _ in enumerate(para_dict_test):
                        para_dict_ex[para_dict[i][0]] = para_dict_test[i][1]
                    embedder.load_state_dict(para_dict_ex)

                    embedding_test = test_embedder(t, pooled_projection)
                    embedding = embedder(t, pooled_projection)
                    self.assertEqual(embedding.shape, embedding_test.shape)

                    embedding_test = embedding_test.reshape(1, -1).to(torch.float32)
                    embedding = embedding.reshape(1, -1).to(torch.float32)
                    result, _, max_err = data_compare(embedding.cpu(), embedding_test.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    @torch.no_grad()
    def test_position_embedding2d_opensora(self):
        '''
        Test Open-Sora PositionEmbedding2D
        '''
        devices = ["npu", "cpu"]
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        grid_sizes = [(64, 32), (224, 224), (320, 180), (256, 256)]
        for device in devices:
            for h, w in grid_sizes:
                for dtype in dtypes:
                    x = torch.randn([32, 3, 224, 224], dtype=dtype).to(device)
                    test_positionembedding2d = OpenSoraPositionEmbedding2D(dim=256).to(device)
                    positionembedding2d = PositionEmbedding2D(dim=256).to(device)

                    embedding_test = test_positionembedding2d(x, h, w)
                    embedding = positionembedding2d(x, h, w)
                    self.assertEqual(embedding.shape, embedding_test.shape)

                    embedding_test = embedding_test.reshape(1, -1).to(torch.float32)
                    embedding = embedding.reshape(1, -1).to(torch.float32)
                    result, _, max_err = data_compare(embedding.cpu(), embedding_test.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    @torch.no_grad()
    def test_patch_embed_hunyuan(self):
        devices = ["npu", "cpu"]
        shapes = [(32, 3, 224, 224), (16, 3, 512, 512), (64, 3, 256, 512), (128, 3, 320, 180)]
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        for device in devices:
            for shape in shapes:
                for dtype in dtypes:
                    image = torch.randn(shape, dtype=dtype).to(device)
                    hunyuandit_patchembedder = HunyuanDitPatchEmbed().to(device)
                    patchembedder = PatchEmbed(pos_embed_type=None).to(device)

                    para_dict = hunyuandit_patchembedder.state_dict()
                    patchembedder.load_state_dict(para_dict)

                    embedding_test = hunyuandit_patchembedder(image)
                    embedding = patchembedder(image)
                    self.assertEqual(embedding.shape, embedding_test.shape)

                    embedding_test = embedding_test.reshape(1, -1).to(torch.float32)
                    embedding = embedding.reshape(1, -1).to(torch.float32)
                    result, _, max_err = data_compare(embedding.cpu(), embedding_test.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

    @torch.no_grad()
    def test_patch_embed_sd3(self):
        devices = ["npu", "cpu"]
        shapes = [(32, 3, 224, 224), (16, 3, 512, 512), (64, 3, 256, 512), (128, 3, 320, 180)]
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        sizes = [(224, 224), (256, 256), (512, 512)]
        for device in devices:
            for shape in shapes:
                for dtype in dtypes:
                    for height, width in sizes:
                        image = torch.randn(shape, dtype=dtype).to(device)
                        test_patchembedder = SD3PatchEmbed(height, width).to(device)
                        patchembedder = PatchEmbed(height, width).to(device)

                        para_dict = test_patchembedder.state_dict()
                        patchembedder.load_state_dict(para_dict)

                        embedding_test = test_patchembedder(image)
                        embedding = patchembedder(image)
                        self.assertEqual(embedding.shape, embedding_test.shape)

                        embedding_test = embedding_test.reshape(1, -1).to(torch.float32)
                        embedding = embedding.reshape(1, -1).to(torch.float32)
                        result, _, max_err = data_compare(embedding.cpu(), embedding_test.cpu())
                        self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")

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
                    rotary_pos_emb_test = (rotary_pos_emb_test[0].to(dtype).to(device),
                                           rotary_pos_emb_test[1].to(dtype).to(device))

                    rope = RotaryPositionEmbedding(embed_dim=dim)
                    rotary_pos_emb = rope.get_2d_rotary_pos_embed(grid_height, grid_width, base_size)
                    rotary_pos_emb = (rotary_pos_emb[0].to(dtype).to(device),
                                      rotary_pos_emb[1].to(dtype).to(device))

                    embedding_test = rope_test(hidden_states, rotary_pos_emb_test)
                    embedding = rope(hidden_states, rotary_pos_emb, rotated_mode="rotated_interleaved",
                                     head_first=True, fused=True)
                    self.assertEqual(embedding.shape, embedding_test.shape)

                    embedding_test = embedding_test.reshape(1, -1).to(torch.float32)
                    embedding = embedding.reshape(1, -1).to(torch.float32)
                    result, _, max_err = data_compare(embedding.cpu(), embedding_test.cpu())
                    self.assertEqual(result, "success", msg=f"Data compare failed. Max error is: {max_err}")


if __name__ == '__main__':
    unittest.main()
