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

import functools
import math
from typing import Literal, Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, broadcast_tensors, Tensor
from einops import rearrange
import numpy as np

from .mlp import Mlp
from mindiesd.layers.rope import rotary_position_embedding
from mindiesd.utils import ParametersInvalid, ModelExecError, ModelInitError

LANG_FREQS = 'lang'
PIXEL_FREQS = 'pixel'
CONSTANT_FREQS = 'constant'


def get_embedding_helper(embedding_type: str, embdding_dim: int):
    match embedding_type:
        case None:
            return nn.Identity()
        case 'rope':
            return ReconstitutionRotaryEmbedding(dim=embdding_dim)
        case _:
            raise ParametersInvalid(f"Unsupported embedding_type:{embedding_type}. "
                                    "The supported embedding_type must be None or 'rope'")


class PatchEmbed3D(nn.Module):
    def __init__(
            self,
            patch_size=(2, 4, 4),
            in_chans=3,
            embed_dim=96,
            norm_layer=None,
            flatten=True,
    ):
        """
        Video to Patch Embedding.
        Args:
            patch_size (tuple[int, int, int]): Patch token size. Default: (2,4,4).
            in_chans (int): Number of input video channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None.
        Adapted Models: Open-Sora
        """

        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # padding
        _, _, x_shape2_d, x_shape3_h, x_shape4_w = x.size()
        re_w = x_shape4_w % self.patch_size[2]
        if re_w != 0:
            x = F.pad(x, (0, self.patch_size[2] - re_w))
        re_h = x_shape3_h % self.patch_size[1]
        if re_h != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - re_h))
        re_d = x_shape2_d % self.patch_size[0]
        if re_d != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - re_d))

        # embedding
        # (B, in_chans, D, H, W) -> (B, embed_dim, D//patch_size[0], H//patch_size[1], W//patch_size[2])
        x = self.proj(x)
        if self.norm is not None:
            x_size_d, x_size_h, x_size_w = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, x_size_d, x_size_h, x_size_w)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, embed_dim, x_size_d, x_size_h, x_size_w) -> (B, N, embed_dim)
        return x


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        bias=True,
    ):
        """
        2D Image to Patch Embedding but with video.
        Args:
            patch_size (int): Patch token size. Default: 16.
            in_chans (int): Number of input video channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 768.
            bias (bool): If true, use bias.
        Adapted Models: Open-Sora-Plan
        """

        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=bias
        )

    def forward(self, x):
        b, _, _, _, _ = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b (t h w) c', b=b)
        return x


def cal_1d_sincos_embed(
        items: torch.Tensor,
        embed_dim: int,
        max_period: int = 10000,
        step: int = 1,
        flip: bool = False
    ):
    """
    Calculate 1d sinusoidal embeddings.
    Args:
        items (torch.Tensor): Items includes N indices. Must be a 1D tensor (N,).
        embed_dim (int): The dimension of the embeddings.
        max_period (int): Controls the minimum frequency of the embeddings.
        step (int): The step of frequences.
        flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
    Return:
        embed (torch.Tensor): An (N, embed_dim//step) tensor of item embeddings.
    """

    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ParametersInvalid(f"Embed_dim should be a positive integer, but got {embed_dim}.")
    if step not in [1, 2]:
        raise ParametersInvalid(f"The value of step must be in [1, 2], but got {step}.")
    if embed_dim % (2 * step) != 0:
        raise ParametersInvalid(f"Embed_dim must be divisible by {2 * step}, but got {embed_dim}.")

    half_of_dim = embed_dim // 2
    # generate frequency vectors
    freqs = torch.arange(start=0, end=half_of_dim, step=step, dtype=torch.float32, device=items.device)
    freqs = torch.exp(-math.log(max_period) * freqs / half_of_dim)  # (embed_dim//(2*step))
    # (N, 1) * (1, embed_dim//(2*step)) -> (N, embed_dim//(2*step))
    freqs = items[:, None].float() * freqs[None, :]
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    # (N, embed_dim//step)
    if flip:
        embed = torch.cat([cos, sin], dim=-1)
    else:
        embed = torch.cat([sin, cos], dim=-1)

    return embed


class SinCosPositionEmbed1D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        step: int = 1,
        flip: bool = False,
        max_period: int = 10000,
        cache1d: bool = True,
        size: int = 128
    ):
        """
        Create 1d sinusoidal embeddings.
        Args:
            embed_dim (int): The dimension of the embeddings.
            step (int): The step of frequences.
            flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
            max_period (int): Controls the minimum frequency of the embeddings.
            cache1d (bool): If true, use cache.
            size (int): The size of cache.
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.step = step
        self.flip = flip
        self.max_period = max_period
        self.cache1d = cache1d
        self.size = size
        if self.cache1d:
            items = torch.arange(self.size)
            # (size, embed_dim//step)
            embed = cal_1d_sincos_embed(items, self.embed_dim, self.max_period, self.step, self.flip)
            self.register_buffer("embed", embed, persistent=False)
        else:
            self.embed = None

    def get_1d_sincos_embed(self, items: torch.Tensor):
        """
        Calculate 1d sinusoidal embeddings.
        Args:
            items (torch.Tensor): Items includes N indices. Must be a 1D tensor (N,).
        Return:
            embed (torch.Tensor): An (N, embed_dim//step) tensor of item embeddings.
        """

        if len(items.shape) != 1:
            raise ParametersInvalid(f"Items should be a 1D tensor, but got a {len(items.shape)}D tensor.")

        items_max = torch.max(items)
        dytpes = [torch.int, torch.long]
        if self.cache1d and items_max < self.size and items.dtype in dytpes:
            embed = self.embed[items]
        else:
            embed = cal_1d_sincos_embed(items, self.embed_dim, self.max_period, self.step, self.flip)

        return embed


class TimestepEmbedder(SinCosPositionEmbed1D):
    def __init__(self, hidden_size, frequency_embedding_size=256, flip=True, cache1d=True, size=128):
        """
        Embeds scalar timesteps into vector representations.
        Args:
            hidden_size (int): Number of linear projection output channels.
            frequency_embedding_size (int): Number of frequency embedding size. Default: 256.
            flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
            cache1d (bool): If true, use cache.
            size (int): The size of cache.
        Adapted Models: Open-Sora, HunyuanDit, SD3
        """

        super().__init__(frequency_embedding_size, flip=flip, cache1d=cache1d, size=size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t, dtype):
        t_freq = self.get_1d_sincos_embed(t)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SizeEmbedder(SinCosPositionEmbed1D):
    def __init__(self, hidden_size, frequency_embedding_size=256, flip=True, cache1d=True, size=128):
        """
        Embeds scalar size into vector representations.
        Args:
            hidden_size (int): Number of linear projection output channels.
            frequency_embedding_size (int): Number of frequency embedding size. Default: 256.
            flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
            cache1d (bool): If true, use cache.
            size (int): The size of cache.
        Adapted Models: Open-Sora
        """

        super().__init__(frequency_embedding_size, flip=flip, cache1d=cache1d, size=size)
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
        if s.ndim != 2:
            raise ModelExecError(f"The dimension of s should be 2, but got {s.ndim}")
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            if s.shape[0] != bs:
                raise ModelExecError(f"The first dimension of the input s must be equal to bs,"
                    f"but got {s.shape[0]} and {bs}")
        b, dims = s.shape[0], s.shape[1]
        s = s.reshape(b * dims)
        s_freq = self.get_1d_sincos_embed(s).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = s_emb.view(b, dims, self.outdim)
        s_emb = s_emb.view(b, dims * self.outdim)
        return s_emb


class CombinedTimestepTextProjEmbeddings(SinCosPositionEmbed1D):
    def __init__(self, embedding_dim, pooled_projection_dim, flip=True, cache1d=True, size=128):
        """
        Args:
            embedding_dim (int): Number of frequency embedding size.
            pooled_projection_dim (int): Number of pooled projection channels.
            flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
            cache1d (bool): If true, use cache.
            size (int): The size of cache.
        Adapted Models: SD3
        """

        super().__init__(embed_dim=256, flip=flip, cache1d=cache1d, size=size)

        self.timestep_embedder = Mlp(
            features_in=256,
            features_hidden=embedding_dim,
            features_out=embedding_dim,
            act_layer=nn.SiLU,
        )
        self.text_embedder = Mlp(
            features_in=pooled_projection_dim,
            features_hidden=embedding_dim,
            features_out=embedding_dim,
            act_layer=nn.SiLU,
        )

    def forward(self, timestep, pooled_projection):
        embedding = self.get_1d_sincos_embed(timestep)
        timesteps_emb = self.timestep_embedder(embedding.to(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = timesteps_emb + pooled_projections

        return conditioning


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
            self,
            in_channels,
            hidden_size,
            act_layer=nn.GELU(approximate="tanh"),
            token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            features_in=in_channels,
            features_hidden=hidden_size,
            features_out=hidden_size,
            act_layer=act_layer,
        )

        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels ** 0.5,
        )

    def forward(self, caption):
        caption = self.y_proj(caption)
        return caption


class SinCosPositionEmbed2D(SinCosPositionEmbed1D):
    def __init__(
        self,
        embed_dim: int = 256,
        step: int = 1,
        flip: bool = False,
        max_period: int = 10000,
        cache2d: bool = True,
        grid_size: Union[Tuple[int, int], int] = (224, 224),
        base_size: Union[int, None] = None,
        interpolation_scale: float = 1.0,
        persistent = False,
    ):
        """
        Create 2d sinusoidal embeddings.
        Args:
            embed_dim (int): The dimension of the embeddings.
            step (int): The step of frequences.
            flip (bool): If true, return [cos, cos, ..., sin, sin], else return [sin, sin ..., cos, cos].
            max_period (int): Controls the minimum frequency of the embeddings.
            cache2d (bood): If true, use cache
            grid_size (Tuple[int, int] or int): The size of grid.
            base_size (int or None): The size of basic patches.
            interpolation_scale (float): The scale parameter.
            persistent (bool): If true, save the cache in dict.
        """

        self.embed_dim = embed_dim
        self.step = step
        self.flip = flip
        self.max_period = max_period
        self.cache2d = cache2d
        self.interpolation_scale = interpolation_scale

        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size)
        else:
            self.grid_size = grid_size
        if base_size is None:
            self.base_size = round((self.grid_size[0] * self.grid_size[1]) ** 0.5)
        else:
            self.base_size = base_size

        if not isinstance(self.embed_dim, int) or self.embed_dim <= 0:
            raise ParametersInvalid(f"Embed_dim should be a positive integer, but got {self.embed_dim}.")
        if self.step not in [1, 2]:
            raise ParametersInvalid(f"The value of step must be in [1, 2], but got {self.step}.")
        if self.embed_dim % (2 * self.step) != 0:
            raise ParametersInvalid(f"Embed_dim must be divisible by {2 * self.step}, but got {self.embed_dim}.")

        self.dim = self.embed_dim // (2 // self.step)
        super().__init__(self.dim, self.step, self.flip, self.max_period, cache1d=False)

        if self.cache2d:
            pos_embed = self._get_2d_sincos_embed(self.grid_size, self.base_size, self.interpolation_scale)
            self.register_buffer("pos_embed", pos_embed, persistent=persistent)
        else:
            self.pos_embed = None

    def get_2d_sincos_embed(self, grid_size, base_size=None, interpolation_scale=1.0, device="cpu"):
        """
        Initialize frequences.
        Args:
            grid_size (Tuple[int, int] or int): The size of grid.
            base_size (int or None): The size of basic patches.
            interpolation_scale (float): The scale parameter.
        Return:
            emb (torch.Tensor): An (1, H*W, embed_dim) tensor of embeddings.
        """

        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)

        is_shape_same = grid_size[0] == self.grid_size[0] and grid_size[1] == self.grid_size[1] \
            and base_size == self.base_size
        if self.cache2d and is_shape_same and interpolation_scale == self.interpolation_scale:
            embed = self.pos_embed
        else:
            embed = self._get_2d_sincos_embed(grid_size, base_size, interpolation_scale, device)

        return embed

    @functools.lru_cache(maxsize=512)
    def _get_2d_sincos_embed(self, grid_size, base_size, interpolation_scale, device="cpu"):
        """
        Initialize frequences.
        Args:
            grid_size (Tuple[int, int]): The size of grid.
            base_size (int or None): The size of basic patches.
            interpolation_scale (float): The scale parameter.
        Return:
            emb (torch.Tensor): An (H*W, embed_dim) tensor of embeddings.
        """

        grid_h = torch.arange(grid_size[0], dtype=torch.float32, device=device) / interpolation_scale
        grid_w = torch.arange(grid_size[1], dtype=torch.float32, device=device) / interpolation_scale

        if base_size is not None:
            grid_h *= base_size / grid_size[0]
            grid_w *= base_size / grid_size[1]

        grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
        grid = torch.stack([grid_h.t().reshape(-1), grid_w.t().reshape(-1)], dim=0)  # (2, H*W)
        emb_h = self.get_1d_sincos_embed(grid[0])  # (H*W, embed_dim//2)
        emb_w = self.get_1d_sincos_embed(grid[1])  # (H*W, embed_dim//2)
        emb = torch.cat([emb_h, emb_w], dim=-1)  # (H*W, embed_dim)
        return emb


class PositionEmbedding2D(SinCosPositionEmbed2D):
    def __init__(self, dim: int):
        """
        Embeds position of 2D images into vector representations.
        Args:
            dim (int): The dimension of embedding.
        Adapted Models: Open-Sora
        """

        if dim % 4 != 0:
            raise ParametersInvalid(f"Input dim must be divisible by 4, but got {dim}.")

        super().__init__(embed_dim=dim, step=2)

    def forward(self, x: torch.Tensor, h: int, w: int, scale: Optional[float]=1.0):
        grid_size = (h, w)
        base_size = round((h * w) ** 0.5)
        embed = self.get_2d_sincos_embed(grid_size, base_size, scale, x.device)
        return embed.unsqueeze(0).to(x.dtype)


class PatchEmbed(SinCosPositionEmbed2D):
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
        """
        2D Image to Patch Embedding with support for position embedding.
        Args:
            height (int): Height of images.
            width (int): Weight of images.
            patch_size (int): The size of patches.
            in_channels (int): Number of input image channels.
            embed_dim (int): Number of linear projection output channels.
            layer_norm (bool): If true, use layernorm.
            flatten (bool): If true, flatten the latent.
            bias (bool): If true, use bias.
            interpolation_scale: Scale coefficient.
            pos_embed_type (str): The type of postion embedding.
            pos_embed_max_size: The size of max postion embedding.
        Adapted Models: SD3, HuanyuanDit
        """

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size
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
            self.cache2d = False
        elif pos_embed_type == "sincos":
            self.cache2d = True
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

        super().__init__(
            embed_dim=embed_dim,
            step=1,
            cache2d=self.cache2d,
            grid_size=grid_size,
            base_size=self.base_size,
            interpolation_scale=self.interpolation_scale,
            persistent=True if pos_embed_max_size else False,
        )

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ParametersInvalid(f"Parameter:`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ParametersInvalid(
                f"The value of height ({height}) cannot be > `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ParametersInvalid(
                f"The value of width ({width}) cannot be > `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        dtype_latent = latent.dtype
        latent = self.proj(latent.to(self.dtype))
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(dtype_latent)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            pos_embed = self.get_2d_sincos_embed(
                (height, width),
                self.base_size,
                interpolation_scale=self.interpolation_scale,
                device=latent.device
            ).unsqueeze(0)

        return (latent + pos_embed).to(dtype_latent)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class RotaryEmbedding(nn.Module):
    def __init__(self,
                 dim,
                 custom_freqs: Optional[Tensor] = None,
                 freqs_for: Union[
                     Literal[LANG_FREQS],
                     Literal[PIXEL_FREQS],
                     Literal[CONSTANT_FREQS]
                 ] = LANG_FREQS,
                 theta=10000,
                 max_freq=10,
                 num_freqs=1,
                 learned_freq=False,
                 xpos_scale_base=512,
                 interpolate_factor=1.,
                 theta_rescale_factor=1.,
                 seq_before_head_dim=False,
                 cache_if_possible=True
                 ):
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ModelInitError(f"Input freqs_for: {freqs_for} is unsupported.")

        self.cache_if_possible = cache_if_possible

        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store('dummy', torch.tensor(0))

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors
        if interpolate_factor < 1.:
            raise ParametersInvalid(
                f"The value of input interpolate_factor must be >= 1, but got {interpolate_factor}.")

        self.interpolate_factor = interpolate_factor

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rearrange_nd_2_n1d(self, x, transform_type='n d -> n 1 d'):
        if transform_type == 'n d -> n 1 d':
            shape = x.shape
            x = x.view(shape[0], shape[1])
            return x.view(shape[0], 1, shape[1])
        return x

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, freq_seq_len=None):
        # 进入这个函数
        seq_dim = default(seq_dim, self.default_seq_dim)

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        if exists(freq_seq_len):
            if freq_seq_len < seq_len:
                raise ModelExecError(
                    f"The value of input freq_seq_len:{freq_seq_len} must be >= seq_len:{seq_len}.")

            seq_len = freq_seq_len

        freqs = self.forward(self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset), seq_len=seq_len,
                             offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return self.apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def get_axial_freqs(self, *dims):
        colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = colon

            new_axis_slice = (Ellipsis, *all_axis, colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    def rotate_half(self, x):
        shape = x.shape
        new_shape = shape[:-1] + (shape[-1] // 2, 2)
        x = x.view(new_shape)

        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        shape = x.shape
        new_shape = shape[:-2] + (shape[-1] * shape[-2],)
        x = x.view(new_shape)
        return x

    def apply_rotary_emb(self, freqs, t, start_index=0, scale=1., seq_dim=-2):
        if t.ndim == 3:
            seq_len = t.shape[seq_dim]
            freqs = freqs[-seq_len:].to(t)

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        if rot_dim > t.shape[-1]:
            raise ModelExecError(
                f"Feature dimension:{t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}.")

        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

        cos = freqs.cos() * scale
        sin = freqs.sin() * scale
        t = (t * cos) + (self.rotate_half(t) * sin)

        return torch.cat((t_left, t, t_right), dim=-1)

    def forward(
            self,
            t: Tensor,
            seq_len=None,
            offset=0
    ):
        should_cache = (
                self.cache_if_possible and \
                not self.learned_freq and \
                exists(seq_len) and \
                self.freqs_for != 'pixel'
        )

        if (
                should_cache and \
                exists(self.cached_freqs) and \
                (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        return freqs


# rope重构类，用来替换RotaryEmbedding功能
# 1.简化Rope为最原简洁的rope功能，如果有新功能如cache再添加新接口
# 2.修改原始rotate_queries_or_keys接口为forward接口，原始foward接口改为get_freqs
class ReconstitutionRotaryEmbedding:
    def __init__(
        self,
        dim: int,
        theta: int = 10000,
        interpolate_factor: float = 1.0,
        theta_rescale_factor: float = 1.0,
        seq_before_head_dim=False,
    ):
        super().__init__()
        # check inputs
        if dim <= 2:
            raise ParametersInvalid(f"The value of input dim must be > 2, but got {dim}.")
        if theta <= 0.:
            raise ParametersInvalid(f"The value of input theta must be > 0, but got {theta}.")
        if interpolate_factor < 1.:
            raise ParametersInvalid(
                f"The value of input interpolate_factor must be >= 1, but got {interpolate_factor}.")
        if theta_rescale_factor <= 0.:
            raise ParametersInvalid(
                f"The value of input theta_rescale_factor must be > 0, but got {theta_rescale_factor}.")

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

        # default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        self.interpolate_factor = interpolate_factor

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_half(self, x):
        shape = x.shape
        new_shape = shape[:-1] + (shape[-1] // 2, 2)
        x = x.view(new_shape)

        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        shape = x.shape
        new_shape = shape[:-2] + (shape[-1] * shape[-2],)
        x = x.view(new_shape)
        return x

    def apply_rotary_emb(self, freqs, t, start_index=0, scale=1., seq_dim=-2):
        freqs_cos = freqs[0]
        freqs_sin = freqs[1]
        if t.ndim == 3:
            seq_len = t.shape[seq_dim]
            freqs_cos = freqs_cos[-seq_len:].to(t)
            freqs_sin = freqs_sin[-seq_len:].to(t)

        rot_dim = freqs_cos.shape[-1]
        end_index = start_index + rot_dim

        if rot_dim > t.shape[-1]:
            raise ModelExecError(
                f"Feature dimension:{t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}.")

        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

        cos = freqs_cos * scale
        sin = freqs_sin * scale
        t = (t * cos) + (self.rotate_half(t) * sin)

        return torch.cat((t_left, t, t_right), dim=-1)

    def get_freqs(
            self,
            t: Tensor,
            seq_len=None,
            offset=0
    ):
        freqs = self.freqs
        freqs = freqs.to(t.device).to(t.dtype)
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        return freqs

    def __call__(self, t, freqs=None, seq_dim=None, offset=0):
        # 进入这个函数
        seq_dim = default(seq_dim, self.default_seq_dim)

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        if freqs is None:
            freqs = self.get_freqs(
                self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset),
                seq_len=seq_len,
                offset=offset
            )
            freqs = (freqs.cos(), freqs.sin())

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return self.apply_rotary_emb(freqs, t, seq_dim=seq_dim)


class RotaryCosSinEmbed:
    """
    RotaryCosSinEmbed get cos_sin tables of rope.
    """
    def __init__(
        self,
        embed_dim: int,
        use_real: bool = True,
        repeat_interleave_real: bool = True,
        theta: float = 10000.0,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        freqs_dtype = torch.float32,
    ):
        """
        Args:
        embed_dim (int): The embedding dimension size.
        use_real (bool): If `True`, return real part and imaginary part separately. Otherwise, return complex numbers.
        repeat_interleave_real (bool):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        linear_factor (float): Scaling factor for the context extrapolation. Defaults to 1.0. Use for `lumina`.
        ntk_factor (float): Scaling factor for the NTK-Aware RoPE. Defaults to 1.0. Use for `lumina`.
        freqs_dtype: Defaults to torch.float32. Only be torch.float64 for Flux.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.use_real = use_real
        self.repeat_interleave_real = repeat_interleave_real
        self.theta = theta
        self.linear_factor = linear_factor    # Use for lumina.
        self.ntk_factor = ntk_factor          # Use for lumina.
        self.freqs_dtype = freqs_dtype        # Flux: torch.float64


    def get_resize_crop_region_for_grid(self, src_h: int, src_w: int, base_size: int):
        """
        Get resize and crop region for grid.

        Args:
            src_h (int): The grid height of the positional embedding.
            src_w (int): The grid width of the positional embedding.
            base_size (int): The target size of resizing and cropping region for grid.

        Returns:
            Tuple[int]: The top-left and bottom-right coordinates of the crop.
        """
        if not isinstance(src_h, int):
            raise ParametersInvalid(f"The data type of input src_h must be int, but got {type(src_h)}.")
        if not isinstance(src_w, int):
            raise ParametersInvalid(f"The data type of input src_w must be int, but got {type(src_w)}.")
        if not isinstance(base_size, int):
            raise ParametersInvalid(f"The data type of input base_size must be int, but got {type(base_size)}.")
        if src_h <= 0:
            raise ParametersInvalid(f"The value of input src_h must be > 0, but got {src_h}.")
        if src_w <= 0:
            raise ParametersInvalid(f"The value of input src_w must be > 0, but got {src_w}.")
        if base_size <= 0:
            raise ParametersInvalid(f"The value of input base_size must be > 0, but got {base_size}.")

        ratio = src_h / src_w
        # resize
        if ratio > 1:
            resize_height = base_size
            resize_width = int(round(base_size / src_h * src_w))
        else:
            resize_width = base_size
            resize_height = int(round(base_size / src_w * src_h))
        crop_top = int(round((base_size - resize_height) / 2.0))
        crop_left = int(round((base_size - resize_width) / 2.0))
        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


    def get_1d_rotary_pos_embed(self, pos: Union[np.ndarray, int]) -> torch.Tensor:
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        Args:
            pos (np.ndarray or int): Position indices for the frequency tensor. [S] or scalar.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex exponentials. [S, D/2].
        """
        if isinstance(pos, int):
            pos = torch.arange(pos)
        elif isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)  # type: ignore  # [S]
        else:
            raise ParametersInvalid(f"The data type of input pos must be np.ndarray or int, but got {type(pos)}.")

        half_of_dim = self.embed_dim // 2

        theta = self.theta * self.ntk_factor
        freqs = torch.arange(start=0, end=half_of_dim, step=2, dtype=self.freqs_dtype, device=pos.device)  # [D/4]
        freqs = (1.0 / (theta ** (freqs[: (half_of_dim // 2)] / half_of_dim)) / self.linear_factor)  # [D/4]
        freqs = torch.outer(pos, freqs)  # [S, D/4]

        if self.use_real and self.repeat_interleave_real:
            # HunyuanDiT, Flux, CogVideox
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D/2]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D/2]
            return freqs_cos, freqs_sin
        elif self.use_real:
            # Stable Audio, Allegro
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [S, D/2]
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [S, D/2]
            return freqs_cos, freqs_sin
        else:
            # lumina
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/4]
            return freqs_cis


    def get_2d_rotary_pos_embed(self, grid_h: int, grid_w: int, base_size: int):
        """
        RoPE for image tokens with 2d structure.

        Args:
            grid_h (int): The grid height of the positional embedding.
            grid_w (int): The grid width of the positional embedding.
            base_size (int): The target size of resizing and cropping region for grid.

        Returns:
            torch.Tensor: positional embedding with shape (grid_size * grid_size, embed_dim/2).
        """
        if not isinstance(grid_h, int):
            raise ParametersInvalid(f"The data type of input grid_h must be int, but got {type(grid_h)}.")
        if not isinstance(grid_w, int):
            raise ParametersInvalid(f"The data type of input grid_w must be int, but got {type(grid_w)}.")
        if not isinstance(base_size, int):
            raise ParametersInvalid(f"The data type of input base_size must be int, but got {type(base_size)}.")
        if grid_h <= 0:
            raise ParametersInvalid(f"The value of input grid_h must be > 0, but got {grid_h}.")
        if grid_w <= 0:
            raise ParametersInvalid(f"The value of input grid_w must be > 0, but got {grid_w}.")
        if base_size <= 0:
            raise ParametersInvalid(f"The value of input base_size must be > 0, but got {base_size}.")

        start, stop = self.get_resize_crop_region_for_grid(grid_h, grid_w, base_size)
        grid_h = np.linspace(start[0], stop[0], grid_h, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(start[1], stop[1], grid_w, endpoint=False, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)  # [2, W, H]

        grid = grid.reshape([2, 1, *grid.shape[1:]])
        # use half of dimensions to encode grid_h and grid_w
        emb_h = self.get_1d_rotary_pos_embed(grid[0].reshape(-1))  # (H*W, D/2) if use_real else (H*W, D/4)
        emb_w = self.get_1d_rotary_pos_embed(grid[1].reshape(-1))  # (H*W, D/2) if use_real else (H*W, D/4)

        if self.use_real:
            cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D)
            sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D)
            pos_embed = (cos, sin)
        else:
            pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)

        return pos_embed


class RotaryPositionEmbedding(RotaryCosSinEmbed, nn.Module):
    """
    RotaryPositionEmbedding apply rotary embeddings to input tensors using the given frequency tensor.
    """
    def __init__(
        self,
        embed_dim: int,
        grid_h: int = 64,
        grid_w: int = 64,
        base_size: int = 32,
        use_real: bool = True,
        repeat_interleave_real: bool = True,
        theta: float = 10000.0,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
    ):
        """
        Args:
        embed_dim (int): The embedding dimension size.
        grid_h (int): The grid height of the positional embedding.
        grid_w (int): The grid width of the positional embedding.
        base_size (int): The target size of resizing and cropping region for grid.
        use_real (bool): If `True`, return real part and imaginary part separately. Otherwise, return complex numbers.
        repeat_interleave_real (bool):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        linear_factor (float): Scaling factor for the context extrapolation. Defaults to 1.0. Use for `lumina`.
        ntk_factor (float): Scaling factor for the NTK-Aware RoPE. Defaults to 1.0. Use for `lumina`.
        """
        # check inputs
        if embed_dim % 4 != 0 or embed_dim <= 2:
            raise ParametersInvalid(f"The value of input embed_dim must be divisible by 4 and > 2, but got {embed_dim}.")
        if grid_h <= 0 or grid_w <= 0:
            raise ParametersInvalid(f"The value of input grid_size must be > 0, but got ({grid_h}, {grid_w}).")
        if base_size <= 0:
            raise ParametersInvalid(f"The value of input base_size must be > 0, but got {base_size}.")
        if theta <= 0.:
            raise ParametersInvalid(f"The value of input theta must be > 0, but got {theta}.")
        if linear_factor <= 0.:
            raise ParametersInvalid(f"The value of input linear_factor must be > 0, but got {linear_factor}.")
        if ntk_factor <= 0.:
            raise ParametersInvalid(f"The value of input ntk_factor must be > 0, but got {ntk_factor}.")

        self.use_real = use_real
        super().__init__(embed_dim, use_real, repeat_interleave_real, theta, linear_factor, ntk_factor)

        self.freqs_cis_img = self.get_2d_rotary_pos_embed(grid_h, grid_w, base_size)


    def reshape_for_broadcast(self, x, cos, sin, head_first):
        ndim = x.ndim
        if head_first:
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return cos.view(*shape), sin.view(*shape)


    def forward(self,
                x: torch.Tensor,
                freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
                rotated_mode: str = "rotated_half",
                head_first: bool = False,
                fused: bool = True) -> torch.Tensor:
        """
        The input tensors are reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting
        compatibility. The resulting tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`): Query or key tensor to apply rotary embeddings. [B, H, S, D].
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        freqs_cis = freqs_cis if freqs_cis is not None else self.freqs_cis_img

        if self.use_real:
            cos, sin = freqs_cis  # [S, D]
            cos, sin = cos.to(x.device), sin.to(x.device)
            cos, sin = self.reshape_for_broadcast(x, cos, sin, head_first)
            x_out = rotary_position_embedding(x, cos, sin, rotated_mode, head_first, fused)
            return x_out
        else:
            # used for lumina
            x_rotated = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
            return x_out.type_as(x)
