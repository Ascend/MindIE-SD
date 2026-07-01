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
from fnmatch import fnmatch as std_fnmatch
from typing import Dict, Optional
import contextvars

import torch
from inspect import signature, Parameter
from torch import distributed as dist
from ..utils import ParametersInvalid, ConfigError
from ..utils.logs.logging import logger

MAX_WEIGHT_SIZE = 100 * 1024 * 1024 * 1024  # 工具对SD量化还不能分片保存

try:
    from wcmatch import fnmatch as wc_fnmatch
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    wc_fnmatch = None

MXFP4_DST_TYPE_MAX_C7 = 7.25
ONLINE_FA_ROT_SEED = 1234


def _expand_brace_pattern(pattern):
    start = pattern.find('{')
    if start == -1:
        return [pattern]
    end = pattern.find('}', start + 1)
    if end == -1:
        return [pattern]
    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    expanded = []
    for item in pattern[start + 1 : end].split(','):
        for tail in _expand_brace_pattern(suffix):
            expanded.append(f"{prefix}{item}{tail}")
    return expanded


def match_layer_pattern(layer_name, pattern):
    if layer_name == pattern:
        return True
    if wc_fnmatch is not None:
        return wc_fnmatch.fnmatch(layer_name, pattern, flags=wc_fnmatch.BRACE)
    return any(std_fnmatch(layer_name, expanded) for expanded in _expand_brace_pattern(pattern))


def match_layer_config(layer_name, layer_config):
    if not layer_config:
        return None
    if layer_name in layer_config:
        return layer_config[layer_name]
    for pattern, value in layer_config.items():
        if match_layer_pattern(layer_name, pattern):
            return value
    return None


def match_layer_patterns(layer_name, layer_patterns):
    if not layer_patterns:
        return False
    return any(match_layer_pattern(layer_name, pattern) for pattern in layer_patterns)


def match_fa_layer(name, layer, fa_layers):
    if not fa_layers:
        return False
    class_name = layer.__class__.__name__
    candidates = (name, class_name, f'{layer.__class__.__module__}.{class_name}')
    return any(match_layer_patterns(candidate, fa_layers) for candidate in candidates if candidate)


def get_mxfp4_quant_kwargs(quant_config=None):
    if quant_config is None or getattr(quant_config, 'mxfp4_scale_alg', None) is None:
        return {}
    return {
        'scale_alg': quant_config.mxfp4_scale_alg,
        'dst_type_max': getattr(quant_config, 'mxfp4_dst_type_max', MXFP4_DST_TYPE_MAX_C7),
    }


class OnlineFARotWeights:
    def __init__(self, weights: Dict[str, torch.Tensor]):
        self._weights = weights

    def keys(self):
        return self._weights.keys()

    def get_tensor(self, key):
        return self._weights[key]


def _is_power_of_two(value):
    return isinstance(value, int) and value > 0 and (value & (value - 1) == 0)


def _walsh_matrix(size, dtype, device):
    if size == 1:
        return torch.tensor([[1.0]], dtype=dtype, device=device)
    had = _walsh_matrix(size // 2, dtype, device)
    return torch.cat([torch.cat([had, had], dim=1), torch.cat([had, -had], dim=1)], dim=0)


def _matmul_had_u(x):
    n = x.shape[-1]
    had = _walsh_matrix(n, x.dtype, x.device)
    return torch.matmul(x, had) / torch.tensor(n, dtype=x.dtype, device=x.device).sqrt()


def create_online_hadamard_rot(size, dtype=torch.float32, seed=ONLINE_FA_ROT_SEED, device=None):
    if not _is_power_of_two(size):
        raise ParametersInvalid(f"Online FA rot only supports power-of-two head_dim, but got {size}.")
    device = torch.device('cpu') if device is None else torch.device(device)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64, device='cpu').to(torch.float32)
    signs = signs * 2 - 1
    rot = _matmul_had_u(torch.diag(signs))
    return rot.to(device=device, dtype=dtype)


def infer_attention_head_dim(layer):
    head_dim = getattr(layer, 'head_dim', None)
    if isinstance(head_dim, int) and head_dim > 0:
        return head_dim
    hidden_size = (
        getattr(layer, 'hidden_size', None) or getattr(layer, 'inner_dim', None) or getattr(layer, 'dim', None)
    )
    num_heads = (
        getattr(layer, 'num_heads', None)
        or getattr(layer, 'heads', None)
        or getattr(layer, 'num_attention_heads', None)
    )
    if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0 and hidden_size % num_heads == 0:
        return hidden_size // num_heads
    for proj_name in ('q_proj', 'to_q', 'q'):
        proj = getattr(layer, proj_name, None)
        out_features = getattr(proj, 'out_features', None)
        if (
            isinstance(out_features, int)
            and isinstance(num_heads, int)
            and num_heads > 0
            and out_features % num_heads == 0
        ):
            return out_features // num_heads
    return None


def _get_module_device(layer):
    for tensor in list(layer.parameters()) + list(layer.buffers()):
        return tensor.device
    return torch.device('cpu')


def build_online_fa_rot_weights(prefix, layer, dtype=torch.bfloat16, seed=ONLINE_FA_ROT_SEED):
    head_dim = infer_attention_head_dim(layer)
    if head_dim is None:
        raise ParametersInvalid(f"Cannot infer head_dim for online FA layer {prefix}.")
    rot = create_online_hadamard_rot(head_dim, dtype=dtype, seed=seed, device=_get_module_device(layer))
    return OnlineFARotWeights(
        {
            f'{prefix}.q_rot': rot,
            f'{prefix}.k_rot': rot.clone(),
        }
    )


def extract_constructor_args(instance, base_class=None):
    cls = instance.__class__
    init_params = signature(cls.__init__).parameters
    if not init_params:
        raise ParametersInvalid("init_params is none!")
    param_names = [k for k, v in init_params.items() if v.kind == Parameter.POSITIONAL_OR_KEYWORD and k != 'self']

    if base_class:
        base_params = signature(base_class.__init__).parameters
        base_param_names = {k for k in base_params if k != 'self'}
        param_names = [n for n in param_names if n in base_param_names]

    return {name: getattr(instance, name) for name in param_names if hasattr(instance, name)}


def replace_rank_suffix(file_path):
    # 分离目录路径和文件名（处理多级目录）
    dir_path, filename = os.path.split(file_path)
    # 分离主文件名和扩展名（如 .json）
    basename, ext = os.path.splitext(filename)

    # 按最后一个下划线分割文件名
    parts = basename.rsplit('_', 1)

    rank = -1

    # 检查后缀是否为数字
    if len(parts) > 1 and parts[-1].isdigit():
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            raise ConfigError(f"must init distributed env if use distributed config {filename}")
        new_basename = f"{parts[0]}_{rank}"
    else:
        new_basename = basename  # 不修改原文件名

    # 重组路径
    new_filename = f"{new_basename}{ext}"
    new_path = os.path.join(dir_path, new_filename)
    return new_path, new_filename, rank


def get_quant_weight(weights, key):
    """安全获取量化偏置张量并转换NPU格式
    Args:
        weights (dict): 参数字典
        key (str): 参数前缀标识符

    Returns:
        torch.Tensor: 转换后的张量

    Raises:
        KeyError: 当指定键不存在时抛出
    """
    if key in weights.keys():
        tensor = weights.get_tensor(key)
    else:
        raise ParametersInvalid(f"Critical parameter missing: {key}.")
    return tensor


class TimestepManager:
    """Manages timestep indices for multi-modal quantization processes."""

    _timestep_var = contextvars.ContextVar("timestep_idx", default=None)
    _timestep_var_max = contextvars.ContextVar("timestep_idx_max", default=None)

    @classmethod
    def set_timestep_idx(cls, cur_timestep: int) -> None:
        r"""
        The method is used to set the current timestep.

        Args:
            cur_timestep: Current iteration timestep.
        """
        if cur_timestep is not None and not isinstance(cur_timestep, int):
            raise ParametersInvalid(f"cur_timestep must be the type of int, but currently got {type(cur_timestep)}.")
        current = cls._timestep_var.get()
        max_step = cls._timestep_var_max.get()
        if current is not None and current == cur_timestep:
            logger.debug("Warning: Setting same timestep value consecutively: %r", cur_timestep)
        if max_step is not None and cur_timestep > max_step:
            raise ParametersInvalid(f"max timestep set in quant weight: {max_step}.")
        cls._timestep_var.set(cur_timestep)
        logger.debug("Timestep index set to: %r", cur_timestep)

    @classmethod
    def get_timestep_idx(cls) -> Optional[int]:
        r"""
        Get the current timestep index.
        Returns:
            The current timestep index.
        """
        t_idx = cls._timestep_var.get()
        if t_idx is None:
            logger.debug("Warning: Timestep index not set. Call set_timestep_idx() before each timestep.")
        return t_idx

    @classmethod
    def set_timestep_idx_max(cls, t_idx: int) -> None:
        r"""
        Set the max timestep index.
        Args:
            t_idx: The max current timestep index.
        """
        if t_idx is not None and not isinstance(t_idx, int):
            raise ParametersInvalid(f"t_idx must be the type of int, but currently got {type(t_idx)}.")
        current = cls._timestep_var_max.get()
        if current is not None and current == t_idx:
            logger.debug("Warning: Setting same Max timestep value consecutively: %r", t_idx)
        cls._timestep_var_max.set(t_idx)
        logger.debug("Max Timestep index set to: %r", t_idx)

    @classmethod
    def get_timestep_idx_max(cls) -> Optional[int]:
        r"""
        Get the max timestep index.
        Returns:
            The max current timestep index.
        """
        t_idx = cls._timestep_var_max.get()
        if t_idx is None:
            logger.debug(
                "Warning: Max Timestep index not set. Call set_timestep_idx_max() before get_timestep_idx_max."
            )
        return t_idx
