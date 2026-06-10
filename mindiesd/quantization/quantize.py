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

import json
import os
from fnmatch import fnmatch
from typing import Dict, Optional
from collections import OrderedDict
from functools import wraps
import torch
from torch import nn
import safetensors
from .mode import QuantAlgorithm
from .config import QuantConfig, LayerQuantConfig, TimestepPolicyConfig, OnlineQuantConfig
from .mode import W4A4_LIST, W8A8_LIST
from .utils import replace_rank_suffix, get_quant_weight, extract_constructor_args, MAX_WEIGHT_SIZE
from .layer import (
    W4A4QuantLinear,
    W4A4MXFP4DualQuantLinear,
    W8A8QuantLinear,
    W8A8TimeStepQuantLinear,
    WeightQuantLinear,
    FP8RotateQuantFA,
    W8A8MXFP8QuantLinear,
    W4A4MXFP4QuantLinear,
    W8A8OnlineQuantLinear,
    W8A8MXFP8OnlineQuantLinear,
    W4A4MXFP4OnlineQuantLinear,
    W4A4MXFP4DualOnlineQuantLinear,
)
from ..utils import ParametersInvalid, ConfigError
from ..utils import file_utils
from ..utils.logs.logging import logger


def get_key_patterns(layer_name):
    key_patterns = [
        f'{layer_name}.linear.weight',
        f'{layer_name}.weight',
        f'{layer_name}',
        f'{layer_name}.fa_q.scale',
        f'{layer_name}.quant_type',
    ]
    return key_patterns


def weight_quantize(name, layer, cfg, quant_weights, **kwargs):
    if cfg.quant_algo in [QuantAlgorithm.W8A16, QuantAlgorithm.W4A16]:
        return w8a16_quantize(name, layer, cfg, quant_weights, **kwargs)
    return layer, False


def w8a16_quantize(name, layer, cfg, quant_weights, **kwargs):
    quant_map = OrderedDict([(nn.Linear, WeightQuantLinear)])

    # 如果模型指定了类的匹配规则，优先匹配模型指定的
    user_dict = kwargs.get('map', None)
    if user_dict:
        for key, value in user_dict.items():
            quant_map[key] = value
        for key in user_dict.keys():
            quant_map.move_to_end(key, last=False)

    # 寻找匹配的规则
    quant_cls = next((quant_map[cls] for cls in quant_map if isinstance(layer, cls)), None)

    if quant_cls is None:
        return layer, False

    # 获取浮点类的入参
    init_params = extract_constructor_args(layer, quant_cls)
    bias = 'bias'
    if bias in init_params and isinstance(init_params[bias], nn.Parameter):
        init_params[bias] = True
    else:
        init_params[bias] = False

    init_params['weights'] = quant_weights
    init_params['prefix'] = name
    if cfg.quant_algo == QuantAlgorithm.W4A16:
        init_params['is_w4'] = True

    # 抑制算法需要的属性
    if f'{name}.div.mul_scale' in quant_weights.keys():
        init_params['mul_scale'] = get_quant_weight(quant_weights, f'{name}.div.mul_scale')
        init_params['prefix'] = f'{name}.linear'

    del layer.weight
    if hasattr(layer, 'bias'):
        del layer.bias
    quant_layer = quant_cls(**init_params, **kwargs)

    return quant_layer, True


def smooth_quantize_w8a8(name, layer, cfg, quant_weights, **kwargs):
    if cfg.quant_algo == QuantAlgorithm.W8A8_TIMESTEP:
        quant_map = OrderedDict([(nn.Linear, W8A8TimeStepQuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W8A8_MXFP8:
        quant_map = OrderedDict([(nn.Linear, W8A8MXFP8QuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_DYNAMIC:
        quant_map = OrderedDict([(nn.Linear, W4A4QuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_MXFP4_DUALSCALE:
        quant_map = OrderedDict([(nn.Linear, W4A4MXFP4DualQuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_MXFP4_DYNAMIC:
        quant_map = OrderedDict([(nn.Linear, W4A4MXFP4QuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_MXFP4_SVD:
        raise ParametersInvalid("SVD Quant algorithm not supported!")
    else:
        quant_map = OrderedDict([(nn.Linear, W8A8QuantLinear)])

    # 如果模型指定了类的匹配规则，优先匹配模型指定的
    user_dict = kwargs.get('map', None)
    if user_dict:
        for key, value in user_dict.items():
            quant_map[key] = value
        for key in user_dict.keys():
            quant_map.move_to_end(key, last=False)

    # 寻找匹配的规则
    quant_cls = next((quant_map[cls] for cls in quant_map if isinstance(layer, cls)), None)

    if quant_cls is None:
        return layer, False

    # 获取浮点类的入参
    init_params = extract_constructor_args(layer, quant_cls)
    bias = 'bias'
    if bias in init_params and isinstance(init_params[bias], nn.Parameter):
        init_params[bias] = True
    else:
        init_params[bias] = False

    if cfg.quant_algo in [
        QuantAlgorithm.W8A8_DYNAMIC,
        QuantAlgorithm.W8A8_MXFP8,
        QuantAlgorithm.W4A4_DYNAMIC,
        QuantAlgorithm.W4A4_MXFP4_DUALSCALE,
        QuantAlgorithm.W4A4_MXFP4_DYNAMIC,
    ]:
        init_params['is_dynamic'] = True

    init_params['weights'] = quant_weights
    init_params['prefix'] = name
    # 抑制算法需要的属性
    if f'{name}.div.mul_scale' in quant_weights.keys():
        init_params['mul_scale'] = get_quant_weight(quant_weights, f'{name}.div.mul_scale')
        init_params['prefix'] = f'{name}.linear'

    del layer.weight
    if hasattr(layer, 'bias'):
        del layer.bias

    quant_layer = quant_cls(**init_params, **kwargs)

    return quant_layer, True


def smooth_quantize(name, layer, cfg, quant_weights, **kwargs):
    if cfg.quant_algo in W8A8_LIST or cfg.quant_algo in W4A4_LIST:
        return smooth_quantize_w8a8(name, layer, cfg, quant_weights, **kwargs)
    return layer, False


def add_fa_quant(layer, cfg, prefix, quant_weights):
    if cfg.quant_algo in [QuantAlgorithm.FP8_DYNAMIC]:
        layer.fa_quant = FP8RotateQuantFA(prefix, quant_weights)


def get_layer_quant_mode(name, layer, cfg):
    layer_quant_mode = None

    for pattern in get_key_patterns(name):
        if pattern in cfg.layer_quantization_mode:
            return cfg.layer_quantization_mode[pattern]
    return layer_quant_mode


def get_layer_quant_cfg(cfg, name, layer):
    layer_quant_cfg = None

    if cfg.quantized_layers is None:
        return None
    for pattern in get_key_patterns(name):
        if pattern in cfg.quantized_layers:
            return cfg.quantized_layers[pattern]
    return layer_quant_cfg


def check_exclude_layers(cfg, name, layer):
    if cfg.exclude_layers is None:
        return False
    return any(pattern in cfg.exclude_layers for pattern in get_key_patterns(name))


def modify_graph(model, modified_layers):
    for name, layer in modified_layers:
        submodules = name.split('.')[:-1]
        layer_name = name.split('.')[-1]
        setattr(model.get_submodule('.'.join(submodules)), layer_name, layer)


# 读取配置文件，获取量化配置和权重
def get_cfg_and_weights(quant_des_path):
    quant_des_path, filename, rank = replace_rank_suffix(quant_des_path)
    quant_algo_str = "quant_algo"
    with file_utils.safe_open(
        quant_des_path, "r", encoding="utf-8", permission_mode=file_utils.CONFIG_FILE_PERMISSION
    ) as reader:
        data = reader.read()
    quant_des_dict = json.loads(data, strict=False)
    logger.debug("[MindIE-SD/quantization] Quant description loaded. filename=%s.", filename)

    if not quant_des_dict:
        raise ParametersInvalid("quant_des_dict is none!")
    exclude_layers = [k for k, v in quant_des_dict.items() if v == "FLOAT"]
    valid_values = {item.value for item in QuantAlgorithm}  # 预计算有效值集合
    quantized_layers = {
        k: {quant_algo_str: QuantAlgorithm(v.upper())}
        for k, v in quant_des_dict.items()
        if isinstance(v, str) and v.upper() in valid_values
    }
    quant_algo = quant_des_dict.get("model_quant_type", None)
    if quant_algo is None:
        raise ParametersInvalid("quant_algo must be the type of QuantAlgorithm.")

    quant_config = {"quant_algo": quant_algo}
    quant_config.update({'exclude_layers': tuple(exclude_layers)})
    quant_config.update({'quantized_layers': quantized_layers})
    quant_config.update({quant_algo_str: QuantAlgorithm(quant_algo)})
    if isinstance(quant_config, dict):
        cfg = LayerQuantConfig.parse_from_dict(quant_config)
    else:
        cfg = quant_config

    quant_weight_dir = os.path.dirname(quant_des_path)
    if rank != -1:
        weight_name = f'quant_model_weight_{quant_algo.lower()}_{rank}.safetensors'
    else:
        weight_name = f'quant_model_weight_{quant_algo.lower()}.safetensors'
    quant_weight_path = os.path.join(quant_weight_dir, weight_name)
    quant_weight_path = file_utils.standardize_path(quant_weight_path)
    file_utils.check_file_safety(
        quant_weight_path, permission_mode=file_utils.MODELDATA_FILE_PERMISSION, max_file_size=MAX_WEIGHT_SIZE
    )
    quant_weights = safetensors.safe_open(quant_weight_path, framework="pytorch")
    logger.debug("[MindIE-SD/quantization] Quant weight file loaded. path=%s.", quant_weight_path)

    return cfg, quant_weights


def validate_quantize_params(func):
    @wraps(func)
    def wrapper(
        model: nn.Module,
        quant_des_path: Optional[str] = None,
        online_config: Optional[OnlineQuantConfig] = None,
        **kwargs,
    ):
        if not isinstance(model, nn.Module):
            raise ParametersInvalid(f"The model must be the type of nn.Module, but currently got {type(model)}.")

        if quant_des_path is not None and online_config is not None:
            raise ParametersInvalid(
                "quant_des_path and online_config are mutually exclusive. Please provide only one of them."
            )

        if quant_des_path is None and online_config is None:
            raise ParametersInvalid("Either quant_des_path or online_config must be provided.")

        dtype = kwargs.get('dtype', torch.bfloat16)
        if not isinstance(dtype, torch.dtype) or dtype not in (torch.float16, torch.bfloat16):
            raise ParametersInvalid(f"Dtype must be torch.float16 or torch.bfloat16, but currently got {type(dtype)}.")

        if quant_des_path is not None:
            if not isinstance(quant_des_path, str) or not quant_des_path.strip():
                raise ConfigError("Invalid string path for quant_des_path.")
            quant_des_path = file_utils.standardize_path(quant_des_path)
            file_utils.check_file_safety(quant_des_path, permission_mode=file_utils.MODELDATA_FILE_PERMISSION)

            timestep_config = kwargs.get('timestep_config')
            if timestep_config is not None and not isinstance(timestep_config, TimestepPolicyConfig):
                raise ParametersInvalid(
                    "Timestep_config must be the type of TimestepPolicyConfig,"
                    "but currently got {type(timestep_config)}."
                )

            module_map = kwargs.get('map', None)
            if module_map is not None:
                if (
                    not isinstance(module_map, Dict)
                    or not all(isinstance(v, nn.Module) for v in module_map.values())
                    or not all(isinstance(k, nn.Module) for k in module_map.keys())
                ):
                    raise ParametersInvalid(
                        "The data type of map must be dictionary, and its KVType must be nn.Module."
                    )

        if online_config is not None:
            if not isinstance(online_config, OnlineQuantConfig):
                raise ParametersInvalid(
                    f"online_config must be the type of OnlineQuantConfig, but currently got {type(online_config)}."
                )

        return func(model, quant_des_path, online_config, **kwargs)

    return wrapper


@validate_quantize_params
def quantize(model, quant_des_path=None, online_config=None, **kwargs):
    r"""
    The method is used to quantize model. Supports two mutually exclusive modes:

    1. Offline quantization: provide quant_des_path (path to msModelSlim exported quantization descriptor).
    2. Online quantization: provide online_config (OnlineQuantConfig specifying quantization type and fallback layers).

    Args:
        model: Floating point models that need to be quantized.
        quant_des_path: The absolute path of the quantized weight descriptor exported by modelslim.
                        Mutually exclusive with online_config.
        online_config: OnlineQuantConfig specifying the quantization type and fallback layers.
                       Mutually exclusive with quant_des_path.
        **kwargs:
            timestep_config: When using timestep quantization, TimestepPolicyConfig needs to be passed in.
            dtype: Dtype specifies the type of the inverse quantization (default: torch.bfloat16).
            map: Custom layer matching dictionary (offline mode only).
    Returns:
        Quantized Model.
    """
    if online_config is not None:
        return _online_quantize_impl(model, online_config, **kwargs)

    cfg, quant_weights = get_cfg_and_weights(quant_des_path)

    if not isinstance(cfg, QuantConfig):
        logger.debug("cfg is not QuantConfig, Without enabling quantization.")
        return model

    if not cfg.layer_quantization_mode:
        logger.debug("Quantization content is none, Without enabling quantization.")
        return model

    modified_layers = []
    rank = int(os.getenv("RANK", "0"))

    for name, layer in model.named_modules():
        # 跳过回退层
        if check_exclude_layers(cfg, name, layer):
            logger.debug("Skipping layer %s due to excluded configuration.", name)
            continue
        # 如果模型显式指定了融合层，以融合层指定的算法为最高优先级配置，否则从config里读取配置
        layer_quant_cfg = get_layer_quant_cfg(cfg, name, layer)
        if layer_quant_cfg is None:
            logger.debug("Cannot find the quantization configuration corresponding to %s.", name)
            continue

        # 以用户申明的融合算法为第一优先级，其次是读取配置中的
        layer_quant_mode = get_layer_quant_mode(name, layer, cfg)
        if layer_quant_mode is None:
            logger.debug("Cannot find the quantization mode corresponding to %s.", name)
            continue

        # 根据算法的要素dispatch到不同分支
        if layer_quant_mode.contains_activation_and_weight_quant():
            quant_layer, is_modified = smooth_quantize(name, layer, layer_quant_cfg, quant_weights, **kwargs)
            if is_modified:
                logger.debug("W8A8 Quant layer name:%s, Quant class name:%s.", name, quant_layer.__class__.__name__)
                modified_layers.append((name, quant_layer))
        elif layer_quant_mode.check_weight_only_mode():
            quant_layer, is_modified = weight_quantize(name, layer, layer_quant_cfg, quant_weights, **kwargs)
            if is_modified:
                logger.debug("Weight Quant layer name:%s, Quant class name:%s.", name, quant_layer.__class__.__name__)
                modified_layers.append((name, quant_layer))
        elif layer_quant_mode.contains_fa_quantization():
            add_fa_quant(layer, layer_quant_cfg, name, quant_weights)
            if rank == 0:
                logger.debug(
                    "FA Quant layer name:%s, Quant class name:%s, Quant algo:%s.",
                    name,
                    layer.__class__.__name__,
                    layer_quant_cfg.quant_algo,
                )

    # 执行改图
    modify_graph(model, modified_layers)
    torch.npu.empty_cache()

    return model


_ONLINE_QUANT_LAYER_MAP = {
    QuantAlgorithm.W8A8_DYNAMIC: W8A8OnlineQuantLinear,
    QuantAlgorithm.W8A8_MXFP8: W8A8MXFP8OnlineQuantLinear,
    QuantAlgorithm.W4A4_MXFP4_DYNAMIC: W4A4MXFP4OnlineQuantLinear,
    QuantAlgorithm.W4A4_MXFP4_DUALSCALE: W4A4MXFP4DualOnlineQuantLinear,
    QuantAlgorithm.W8A8: W8A8OnlineQuantLinear,
}

_W4A4_QUANT_TYPES = (
    QuantAlgorithm.W4A4_MXFP4_DYNAMIC,
    QuantAlgorithm.W4A4_MXFP4_DUALSCALE,
)


def _match_fallback(layer_name, fallback_layers):
    for pattern, algo in fallback_layers.items():
        if fnmatch(layer_name, pattern):
            return algo
    return None


def _online_quantize_impl(model, online_config, **kwargs):
    quant_type = online_config.quant_type
    fallback_layers = online_config.fallback_layers or {}
    fallback_timesteps = online_config.fallback_timesteps
    dtype = kwargs.get('dtype', torch.bfloat16)

    main_quant_cls = _ONLINE_QUANT_LAYER_MAP.get(quant_type)
    if main_quant_cls is None:
        raise ParametersInvalid(
            f"Unsupported online quantization type: {quant_type}. "
            f"Supported types: {list(_ONLINE_QUANT_LAYER_MAP.keys())}"
        )

    logger.info(
        "Online quantization started with quant_type=%s, fallback_layers=%s, fallback_timesteps=%s",
        quant_type,
        fallback_layers,
        fallback_timesteps,
    )

    modified_layers = []
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue

        fallback_algo = _match_fallback(name, fallback_layers)

        if fallback_algo == QuantAlgorithm.W16A16:
            logger.debug("Layer %s keeps W16A16 (no quantization).", name)
            continue

        if fallback_algo is not None:
            fallback_cls = _ONLINE_QUANT_LAYER_MAP.get(fallback_algo)
            if fallback_cls is None:
                raise ParametersInvalid(
                    f"Unsupported fallback algorithm: {fallback_algo} for layer {name}. "
                    f"Supported fallback types: {list(_ONLINE_QUANT_LAYER_MAP.keys())}"
                )
            quant_layer = fallback_cls(layer, dtype=dtype)
            logger.debug(
                "Fallback layer name:%s, algo:%s, class:%s.", name, fallback_algo, quant_layer.__class__.__name__
            )
        elif quant_type in _W4A4_QUANT_TYPES:
            quant_layer = main_quant_cls(layer, dtype=dtype, fallback_timesteps=fallback_timesteps)
            logger.debug(
                "Online quant layer name:%s, class:%s, fallback_timesteps:%s.",
                name,
                quant_layer.__class__.__name__,
                fallback_timesteps,
            )
        else:
            quant_layer = main_quant_cls(layer, dtype=dtype)
            logger.debug("Online quant layer name:%s, class:%s.", name, quant_layer.__class__.__name__)

        modified_layers.append((name, quant_layer))

    modify_graph(model, modified_layers)
    torch.npu.empty_cache()

    logger.info("Online quantization completed. %d layers quantized.", len(modified_layers))
    return model
