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
from collections import OrderedDict
from functools import wraps
from typing import Optional
import torch
from torch import nn
import safetensors
from .mode import QuantAlgorithm
from .config import OnlineQuantConfig, QuantConfig, TimestepPolicyConfig
from .mode import W4A4_LIST, W8A8_LIST
from .utils import (
    replace_rank_suffix,
    get_quant_weight,
    extract_constructor_args,
    MAX_WEIGHT_SIZE,
    build_online_fa_rot_weights,
    match_fa_layer,
    match_layer_config,
)
from .layer import (
    W4A4QuantLinear,
    W4A4MXFP4DualQuantLinear,
    W8A8QuantLinear,
    W8A8TimeStepQuantLinear,
    WeightQuantLinear,
    FP8RotateQuantFA,
    MXFP8RotateQuantFA,
    MXFP4QuantFA,
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
    quant_candidates = OrderedDict([(nn.Linear, WeightQuantLinear)])

    # 寻找匹配的规则
    quant_cls = next((quant_candidates[cls] for cls in quant_candidates if isinstance(layer, cls)), None)

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
        quant_candidates = OrderedDict([(nn.Linear, W8A8TimeStepQuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W8A8_MXFP8:
        quant_candidates = OrderedDict([(nn.Linear, W8A8MXFP8QuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_DYNAMIC:
        quant_candidates = OrderedDict([(nn.Linear, W4A4QuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_MXFP4_DUALSCALE:
        quant_candidates = OrderedDict([(nn.Linear, W4A4MXFP4DualQuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_MXFP4_DYNAMIC:
        quant_candidates = OrderedDict([(nn.Linear, W4A4MXFP4QuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_MXFP4_SVD:
        raise ParametersInvalid("SVD Quant algorithm not supported!")
    else:
        quant_candidates = OrderedDict([(nn.Linear, W8A8QuantLinear)])

    # 寻找匹配的规则
    quant_cls = next((quant_candidates[cls] for cls in quant_candidates if isinstance(layer, cls)), None)

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


def add_fa_quant(layer, cfg, prefix, quant_weights, **kwargs):
    if cfg.quant_algo in [QuantAlgorithm.MXFP4_DYNAMIC]:
        layer.fa_quant = MXFP4QuantFA(prefix, quant_weights, **kwargs)
    elif cfg.quant_algo in [QuantAlgorithm.FP8_DYNAMIC]:
        layer.fa_quant = FP8RotateQuantFA(prefix, quant_weights)
    elif cfg.quant_algo in [QuantAlgorithm.MXFP8_DYNAMIC]:
        layer.fa_quant = MXFP8RotateQuantFA(prefix, quant_weights)


def normalize_quant_config(kwargs):
    quant_config = kwargs.get('quant_config', None)
    if quant_config is None:
        quant_config = QuantConfig.from_kwargs(kwargs)
    elif not isinstance(quant_config, QuantConfig):
        raise ParametersInvalid(f"quant_config must be QuantConfig, but currently got {type(quant_config)}.")
    else:
        timestep_config = kwargs.get('timestep_config', None)
        timestep_policy = kwargs.get('timestep_policy', None)
        if timestep_config is not None and timestep_policy is not None and timestep_config is not timestep_policy:
            raise ParametersInvalid("timestep_config and timestep_policy cannot both be set to different objects.")
        if quant_config.timestep_config is None:
            quant_config.timestep_config = timestep_config if timestep_config is not None else timestep_policy
        if quant_config.use_nz is None:
            quant_config.use_nz = kwargs.get('use_nz', None)
    kwargs['quant_config'] = quant_config
    kwargs['timestep_config'] = quant_config.timestep_config
    kwargs['dtype'] = quant_config.dtype
    if quant_config.use_nz is not None:
        kwargs['use_nz'] = quant_config.use_nz
    return kwargs


def resolve_quant_des_path(quant_des_path, quant_config, check_path=True):
    if quant_des_path is not None:
        if not isinstance(quant_des_path, str) or not quant_des_path.strip():
            raise ConfigError("Invalid string path for quant_des_path.")
        quant_config.quant_des_path = quant_des_path
    quant_des_path = quant_config.quant_des_path
    if not isinstance(quant_des_path, str) or not quant_des_path.strip():
        raise ConfigError("Invalid string path for quant_des_path.")
    quant_des_path = file_utils.standardize_path(quant_des_path)
    quant_config.quant_des_path = quant_des_path
    if check_path:
        file_utils.check_file_safety(quant_des_path, permission_mode=file_utils.MODELDATA_FILE_PERMISSION)
    return quant_des_path


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

    quant_config = {
        "quant_des_path": quant_des_path,
        "quant_algo": QuantAlgorithm(quant_algo),
        "exclude_layers": tuple(exclude_layers),
        "quantized_layers": quantized_layers,
    }
    cfg = QuantConfig.parse_from_dict(quant_config)

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
    def wrapper(model: nn.Module, quant_des_path=None, online_config: Optional[OnlineQuantConfig] = None, **kwargs):
        # 检查 model 类型
        if not isinstance(model, nn.Module):
            raise ParametersInvalid(f"The model must be the type of nn.Module, but currently got {type(model)}.")

        if online_config is not None:
            if quant_des_path is not None or kwargs.get('quant_config') is not None:
                raise ParametersInvalid("online_config is mutually exclusive with quant_des_path and quant_config.")
            if not isinstance(online_config, OnlineQuantConfig):
                raise ParametersInvalid(
                    f"online_config must be the type of OnlineQuantConfig, but currently got {type(online_config)}."
                )
            dtype = kwargs.get('dtype', torch.bfloat16)
            if not isinstance(dtype, torch.dtype) or dtype not in (torch.float16, torch.bfloat16):
                raise ParametersInvalid(
                    f"Dtype must be torch.float16 or torch.bfloat16, but currently got {type(dtype)}."
                )
            return func(model, quant_des_path, online_config, **kwargs)

        quant_config = kwargs.get('quant_config')
        if quant_config is not None and not isinstance(quant_config, QuantConfig):
            raise ParametersInvalid(f"quant_config must be QuantConfig, but currently got {type(quant_config)}.")

        timestep_config = kwargs.get('timestep_config')
        if timestep_config is not None and not isinstance(timestep_config, TimestepPolicyConfig):
            raise ParametersInvalid(
                f"Timestep_config must be the type of TimestepPolicyConfig,but currently got {type(timestep_config)}."
            )

        timestep_policy = kwargs.get('timestep_policy')
        if timestep_policy is not None and not isinstance(timestep_policy, TimestepPolicyConfig):
            raise ParametersInvalid(
                f"Timestep_policy must be the type of TimestepPolicyConfig,but currently got {type(timestep_policy)}."
            )

        config_dtype = quant_config.dtype if quant_config is not None else None
        dtype = kwargs.get('dtype', config_dtype if config_dtype is not None else torch.bfloat16)
        if not isinstance(dtype, torch.dtype) or dtype not in (torch.float16, torch.bfloat16):
            raise ParametersInvalid(f"Dtype must be torch.float16 or torch.bfloat16, but currently got {type(dtype)}.")

        kwargs = normalize_quant_config(kwargs)
        quant_des_path = resolve_quant_des_path(quant_des_path, kwargs['quant_config'], check_path=True)

        return func(model, quant_des_path, None, **kwargs)

    return wrapper


@validate_quantize_params
def quantize(model, quant_des_path=None, online_config=None, **kwargs):
    r"""
    The method is used to quant model.

    Args:
        model: Floating point models that need to be quantized.
        quant_des_path: The absolute path of the quantized weight descripter exported by modelslim.
        **kwargs:
            quant_config: QuantConfig carries JSON path, JSON overrides, timestep, dtype and MXFP4 runtime settings.
            timestep_config: Compatibility input. Prefer quant_config.timestep_config.
            timestep_policy: Compatibility alias of timestep_config.
            dtype: Dtype specifies the type of the inverse quantization.
    Returns:
        Quantntifild Model.
    """
    if online_config is not None:
        return _online_quantize_impl(model, online_config, **kwargs)

    kwargs = normalize_quant_config(kwargs)
    user_quant_config = kwargs['quant_config']
    quant_des_path = resolve_quant_des_path(quant_des_path, user_quant_config, check_path=False)
    cfg, quant_weights = get_cfg_and_weights(quant_des_path)
    cfg = cfg.merged_with_user(user_quant_config)
    cfg.quant_des_path = quant_des_path
    kwargs.update(cfg.to_kwargs())

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
            add_fa_quant(layer, layer_quant_cfg, name, quant_weights, **kwargs)
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


def _make_online_quant_config(online_config, quant_algo, dtype):
    return QuantConfig(
        quant_algo=quant_algo,
        dtype=dtype,
        timestep_config=online_config.timestep_config,
        mxfp4_scale_alg=online_config.mxfp4_scale_alg,
        mxfp4_dst_type_max=online_config.mxfp4_dst_type_max,
    )


def _create_online_quant_layer(quant_cls, layer, dtype, quant_config=None):
    if quant_config is not None:
        try:
            return quant_cls(layer, dtype=dtype, quant_config=quant_config)
        except TypeError as exc:
            if 'quant_config' not in str(exc):
                raise
    return quant_cls(layer, dtype=dtype)


def _match_fallback(layer_name, fallback_layers):
    return match_layer_config(layer_name, fallback_layers)


def _online_quantize_impl(model, online_config, **kwargs):
    quant_type = online_config.quant_type
    fallback_layers = online_config.fallback_layers or {}
    fa_layers = online_config.fa_layers or ()
    fa_quant_type = online_config.fa_quant_type
    dtype = kwargs.get('dtype', torch.bfloat16)
    mm_quant_config = _make_online_quant_config(online_config, quant_type, dtype)

    main_quant_cls = _ONLINE_QUANT_LAYER_MAP.get(quant_type)
    if main_quant_cls is None:
        raise ParametersInvalid(
            f"Unsupported online quantization type: {quant_type}. "
            f"Supported types: {list(_ONLINE_QUANT_LAYER_MAP.keys())}"
        )

    logger.info(
        "Online quantization started with quant_type=%s, fa_quant_type=%s, fa_layers=%s, fallback_layers=%s",
        quant_type,
        fa_quant_type,
        fa_layers,
        fallback_layers,
    )

    modified_layers = []
    fa_count = 0
    for name, layer in model.named_modules():
        fallback_algo = _match_fallback(name, fallback_layers)

        if fa_quant_type is not None and match_fa_layer(name, layer, fa_layers):
            if fallback_algo == QuantAlgorithm.W16A16:
                logger.debug("FA layer %s keeps W16A16 (no FA quantization).", name)
            else:
                fa_quant_config = _make_online_quant_config(online_config, fa_quant_type, dtype)
                rot_weights = build_online_fa_rot_weights(name, layer, dtype=dtype)
                add_fa_quant(layer, fa_quant_config, name, rot_weights, quant_config=fa_quant_config)
                fa_count += 1
                logger.debug("Online FA quant layer name:%s, algo:%s.", name, fa_quant_type)

        if not isinstance(layer, nn.Linear):
            continue

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
            quant_layer = _create_online_quant_layer(fallback_cls, layer, dtype)
            logger.debug(
                "Fallback layer name:%s, algo:%s, class:%s.", name, fallback_algo, quant_layer.__class__.__name__
            )
        elif quant_type in _W4A4_QUANT_TYPES:
            quant_layer = _create_online_quant_layer(main_quant_cls, layer, dtype, mm_quant_config)
            logger.debug("Online quant layer name:%s, class:%s.", name, quant_layer.__class__.__name__)
        else:
            quant_layer = _create_online_quant_layer(main_quant_cls, layer, dtype)
            logger.debug("Online quant layer name:%s, class:%s.", name, quant_layer.__class__.__name__)

        modified_layers.append((name, quant_layer))

    modify_graph(model, modified_layers)
    torch.npu.empty_cache()

    logger.info(
        "Online quantization completed. %d linear layers quantized, %d FA layers quantized.",
        len(modified_layers),
        fa_count,
    )
    return model
