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
import dataclasses
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import torch

from .mode import QuantAlgorithm, QuantMode
from ..utils import ModelInitError, ParametersInvalid

W8A8_STATIC_LINEAR_STRATEGIES = ("dynamic", "static")
W4A4_LINEAR_STRATEGIES = ("W4A4", "W4A8")
FA_STRATEGIES = ("MXFP4", "FP8", "FLOAT")
VALID_STRATEGIES = {
    "w8a8_static_linear": W8A8_STATIC_LINEAR_STRATEGIES,
    "w4a4_linear": W4A4_LINEAR_STRATEGIES,
    "fa": FA_STRATEGIES,
}


class TimestepPolicyConfig:
    def __init__(self, default_strategy="dynamic", w4a4_default_strategy="W4A4", fa_default_strategy="MXFP4"):
        r"""
        The method is used to init TimestepPolicyConfig.
        """
        self._strategies = {target: {} for target in VALID_STRATEGIES}
        self._default_strategy = {
            "w8a8_static_linear": default_strategy,
            "w4a4_linear": w4a4_default_strategy,
            "fa": fa_default_strategy,
        }
        self._validate_strategy(default_strategy, "w8a8_static_linear")
        self._validate_strategy(w4a4_default_strategy, "w4a4_linear")
        self._validate_strategy(fa_default_strategy, "fa")

    @staticmethod
    def _resolve_legacy_linear_target(strategy):
        if strategy in W8A8_STATIC_LINEAR_STRATEGIES:
            return "w8a8_static_linear"
        if strategy in W4A4_LINEAR_STRATEGIES:
            return "w4a4_linear"
        return "linear"

    @classmethod
    def _normalize_target(cls, target, strategy=None):
        if not isinstance(target, str):
            raise TypeError(f"target必须是字符串类型，实际类型：{type(target)}")
        if target == "linear":
            target = cls._resolve_legacy_linear_target(strategy)
        if target not in VALID_STRATEGIES:
            raise ValueError(f"无效的target：{target}，允许值：{tuple(VALID_STRATEGIES)}")
        return target

    @staticmethod
    def _validate_strategy(strategy, target):
        if not isinstance(strategy, str):
            raise TypeError(f"strategy必须是字符串类型，实际类型：{type(strategy)}")
        if strategy not in VALID_STRATEGIES[target]:
            raise ValueError(f"无效的策略类型：{strategy}，允许值：{VALID_STRATEGIES[target]}")

    @staticmethod
    def _normalize_step_range(step_range):
        if isinstance(step_range, int):
            return [step_range]
        if isinstance(step_range, (list, range)):
            if not all(isinstance(s, int) for s in step_range):
                raise TypeError("step_range列表必须包含整数元素")
            return step_range
        raise TypeError(f"step_range必须是int, list, range类型，实际类型：{type(step_range)}")

    def register(self, step_range, strategy, target="w4a4_linear"):
        r"""
        The method is used to register strategy.

        Args:
            step_range: Timestep range, the type can be int, range, or list.
            strategy: Strategy string. Timestep linear supports dynamic/static; W4A4 linear supports W4A4/W4A8;
                FA supports MXFP4/FP8/FLOAT.
            target: w8a8_static_linear, w4a4_linear, or fa. Defaults to w4a4_linear.
        """
        target = self._normalize_target(target, strategy)
        self._validate_strategy(strategy, target)
        step_range = self._normalize_step_range(step_range)
        for step in step_range:
            self._strategies[target][step] = strategy

    def get_strategy(self, step, target="w4a4_linear"):
        r"""
        The method is used to get strategy.

        Args:
            step: Specifild timestep.
            target: w8a8_static_linear, w4a4_linear, or fa. Defaults to w4a4_linear.
        Returns:
            The strategy corresponding to the specifiled timestep.
        """
        target = self._normalize_target(target)
        if step is not None and not isinstance(step, int):
            raise ParametersInvalid(f"step must be the type of int, but currently got {type(step)}.")
        return self._strategies[target].get(step, self._default_strategy[target])


@dataclasses.dataclass
class QuantConfig:
    quant_des_path: Optional[str] = None
    quant_algo: Optional[QuantAlgorithm] = None
    quantized_layers: Optional[Dict[str, "QuantConfig"]] = None
    exclude_layers: Optional[Tuple[str, ...]] = None

    dtype: torch.dtype = torch.bfloat16
    use_nz: Optional[bool] = None
    timestep_config: Optional[TimestepPolicyConfig] = None

    mxfp4_scale_alg: Optional[int] = None
    mxfp4_dst_type_max: float = 7.25

    def __post_init__(self):
        self.quant_algo = self._normalize_quant_algo(self.quant_algo)
        self.exclude_layers = self._normalize_exclude_layers(self.exclude_layers)
        self.quantized_layers = self._normalize_quantized_layers(self.quantized_layers)

        if self.quant_des_path is not None and not isinstance(self.quant_des_path, str):
            raise ModelInitError("self.quant_des_path must be a string or None.")
        if self.quant_algo is not None and not isinstance(self.quant_algo, QuantAlgorithm):
            raise ModelInitError(
                f'self.quant_algo must be an instance of QuantAlgorithm, but actually got {type(self.quant_algo)}.'
            )
        if not isinstance(self.dtype, torch.dtype) or self.dtype not in (torch.float16, torch.bfloat16):
            raise ModelInitError("self.dtype must be torch.float16 or torch.bfloat16.")
        if self.timestep_config is not None and not isinstance(self.timestep_config, TimestepPolicyConfig):
            raise ModelInitError("self.timestep_config must be an instance of TimestepPolicyConfig.")
        if self.use_nz is not None and not isinstance(self.use_nz, bool):
            raise ModelInitError("self.use_nz must be a bool.")
        if self.mxfp4_scale_alg is not None and not isinstance(self.mxfp4_scale_alg, int):
            raise ModelInitError("mxfp4_scale_alg must be an int or None.")
        if isinstance(self.mxfp4_dst_type_max, bool) or not isinstance(self.mxfp4_dst_type_max, (int, float)):
            raise ModelInitError("mxfp4_dst_type_max must be a float.")
        self.mxfp4_dst_type_max = float(self.mxfp4_dst_type_max)

    @staticmethod
    def _normalize_quant_algo(quant_algo):
        if isinstance(quant_algo, str):
            return QuantAlgorithm(quant_algo.upper())
        return quant_algo

    @staticmethod
    def _normalize_exclude_layers(exclude_layers):
        if exclude_layers is None:
            return None
        if isinstance(exclude_layers, list):
            exclude_layers = tuple(exclude_layers)
        if not isinstance(exclude_layers, tuple):
            raise ModelInitError("self.exclude_layers must be a tuple")
        for layer in exclude_layers:
            if not isinstance(layer, str):
                raise ModelInitError("Items in exclude_layers must be strings")
        return exclude_layers

    @classmethod
    def _normalize_quantized_layers(cls, quantized_layers):
        if quantized_layers is None:
            return None
        if not isinstance(quantized_layers, dict):
            raise ModelInitError("self.quantized_layers must be a dictionary")
        normalized = {}
        for name, layer_config in quantized_layers.items():
            if not isinstance(name, str):
                raise ModelInitError("Keys in self.quantized_layers must be strings")
            if isinstance(layer_config, dict):
                layer_config = cls.parse_from_dict(layer_config)
            if not isinstance(layer_config, QuantConfig):
                raise ModelInitError("Values in self.quantized_layers must be instances of QuantConfig")
            normalized[name] = layer_config
        return normalized

    @classmethod
    def parse_from_dict(cls, config: dict):
        return cls(**dict(config))

    @classmethod
    def from_kwargs(cls, kwargs: dict):
        timestep_config = kwargs.get('timestep_config', None)
        timestep_policy = kwargs.get('timestep_policy', None)
        if timestep_config is not None and timestep_policy is not None and timestep_config is not timestep_policy:
            raise ParametersInvalid("timestep_config and timestep_policy cannot both be set to different objects.")
        config_kwargs = {}
        for name in (
            'quant_des_path',
            'quant_algo',
            'quantized_layers',
            'exclude_layers',
            'dtype',
            'use_nz',
            'mxfp4_scale_alg',
            'mxfp4_dst_type_max',
        ):
            if name in kwargs:
                config_kwargs[name] = kwargs[name]
        config_kwargs['timestep_config'] = timestep_config if timestep_config is not None else timestep_policy
        return cls(**config_kwargs)

    def merged_with_user(self, user_config: Optional["QuantConfig"]):
        if user_config is None:
            return self
        merged = dataclasses.replace(self)
        for field in dataclasses.fields(QuantConfig):
            value = getattr(user_config, field.name)
            if value is not None:
                setattr(merged, field.name, value)
        if user_config.quant_algo is not None and user_config.quantized_layers is None and merged.quantized_layers:
            merged.quantized_layers = {
                name: dataclasses.replace(layer_config, quant_algo=user_config.quant_algo)
                for name, layer_config in merged.quantized_layers.items()
            }
        return merged

    def to_kwargs(self):
        kwargs = {'dtype': self.dtype}
        if self.use_nz is not None:
            kwargs['use_nz'] = self.use_nz
        if self.timestep_config is not None:
            kwargs['timestep_config'] = self.timestep_config
        kwargs['quant_config'] = self
        return kwargs

    @cached_property
    def layer_quantization_mode(self):
        if self.quantized_layers is not None:
            return {
                name: QuantMode.from_quant_algo(layer_config.quant_algo)
                for name, layer_config in self.quantized_layers.items()
            }
        return QuantMode.from_quant_algo(self.quant_algo)

    @cached_property
    def quant_algorithms_list(self):
        if not self.quantized_layers:
            return []
        return list(set(layer_config.quant_algo for _, layer_config in self.quantized_layers.items()))

    def serialize_to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class LayerQuantConfig(QuantConfig):
    def __post_init__(self):
        super().__post_init__()
        self.auto_quant_mode = self.layer_quantization_mode if self.quantized_layers else {}

    @cached_property
    def layer_quantization_mode(self):
        if self.quantized_layers is not None:
            return {
                name: QuantMode.from_quant_algo(layer_config.quant_algo)
                for name, layer_config in self.quantized_layers.items()
            }
        return {}

    def serialize_to_dict(self):
        output = {'quant_algo': self.quant_algo, 'quantized_layers': {}}
        if self.quantized_layers:
            output['quantized_layers'] = {
                name: layer_config.serialize_to_dict() for name, layer_config in self.quantized_layers.items()
            }
            for layer_dict in output['quantized_layers'].values():
                layer_dict.pop('exclude_layers', None)
        return output


SUPPORTED_ONLINE_QUANT_TYPES = (
    QuantAlgorithm.W8A8_DYNAMIC,
    QuantAlgorithm.W8A8_MXFP8,
    QuantAlgorithm.W4A4_MXFP4_DYNAMIC,
    QuantAlgorithm.W4A4_MXFP4_DUALSCALE,
)

SUPPORTED_ONLINE_FALLBACK_TYPES = (
    QuantAlgorithm.W8A8,
    QuantAlgorithm.W16A16,
)

SUPPORTED_ONLINE_FA_TYPES = (
    QuantAlgorithm.FP8_DYNAMIC,
    QuantAlgorithm.MXFP4_DYNAMIC,
)


@dataclasses.dataclass
class OnlineQuantConfig:
    quant_type: QuantAlgorithm = QuantAlgorithm.W8A8_DYNAMIC
    fallback_layers: Optional[Dict[str, QuantAlgorithm]] = None
    fa_layers: Optional[Tuple[str, ...]] = None
    fa_quant_type: Optional[QuantAlgorithm] = None
    timestep_config: Optional[TimestepPolicyConfig] = None
    mxfp4_scale_alg: Optional[int] = None
    mxfp4_dst_type_max: float = 7.25
    fallback_timesteps: dataclasses.InitVar[Optional[List[int]]] = None

    def __post_init__(self, fallback_timesteps):
        if fallback_timesteps is not None:
            raise ModelInitError("fallback_timesteps is no longer supported. Use timestep_config instead.")

        self.quant_type = QuantConfig._normalize_quant_algo(self.quant_type)
        if not isinstance(self.quant_type, QuantAlgorithm):
            raise ModelInitError(
                f'self.quant_type must be an instance of QuantAlgorithm, but actually got {type(self.quant_type)}.'
            )
        if self.quant_type not in SUPPORTED_ONLINE_QUANT_TYPES:
            raise ModelInitError(
                f'self.quant_type must be one of {SUPPORTED_ONLINE_QUANT_TYPES}, but actually got {self.quant_type}.'
            )

        self.fa_layers = self._normalize_fa_layers(self.fa_layers)
        self.fa_quant_type = QuantConfig._normalize_quant_algo(self.fa_quant_type)
        if self.fa_quant_type is not None:
            if not isinstance(self.fa_quant_type, QuantAlgorithm):
                raise ModelInitError(
                    f'self.fa_quant_type must be an instance of QuantAlgorithm, '
                    f'but actually got {type(self.fa_quant_type)}.'
                )
            if self.fa_quant_type not in SUPPORTED_ONLINE_FA_TYPES:
                raise ModelInitError(
                    f'self.fa_quant_type must be one of {SUPPORTED_ONLINE_FA_TYPES}, '
                    f'but actually got {self.fa_quant_type}.'
                )
            if not self.fa_layers:
                raise ModelInitError("self.fa_layers must be configured when fa_quant_type is set.")

        if self.timestep_config is not None and not isinstance(self.timestep_config, TimestepPolicyConfig):
            raise ModelInitError("self.timestep_config must be an instance of TimestepPolicyConfig.")
        if self.mxfp4_scale_alg is not None and not isinstance(self.mxfp4_scale_alg, int):
            raise ModelInitError("mxfp4_scale_alg must be an int or None.")
        if isinstance(self.mxfp4_dst_type_max, bool) or not isinstance(self.mxfp4_dst_type_max, (int, float)):
            raise ModelInitError("mxfp4_dst_type_max must be a float.")
        self.mxfp4_dst_type_max = float(self.mxfp4_dst_type_max)

        if self.fallback_layers is None:
            self.fallback_layers = {}
        if not isinstance(self.fallback_layers, dict):
            raise ModelInitError("self.fallback_layers must be a dict mapping layer name patterns to QuantAlgorithm")
        parsed_fallback = {}
        for pattern, algo in self.fallback_layers.items():
            if not isinstance(pattern, str):
                raise ModelInitError("Keys in fallback_layers must be strings")
            algo = QuantConfig._normalize_quant_algo(algo)
            if not isinstance(algo, QuantAlgorithm):
                raise ModelInitError(f"Values in fallback_layers must be QuantAlgorithm, got {type(algo)}")
            if algo not in SUPPORTED_ONLINE_FALLBACK_TYPES:
                raise ModelInitError(
                    f"Fallback algorithm for '{pattern}' must be one of {SUPPORTED_ONLINE_FALLBACK_TYPES}, "
                    f"but got {algo}"
                )
            parsed_fallback[pattern] = algo
        self.fallback_layers = parsed_fallback

    @staticmethod
    def _normalize_fa_layers(fa_layers):
        if fa_layers is None:
            return ()
        if isinstance(fa_layers, str):
            return (fa_layers,)
        if not isinstance(fa_layers, (list, tuple)):
            raise ModelInitError("self.fa_layers must be a list or tuple of layer/class name patterns.")
        normalized = tuple(fa_layers)
        for pattern in normalized:
            if not isinstance(pattern, str):
                raise ModelInitError("Items in fa_layers must be strings.")
        return normalized

    @classmethod
    def parse_from_dict(cls, config: dict):
        return cls(**dict(config))

    def serialize_to_dict(self):
        fallback = {
            pattern: algo.value if isinstance(algo, QuantAlgorithm) else algo
            for pattern, algo in self.fallback_layers.items()
        }
        result = {
            'quant_type': self.quant_type.value if isinstance(self.quant_type, QuantAlgorithm) else self.quant_type,
            'fallback_layers': fallback,
        }
        if self.fa_quant_type is not None:
            result['fa_quant_type'] = (
                self.fa_quant_type.value if isinstance(self.fa_quant_type, QuantAlgorithm) else self.fa_quant_type
            )
            result['fa_layers'] = list(self.fa_layers)
        if self.timestep_config is not None:
            result['timestep_config'] = self.timestep_config
        if self.mxfp4_scale_alg is not None:
            result['mxfp4_scale_alg'] = self.mxfp4_scale_alg
            result['mxfp4_dst_type_max'] = self.mxfp4_dst_type_max
        return result
