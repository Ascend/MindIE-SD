#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU configuration, descriptor and pattern contracts; no device execution."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mindiesd.quantization.config import LayerQuantConfig, OnlineQuantConfig, QuantConfig, TimestepPolicyConfig
from mindiesd.quantization.description import parse_quant_description
from mindiesd.quantization.mode import FP8FAMode, QuantAlgorithm
from mindiesd.utils import ModelInitError, ParametersInvalid

utils = importlib.import_module('mindiesd.quantization.utils')
description = importlib.import_module('mindiesd.quantization.description')


@pytest.mark.parametrize(
    'options',
    [
        {'quant_des_path': 3},
        {'quant_algo': 4},
        {'dtype': torch.float32},
        {'timestep_config': object()},
        {'use_nz': 1},
        {'mxfp4_scale_alg': '2'},
        {'mxfp4_dst_type_max': True},
        {'mxfp4_dst_type_max': '7'},
        {'fp8_fa_mode': 'invalid'},
        {'fp8_fa_mode': []},
        {'exclude_layers': 'a'},
        {'exclude_layers': [3]},
        {'quantized_layers': []},
        {'quantized_layers': {1: QuantConfig()}},
        {'quantized_layers': {'a': 1}},
    ],
)
def test_quant_config_rejects_invalid_field_without_device_access(options):
    with pytest.raises(ModelInitError):
        QuantConfig(**options)


def test_config_normalizes_nested_entries_and_explicit_overrides():
    policy = TimestepPolicyConfig()
    baseline = QuantConfig(
        quant_algo='w8a8', quantized_layers={'proj': {'quant_algo': 'w8a8'}}, exclude_layers=['skip']
    )
    assert baseline.exclude_layers == ('skip',)
    assert baseline.quantized_layers['proj'].quant_algo == QuantAlgorithm.W8A8
    assert baseline.merged_with_user(None) is baseline
    override = QuantConfig(
        quant_algo='w8a8_dynamic', use_nz=False, timestep_config=policy, fp8_fa_mode='high_precision'
    )
    merged = baseline.merged_with_user(override)
    assert merged is not baseline
    assert merged.quantized_layers['proj'].quant_algo == QuantAlgorithm.W8A8_DYNAMIC
    assert baseline.quantized_layers['proj'].quant_algo == QuantAlgorithm.W8A8
    assert merged.to_kwargs()['timestep_config'] is policy
    assert merged.to_kwargs()['fp8_fa_mode'] == FP8FAMode.HIGH_PRECISION
    assert merged.to_kwargs()['use_nz'] is False
    assert merged.quant_algorithms_list == [QuantAlgorithm.W8A8_DYNAMIC]
    assert not QuantConfig().quant_algorithms_list
    layers = LayerQuantConfig(quantized_layers={'proj': QuantConfig(quant_algo='w8a8')})
    serialized = layers.serialize_to_dict()
    assert serialized['quantized_layers']['proj']['quant_algo'] == QuantAlgorithm.W8A8
    assert 'exclude_layers' not in serialized['quantized_layers']['proj']
    assert not LayerQuantConfig().serialize_to_dict()['quantized_layers']


def test_legacy_policy_conflicts_are_rejected_and_alias_identity_is_preserved():
    one, two = TimestepPolicyConfig(), TimestepPolicyConfig()
    with pytest.raises(ParametersInvalid):
        QuantConfig.from_kwargs({'timestep_config': one, 'timestep_policy': two})
    assert QuantConfig.from_kwargs({'timestep_config': one, 'timestep_policy': one}).timestep_config is one
    assert QuantConfig.from_kwargs({'timestep_policy': one}).timestep_config is one


@pytest.mark.parametrize(
    'options',
    [
        {'quant_type': None},
        {'quant_type': 'w8a16'},
        {'fa_layers': 4},
        {'fa_layers': [4]},
        {'fa_quant_type': 3},
        {'fa_quant_type': 'mxfp8_dynamic', 'fa_layers': ('attn',)},
        {'fa_quant_type': 'fp8_dynamic'},
        {'timestep_config': object()},
        {'mxfp4_scale_alg': '2'},
        {'mxfp4_dst_type_max': False},
        {'mxfp4_dst_type_max': '7'},
        {'fallback_layers': []},
        {'fallback_layers': {1: 'W16A16'}},
        {'fallback_layers': {'proj': object()}},
        {'fallback_layers': {'proj': 'W8A8_MXFP8'}},
        {'fallback_timesteps': [1]},
    ],
)
def test_online_config_rejects_inconsistent_combinations(options):
    with pytest.raises(ModelInitError):
        OnlineQuantConfig(**options)


def test_online_config_roundtrip_preserves_fa_and_fallback_contracts():
    policy = TimestepPolicyConfig()
    cfg = OnlineQuantConfig.parse_from_dict(
        {
            'quant_type': 'w8a8_dynamic',
            'fallback_layers': {'skip': 'w16a16'},
            'fa_layers': 'attn',
            'fa_quant_type': 'fp8_dynamic',
            'fp8_fa_mode': 'c8v16_tiling512',
            'timestep_config': policy,
            'mxfp4_scale_alg': 2,
            'mxfp4_dst_type_max': 7,
        }
    )
    result = cfg.serialize_to_dict()
    assert cfg.fa_layers == ('attn',)
    assert result['fallback_layers'] == {'skip': 'W16A16'}
    assert result['fa_layers'] == ['attn']
    assert result['timestep_config'] is policy
    assert result['mxfp4_dst_type_max'] == 7.0
    assert result['fp8_fa_mode'] == 'C8V16_TILING512'


@pytest.mark.parametrize('step_range', [[0, '1'], (0, 1), 1.5])
def test_policy_rejects_invalid_range_types(step_range):
    with pytest.raises(TypeError):
        TimestepPolicyConfig().register(step_range, 'FP8', target='fa')


def test_policy_explicit_targets_legacy_aliases_and_invalid_lookup():
    policy = TimestepPolicyConfig()
    policy.register(1, 'static', target='linear')
    policy.register(range(2, 4), 'W4A8', target='linear')
    assert policy.get_strategy(1, 'w8a8_static_linear') == 'static'
    assert policy.get_strategy(2) == 'W4A8'
    assert policy.get_strategy(None, 'fa') == 'MXFP4'
    for target, strategy, error in (
        (1, 'FP8', TypeError),
        ('missing', 'FP8', ValueError),
        ('linear', 'FP8', ValueError),
        ('fa', 3, TypeError),
        ('fa', 'W4A4', ValueError),
    ):
        with pytest.raises(error):
            policy.register([1], strategy, target)
    with pytest.raises(ParametersInvalid):
        policy.get_strategy('1')


@pytest.mark.parametrize(
    'data',
    [
        '[]',
        '{}',
        '{',
        '{"model_quant_type": 1}',
        '{"model_quant_type": "w8a8"}',
        '{"model_quant_type":"W8A8","proj":null}',
        '{"model_quant_type":"W8A8","proj":"invalid"}',
        '{"model_quant_type":"W8A8","proj":"FLOAT","proj":"W8A8"}',
        '{"model_quant_type":"W8A8","metadata":{"value":NaN}}',
        '{"model_quant_type":"W8A8","metadata":{"value":1e309}}',
    ],
)
def test_descriptor_rejects_malformed_or_ambiguous_json(data):
    with pytest.raises(ParametersInvalid):
        parse_quant_description(data)


@pytest.mark.parametrize(
    'name,value',
    [
        ('version', 1),
        ('group_size', -1),
        ('group_size', True),
        ('metadata', []),
        ('optional', 1),
        (' ', 'FLOAT'),
        ('a\x7f', 'FLOAT'),
        ('a\x01', 'FLOAT'),
        ('x' * 90, 'unknown'),
    ],
)
def test_descriptor_validates_metadata_types_and_untrusted_names(name, value):
    with pytest.raises(ParametersInvalid):
        parse_quant_description(json.dumps({'model_quant_type': 'W8A8', name: value}))


def test_descriptor_keeps_forward_compatible_metadata_and_case_insensitive_layers():
    data = {
        'model_quant_type': 'W8A8',
        'version': 'AscendV1',
        'group_size': 0,
        'kv_cache_type': 'future',
        'kv_quant_type': 'future',
        'fa_quant_type': 'future',
        'reduce_quant_type': 'future',
        'metadata': {'threshold': 1.5},
        'optional': {},
        'proj': 'w8a8',
        'bias': 'FLOAT',
    }
    assert parse_quant_description(json.dumps(data)) == data


@pytest.mark.parametrize('exception', [ValueError('numeric limit'), RecursionError('nesting')])
def test_descriptor_translates_standard_parser_resource_errors(exception):
    with patch.object(description.json, 'loads', side_effect=exception), pytest.raises(ParametersInvalid):
        parse_quant_description('{}')
    with pytest.raises(ParametersInvalid):
        description._validate_metadata('unknown', 1)


def test_fallback_pattern_expansion_without_optional_wcmatch():
    with patch.object(utils, 'wc_fnmatch', None):
        assert utils.match_layer_pattern('blocks.2.attn.q', 'blocks.{1,2}.{attn,mlp}.q')
        assert not utils.match_layer_pattern('blocks.3.attn.q', 'blocks.{1,2}.{attn,mlp}.q')
        assert utils.match_layer_pattern('literal{', 'literal{')
        assert not utils.match_layer_pattern('other', 'literal{')
        assert not utils.match_layer_patterns('layer', ())
        assert utils.match_layer_config('layer', {}) is None
        assert not utils.match_fa_layer('layer', torch.nn.Linear(2, 2), ())


def test_attention_dimension_inference_and_rotation_rejection():
    for obj, expected in (
        (SimpleNamespace(head_dim=8), 8),
        (SimpleNamespace(hidden_size=32, num_heads=4), 8),
        (SimpleNamespace(heads=4, q_proj=torch.nn.Linear(8, 32)), 8),
    ):
        assert utils.infer_attention_head_dim(obj) == expected
    for obj in (SimpleNamespace(), SimpleNamespace(hidden_size=31, num_heads=4)):
        assert utils.infer_attention_head_dim(obj) is None
        with pytest.raises(ParametersInvalid):
            utils.build_online_fa_rot_weights('attn', obj)
    for size in (0, -1, 3, '8'):
        with pytest.raises(ParametersInvalid):
            utils.create_online_hadamard_rot(size)
    rot = utils.create_online_hadamard_rot(4)
    torch.testing.assert_close(rot @ rot.T, torch.eye(4))
