# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

import importlib
import inspect
from unittest.mock import MagicMock

import pytest
import torch

from mindiesd.utils import ParametersInvalid

module = importlib.import_module('mindiesd.layers.fused_moe')


@pytest.mark.parametrize('use_fused', [False, True])
def test_public_wrapper_forwards_all_options_to_moe(monkeypatch, use_fused):
    values = dict(
        hidden_states=torch.randn(2, 4),
        router_logits=torch.randn(2, 3),
        num_experts=3,
        top_k=2,
        w13_weight=torch.randn(3, 4, 8),
        w2_weight=torch.randn(3, 4, 4),
        dispatcher_type='static',
        inputs_sharded=True,
        k_group=2,
        group_count=2,
        group_select_mode=1,
        routing_method='sigmoid',
        renormalize=True,
        routed_scaling_factor=0.5,
        reduce_routed_out=False,
        return_dispatcher_type=True,
    )
    expected = (torch.randn(2, 4), 'static')
    implementation = MagicMock(return_value=expected)
    monkeypatch.setattr(module, 'moe', implementation)
    actual = module.fused_moe(**values, use_fused_op=use_fused)
    assert actual is expected
    bound = inspect.signature(module.fused_moe).bind(**values, use_fused_op=use_fused)
    bound.apply_defaults()
    forwarded = dict(bound.arguments)
    del forwarded['use_fused_op']
    implementation.assert_called_once_with(**forwarded)


@pytest.mark.parametrize('invalid', [1, None, 'yes'])
def test_non_boolean_fused_switch_fails_before_dispatch(monkeypatch, invalid):
    implementation = MagicMock()
    monkeypatch.setattr(module, 'moe', implementation)
    with pytest.raises(ParametersInvalid, match='must be a bool'):
        module.fused_moe(torch.empty(0), torch.empty(0), 1, 1, torch.empty(0), torch.empty(0), use_fused_op=invalid)
    implementation.assert_not_called()
