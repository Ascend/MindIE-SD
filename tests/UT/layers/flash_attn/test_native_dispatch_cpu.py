# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

import importlib
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu

from mindiesd.compilation.compiliation_config import CompilationConfig
from mindiesd.layers.flash_attn.common import AttentionParam
from mindiesd.utils import ParametersInvalid

fas = importlib.import_module('mindiesd.layers.flash_attn.fused_attn_score')
pfa = importlib.import_module('mindiesd.layers.flash_attn.prompt_flash_attn')
pre = importlib.import_module('mindiesd.layers.flash_attn.ascend_laser_preprocess')


@pytest.mark.parametrize('backend', ['fas', 'pfa'])
@pytest.mark.parametrize('layout', ['BNSD', 'BSND', 'BSH'])
@pytest.mark.parametrize('head_first', [False, True])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('graph', ['off', 'only', 'compile'])
def test_native_boundary_layout_masks_and_graph_sync(monkeypatch, backend, layout, head_first, masked, graph):
    monkeypatch.setattr(CompilationConfig, 'aclgraph_only', graph == 'only')
    monkeypatch.setattr(CompilationConfig, 'aclgraph_with_compile', graph == 'compile')
    monkeypatch.setattr(pfa, 'is_a5_device', lambda: False)
    boundary = MagicMock(side_effect=lambda q, k, v, **kwargs: (q * 2,) if backend == 'fas' else q * 2)
    monkeypatch.setattr(
        torch_npu, 'npu_fusion_attention' if backend == 'fas' else 'npu_prompt_flash_attention', boundary
    )
    param = AttentionParam(2, 3, 5, 4, 4, torch.float32, head_first)
    query = torch.randn(2, 3, 4, 5)
    if not head_first:
        query = query.transpose(1, 2)
    mask = torch.eye(4, dtype=torch.bool) if masked else None
    cls = fas.FlashAttentionScore if backend == 'fas' else pfa.PromptFlashAttention
    assert cls.is_supported_shape(param)
    actual = getattr(cls, 'forward_attn_' + layout.lower())(param, query, query, query, mask, 0.5)
    torch.testing.assert_close(actual, query * 2)
    args, kwargs = boundary.call_args
    expected_shape = {'BNSD': (2, 3, 4, 5), 'BSND': (2, 4, 3, 5), 'BSH': (2, 4, 15)}[layout]
    assert args[0].shape == expected_shape
    assert args[0].device.type == 'cpu'
    assert kwargs['input_layout'] == layout
    if masked:
        torch.testing.assert_close(kwargs['atten_mask'], ~mask)
    else:
        assert kwargs['atten_mask'] is None
    assert kwargs['scale' if backend == 'fas' else 'scale_value'] == 0.5
    if backend == 'fas':
        assert kwargs['sync'] == (graph != 'off')


@pytest.mark.parametrize('layout', ['bnsd', 'bsnd', 'bsh'])
def test_pfa_rejects_a5_before_native_call(monkeypatch, layout):
    monkeypatch.setattr(pfa, 'is_a5_device', lambda: True)
    native = MagicMock()
    monkeypatch.setattr(torch_npu, 'npu_prompt_flash_attention', native)
    with pytest.raises(ParametersInvalid, match='not supported on A5'):
        getattr(pfa.PromptFlashAttention, 'forward_attn_' + layout)(None, None, None, None)
    native.assert_not_called()


def test_laser_preprocess_returns_contiguous_cpu_outputs(monkeypatch):
    monkeypatch.setattr(pre, 'is_a5_device', lambda: False)
    query = torch.randn(2, 4, 3, 5)
    outputs = tuple(query.transpose(0, 1) for _ in range(3))
    native = MagicMock(return_value=outputs)
    monkeypatch.setattr(pre.ops, 'laser_attention_preprocess', native)
    actual = pre.la_preprocess(query, query, query, align_len=128)
    native.assert_called_once_with(query, query, query, 128)
    for output, expected in zip(actual, outputs):
        assert output.is_contiguous()
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize('kind', ['a5', 'query_rank', 'key_rank', 'value_rank', 'key_heads', 'value_heads'])
def test_laser_preprocess_rejects_invalid_contract_before_kernel(monkeypatch, kind):
    monkeypatch.setattr(pre, 'is_a5_device', lambda: kind == 'a5')
    query = torch.empty(1, 4, 3, 5)
    key, value = query, query
    if kind == 'query_rank':
        query = query.flatten()
    elif kind == 'key_rank':
        key = key.flatten()
    elif kind == 'value_rank':
        value = value.flatten()
    elif kind == 'key_heads':
        key = torch.empty(1, 4, 2, 5)
    elif kind == 'value_heads':
        value = torch.empty(1, 4, 3, 6)
    native = MagicMock()
    monkeypatch.setattr(pre.ops, 'laser_attention_preprocess', native)
    with pytest.raises(ParametersInvalid):
        pre.la_preprocess(query, key, value)
    native.assert_not_called()
