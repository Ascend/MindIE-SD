# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

from unittest.mock import MagicMock

import pytest
import torch
import torch_npu

from mindiesd.layers.moe.experts_selector import select_experts
from mindiesd.layers.moe.moe_dataclass import MoERoutingInput


@pytest.mark.parametrize(
    'grouped,norm,renormalize,scale',
    [
        (False, 0, False, 1.0),
        (False, 0, True, 1.75),
        (True, 0, True, 1.0),
        (False, 1, True, 0.5),
        (True, 1, False, 1.0),
    ],
)
def test_gating_dispatch_and_postprocessing_on_cpu(monkeypatch, grouped, norm, renormalize, scale):
    weights = torch.tensor([[[0.2, 0.3]], [[0.1, 0.7]]])
    ids = torch.tensor([[[3, 1]], [[2, 0]]], dtype=torch.int64)
    softmax = MagicMock(side_effect=lambda *args, **kwargs: (weights.clone(), ids.clone(), None))
    generic = MagicMock(side_effect=lambda *args, **kwargs: (weights.clone(), ids.clone(), None))
    monkeypatch.setattr(torch_npu, 'npu_moe_gating_top_k_softmax', softmax)
    monkeypatch.setattr(torch_npu, 'npu_moe_gating_top_k', generic)
    logits = torch.randn(2, 4)
    routing = MoERoutingInput(
        torch.randn(2, 8),
        logits,
        2,
        renormalize=renormalize,
        norm_type=norm,
        k_group=2 if grouped else 1,
        group_count=2 if grouped else 1,
        routed_scaling_factor=scale,
    )
    actual_weights, actual_ids = select_experts(routing)
    expected = weights / weights.sum(-1, keepdim=True) if norm == 0 and renormalize else weights
    torch.testing.assert_close(actual_weights, (expected * scale).reshape(2, 2))
    torch.testing.assert_close(actual_ids, ids.reshape(2, 2).to(torch.int32))
    assert actual_weights.device.type == 'cpu'
    if not grouped and norm == 0:
        softmax.assert_called_once_with(logits, None, k=2)
        generic.assert_not_called()
    else:
        generic.assert_called_once_with(
            logits,
            k=2,
            k_group=routing.k_group,
            group_count=routing.group_count,
            group_select_mode=0,
            norm_type=norm,
            renorm=0,
            out_flag=False,
            eps=1e-20,
        )
        softmax.assert_not_called()
