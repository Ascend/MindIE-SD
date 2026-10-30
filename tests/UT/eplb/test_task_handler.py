# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch_npu

from mindiesd.eplb.task_handler import handle_profile_task, handle_update_layout_task, handle_unknown_task
from mindiesd.utils.exception import ParametersInvalid


@pytest.mark.parametrize("index,has_collector,uploads", [(None, True, 0), (2, True, 0), (0, False, 0), (0, True, 1)])
def test_profile_reads_cpu_load_and_dispatcher_under_lock(index, has_collector, uploads):
    queue = MagicMock()
    collector = MagicMock()
    collector.get_expert_load.return_value = torch.tensor([3, 8])
    dispatcher = SimpleNamespace(update_lock=nullcontext(), local_expert_list=[2, 4])
    handle_profile_task(
        SimpleNamespace(moe_layer_idx=index), queue, [collector if has_collector else None], [dispatcher], object()
    )
    assert queue.put.call_count == uploads
    if uploads:
        payload = queue.put.call_args.args[0]
        np.testing.assert_array_equal(payload['load'], [3, 8])
        assert payload['local_expert_list'] == [2, 4]
        assert payload['moe_layer_idx'] == 0


@pytest.mark.parametrize("index,already_valid", [(None, False), (3, False), (0, True), (0, False)])
def test_layout_copies_selected_cpu_weights_and_maps(monkeypatch, index, already_valid):
    stream = object()
    context = MagicMock(side_effect=lambda value: nullcontext())
    monkeypatch.setattr(torch_npu.npu, 'stream', context)
    dispatcher = SimpleNamespace(
        update_valid_tensor=torch.tensor([int(already_valid)]),
        weight1_list_cpu=[torch.full((2, 2), float(i)) for i in range(3)],
        weight2_list_cpu=[torch.full((2, 2), float(i + 10)) for i in range(3)],
        copy_module_weight_and_map=MagicMock(),
    )
    data = dict(
        device_indices=[[0, 1]],
        local_expert_indices=[[2, 0]],
        local_expert_list=[2, 0],
        expert_trans_tensor=torch.eye(3),
    )
    handle_update_layout_task(SimpleNamespace(moe_layer_idx=index, data=data), None, [], [dispatcher], stream)
    if index != 0 or already_valid:
        dispatcher.copy_module_weight_and_map.assert_not_called()
        context.assert_not_called()
    else:
        context.assert_called_once_with(stream)
        dispatcher.copy_module_weight_and_map.assert_called_once()
        kwargs = dispatcher.copy_module_weight_and_map.call_args.kwargs
        torch.testing.assert_close(kwargs['weight1'][0], torch.full((2, 2), 2.0))
        torch.testing.assert_close(kwargs['weight2'][1], torch.full((2, 2), 10.0))
        assert kwargs['weight1'][0].device.type == 'cpu'
        assert kwargs['local_expert_list'] == [2, 0]
        assert kwargs['layer_idx'] == 0
        torch.testing.assert_close(kwargs['device_indices_map'], torch.tensor([[0, 1]]))
        torch.testing.assert_close(kwargs['local_expert_indices_map'], torch.tensor([[2, 0]]))


def test_unknown_task_is_explicit_error():
    with pytest.raises(ParametersInvalid, match='Unknown task type'):
        handle_unknown_task(SimpleNamespace(task_type='invalid'), None, [], [], None)
