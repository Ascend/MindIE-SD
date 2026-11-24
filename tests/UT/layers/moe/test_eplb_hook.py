#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""CPU contracts for the optional MoE EPLB hook.

These tests never touch the NPU: the collaborators are plain mocks, so they
guard the hook contract (who is called when, what the payload must carry,
which re-placements are refused and which weights stay applied) rather than
the forward-path wiring or the EPLB algorithm itself.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from mindiesd.layers.moe.eplb_hook import EplbWeightUpdate, MoEEplbHook
from mindiesd.utils import ParametersInvalid
from mindiesd.utils.expert_update import ExpertUpdate


def make_dispatcher(update_flag=False, local_experts=None, payload=None, ep_size=1):
    """Build a dispatcher stand-in with the attributes the hook reads."""
    dispatcher = MagicMock()
    dispatcher.update_flag = update_flag
    dispatcher.ep_size = ep_size
    dispatcher.local_expert_list = list(local_experts if local_experts is not None else [0, 1])
    dispatcher.update_module_weight_and_map.return_value = payload
    return dispatcher


def make_payload(w13, w2, local_experts=None):
    local_experts = list(local_experts if local_experts is not None else [0, 1])
    return (w13, w2, len(local_experts), None, None, local_experts)


class TestExpertUpdateContract(unittest.TestCase):
    """The dispatcher payload is a type, not a positional convention."""

    @staticmethod
    def _weights():
        return (
            torch.nn.ParameterList([torch.zeros(2, 2) for _ in range(2)]),
            torch.nn.ParameterList([torch.zeros(2, 2) for _ in range(2)]),
        )

    def test_typed_payload_is_applied(self):
        weight1, weight2 = self._weights()
        payload = ExpertUpdate(
            weight1=weight1,
            weight2=weight2,
            local_expert_num=2,
            local_expert_list=[0, 1],
        )
        hook = MoEEplbHook(dispatcher=make_dispatcher(update_flag=True, payload=payload))

        update = hook.poll_weight_update()

        self.assertIsNotNone(update)
        self.assertEqual(hook.refused_updates, 0)

    def test_legacy_tuple_is_still_accepted(self):
        weight1, weight2 = self._weights()
        hook = MoEEplbHook(dispatcher=make_dispatcher(update_flag=True, payload=make_payload(weight1, weight2)))

        self.assertIsNotNone(hook.poll_weight_update())

    def test_malformed_payload_names_the_contract(self):
        hook = MoEEplbHook(dispatcher=make_dispatcher(update_flag=True, payload=(torch.zeros(2),)))

        with self.assertRaises(ParametersInvalid) as raised:
            hook.poll_weight_update()
        self.assertIn("ExpertUpdate", str(raised.exception))

    def test_published_layout_accessors(self):
        hook = MoEEplbHook(dispatcher=make_dispatcher())
        self.assertIsNone(hook.published_device_map())

        hook.device_indices_map = torch.tensor([0, 0], dtype=torch.int32)
        self.assertIsNotNone(hook.published_device_map())

        hook.clear_published_layout()
        self.assertIsNone(hook.published_device_map())


class TestMoEEplbHookConstruction(unittest.TestCase):
    """The hook must be opt-in and must not be constructible half-empty."""

    def test_rejects_hook_without_collaborators(self):
        with self.assertRaises(ParametersInvalid):
            MoEEplbHook()

    def test_accepts_collector_only(self):
        self.assertIsNotNone(MoEEplbHook(collector=MagicMock()))

    def test_accepts_dispatcher_only(self):
        self.assertIsNotNone(MoEEplbHook(dispatcher=make_dispatcher()))

    def test_snapshots_the_supported_local_experts(self):
        hook = MoEEplbHook(dispatcher=make_dispatcher(local_experts=[4, 5]))
        self.assertEqual(hook._supported_local_experts, [4, 5])

    def test_keeps_layer_idx_for_logging(self):
        self.assertEqual(MoEEplbHook(collector=MagicMock(), layer_idx=7).layer_idx, 7)


class TestMoEEplbHookOnRouting(unittest.TestCase):
    """on_routing forwards sorted indices and polls the scheduler."""

    def test_flattens_and_sorts_2d_routing_ids(self):
        collector = MagicMock()
        MoEEplbHook(collector=collector).on_routing(torch.tensor([[3, 0], [2, 1]]))
        forwarded = collector.collect_expert_load.call_args.args[0]
        self.assertEqual(tuple(forwarded.shape), (4,))
        self.assertEqual(forwarded.tolist(), [0, 1, 2, 3], "collector requires sorted expert indices")

    def test_sorts_already_flattened_ids(self):
        collector = MagicMock()
        MoEEplbHook(collector=collector).on_routing(torch.tensor([5, 1, 3, 1]))
        self.assertEqual(collector.collect_expert_load.call_args.args[0].tolist(), [1, 1, 3, 5])

    def test_single_index_is_not_sorted(self):
        collector = MagicMock()
        MoEEplbHook(collector=collector).on_routing(torch.tensor([3]))
        self.assertEqual(collector.collect_expert_load.call_args.args[0].tolist(), [3])

    def test_polls_scheduler(self):
        dispatcher = make_dispatcher()
        MoEEplbHook(dispatcher=dispatcher).on_routing(torch.zeros(2, 2, dtype=torch.long))
        dispatcher.check_consistency.assert_called_once()

    def test_does_not_poll_without_dispatcher(self):
        collector = MagicMock()
        hook = MoEEplbHook(collector=collector)

        hook.on_routing(torch.zeros(2, 2, dtype=torch.long))

        # Collection still happens, but there is no dispatcher to ask for a layout.
        self.assertIsNone(hook.poll_weight_update())
        self.assertIsNone(hook.published_device_map())
        self.assertEqual(collector.collect_expert_load.call_count, 1)


class TestMoEEplbHookCountsFastPath(unittest.TestCase):
    """A collector that exposes routed_expert_num gets cumulative counts, not indices."""

    def test_uses_counts_entry_point_when_available(self):
        collector = MagicMock()
        collector.routed_expert_num = 4
        collector.lb_interval = 1
        hook = MoEEplbHook(collector=collector)

        hook.on_routing(torch.tensor([[0, 3], [3, 1]]))

        collector.collect_expert_load.assert_not_called()
        cumulative = collector.collect_expert_counts.call_args.args[0]
        # counts = [1, 1, 0, 2] -> cumulative [1, 2, 2, 4]
        self.assertEqual(cumulative.tolist(), [1, 2, 2, 4])

    def test_falls_back_to_sorted_indices_without_the_entry_point(self):
        collector = MagicMock()
        del collector.routed_expert_num
        del collector.collect_expert_counts
        hook = MoEEplbHook(collector=collector)

        hook.on_routing(torch.tensor([[3, 0], [2, 1]]))

        sent = collector.collect_expert_load.call_args.args[0]
        self.assertEqual(sent.tolist(), [0, 1, 2, 3])


class TestMoEEplbHookWeightUpdate(unittest.TestCase):
    """poll_weight_update only fires when the scheduler flagged a new layout."""

    def test_no_update_when_flag_clear(self):
        dispatcher = make_dispatcher(update_flag=False)
        self.assertIsNone(MoEEplbHook(dispatcher=dispatcher).poll_weight_update())
        dispatcher.update_module_weight_and_map.assert_not_called()

    def test_no_update_without_dispatcher(self):
        self.assertIsNone(MoEEplbHook(collector=MagicMock()).poll_weight_update())

    def test_returns_replaced_weights_when_flag_set(self):
        weight1 = torch.zeros(2, 3)
        weight2 = torch.zeros(3, 2)
        dispatcher = make_dispatcher(update_flag=True, payload=make_payload(weight1, weight2))

        update = MoEEplbHook(dispatcher=dispatcher).poll_weight_update()

        self.assertIsInstance(update, EplbWeightUpdate)
        self.assertIs(update.w13_weight, weight1)
        self.assertIs(update.w2_weight, weight2)

    def test_parameter_list_payload_is_stacked(self):
        local = [torch.full((3, 2), float(i)) for i in range(2)]
        dispatcher = make_dispatcher(
            update_flag=True,
            payload=make_payload(torch.nn.ParameterList(local), torch.nn.ParameterList(local)),
        )

        update = MoEEplbHook(dispatcher=dispatcher).poll_weight_update()

        self.assertEqual(tuple(update.w13_weight.shape), (2, 3, 2))
        self.assertTrue(torch.equal(update.w13_weight[1], local[1]))

    def test_empty_result_is_tolerated(self):
        dispatcher = make_dispatcher(update_flag=True, payload=None)
        self.assertIsNone(MoEEplbHook(dispatcher=dispatcher).poll_weight_update())

    def test_empty_weight_list_is_refused_not_raised(self):
        dispatcher = make_dispatcher(
            update_flag=True,
            local_experts=[],
            payload=(torch.nn.ParameterList([]), torch.nn.ParameterList([]), 0, None, None, []),
        )
        self.assertIsNone(MoEEplbHook(dispatcher=dispatcher).poll_weight_update())


class TestMoEEplbHookMapPath(unittest.TestCase):
    """A published device map lets the hook move experts across ranks."""

    @staticmethod
    def _device_map():
        # 8 experts on 4 ranks; rank 0 ends up owning experts 5 and 7.
        return torch.tensor([1, 2, 3, 1, 2, 0, 3, 0], dtype=torch.int32)

    @staticmethod
    def _weights(value_by_row):
        rows = [torch.full((2, 2), float(value)) for value in value_by_row]
        return torch.nn.ParameterList(rows), torch.nn.ParameterList(rows)

    def test_sorts_weight_rows_by_ascending_expert_id(self):
        device_map = self._device_map()
        weight1, weight2 = self._weights([7.0, 5.0])  # dispatcher order: 7 then 5
        payload = ExpertUpdate(
            weight1=weight1,
            weight2=weight2,
            local_expert_num=2,
            device_indices_map=device_map,
            local_expert_list=[7, 5],
        )
        hook = MoEEplbHook(dispatcher=make_dispatcher(update_flag=True, payload=payload, ep_size=4))

        update = hook.poll_weight_update()

        self.assertIsNotNone(update)
        self.assertEqual(hook.refused_updates, 0)
        self.assertEqual(update.local_expert_list, [5, 7], "rows follow ascending expert id")
        self.assertEqual(int(update.w13_weight[0].sum()), 20, "row 0 is expert 5")
        self.assertEqual(int(update.w13_weight[1].sum()), 28, "row 1 is expert 7")
        self.assertTrue(torch.equal(hook.published_device_map(), device_map))

    def test_rejects_a_layout_with_a_different_expert_count(self):
        weight1, weight2 = self._weights([0.0, 0.0, 0.0])
        payload = ExpertUpdate(
            weight1=weight1,
            weight2=weight2,
            local_expert_num=3,
            device_indices_map=self._device_map(),
            local_expert_list=[7, 5, 3],
        )
        hook = MoEEplbHook(dispatcher=make_dispatcher(update_flag=True, payload=payload, ep_size=4))

        self.assertIsNone(hook.poll_weight_update())
        self.assertEqual(hook.refused_updates, 1)
        self.assertIsNone(hook.published_device_map(), "a refused update must not publish a layout")

    def test_rejects_weight_rows_that_do_not_match(self):
        weight1, weight2 = self._weights([0.0])  # one row, but the list names two experts
        payload = ExpertUpdate(
            weight1=weight1,
            weight2=weight2,
            local_expert_num=2,
            device_indices_map=self._device_map(),
            local_expert_list=[7, 5],
        )
        hook = MoEEplbHook(dispatcher=make_dispatcher(update_flag=True, payload=payload, ep_size=4))

        self.assertIsNone(hook.poll_weight_update())
        self.assertEqual(hook.refused_updates, 1)

    def test_rejects_a_lopsided_device_map(self):
        # 8 experts over 4 ranks must be 2 per rank; this map gives rank 0 five.
        lopsided = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3], dtype=torch.int32)
        weight1, weight2 = self._weights([0.0, 0.0])
        payload = ExpertUpdate(
            weight1=weight1,
            weight2=weight2,
            local_expert_num=2,
            device_indices_map=lopsided,
            local_expert_list=[5, 6],
        )
        hook = MoEEplbHook(dispatcher=make_dispatcher(update_flag=True, payload=payload, ep_size=4))

        self.assertIsNone(hook.poll_weight_update())
        self.assertEqual(hook.refused_updates, 1, "an uneven split would desynchronise the all-to-all")
        self.assertIsNone(hook.published_device_map())


class TestMoEEplbHookCollectiveScope(unittest.TestCase):
    """The scope decision must never be taken on one rank alone."""

    def test_reduction_failure_refuses_the_update(self):
        hook = MoEEplbHook(collector=MagicMock(lb_interval=1))
        fake_dist = MagicMock()
        fake_dist.is_available.return_value = True
        fake_dist.is_initialized.return_value = True
        fake_dist.all_reduce.side_effect = RuntimeError("MIN unsupported")
        with (
            patch("mindiesd.layers.moe.eplb_hook.dist", fake_dist),
            patch("mindiesd.layers.moe.eplb_hook.logger") as logger,
        ):
            accepted = hook._collective_scope(True, torch.ones(1))

        self.assertFalse(accepted, "a failed reduction must refuse, not trust the local rank")
        logger.warning.assert_called_once()

    def test_reduction_verdict_is_honoured(self):
        hook = MoEEplbHook(collector=MagicMock(lb_interval=1))

        def all_reduce(flag, op=None):
            flag[0] = 0  # another rank rejected the update

        fake_dist = MagicMock()
        fake_dist.is_available.return_value = True
        fake_dist.is_initialized.return_value = True
        fake_dist.all_reduce.side_effect = all_reduce
        with patch("mindiesd.layers.moe.eplb_hook.dist", fake_dist):
            self.assertFalse(hook._collective_scope(True, torch.ones(1)))

        def accept(flag, op=None):
            flag[0] = 1

        fake_dist.all_reduce.side_effect = accept
        with patch("mindiesd.layers.moe.eplb_hook.dist", fake_dist):
            self.assertTrue(hook._collective_scope(False, torch.ones(1)))


class TestMoEEplbHookScopeGuard(unittest.TestCase):
    """Updates that would silently mis-route tokens must be refused."""

    def test_refuses_layout_that_changes_the_local_expert_set(self):
        weight1 = torch.zeros(2, 3)
        weight2 = torch.zeros(3, 2)
        dispatcher = make_dispatcher(
            update_flag=True,
            local_experts=[0, 1],
            payload=make_payload(weight1, weight2, local_experts=[1, 3]),
        )
        hook = MoEEplbHook(dispatcher=dispatcher)

        self.assertIsNone(hook.poll_weight_update())
        self.assertEqual(hook.refused_updates, 1)

    def test_refuses_weight_rows_that_do_not_match_the_local_expert_count(self):
        dispatcher = make_dispatcher(
            update_flag=True,
            local_experts=[0, 1],
            payload=make_payload(torch.zeros(3, 3), torch.zeros(3, 2), local_experts=[0, 1]),
        )
        hook = MoEEplbHook(dispatcher=dispatcher)

        self.assertIsNone(hook.poll_weight_update())
        self.assertEqual(hook.refused_updates, 1)

    def test_accepts_layout_that_keeps_the_local_expert_set(self):
        dispatcher = make_dispatcher(
            update_flag=True,
            local_experts=[0, 1],
            payload=make_payload(torch.ones(2, 3), torch.ones(3, 2), local_experts=[0, 1]),
        )
        hook = MoEEplbHook(dispatcher=dispatcher)

        self.assertIsNotNone(hook.poll_weight_update())
        self.assertEqual(hook.refused_updates, 0)


class TestMoEEplbHookKeepsAppliedWeights(unittest.TestCase):
    """The layout stays in force, so the hook must keep the weights it belongs to."""

    @staticmethod
    def _hook():
        return MoEEplbHook(collector=MagicMock(lb_interval=1))

    def test_defaults_before_any_update(self):
        hook = self._hook()
        w13, w2 = torch.ones(2, 8, 16), torch.ones(2, 8, 8)

        got13, got2 = hook.current_weights(w13, w2)

        self.assertIs(got13, w13)
        self.assertIs(got2, w2)

    def test_remembered_weights_replace_the_defaults(self):
        hook = self._hook()
        new13, new2 = torch.zeros(2, 8, 16), torch.zeros(2, 8, 8)
        hook.remember_weights(EplbWeightUpdate(w13_weight=new13, w2_weight=new2))

        got13, got2 = hook.current_weights(torch.ones(2, 8, 16), torch.ones(2, 8, 8))

        self.assertIs(got13, new13)
        self.assertIs(got2, new2)

    def test_shape_change_falls_back_to_the_defaults(self):
        hook = self._hook()
        hook.remember_weights(EplbWeightUpdate(w13_weight=torch.zeros(2, 8, 16), w2_weight=torch.zeros(2, 8, 8)))
        w13, w2 = torch.ones(4, 8, 16), torch.ones(4, 8, 8)

        got13, got2 = hook.current_weights(w13, w2)

        self.assertIs(got13, w13, "a stale shape must never be handed to the forward")
        self.assertIs(got2, w2)

    def test_collective_scope_without_a_process_group(self):
        hook = self._hook()
        self.assertTrue(hook._collective_scope(True, torch.ones(1)))
        self.assertFalse(hook._collective_scope(False, torch.ones(1)))

    def test_check_interval_follows_the_collector(self):
        dispatcher = MagicMock()
        hook = MoEEplbHook(collector=MagicMock(lb_interval=3), dispatcher=dispatcher)

        for _ in range(5):
            hook.on_routing(torch.zeros(2, 2, dtype=torch.int64))

        self.assertEqual(dispatcher.check_consistency.call_count, 1, "one check per lb_interval routing calls")


if __name__ == "__main__":
    unittest.main()
