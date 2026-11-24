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
"""CPU contracts for wiring the EPLB hook into the MoE forward path.

These tests cover the integration side: what `fused_moe()` forwards, which
dispatcher `moe()` picks when EPLB is on, and how the forward path rebuilds
weights after an update. They run on CPU with the dispatcher mocked out.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from mindiesd.layers.fused_moe import fused_moe
from mindiesd.layers.moe.eplb_hook import EplbWeightUpdate, MoEEplbHook
from mindiesd.layers.moe.moe import moe, resolve_dispatcher_class
from mindiesd.layers.moe.moe_context import MoECommType
from mindiesd.layers.moe.token_dispatcher import DynamicDispatcher, StaticDispatcher
from mindiesd.quantization.mode import QuantAlgorithm
from mindiesd.utils import ParametersInvalid


class TestFusedMoeEplbWiring(unittest.TestCase):
    """The public entry must forward the hook, and default to disabled."""

    @staticmethod
    def _call_fused_moe(**overrides):
        kwargs = {
            "hidden_states": torch.zeros(4, 8),
            "router_logits": torch.zeros(4, 4),
            "num_experts": 4,
            "top_k": 2,
            "w13_weight": torch.zeros(4, 8),
            "w2_weight": torch.zeros(4, 8),
        }
        kwargs.update(overrides)
        return kwargs

    def test_hook_is_forwarded_to_moe(self):
        hook = MoEEplbHook(collector=MagicMock())
        with patch("mindiesd.layers.fused_moe.moe") as mocked_moe:
            fused_moe(**self._call_fused_moe(eplb_hook=hook))
        self.assertIs(mocked_moe.call_args.kwargs["eplb_hook"], hook)

    def test_hook_defaults_to_none(self):
        with patch("mindiesd.layers.fused_moe.moe") as mocked_moe:
            fused_moe(**self._call_fused_moe())
        self.assertIsNone(mocked_moe.call_args.kwargs["eplb_hook"])


class TestDispatcherSelectionWithEplb(unittest.TestCase):
    """An EPLB hook must run on the map-aware dispatcher."""

    def _ep(self, ep_size=4, comm=MoECommType.EP):
        return (
            patch("mindiesd.layers.moe.moe.get_moe_comm_type", return_value=comm),
            patch("mindiesd.layers.moe.moe.get_moe_group", return_value=object()),
            patch("mindiesd.layers.moe.moe.dist.get_world_size", return_value=ep_size),
        )

    def test_top_k_not_below_ep_size_switches_to_dynamic_when_eplb_is_on(self):
        comm, group, size = self._ep(ep_size=4)
        with comm, group, size:
            # Without EPLB the historical rule picks the static path for top_k >= ep_size.
            self.assertIs(resolve_dispatcher_class(top_k=8, eplb=False), StaticDispatcher)
            # With EPLB it must not: the static dispatcher ignores the published layout.
            self.assertIs(resolve_dispatcher_class(top_k=8, eplb=True), DynamicDispatcher)

    def test_explicit_static_dispatcher_is_rejected_with_eplb(self):
        comm, group, size = self._ep()
        with comm, group, size, self.assertRaises(ParametersInvalid):
            resolve_dispatcher_class(top_k=8, dispatcher_type="static", eplb=True)


class TestMoeForwardEplbBranch(unittest.TestCase):
    """Exercise the real `moe()` branch: collect -> swap weights -> rebuild payload."""

    @staticmethod
    def _moe_kwargs(hook, **overrides):
        kwargs = {
            "hidden_states": torch.zeros(4, 8),
            "router_logits": torch.zeros(4, 4),
            "num_experts": 4,
            "top_k": 2,
            "w13_weight": torch.zeros(4, 8, 16),
            "w2_weight": torch.zeros(4, 8, 8),
            "dispatcher_type": "static",
            "eplb_hook": hook,
        }
        kwargs.update(overrides)
        # A MagicMock hook would return a Mock from `current_weights`, which cannot be
        # unpacked; make the default identity so tests exercise the real call site.
        if isinstance(getattr(hook, "current_weights", None), MagicMock) and isinstance(
            hook.current_weights.return_value, MagicMock
        ):
            hook.current_weights.side_effect = lambda w13, w2: (w13, w2)
        return kwargs

    def _run_moe(self, hook, **overrides):
        topk_ids = torch.zeros(4, 2, dtype=torch.int32)
        with (
            patch("mindiesd.layers.moe.moe.build_moe_weights", return_value=MagicMock()) as build_weights,
            patch("mindiesd.layers.moe.moe.build_mlp_compute_input", return_value=MagicMock()),
            patch("mindiesd.layers.moe.moe.unified_apply_mlp", return_value=torch.zeros(4, 8)),
            patch("mindiesd.layers.moe.moe.select_experts", return_value=(torch.ones(4, 2), topk_ids)),
            patch.object(StaticDispatcher, "dispatch", return_value=MagicMock()),
            patch.object(StaticDispatcher, "combine", return_value=torch.zeros(4, 8)),
            patch.object(StaticDispatcher, "finalize", return_value=torch.zeros(4, 8)),
        ):
            moe(**self._moe_kwargs(hook, **overrides))
        return build_weights

    def _capture_dispatch_input(self, hook, ep_size=1):
        """Run moe() with the dispatch-input builder replaced by a recorder."""
        captured = []

        def fake_builder(**kwargs):
            captured.append(kwargs)
            return MagicMock()

        ep_group = object() if ep_size > 1 else None
        with (
            patch("mindiesd.layers.moe.moe.build_token_dispatch_input", side_effect=fake_builder),
            patch("mindiesd.layers.moe.moe.build_moe_weights", return_value=MagicMock()),
            patch("mindiesd.layers.moe.moe.build_mlp_compute_input", return_value=MagicMock()),
            patch("mindiesd.layers.moe.moe.unified_apply_mlp", return_value=torch.zeros(4, 8)),
            patch(
                "mindiesd.layers.moe.moe.select_experts",
                return_value=(torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int32)),
            ),
            patch.object(StaticDispatcher, "dispatch", return_value=MagicMock()),
            patch.object(StaticDispatcher, "combine", return_value=torch.zeros(4, 8)),
            patch.object(StaticDispatcher, "finalize", return_value=torch.zeros(4, 8)),
            patch.object(DynamicDispatcher, "dispatch", return_value=MagicMock()),
            patch.object(DynamicDispatcher, "combine", return_value=torch.zeros(4, 8)),
            patch.object(DynamicDispatcher, "finalize", return_value=torch.zeros(4, 8)),
            # ep_group is only accepted as a real ProcessGroup, which a CPU-only unit
            # test cannot create; the validator and the world size are stubbed so the
            # branch that decides whether to forward the layout still runs.
            patch("mindiesd.layers.moe.moe.validate_moe_inputs", return_value=QuantAlgorithm.NO_QUANT),
            # The real context setter asks the group for its size; a CPU-only test has
            # no process group, so both module-level references are stubbed.
            patch("mindiesd.layers.moe.moe.dist.get_world_size", return_value=ep_size),
            patch("mindiesd.layers.moe.moe_context.dist.get_world_size", return_value=ep_size),
            # The dynamic dispatcher asks for its rank while preparing; there is no
            # process group in a CPU-only unit test.
            patch("mindiesd.layers.moe.token_dispatcher.dist.get_rank", return_value=0),
            patch("mindiesd.layers.moe.token_dispatcher.dist.get_world_size", return_value=ep_size),
            patch("mindiesd.layers.moe.moe._log_moe_config_once"),
        ):
            # Under EP an EPLB hook must resolve to the dynamic (map-aware) dispatcher,
            # so the type is deliberately left unset here.
            moe(**self._moe_kwargs(hook, ep_group=ep_group, dispatcher_type=None))
        return captured[0]

    def test_single_rank_does_not_consume_a_published_layout(self):
        published = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        hook = MagicMock()
        hook.poll_weight_update.return_value = None
        hook.published_device_map.return_value = published

        captured = self._capture_dispatch_input(hook)

        self.assertIsNone(captured["device_indices_map"], "the map must not leak into the single-rank path")

    def test_hook_runs_and_weights_are_rebuilt(self):
        new_w13 = torch.ones(2, 8, 16)
        new_w2 = torch.ones(2, 8, 8)
        hook = MagicMock()
        hook.poll_weight_update.return_value = EplbWeightUpdate(w13_weight=new_w13, w2_weight=new_w2)

        build_weights = self._run_moe(hook)

        hook.on_routing.assert_called_once()
        hook.poll_weight_update.assert_called_once()
        self.assertEqual(build_weights.call_count, 2, "weights must be rebuilt after a re-placement")
        self.assertIs(build_weights.call_args.args[0], new_w13)
        self.assertIs(build_weights.call_args.args[1], new_w2)

    def test_no_update_keeps_single_weight_build(self):
        hook = MagicMock()
        hook.poll_weight_update.return_value = None

        build_weights = self._run_moe(hook)

        hook.on_routing.assert_called_once()
        self.assertEqual(build_weights.call_count, 1, "no re-placement must not rebuild the payload")

    def test_biased_layer_skips_the_replacement(self):
        hook = MagicMock()
        hook.poll_weight_update.return_value = EplbWeightUpdate(
            w13_weight=torch.ones(4, 8, 16), w2_weight=torch.ones(4, 8, 8)
        )

        build_weights = self._run_moe(hook, w13_bias=torch.zeros(4, 16))

        self.assertEqual(build_weights.call_count, 1, "bias cannot be permuted yet, so weights must stay")
        self.assertIsNot(build_weights.call_args.args[0], hook.poll_weight_update.return_value.w13_weight)
        # The refused layout must not stay in force, and the refused weights must not
        # be remembered for later calls.
        hook.clear_published_layout.assert_called_once()
        hook.remember_weights.assert_not_called()

    def test_applied_weights_are_used_on_later_calls(self):
        applied13 = torch.ones(2, 8, 16)
        applied2 = torch.ones(2, 8, 8)
        hook = MagicMock()
        hook.poll_weight_update.return_value = None
        hook.current_weights.return_value = (applied13, applied2)

        build_weights = self._run_moe(hook)

        self.assertIs(build_weights.call_args.args[0], applied13, "the stored weights must win over the stale ones")
        self.assertIs(build_weights.call_args.args[1], applied2)

    def test_ep_publishes_the_layout_to_the_dispatcher(self):
        published = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        hook = MagicMock()
        hook.poll_weight_update.return_value = None
        hook.published_device_map.return_value = published

        captured = self._capture_dispatch_input(hook, ep_size=2)

        self.assertIsNotNone(captured["device_indices_map"], "an EP run must forward the published layout")
        self.assertTrue(torch.equal(captured["device_indices_map"], published))

    def test_quantized_layer_does_not_ask_for_the_map_aware_dispatcher(self):
        # A quantized layer never applies a re-placement (its scales cannot be
        # permuted), so forcing the dynamic dispatcher there would only change the
        # numeric path and cost.
        hook = MagicMock()
        hook.poll_weight_update.return_value = None
        with (
            patch("mindiesd.layers.moe.moe.resolve_dispatcher_class", return_value=StaticDispatcher) as resolver,
            patch("mindiesd.layers.moe.moe.validate_moe_inputs", return_value=QuantAlgorithm.NO_QUANT),
        ):
            self._run_moe(hook, w13_weight_scale=torch.ones(4, 16))
        self.assertFalse(resolver.call_args.kwargs["eplb"], "a layer with per-expert scales cannot take an update")

        with (
            patch("mindiesd.layers.moe.moe.resolve_dispatcher_class", return_value=StaticDispatcher) as resolver,
            patch("mindiesd.layers.moe.moe.validate_moe_inputs", return_value=QuantAlgorithm.NO_QUANT),
        ):
            self._run_moe(hook)
        self.assertTrue(resolver.call_args.kwargs["eplb"], "a plain layer must still use the map-aware dispatcher")

    def test_rejects_object_that_is_not_a_hook(self):
        with self.assertRaises(ParametersInvalid):
            moe(**self._moe_kwargs(object()))

    def test_rejects_a_hook_without_the_weight_persistence_methods(self):
        class OldProtocolHook:
            def on_routing(self, topk_ids):
                pass

            def poll_weight_update(self):
                return None

            def published_device_map(self):
                return None

            def clear_published_layout(self):
                pass

        with self.assertRaises(ParametersInvalid) as raised:
            moe(**self._moe_kwargs(OldProtocolHook()))
        self.assertIn("current_weights", str(raised.exception))
