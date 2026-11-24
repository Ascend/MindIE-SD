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
"""Optional EPLB (dynamic expert load balancing) hook for the MoE forward path.

The hook is *injected* by the caller and never constructed implicitly, so the
default `eplb_hook=None` leaves the legacy forward path unchanged apart from one
identity check. It is the framework-side seam of the adaptation flow documented in
`docs/zh/features/DyEPLB.md` (English page: `docs/en/features/DyEPLB.md`):
collect per-expert load right after routing, then hand the re-placed expert
weights to the grouped matrix multiplication.

Two collaborators from `mindiesd.eplb` are wrapped:

* `ExpertLoadCollector` -- accumulates how many tokens hit every expert.
* `DynamicDispatcher` -- checks whether the scheduler published a new layout and
  hands back the re-placed expert weights.

Only object protocols are required, so both collaborators stay replaceable (and
testable) without importing the NPU-only implementations.

Scope limit (enforced, not just documented):
    A re-placement is applied as long as every rank keeps the same *number* of
    experts; which experts a rank owns may change. The token dispatcher consumes
    `device_indices_map` to send each token to the new owner and orders the
    exchange by ascending global expert id (see `token_dispatcher`). A per-rank
    row map is not published, because a rank only knows the rows of its own
    experts and stores -1 for the rest. Updates that change the number of experts
    per rank are refused with a warning; so are layered with a per-expert bias or
    quantization scale, whose extra tensors cannot be permuted yet.

Naming note: `mindiesd.eplb.dispatcher.DynamicDispatcher` (expert re-placement)
and `mindiesd.layers.moe.token_dispatcher.DynamicDispatcher` (token dispatch) are
different classes that share a name; import them with an alias.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ...utils import ParametersInvalid
from ...utils.expert_update import ExpertUpdate
from ...utils.logs.logging import logger
from .expert_layout import sort_order


def _as_expert_update(payload) -> ExpertUpdate:
    """Accept the typed payload or the historical 6-tuple and normalise it."""
    if isinstance(payload, ExpertUpdate):
        return payload
    try:
        weight1, weight2, local_num, device_map, local_map, local_list = payload
    except (TypeError, ValueError) as error:
        raise ParametersInvalid(
            "dispatcher.update_module_weight_and_map() must return ExpertUpdate or the legacy 6-tuple "
            "(weight1, weight2, local_expert_num, device_indices_map, local_expert_indices_map, "
            f"local_expert_list), but got {type(payload).__name__}."
        ) from error
    return ExpertUpdate(
        weight1=weight1,
        weight2=weight2,
        local_expert_num=local_num,
        device_indices_map=device_map,
        local_expert_indices_map=local_map,
        local_expert_list=local_list,
    )


def _as_weight_tensor(weight):
    """Normalise a dispatcher weight payload to a single stacked tensor.

    `mindiesd.eplb.DispatcherBase.update_module_weight_and_map` returns
    `nn.ParameterList` objects (one tensor per local expert), while the MoE MLP
    path expects one dense `[local_experts, ...]` tensor. Both shapes are
    accepted here so the hook does not force a dispatcher implementation detail
    onto every caller.

    Raises:
        RuntimeError: If `weight` is an empty sequence; `torch.stack` cannot
            infer a shape from it. Callers turn this into a skipped update.
    """
    if weight is None:
        raise RuntimeError(
            "dispatcher.update_module_weight_and_map() returned no weights; "
            "the payload must carry one tensor per local expert"
        )
    if torch.is_tensor(weight):
        return weight
    return torch.stack(list(weight))


@dataclass
class EplbWeightUpdate:
    """Expert weights produced by one EPLB round.

    Attributes:
        w13_weight (torch.Tensor):
            Re-placed gate/up projection weights, row order = new local expert order.
        w2_weight (torch.Tensor):
            Re-placed down projection weights.
        device_indices_map (torch.Tensor | None):
            `[num_experts]` int tensor mapping a global expert id to the rank that
            now holds it. `None` when the dispatcher does not publish maps.
        local_expert_num (int | None):
            Number of experts this rank holds after the re-placement.
        local_expert_list (list | None):
            Global expert ids held by this rank, in weight-row order.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    device_indices_map: torch.Tensor | None = None
    local_expert_num: int | None = None
    local_expert_list: list | None = None


class MoEEplbHook:
    """Bridge between the MoE forward path and `mindiesd.eplb`.

    One hook instance per MoE layer: the real `DynamicDispatcher` keeps
    per-layer state (`module_state`, `update_lock`, weight lists), and
    `update_flag` is cleared by the first poll, so sharing a hook across layers
    would leave the remaining layers un-updated.

    Args:
        collector (optional):
            Object exposing `collect_expert_load(indices_expert)`. Pass `None`
            to disable load collection.
        dispatcher (optional):
            Object exposing `check_consistency()`, the `update_flag` attribute,
            `local_expert_list` and `update_module_weight_and_map()`. Pass
            `None` to disable weight re-placement.
        layer_idx (int, optional):
            MoE layer index, used for logging and to correlate with the
            dispatcher's `layer_idx`.

    Raises:
        ParametersInvalid: If neither `collector` nor `dispatcher` is given. An
            inert hook would only add a check to the hot path, so it is rejected.
    """

    def __init__(self, collector=None, dispatcher=None, layer_idx: int = 0):
        if collector is None and dispatcher is None:
            raise ParametersInvalid(
                "MoEEplbHook requires at least one of collector or dispatcher; "
                "leave eplb_hook=None on fused_moe to disable EPLB."
            )
        self.collector = collector
        self.dispatcher = dispatcher
        self.layer_idx = layer_idx
        self.refused_updates = 0
        # The consistency check is one device-side all-reduce plus a host sync, so it
        # follows the collector's reporting cadence: the scheduler cannot publish a
        # new layout faster than that anyway.
        self.check_interval = max(1, int(getattr(collector, "lb_interval", 1) or 1))
        self._routing_calls = 0
        # Current "global expert id -> rank" map. It has to stay effective for every
        # following forward, not only for the call that applied the re-placement:
        # `applied_weights` below keeps the matching weight rows, because the caller's
        # tensors are stale after the first re-placement.
        self.device_indices_map = None
        # Weights that belong to the layout above. The MoE forward keeps the layout
        # in force after a re-placement, so the caller's original tensors go stale;
        # `moe()` stores the applied tensors here and reads them back on the next
        # call (`current_weights`), otherwise the next forward would combine the new
        # layout with the old weight rows.
        self.applied_weights = None
        # Snapshot of the local expert set the forward path assumes. The snapshot
        # is taken at construction because the dispatcher mutates
        # `local_expert_list` on every applied update.
        self._supported_local_experts = None
        if dispatcher is not None and hasattr(dispatcher, "local_expert_list"):
            self._supported_local_experts = [int(e) for e in dispatcher.local_expert_list]

    def published_device_map(self):
        """Return the layout in force (global expert id -> rank), or `None`.

        The forward reads the layout through this method instead of touching the
        attribute, so the collaboration stays a documented protocol.
        """
        return self.device_indices_map

    def clear_published_layout(self) -> None:
        """Forget the published layout.

        Called when a re-placement is skipped: keeping the layout would drive the
        token exchange for weights that are still the old ones.
        """
        self.device_indices_map = None

    def _collective_scope(self, valid: bool, reference: torch.Tensor) -> bool:
        """Return True only when every rank accepts this update.

        A refusal is a collective decision: ranks that rejected an update must not
        publish a new layout while the other ranks keep sending tokens along the old
        one, otherwise the all-to-all split sizes disagree. Outside a distributed run
        the local answer is authoritative; when the reduction itself fails the update
        is refused (never accepted on the local answer alone).
        """
        if not (dist.is_available() and dist.is_initialized()):
            return valid
        flag = torch.tensor([1 if valid else 0], dtype=torch.int32, device=reference.device)
        try:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        except (RuntimeError, ValueError) as error:
            # Falling back to the local answer would let one rank accept while another
            # refuses, which is exactly the layout split this check exists to prevent.
            # Every rank hits the same unsupported reduction, so refusing here is at
            # least consistent; the warning names the backend limitation.
            logger.warning(
                "[MindIE-SD/eplb] EPLB update refused: the collective scope check failed, so the "
                "ranks cannot agree on a layout. layer_idx=%s, actual_error=%s.",
                self.layer_idx,
                error,
            )
            return False
        return bool(int(flag.item()))

    def remember_weights(self, weight_update: EplbWeightUpdate) -> None:
        """Remember the weights a just-applied re-placement produced."""
        self.applied_weights = (weight_update.w13_weight, weight_update.w2_weight)

    def current_weights(self, default_w13, default_w2):
        """Return the weights the current layout belongs to.

        Args:
            default_w13, default_w2: The caller's tensors, used while no
                re-placement has been applied (and after a shape change).

        Returns:
            tuple: The applied tensors when they still match, else the defaults.
        """
        if self.applied_weights is None:
            return default_w13, default_w2
        applied_w13, applied_w2 = self.applied_weights
        if applied_w13.shape != default_w13.shape or applied_w2.shape != default_w2.shape:
            return default_w13, default_w2
        return applied_w13, applied_w2

    def on_routing(self, topk_ids: torch.Tensor) -> None:
        """Record routed experts and poll the scheduler for a new layout.

        Call this once per MoE layer right after routing.

        Args:
            topk_ids (torch.Tensor):
                Routed expert ids (global expert ids). Both the routing layout
                `[tokens, top_k]` and the already-expanded `[tokens * top_k]`
                layout are accepted.

        Collection has two paths: a fast one feeds `torch.histc` counts straight to
        `ExpertLoadCollector.collect_expert_counts`, and a fallback feeds sorted
        indices to `collect_expert_load`, whose CANN op binary-searches a sorted
        expert index list. The sort only happens on the fallback path, because it
        costs more than the rest of the bookkeeping.
        """
        if self.collector is not None:
            indices = topk_ids.reshape(-1) if topk_ids.dim() > 1 else topk_ids
            num_experts = getattr(self.collector, "routed_expert_num", 0)
            if isinstance(num_experts, int) and num_experts > 0 and hasattr(self.collector, "collect_expert_counts"):
                # One histc plus a cumsum, instead of sorting the routing tensor:
                # the CANN op behind collect_expert_load binary-searches a sorted
                # list, and the sort alone costs more than the MoE bookkeeping.
                histogram = torch.histc(indices.float(), bins=num_experts, min=0, max=num_experts)
                self.collector.collect_expert_counts(torch.cumsum(histogram, 0))
            else:
                self.collector.collect_expert_load(torch.sort(indices).values if indices.numel() > 1 else indices)
        if self.dispatcher is not None:
            self._routing_calls += 1
            if self._routing_calls % self.check_interval == 0:
                self.dispatcher.check_consistency()

    def poll_weight_update(self) -> EplbWeightUpdate | None:
        """Return the new expert weights when the scheduler flagged an update.

        The dispatcher is responsible for clearing its own `update_flag`; the
        real implementation does it inside `update_module_weight_and_map()`.

        Returns:
            EplbWeightUpdate | None: The re-placed weights, or `None` when the
            current layout is still valid or the update is outside the supported
            scope (see the module docstring).
        """
        if self.dispatcher is None or not getattr(self.dispatcher, "update_flag", False):
            return None

        result = self.dispatcher.update_module_weight_and_map()
        if result is None:
            logger.warning("[MindIE-SD/eplb] Dispatcher reported an update but returned no weights.")
            return None
        update = _as_expert_update(result)

        try:
            w13_weight = _as_weight_tensor(update.weight1)
            w2_weight = _as_weight_tensor(update.weight2)
        except RuntimeError as error:
            logger.warning(
                "[MindIE-SD/eplb] EPLB weight payload could not be stacked, update skipped. "
                "issue=empty or ragged weight list, layer_idx=%s, actual_error=%s.",
                self.layer_idx,
                error,
            )
            return None

        expected = self._supported_local_experts
        updated_local = update.local_expert_list
        local_num = update.local_expert_num
        device_map = update.device_indices_map
        ep_size = int(getattr(self.dispatcher, "ep_size", 1) or 1)

        if device_map is not None:
            # A map-based dispatcher publishes the global expert layout. Everything
            # downstream is ordered by ascending global expert id: the token
            # dispatcher derives the send order from the device map, and the weight
            # rows are sorted here to match. The per-rank row map is deliberately
            # not used -- it stores -1 for experts owned by the other ranks.
            num_experts = int(device_map.numel())
            expected_local = num_experts // ep_size if ep_size > 0 else num_experts
            experts = [int(e) for e in updated_local] if updated_local is not None else None
            valid, reason = True, ""
            if local_num is not None and int(local_num) != expected_local:
                valid = False
                reason = (
                    "the current fast path requires the same number of experts on every rank "
                    f"(redundant experts are not supported yet): expected_local={expected_local}, "
                    f"scheduler_local={local_num}"
                )
            elif experts is None or len(experts) != w13_weight.shape[0]:
                valid = False
                reason = (
                    f"missing or inconsistent local expert list: weight_rows={w13_weight.shape[0]}, experts={experts}"
                )
            else:
                # A lopsided map would break the all-to-all split sizes, which must
                # agree on every rank; reject it before it can desynchronise them.
                counts = torch.bincount(device_map.to(torch.int64), minlength=ep_size)
                if counts.numel() != ep_size or bool((counts != expected_local).any()):
                    valid = False
                    reason = (
                        f"the map is not an even split: per-rank counts={counts.tolist()}, expected={expected_local}"
                    )
            if not self._collective_scope(valid, w13_weight):
                self.refused_updates += 1
                logger.warning(
                    "[MindIE-SD/eplb] EPLB update refused: layer_idx=%s, ep_size=%s, local_valid=%s, reason=%s. "
                    "Troubleshooting=keep the per-rank expert count uniform; the current weights are kept.",
                    self.layer_idx,
                    ep_size,
                    valid,
                    reason or "another rank rejected this update",
                )
                return None
            order = sort_order(experts)
            if order != list(range(len(order))):
                index = torch.tensor(order, device=w13_weight.device)
                w13_weight = w13_weight.index_select(0, index)
                w2_weight = w2_weight.index_select(0, index)
                experts = [experts[position] for position in order]
            self.device_indices_map = device_map.detach().clone()
            logger.debug(
                "[MindIE-SD/eplb] Expert weights re-placed and sorted by expert id, layer_idx=%s, local_experts=%s.",
                self.layer_idx,
                experts,
            )
            return EplbWeightUpdate(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                device_indices_map=self.device_indices_map,
                local_expert_num=int(local_num) if local_num is not None else None,
                local_expert_list=experts,
            )

        if expected is not None:
            if updated_local is not None and [int(e) for e in updated_local] != expected:
                self.refused_updates += 1
                logger.warning(
                    "[MindIE-SD/eplb] EPLB update refused: the new layout moves this rank to a "
                    "different local expert set, which the MoE forward cannot remap yet. "
                    "issue=unsupported re-placement scope, layer_idx=%s, expected_local_experts=%s, "
                    "scheduler_local_experts=%s. possible_cause=the scheduler re-balanced across ranks. "
                    "Troubleshooting=keep EPLB disabled, or enable per-rank remapping support; "
                    "the current weights are kept unchanged.",
                    self.layer_idx,
                    expected,
                    list(updated_local),
                )
                return None
            if w13_weight.shape[0] != len(expected):
                self.refused_updates += 1
                logger.warning(
                    "[MindIE-SD/eplb] EPLB update refused: weight rows do not match the local expert "
                    "count. layer_idx=%s, expected=%s, actual=%s.",
                    self.layer_idx,
                    len(expected),
                    w13_weight.shape[0],
                )
                return None

        logger.debug(
            "[MindIE-SD/eplb] Expert weights re-placed, layer_idx=%s, local_experts=%s.",
            self.layer_idx,
            w13_weight.shape[0],
        )
        return EplbWeightUpdate(w13_weight=w13_weight, w2_weight=w2_weight)
