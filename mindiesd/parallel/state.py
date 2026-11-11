# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Caller-owned chains, caches, stream state and validated distributed context."""

from __future__ import annotations

import torch.distributed as dist

from .config import _validate_attention_mode
from .qkv_layout import QKVLayoutPlan

__all__ = ["ParallelState"]


def _copy_stats(stats):
    """Copy the host-only diagnostic dictionaries without touching device tensors."""
    if isinstance(stats, dict):
        return {key: _copy_stats(value) for key, value in stats.items()}
    if isinstance(stats, (list, tuple)):
        return type(stats)(_copy_stats(value) for value in stats)
    return stats


class ParallelState:
    """State for one communicator; QKV layout selections may change between calls."""

    def __init__(
        self,
        *,
        group=None,
        overlap_comm=False,
        optimize_attention_pipeline=True,
        bsa_backend="cann",
        head_major_qkv: bool = False,
        qkv_layout_plan: QKVLayoutPlan | None = None,
    ):
        if not isinstance(overlap_comm, bool):
            raise ValueError("overlap_comm must be a bool and agree across ranks")
        if not isinstance(optimize_attention_pipeline, bool):
            raise ValueError("optimize_attention_pipeline must be a bool and agree across ranks")
        if bsa_backend not in ("cann", "eagle"):
            raise ValueError("bsa_backend must be 'cann' or 'eagle'")
        if not isinstance(head_major_qkv, bool):
            raise ValueError("head_major_qkv must be a bool and agree across ranks")
        self._overlap_comm = overlap_comm
        self._optimize_attention_pipeline = optimize_attention_pipeline
        self._bsa_backend = bsa_backend
        self._head_major_qkv = head_major_qkv
        self.qkv_layout_plan = qkv_layout_plan
        self._last_call_stats = None
        self._comm_stream = None
        self._execution_active = False
        self._execution_failed = False
        self._last_execution_event = None
        self.group = group
        self.marker_printed = False
        self.allreduce_total = 0
        self.a2a_total = 0
        self._chain_totals = {}
        self.fused_stages_total = 0
        self.token_block_mapping_cache = {}
        self.reorder_mapping_cache = {}
        self.chains = {}
        self._last_chain = None  # the last completed call; cleared before every execution attempt
        self.context = None  # bound by bind_or_check(): (group id, world, rank, device)

    def bind_or_check(self, *, group, world, rank, device):
        """Bind once and reject stale group, rank or device reuse."""
        if self.group is None:
            self.group = group
        elif self.group is not group:
            raise RuntimeError(
                "this ParallelState is bound to a different process group: construct one state per communicator"
            )
        if group is not None and dist.is_initialized():
            try:
                gw, gr = dist.get_world_size(group), dist.get_rank(group)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "the process group is not usable (%r): a state must not be reused across a "
                    "destroy/init cycle" % (exc,)
                ) from exc
            if (int(gw), int(gr)) != (int(world), int(rank)):
                raise RuntimeError(
                    "the process group reports (world=%d, rank=%d) but this call says (world=%d, "
                    "rank=%d)" % (gw, gr, world, rank)
                )
        ctx = (id(group) if group is not None else None, int(world), int(rank), str(device))
        if self.context is None:
            self.context = ctx
        elif self.context != ctx:
            raise RuntimeError(
                "this ParallelState was bound to %r and is now used in %r; construct one state per "
                "live context" % (self.context, ctx)
            )
        return ctx

    def chain(self, reorder_mapping, *, build, precision="mix", sparse_type="rf_v3", pre_rearranged=False):
        """Get or build the chain for geometry, precision, consumer and input order."""
        _validate_attention_mode(precision, sparse_type, pre_rearranged)
        key = (reorder_mapping.config.key, precision, sparse_type)
        if pre_rearranged:
            key += (True,)
        chain = self.chains.get(key)
        if chain is None:
            order_kwargs = {"pre_rearranged": True} if pre_rearranged else {}
            chain = build(
                reorder_mapping,
                state=self,
                precision=precision,
                sparse_type=sparse_type,
                **order_kwargs,
            )
            self.chains[key] = chain
        return chain

    def _record_call(self, config_key, precision, sparse_type, counts):
        """Commit one successful call; keep legacy per-chain diagnostics here too."""
        totals = self._chain_totals.setdefault(
            (config_key, precision, sparse_type), {"ar_calls_lifetime": 0, "a2a_calls_lifetime": 0}
        )
        totals["ar_calls_lifetime"] += counts["allreduce"]
        totals["a2a_calls_lifetime"] += counts["a2a_calls"]
        self.allreduce_total += counts["allreduce"]
        self.a2a_total += counts["a2a_calls"]
        self.fused_stages_total += counts["fused_stages"]

    def stats(self):
        return {
            "allreduce_total": self.allreduce_total,
            "a2a_total": self.a2a_total,
            "fused_stages_total": self.fused_stages_total,
            "engines": len(self.chains),
            "context": self.context,
        }

    @property
    def last_call_stats(self):
        """Defensive host-only snapshot of the last successful public call, or None.

        Cleared before every public attempt, including argument rejection. The
        snapshot is committed only after the full collective protocol succeeds;
        reading it never synchronizes streams or reads tensor values.
        """
        return None if self._last_call_stats is None else _copy_stats(self._last_call_stats)

    @property
    def qkv_layout_plan(self):
        """Optional shared producer plan; absent plans retain the legacy bool decision."""
        return self._qkv_layout_plan

    @qkv_layout_plan.setter
    def qkv_layout_plan(self, plan):
        if plan is not None and not isinstance(plan, QKVLayoutPlan):
            raise ValueError("qkv_layout_plan must be a QKVLayoutPlan or None")
        self._qkv_layout_plan = plan

    @property
    def overlap_comm(self):
        """The caller-selected mode is fixed for this state's lifetime."""
        return self._overlap_comm

    @property
    def optimize_attention_pipeline(self):
        """Whether this state's chains reuse buffers, indices and preparation work."""
        return self._optimize_attention_pipeline

    @property
    def bsa_backend(self):
        """The RF-v3 BF16 operator used by this state's chains."""
        return self._bsa_backend

    @property
    def head_major_qkv(self):
        """Permit head-major sends; explicit assignment invalidates a previous shared plan."""
        return self._head_major_qkv

    @head_major_qkv.setter
    def head_major_qkv(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("head_major_qkv must be a bool and agree across ranks")
        self._head_major_qkv = enabled
        self._qkv_layout_plan = None

    def _check_execution(self):
        if self._execution_failed:
            raise RuntimeError("this ParallelState has a failed execution; construct a new state")
        if self._execution_active:
            raise RuntimeError("this ParallelState already has an active execution")

    def _begin_execution(self, compute_stream, *, build):
        """Own reuse/validity; the execution tier supplies the device stream factory."""
        self._check_execution()
        self._execution_active = True
        try:
            if self._comm_stream is None:
                self._comm_stream = build()
            # Protect the cached producer buffers even when the next layer is
            # called from another compute stream or uses another cached Chain.
            if self._last_execution_event is not None:
                compute_stream.wait_event(self._last_execution_event)
            return self._comm_stream
        except Exception:
            if self._comm_stream is None:
                self._execution_active = False
            else:
                self._fail_execution()
            raise

    def _finish_execution(self, event):
        self._last_execution_event = event
        self._execution_active = False

    def _fail_execution(self):
        # Never replay partial collectives; stream-recorded buffers live until submitted work drains.
        self._execution_failed = True
        self._execution_active = False
        self._last_chain = None
        self._last_call_stats = None
