# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Device-free producer/consumer agreement for the existing QKV send layout."""

from dataclasses import dataclass, field

from .topology import _head_major_qkv_geometry_supported


@dataclass(frozen=True, slots=True)
class QKVLayoutPlan:
    """A shared metadata decision; consumers still validate physical tensors.

    Construct with :func:`plan_qkv_layout`. The supported fields are derived,
    so constructing or replacing metadata cannot bypass the original whitelist.
    Sequence length is intentionally checked only on the final tensors.
    """

    requested: bool
    head_count: int
    head_dim: int
    batch_size: int
    world: int
    chunking: int
    width: int
    precision: str
    sparse_type: str
    pre_rearranged: bool
    overlap_comm: bool
    optimize_attention_pipeline: bool
    supported: bool = field(init=False)
    layout: str = field(init=False)
    fallback_reason: str | None = field(init=False)

    def __post_init__(self):
        for name in ("requested", "pre_rearranged", "overlap_comm", "optimize_attention_pipeline"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        reason = None
        if not self.requested:
            reason = "disabled"
        elif not self.optimize_attention_pipeline:
            reason = "pipeline_optimization_disabled"
        elif not self.overlap_comm:
            reason = "overlap_disabled"
        elif not self.pre_rearranged:
            reason = "input_not_pre_rearranged"
        elif self.precision != "bf16" or self.sparse_type != "rf_v3":
            reason = "attention_mode_unsupported"
        elif self.head_count != 56 or not _head_major_qkv_geometry_supported(
            self.chunking, self.world, self.width, self.batch_size, self.head_dim
        ):
            reason = "geometry_unsupported"
        object.__setattr__(self, "supported", reason is None)
        object.__setattr__(self, "layout", "head_major" if reason is None else "bsnd")
        object.__setattr__(self, "fallback_reason", reason)

    @property
    def output_bnsd(self):
        """The producer may emit BNSD storage exposed as a logical BSND view."""
        return self.supported


def plan_qkv_layout(
    *,
    head_count,
    head_dim,
    batch_size,
    world,
    chunking,
    width=1,
    precision="bf16",
    sparse_type="rf_v3",
    pre_rearranged=False,
    overlap_comm=False,
    optimize_attention_pipeline=True,
    requested=True,
):
    """Accept only 56 heads, D128, B1 with (world,chunks)=(8,7) or (4,14).

    The same immutable instance can choose the producer's physical storage and
    be assigned to ``ParallelState.qkv_layout_plan``. Unsupported metadata uses
    the existing BSND route and records why; final strides are never assumed.
    """
    return QKVLayoutPlan(**locals())


__all__ = ["QKVLayoutPlan", "plan_qkv_layout"]
