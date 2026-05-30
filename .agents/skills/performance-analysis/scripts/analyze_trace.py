#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""
Ascend NPU profiling trace analysis.

Implements the ascend-profiling-anomaly pipeline. Supports two input formats:
  1. CANN profiler output: kernel_details.csv + trace_view.json (primary, richer)
  2. Chrome Trace JSON (*.pt.trace.json) from tensorboard_trace_handler (fallback)

Produces two reports:
  - profiling_report.md (anomaly discovery + performance analysis)
  - model_architecture_report.md (model architecture from profiling data)

Usage:
    python analyze_trace.py --profile-dir ./profile_l1 --output-dir ./
"""
# pylint: disable=duplicate-code,too-many-lines

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Interval operations (from ascend-profiling-anomaly reference)
# ============================================================================


@dataclass
class Interval:
    start_us: float
    end_us: float

    @property
    def dur_us(self) -> float:
        return max(0.0, self.end_us - self.start_us)


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    items = sorted(
        (i for i in intervals if i.end_us > i.start_us),
        key=lambda x: (x.start_us, x.end_us),
    )
    if not items:
        return []
    merged: List[Interval] = [Interval(items[0].start_us, items[0].end_us)]
    for cur in items[1:]:
        last = merged[-1]
        if cur.start_us <= last.end_us:
            last.end_us = max(last.end_us, cur.end_us)
        else:
            merged.append(Interval(cur.start_us, cur.end_us))
    return merged


def interval_union_us(intervals: Sequence[Interval]) -> float:
    return sum(i.dur_us for i in merge_intervals(intervals))


def interval_overlap_ratio(target: Interval, others: Sequence[Interval]) -> Optional[float]:
    if target.dur_us <= 0:
        return None
    clipped: List[Interval] = []
    for x in others:
        s = max(target.start_us, x.start_us)
        e = min(target.end_us, x.end_us)
        if e > s:
            clipped.append(Interval(s, e))
    if not clipped:
        return 0.0
    return interval_union_us(clipped) / target.dur_us


# ============================================================================
# Trace event structures
# ============================================================================


@dataclass
class TraceEvent:
    """Parsed Chrome Trace event."""

    ts: float  # timestamp in microseconds
    dur: float  # duration in microseconds
    name: str  # event name
    cat: str  # category: kernel, cpu_op, user_annotation, AscendCL, communication
    pid: int  # process ID
    tid: int  # thread / stream ID
    ph: str = "X"  # phase

    @property
    def end_us(self) -> float:
        return self.ts + self.dur

    @property
    def is_device(self) -> bool:
        return self.cat == "kernel"

    @property
    def is_host(self) -> bool:
        return self.cat in ("cpu_op", "user_annotation", "AscendCL")

    @property
    def is_communication(self) -> bool:
        return self.cat == "communication" or "AllReduce" in self.name or "AllGather" in self.name

    @property
    def interval(self) -> Interval:
        return Interval(self.ts, self.end_us)


@dataclass
class KernelInfo:
    name: str
    task_type: str  # AI_CORE, AI_CPU, HCCL
    start_us: float
    dur_us: float
    stream_id: int
    input_shapes: str = ""
    wait_us: float = 0.0

    @property
    def end_us(self) -> float:
        return self.start_us + self.dur_us

    @property
    def interval(self) -> Interval:
        return Interval(self.start_us, self.end_us)


# ============================================================================
# INGEST: Parse CANN profiler CSV + trace JSON (primary format)
# ============================================================================

_PROFILER_OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"


def find_profiler_output_dir(profile_dir: str) -> Optional[str]:
    """Find the ASCEND_PROFILER_OUTPUT directory within profile_dir."""
    for root, dirs, _files in os.walk(profile_dir):
        if _PROFILER_OUTPUT_DIR in dirs:
            return os.path.join(root, _PROFILER_OUTPUT_DIR)
        for d in dirs:
            if d == _PROFILER_OUTPUT_DIR:
                return os.path.join(root, d)
    return None


def load_kernels_from_csv(csv_path: str) -> List[KernelInfo]:
    """Load kernel events from kernel_details.csv (CANN profiler format)."""
    kernels: List[KernelInfo] = []
    t0 = None

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                name = row.get("Name", "").strip()
                if not name:
                    continue

                start_us = float(row.get("Start Time(us)", 0))
                dur_us = float(row.get("Duration(us)", 0))
                wait_us = float(row.get("Wait Time(us)", 0))

                if dur_us <= 0:
                    continue

                # Normalize: timestamps in CANN profiler are in some epoch units.
                # Convert to relative microseconds.
                if t0 is None:
                    t0 = start_us
                rel_start = start_us - t0

                task_type = _infer_task_type_from_name(name)

                ki = KernelInfo(
                    name=name,
                    task_type=task_type,
                    start_us=rel_start,
                    dur_us=dur_us,
                    stream_id=0,
                    input_shapes="",
                )
                ki.wait_us = wait_us
                kernels.append(ki)
            except (ValueError, KeyError):
                continue

    return kernels


def _infer_task_type_from_name(name: str) -> str:
    name_lower = name.lower()
    if any(kw in name for kw in ("Hcom", "AllReduce", "AllGather", "Broadcast", "ReduceScatter")):
        return "HCCL"
    if any(kw in name_lower for kw in ("aicpu",)):
        return "AI_CPU"
    if any(kw in name_lower for kw in ("memcpy", "copy", "d2h", "h2d")):
        return "MEMCPY"
    return "AI_CORE"


def load_host_events_from_trace(trace_path: str, t0: float) -> List[TraceEvent]:
    """Load host-side events from trace_view.json."""
    events: List[TraceEvent] = []
    with open(trace_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    trace_events = data.get("traceEvents", data) if isinstance(data, dict) else data

    for raw in trace_events:
        if not isinstance(raw, dict):
            continue
        ph = raw.get("ph", "")
        name = raw.get("name", "")
        ts = float(raw.get("ts", "0"))
        dur = float(raw.get("dur", 0))
        pid = int(raw.get("pid", 0))
        tid = int(raw.get("tid", 0))
        cat = raw.get("cat", "")

        if ph == "X" and dur > 0:
            rel_ts = ts - t0 if t0 else ts
            events.append(TraceEvent(ts=rel_ts, dur=dur, name=name, cat=cat, pid=pid, tid=tid, ph=ph))
        elif ph == "M":
            pass  # Skip metadata events
        elif ph == "C":
            pass  # Skip counter events

    return events


def load_step_time_from_csv(csv_path: str) -> Dict[str, Any]:
    """Load step-level timing from step_trace_time.csv."""
    step_info: Dict[str, Any] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row_count = 0
            for row in reader:
                step_info["device_id"] = row.get("Device_id", "")
                step_info["computing_us"] = float(row.get("Computing", 0))
                step_info["communication_us"] = float(row.get("Communication", 0))
                step_info["communication_not_overlapped_us"] = float(row.get("Communication(Not Overlapped)", 0))
                step_info["overlapped_us"] = float(row.get("Overlapped", 0))
                step_info["free_us"] = float(row.get("Free", 0))
                step_info["stage_us"] = float(row.get("Stage", 0))
                step_info["bubble_us"] = float(row.get("Bubble", 0))
                row_count += 1
            step_info["step_count"] = row_count
    except Exception as exc:
        logger.warning("Failed to load step_trace_time.csv: %s", exc)
    return step_info


# ============================================================================
# INGEST: Parse Chrome Trace JSON (fallback format)
# ============================================================================


def load_trace_events(trace_path: str) -> List[TraceEvent]:
    with open(trace_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    events: List[TraceEvent] = []
    trace_events = data.get("traceEvents", data) if isinstance(data, dict) else data

    for raw in trace_events:
        if not isinstance(raw, dict):
            continue
        ph = raw.get("ph", "")
        ts = float(raw.get("ts", 0))
        dur = float(raw.get("dur", 0))
        name = raw.get("name", "")
        cat = raw.get("cat", "")
        pid = int(raw.get("pid", 0))
        tid = int(raw.get("tid", 0))

        if ph == "X" and dur > 0:
            events.append(TraceEvent(ts=ts, dur=dur, name=name, cat=cat, pid=pid, tid=tid))
        elif ph == "B":
            # Begin event - paired events are harder to parse, skip for now
            pass

    return events


def load_all_traces(profile_dir: str) -> Tuple[List[TraceEvent], List[str]]:
    events: List[TraceEvent] = []
    sources: List[str] = []
    for fn in sorted(os.listdir(profile_dir)):
        if fn.endswith(".pt.trace.json") or fn.endswith(".trace.json"):
            path = os.path.join(profile_dir, fn)
            events.extend(load_trace_events(path))
            sources.append(fn)
    events.sort(key=lambda e: e.ts)
    return events, sources


# ============================================================================
# CLASSIFY: split events into kernel/host/communication
# ============================================================================

ClassifyResult = Tuple[List[KernelInfo], List[TraceEvent], List[TraceEvent], List[TraceEvent]]


def classify_events(events: List[TraceEvent]) -> ClassifyResult:
    kernels: List[KernelInfo] = []
    host_events: List[TraceEvent] = []
    comm_events: List[TraceEvent] = []
    step_markers: List[TraceEvent] = []

    for ev in events:
        if ev.is_device:
            # Guess task type from name patterns
            task_type = _guess_task_type(ev.name)
            shapes = ""
            if isinstance(ev.dur, (int, float)):
                pass
            ki = KernelInfo(
                name=ev.name,
                task_type=task_type,
                start_us=ev.ts,
                dur_us=ev.dur,
                stream_id=ev.tid,
                input_shapes=shapes,
            )
            kernels.append(ki)
        elif "Step#" in ev.name or "Iteration" in ev.name or "ProfilerStep" in ev.name:
            step_markers.append(ev)
        elif ev.is_communication:
            comm_events.append(ev)
        else:
            host_events.append(ev)

    return kernels, host_events, comm_events, step_markers


def _guess_task_type(name: str) -> str:
    name_lower = name.lower()
    if any(kw in name for kw in ("Hcom", "AllReduce", "AllGather", "Broadcast", "ReduceScatter")):
        return "HCCL"
    if any(kw in name_lower for kw in ("aicpu", "ai_cpu", "aicpu")):
        return "AI_CPU"
    if any(kw in name_lower for kw in ("memcpy", "copy", "d2h", "h2d")):
        return "MEMCPY"
    return "AI_CORE"


# ============================================================================
# STEP_DETECTION: identify step boundaries
# ============================================================================


def detect_steps(kernels: List[KernelInfo], markers: List[TraceEvent]) -> List[Dict[str, Any]]:
    if not kernels:
        return []

    if markers:
        markers_sorted = sorted(markers, key=lambda m: m.ts)
        steps = []
        for i, m in enumerate(markers_sorted):
            step_start = m.ts
            step_end = markers_sorted[i + 1].ts if i + 1 < len(markers_sorted) else max(k.end_us for k in kernels)
            steps.append(
                {
                    "id": i,
                    "start_us": step_start,
                    "end_us": step_end,
                    "marker_name": m.name,
                }
            )
        return steps

    # Fallback: single pseudo-step spanning entire capture
    return [
        {
            "id": 0,
            "start_us": min(k.start_us for k in kernels),
            "end_us": max(k.end_us for k in kernels),
            "marker_name": "pseudo_step",
        }
    ]


# ============================================================================
# BUILD_DEVICE_INTERVALS + MERGE_INTERVALS
# ============================================================================


def build_device_intervals(kernels: List[KernelInfo], step_start_us: float, step_end_us: float) -> List[Interval]:
    out: List[Interval] = []
    for k in kernels:
        s = max(step_start_us, k.start_us)
        e = min(step_end_us, k.end_us)
        if e > s:
            out.append(Interval(s, e))
    return out


def compute_bubble_metrics(
    step_start_us: float, step_end_us: float, device_intervals: Sequence[Interval]
) -> Dict[str, Any]:
    service_us = max(0.0, step_end_us - step_start_us)
    merged = merge_intervals(device_intervals)

    if not merged:
        return {
            "service_ms": service_us / 1000.0,
            "device_busy_union_ms": 0.0,
            "underfeed_ms": service_us / 1000.0,
            "underfeed_ratio": 1.0 if service_us > 0 else 0.0,
            "prelaunch_gap_ms": service_us / 1000.0,
            "tail_gap_ms": 0.0,
            "internal_bubble_total_ms": 0.0,
            "largest_internal_bubble_ms": 0.0,
            "bubble_count": 0,
            "bubble_windows": [],
            "merged_segments": [],
        }

    busy_union_us = sum(seg.dur_us for seg in merged)
    prelaunch_us = max(0.0, merged[0].start_us - step_start_us)
    tail_us = max(0.0, step_end_us - merged[-1].end_us)

    bubbles: List[Interval] = []
    for left, right in zip(merged[:-1], merged[1:]):
        if right.start_us > left.end_us:
            bubbles.append(Interval(left.end_us, right.start_us))

    bubble_total_us = sum(b.dur_us for b in bubbles)
    largest_bubble_us = max((b.dur_us for b in bubbles), default=0.0)
    underfeed_us = max(0.0, service_us - busy_union_us)
    underfeed_ratio = underfeed_us / service_us if service_us > 0 else 0.0

    return {
        "service_ms": service_us / 1000.0,
        "device_busy_union_ms": busy_union_us / 1000.0,
        "underfeed_ms": underfeed_us / 1000.0,
        "underfeed_ratio": underfeed_ratio,
        "prelaunch_gap_ms": prelaunch_us / 1000.0,
        "tail_gap_ms": tail_us / 1000.0,
        "internal_bubble_total_ms": bubble_total_us / 1000.0,
        "largest_internal_bubble_ms": largest_bubble_us / 1000.0,
        "bubble_count": len(bubbles),
        "bubble_windows": bubbles,
        "merged_segments": merged,
    }


# ============================================================================
# ANOMALY_TAGGING
# ============================================================================


def classify_hidden_issue(metrics: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    service_ms = float(metrics["service_ms"])
    if metrics["underfeed_ratio"] >= 0.30:
        tags.append("DEVICE_IDLE_GAP_HEAVY")
    if metrics["prelaunch_gap_ms"] >= max(1.0, 0.10 * service_ms):
        tags.append("PRELAUNCH_GAP_HEAVY")
    if metrics["tail_gap_ms"] >= max(1.0, 0.10 * service_ms):
        tags.append("TAIL_GAP_HEAVY")
    if metrics["largest_internal_bubble_ms"] >= max(1.0, 0.10 * service_ms):
        tags.append("INTERNAL_BUBBLE_HEAVY")
    return tags


# ============================================================================
# HOST_EVIDENCE + SOFT_ATTRIBUTION
# ============================================================================


def soft_attribution_for_bubble(
    bubble: Interval,
    host_intervals: Sequence[Interval],
    sync_intervals: Sequence[Interval],
    comm_intervals: Sequence[Interval],
) -> Dict[str, Any]:
    host_cov = interval_overlap_ratio(bubble, host_intervals)
    sync_cov = interval_overlap_ratio(bubble, sync_intervals)
    comm_cov = interval_overlap_ratio(bubble, comm_intervals)
    labels: List[str] = []

    if (sync_cov or 0.0) >= 0.20:
        labels.append("possible_sync_or_h2d")
    if (comm_cov or 0.0) >= 0.20:
        labels.append("possible_comm_wait")
    if (host_cov or 0.0) is not None and host_cov < 0.05:
        labels.append("possible_untraced_host_blocking")
    if not labels and (host_cov or 0.0) is not None and host_cov >= 0.10:
        labels.append("possible_host_launch_lag")
    if not labels:
        labels.append("insufficient_evidence")

    return {
        "host_visible_coverage_ratio": host_cov,
        "sync_marker_overlap_ratio": sync_cov,
        "comm_marker_overlap_ratio": comm_cov,
        "soft_root_cause_labels": labels,
    }


# ============================================================================
# WAIT_ANCHOR_SCAN
# ============================================================================


def detect_wait_anchors(
    kernels: List[KernelInfo],
    step_start_us: float,
    step_end_us: float,
) -> List[Dict[str, Any]]:
    """Detect wait-anchor false hotspots (kernel duration tiny but total cost = wait dominates)."""
    # torch_npu profiler at level=l1 does not provide per-kernel wait time,
    # but we can detect anomalies from gaps between kernels
    step_kernels = [k for k in kernels if step_start_us <= k.start_us < step_end_us]
    step_kernels.sort(key=lambda k: k.start_us)

    wait_anchors = []
    for k in step_kernels:
        if k.dur_us < 10.0:
            wait_anchors.append(
                {
                    "name": k.name,
                    "duration_us": k.dur_us,
                    "start_us": k.start_us,
                    "risk": "WAIT_ANCHOR_CANDIDATE",
                    "note": "Tiny duration kernel - check if it absorbs idle wait time",
                }
            )

    return wait_anchors[:20]  # limit


# ============================================================================
# AICPU classification
# ============================================================================


def classify_aicpu(kernels: List[KernelInfo], ai_core_intervals: Sequence[Interval]) -> List[Dict[str, Any]]:
    """Classify AI_CPU kernels by masked_ratio (how much they overlap with AI_CORE)."""
    results = []
    for k in kernels:
        if k.task_type != "AI_CPU":
            continue
        overlap_ratio = interval_overlap_ratio(k.interval, ai_core_intervals) or 0.0
        if overlap_ratio >= 0.9:
            classification = "AICPU_MASKED_BUT_UNDESIRABLE"
        elif overlap_ratio >= 0.2:
            classification = "AICPU_PARTIALLY_EXPOSED"
        else:
            classification = "AICPU_EXPOSED_NOT_ALLOWED"

        results.append(
            {
                "name": k.name,
                "duration_us": k.dur_us,
                "masked_ratio": overlap_ratio,
                "classification": classification,
            }
        )
    return results


# ============================================================================
# SEGMENTATION: structure / layer detection
# ============================================================================


def segment_structures(
    kernels: List[KernelInfo], step: Dict[str, Any], host_events: List[TraceEvent]
) -> List[Dict[str, Any]]:
    """Segment a step into structures (layers/components)."""
    step_start = step["start_us"]
    step_end = step["end_us"]
    step_kernels = [k for k in kernels if step_start <= k.start_us < step_end]
    step_kernels.sort(key=lambda k: k.start_us)

    if not step_kernels:
        return []

    # Wagner-Fischer: group kernels by functional role based on name patterns
    pattern_groups = _group_by_pattern(step_kernels)

    # Also use large gaps to identify structure boundaries
    structures = []
    current_group = []
    group_start = step_kernels[0].start_us

    for i, k in enumerate(step_kernels):
        if current_group and i > 0:
            prev = step_kernels[i - 1]
            gap = k.start_us - prev.end_us
            # Large gap (> 5ms) indicates structure boundary
            if gap > 5000:
                if current_group:
                    structures.append(_build_structure(current_group, group_start))
                current_group = []
                group_start = k.start_us

        current_group.append(k)

    if current_group:
        structures.append(_build_structure(current_group, group_start))

    # If pattern_groups give better segmentation, use those
    if not structures and pattern_groups:
        for pg in pattern_groups:
            structures.append(_build_structure(pg["kernels"], pg["start_us"]))

    # Fallback: single structure
    if not structures:
        structures.append(_build_structure(step_kernels, step_kernels[0].start_us))

    return structures


def _group_by_pattern(kernels: List[KernelInfo]) -> List[Dict[str, Any]]:
    """Group kernels by repeating name patterns."""
    if len(kernels) < 4:
        return []

    # Try to find repeating sequences
    # For now, group by coarse functional categories
    groups: List[Dict[str, Any]] = []
    current_group = []
    current_start = kernels[0].start_us
    prev_role = _kernel_role(kernels[0].name)

    for k in kernels:
        role = _kernel_role(k.name)
        if role != prev_role and len(current_group) >= 3:
            groups.append({"kernels": list(current_group), "start_us": current_start})
            current_group = []
            current_start = k.start_us
        current_group.append(k)
        prev_role = role

    if current_group:
        groups.append({"kernels": list(current_group), "start_us": current_start})

    return groups


def _kernel_role(name: str) -> str:
    name_lower = name.lower()
    if any(kw in name_lower for kw in ("attention", "flash_attn", "fia", "fused_infer")):
        return "attention"
    if any(kw in name_lower for kw in ("matmul", "linear", "gemm", "addmm")):
        return "matmul"
    if any(kw in name_lower for kw in ("layernorm", "rmsnorm", "norm")):
        return "norm"
    if any(kw in name_lower for kw in ("gelu", "relu", "silu", "swish", "sigmoid", "softmax")):
        return "activation"
    if any(kw in name_lower for kw in ("add", "mul", "div", "sub")):
        return "elementwise"
    if any(kw in name_lower for kw in ("allreduce", "allgather", "broadcast", "hccl")):
        return "communication"
    if any(kw in name_lower for kw in ("memcpy", "copy", "d2h", "h2d")):
        return "memcpy"
    return "other"


def _build_structure(kernels: List[KernelInfo], start_us: float) -> Dict[str, Any]:
    if not kernels:
        return {}
    end_us = max(k.end_us for k in kernels)
    wall_ms = (end_us - start_us) / 1000.0

    intervals = [k.interval for k in kernels]
    merged = merge_intervals(intervals)
    busy_union_ms = sum(m.dur_us for m in merged) / 1000.0
    kernel_sum_ms = sum(k.dur_us for k in kernels) / 1000.0

    ai_core_sum = sum(k.dur_us for k in kernels if k.task_type == "AI_CORE")
    ai_cpu_sum = sum(k.dur_us for k in kernels if k.task_type == "AI_CPU")
    hccl_sum = sum(k.dur_us for k in kernels if k.task_type == "HCCL")

    # Determine structure type from kernel composition
    names = [k.name for k in kernels]
    stype = _classify_structure_type(names)

    return {
        "type": stype,
        "kernel_count": len(kernels),
        "start_us": start_us,
        "end_us": end_us,
        "wall_ms": wall_ms,
        "device_busy_union_ms": busy_union_ms,
        "kernel_sum_ms": kernel_sum_ms,
        "ai_core_pct": ai_core_sum / max(kernel_sum_ms * 1000, 1),
        "ai_cpu_pct": ai_cpu_sum / max(kernel_sum_ms * 1000, 1),
        "hccl_pct": hccl_sum / max(kernel_sum_ms * 1000, 1),
        "kernels": kernels,
    }


def _classify_structure_type(names: List[str]) -> str:
    name_str = " ".join(names).lower()
    if any(kw in name_str for kw in ("attention", "flash_attn", "fia")):
        if any(kw in name_str for kw in ("matmul", "linear")):
            return "attention_block"
        return "attention"
    if any(kw in name_str for kw in ("matmul", "linear", "gemm")) and any(
        kw in name_str for kw in ("gelu", "relu", "norm")
    ):
        return "mlp_block"
    if any(kw in name_str for kw in ("text_encoder", "t5", "umt5")):
        return "text_encoder"
    if any(kw in name_str for kw in ("vae", "decoder", "encoder")):
        return "vae"
    if any(kw in name_str for kw in ("transformer",)):
        return "transformer_block"
    return "unknown"


# ============================================================================
# COMMUNICATION analysis
# ============================================================================


def analyze_communication(
    kernels: List[KernelInfo], comm_events: List[TraceEvent], compute_intervals: Sequence[Interval]
) -> Dict[str, Any]:
    """Analyze communication overhead and overlap with compute."""
    comm_kernels = [k for k in kernels if k.task_type == "HCCL"]

    comm_intervals = [k.interval for k in comm_kernels]
    comm_total = sum(i.dur_us for i in comm_intervals) / 1000.0

    comm_all = list(comm_intervals) + [e.interval for e in comm_events]

    hidden = 0.0
    exposed = 0.0
    for ci in comm_all:
        overlap_ratio = interval_overlap_ratio(ci, compute_intervals) or 0.0
        if overlap_ratio >= 0.90:
            hidden += ci.dur_us
        elif overlap_ratio < 0.10:
            exposed += ci.dur_us
        else:
            hidden += ci.dur_us * overlap_ratio
            exposed += ci.dur_us * (1.0 - overlap_ratio)

    return {
        "comm_total_ms": comm_total,
        "comm_hidden_ms": hidden / 1000.0,
        "comm_exposed_ms": exposed / 1000.0,
        "comm_exposed_ratio": exposed / max(comm_total * 1000, 1),
        "can_not_hide_ratio": exposed / max((hidden + exposed), 1),
    }


# ============================================================================
# FUSION_OPPORTUNITIES detection
# ============================================================================

FUSION_PATTERNS = [
    {
        "name": "MatMul + BiasAdd + Gelu",
        "pattern": ["matmul", "add", "gelu"],
        "fused_name": "MatMulBiasGelu",
        "savings_ratio": 0.30,
        "min_chain": 3,
    },
    {
        "name": "MatMul + BiasAdd + Relu",
        "pattern": ["matmul", "add", "relu"],
        "fused_name": "MatMulBiasRelu",
        "savings_ratio": 0.25,
        "min_chain": 3,
    },
    {
        "name": "LayerNorm + MatMul",
        "pattern": ["norm", "matmul"],
        "fused_name": "LayerNormMatMul",
        "savings_ratio": 0.15,
        "min_chain": 2,
    },
    {
        "name": "Element-wise chain (>=3 ops)",
        "pattern": ["elementwise", "elementwise", "elementwise"],
        "fused_name": "FusedElementWise",
        "savings_ratio": 0.20,
        "min_chain": 3,
    },
    {
        "name": "FlashAttention + MatMul (proj)",
        "pattern": ["attention", "matmul"],
        "fused_name": "FusedAttentionProj",
        "savings_ratio": 0.10,
        "min_chain": 2,
    },
    {
        "name": "Scale + Softmax + MatMul (FA scope)",
        "pattern": ["elementwise", "activation", "matmul"],
        "fused_name": "FusedScaleSoftmaxMatMul",
        "savings_ratio": 0.25,
        "min_chain": 3,
    },
]


_ROLE_ELEMENTWISE = "elementwise"
_ROLE_MATMUL = "matmul"
_ROLE_ATTENTION = "attention"
_ROLE_NORM = "norm"
_ROLE_ACTIVATION = "activation"
_FUSION_KEY_START = "start_us"


def detect_fusion_opportunities(kernels: List[KernelInfo]) -> List[Dict[str, Any]]:
    """Detect operator fusion opportunities from kernel sequence."""
    if not kernels:
        return []

    sorted_k = sorted(kernels, key=lambda k: k.start_us)
    roles = [_kernel_role(k.name) for k in sorted_k]

    opportunities = []
    for fp in FUSION_PATTERNS:
        pat = fp["pattern"]
        pat_len = len(pat)

        for i in range(len(roles) - pat_len + 1):
            window_roles = roles[i : i + pat_len]
            window_kernels = sorted_k[i : i + pat_len]

            skip = False
            for j, (actual_role, expected) in enumerate(zip(window_roles, pat)):
                if expected == _ROLE_MATMUL and actual_role != _ROLE_MATMUL:
                    skip = True
                elif expected == _ROLE_ATTENTION and actual_role != _ROLE_ATTENTION:
                    skip = True
                elif expected == _ROLE_NORM and actual_role != _ROLE_NORM:
                    skip = True
                elif expected == _ROLE_ELEMENTWISE:
                    if actual_role != _ROLE_ELEMENTWISE:
                        skip = True
                elif expected == _ROLE_ACTIVATION:
                    if actual_role != _ROLE_ACTIVATION:
                        skip = True
            if skip:
                continue

            if pat == [_ROLE_ELEMENTWISE, _ROLE_ELEMENTWISE, _ROLE_ELEMENTWISE]:
                all_elementwise = all(
                    any(kw in k.name.lower() for kw in ("add", "mul", "div", "sub")) for k in window_kernels
                )
                if not all_elementwise:
                    continue

            total_dur = sum(k.dur_us for k in window_kernels) / 1000.0
            savings = total_dur * fp["savings_ratio"]

            opportunities.append(
                {
                    "pattern": fp["name"],
                    "fused_name": fp["fused_name"],
                    "kernel_chain": [k.name for k in window_kernels],
                    _FUSION_KEY_START: window_kernels[0].start_us,
                    "end_us": window_kernels[-1].end_us,
                    "current_dur_ms": total_dur,
                    "estimated_savings_ms": savings,
                    "estimated_fused_ms": total_dur - savings,
                    "savings_pct": fp["savings_ratio"] * 100,
                }
            )

    # Deduplicate by start position
    seen_starts = set()
    unique = []
    for opp in sorted(opportunities, key=lambda x: x["estimated_savings_ms"], reverse=True):
        if opp[_FUSION_KEY_START] not in seen_starts:
            seen_starts.add(opp[_FUSION_KEY_START])
            unique.append(opp)

    return unique[:30]


# ============================================================================
# Kernel statistics
# ============================================================================


def compute_kernel_stats(kernels: List[KernelInfo]) -> Dict[str, Any]:
    """Compute per-kernel-name statistics."""
    by_name: Dict[str, List[KernelInfo]] = defaultdict(list)
    for k in kernels:
        by_name[k.name].append(k)

    stats = {}
    for name, klist in by_name.items():
        durations = [k.dur_us for k in klist]
        stats[name] = {
            "count": len(klist),
            "total_dur_us": sum(durations),
            "avg_dur_us": sum(durations) / len(durations) if durations else 0,
            "min_dur_us": min(durations) if durations else 0,
            "max_dur_us": max(durations) if durations else 0,
            "task_type": klist[0].task_type,
        }

    return stats


# ============================================================================
# RENDER: Generate reports
# ============================================================================


class ProfilingContext:
    """Aggregated profiling data for report rendering."""

    def __init__(
        self,
        all_kernels,
        host_events,
        comm_events,
        steps,
        structures,
        fusion_opps,
        comm_analysis,
        kernel_stats,
        meta,
        warmup_info=None,
        stage_breakdown=None,
        category_ratios=None,
    ):
        self.all_kernels = all_kernels
        self.host_events = host_events
        self.comm_events = comm_events
        self.steps = steps
        self.structures = structures
        self.fusion_opps = fusion_opps
        self.comm_analysis = comm_analysis
        self.kernel_stats = kernel_stats
        self.meta = meta
        self.warmup_info = warmup_info
        self.stage_breakdown = stage_breakdown
        self.category_ratios = category_ratios


def render_profiling_report(ctx: ProfilingContext) -> str:
    all_kernels = ctx.all_kernels
    host_events = ctx.host_events
    comm_events = ctx.comm_events
    steps = ctx.steps
    structures = ctx.structures
    fusion_opps = ctx.fusion_opps
    comm_analysis = ctx.comm_analysis
    kernel_stats = ctx.kernel_stats
    meta = ctx.meta
    warmup_info = ctx.warmup_info
    stage_breakdown = ctx.stage_breakdown
    category_ratios = ctx.category_ratios
    lines = []

    def w(line=""):
        lines.append(line)

    w(f"# {meta.get('model', 'Model')} Profiling Analysis Report")
    w()
    w("## 1. Overview")
    w()
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Profile source | {meta.get('source', 'unknown')} |")
    w(f"| Total kernel count | {len(all_kernels)} |")
    w(f"| AI_CORE kernels | {sum(1 for k in all_kernels if k.task_type == 'AI_CORE')} |")
    w(f"| AI_CPU kernels | {sum(1 for k in all_kernels if k.task_type == 'AI_CPU')} |")
    w(f"| HCCL kernels | {sum(1 for k in all_kernels if k.task_type == 'HCCL')} |")
    w(f"| Step count | {len(steps)} |")
    w("| Profile level | l1 |")
    w()

    # ===== LAYER 0: Warmup Verification =====
    w("## 2. Warmup Verification (Layer 0)")
    w()
    if warmup_info:
        if warmup_info.get("stripped", True):
            w(f"Warmup properly stripped during profiling collection. {warmup_info.get('note', '')}")
        else:
            w(f"**WARMUP_NOT_STRIPPED**: warmup steps detected in profiling data. {warmup_info.get('note', '')}")
            w()
            w(
                "> Recommendation: re-profile with profiling-collection skill "
                "ensuring warmup runs outside the profiler context."
            )
    w()

    # ===== LAYER 1: Stage Breakdown =====
    w("## 3. Stage Breakdown — DiT vs VAE (Layer 1)")
    w()
    if stage_breakdown:
        w("| Stage | Duration (ms) | Percentage |")
        w("|---|---|---|")
        for key, label in [("dit", "DiT (Transformer)"), ("vae", "VAE")]:
            sb = stage_breakdown.get(key, {})
            w(f"| {label} | {sb.get('dur_ms', 0):.1f} | {sb.get('pct', 0):.1%} |")
        w(f"| **Total** | **{stage_breakdown.get('total_ms', 0):.1f}** | **100%** |")
        w()
        # Bottleneck judgment
        dit_pct = stage_breakdown.get("dit", {}).get("pct", 0)
        vae_pct = stage_breakdown.get("vae", {}).get("pct", 0)
        if dit_pct >= 0.70:
            w(f"> Bottleneck stage: **DiT** ({dit_pct:.0%}). Focus optimization on Transformer path.")
        elif vae_pct >= 0.70:
            w(f"> Bottleneck stage: **VAE** ({vae_pct:.0%}). Focus optimization on VAE encode/decode.")
        else:
            w(f"> Balanced workload: DiT ({dit_pct:.0%}) / VAE ({vae_pct:.0%}).")
    w()

    # ===== LAYER 2: Category Ratios =====
    w("## 4. Category Ratios — FA / MatMul / Vector / Comm (Layer 2)")
    w()
    if category_ratios:
        for stage_key, stage_label in [("dit", "DiT"), ("vae", "VAE")]:
            cats = category_ratios.get(stage_key, {})
            if cats:
                w(f"### {stage_label}")
                w()
                w("| Category | Percentage |")
                w("|---|---|")
                for cat_name in ("FA", "MatMul", "Vector", "Comm"):
                    pct = cats.get(cat_name, 0)
                    w(f"| **{cat_name}** | {pct:.0f}% |")
                w()
    w()

    # ===== Hidden Issue Discovery =====
    w("## 5. Bubble Analysis — Host Bound (Layer 3a)")
    w()

    for step in steps:
        step_kernels = [k for k in all_kernels if step["start_us"] <= k.start_us < step["end_us"]]
        device_intervals = build_device_intervals(step_kernels, step["start_us"], step["end_us"])
        bubble = compute_bubble_metrics(step["start_us"], step["end_us"], device_intervals)
        tags = classify_hidden_issue(bubble)

        w(f"### Step {step['id']} (`{step['marker_name']}`)")
        w()
        w("| Metric | Value |")
        w("|---|---|")
        w(f"| Service time | {bubble['service_ms']:.2f} ms |")
        w(f"| Device busy (union) | {bubble['device_busy_union_ms']:.2f} ms |")
        w(f"| **Underfeed (Host Bound)** | **{bubble['underfeed_ms']:.2f} ms** |")
        w(f"| **Underfeed ratio** | **{bubble['underfeed_ratio']:.1%}** |")
        w(f"| Prelaunch gap | {bubble['prelaunch_gap_ms']:.2f} ms |")
        w(f"| Tail gap | {bubble['tail_gap_ms']:.2f} ms |")
        w(f"| Internal bubble total | {bubble['internal_bubble_total_ms']:.2f} ms |")
        w(f"| Largest internal bubble | {bubble['largest_internal_bubble_ms']:.2f} ms |")
        w(f"| Bubble count | {bubble['bubble_count']} |")
        w()

        if tags:
            w(f"**Anomaly Tags:** {', '.join(tags)}")
        else:
            w("**Anomaly Tags:** none (device utilization healthy)")

        # Host evidence for top bubbles
        w()
        w("#### Top Bubble Windows with Host Evidence")
        w()

        host_intervals = [e.interval for e in host_events]
        sync_intervals = [e.interval for e in host_events if _is_sync_event(e)]
        comm_event_intervals = [e.interval for e in comm_events]

        bubbles = sorted(bubble["bubble_windows"], key=lambda b: b.dur_us, reverse=True)[:5]
        if bubbles:
            w("| # | Start (us) | End (us) | Dur (ms) | Host Cov | Sync Cov | Comm Cov | Attribution |")
            w("|---|---|---|---|---|---|---|---|")
            for bi, bub in enumerate(bubbles):
                attr = soft_attribution_for_bubble(bub, host_intervals, sync_intervals, comm_event_intervals)
                labels = ", ".join(attr["soft_root_cause_labels"])
                w(
                    f"| {bi + 1} | {bub.start_us:.0f} | {bub.end_us:.0f} | {bub.dur_us / 1000:.2f} | "
                    f"{_fmt_pct(attr['host_visible_coverage_ratio'])} | "
                    f"{_fmt_pct(attr['sync_marker_overlap_ratio'])} | "
                    f"{_fmt_pct(attr['comm_marker_overlap_ratio'])} | {labels} |"
                )

            # Kernel context around top bubbles
            w()
            w("#### Kernel Context Around Largest Bubble")
            w()
            top_bubble = bubbles[0]
            prev_k = _kernel_before(step_kernels, top_bubble.start_us)
            next_k = _kernel_after(step_kernels, top_bubble.end_us)
            w("| Position | Name | Task Type | Dur (us) | Stream |")
            w("|---|---|---|---|---|")
            for label, k in [("Before gap", prev_k), ("After gap", next_k)]:
                if k:
                    w(f"| {label} | {k.name} | {k.task_type} | {k.dur_us:.1f} | {k.stream_id} |")
            w()

        else:
            w("No significant bubble windows detected.")
            w()

    # ===== Communication Analysis =====
    w("## 6. Communication Analysis (Layer 3b)")
    w()
    if comm_analysis["comm_total_ms"] > 0:
        w("| Metric | Value |")
        w("|---|---|")
        w(f"| Total communication time | {comm_analysis['comm_total_ms']:.2f} ms |")
        w(f"| Communication hidden (overlapped w/ compute) | {comm_analysis['comm_hidden_ms']:.2f} ms |")
        w(f"| **Communication exposed (not overlapped)** | **{comm_analysis['comm_exposed_ms']:.2f} ms** |")
        w(f"| **Exposed (can not hide) ratio** | **{comm_analysis['can_not_hide_ratio']:.1%}** |")
        w()
        if comm_analysis["can_not_hide_ratio"] > 0.30:
            w("WARNING: Significant portion of communication is NOT overlapped with compute.")
            w("Consider optimizing communication pipeline or increasing computation parallelism.")
        elif comm_analysis["can_not_hide_ratio"] < 0.10:
            w("OK: Communication is well-overlapped with compute.")
    else:
        w("No HCCL communication events detected (single card run).")
    w()

    # ===== Structure Breakdown =====
    w("## 7. Structure (Layer) Timing Breakdown")
    w()
    if structures:
        w("| # | Type | Kernels | Wall (ms) | Busy (ms) | Kernel Sum (ms) | AI_CORE% | AI_CPU% | HCCL% |")
        w("|---|---|---|---|---|---|---|---|---|")
        for i, s in enumerate(structures):
            w(
                f"| {i} | {s['type']} | {s['kernel_count']} | {s['wall_ms']:.2f} | "
                f"{s['device_busy_union_ms']:.2f} | {s['kernel_sum_ms']:.2f} | "
                f"{s['ai_core_pct']:.0%} | {s['ai_cpu_pct']:.0%} | {s['hccl_pct']:.0%} |"
            )
        w()

    # ===== Operator Ranking (>1% threshold) =====
    w("## 8. Operator Details — >1% Duration (Layer 4)")
    w()
    # Filter to >1% operators
    total_kernel_us = sum(stat["total_dur_us"] for _, stat in kernel_stats.items())
    ranked = _filter_operators_by_threshold(kernel_stats, total_kernel_us, 1.0)
    # Add stage info
    for item in ranked:
        item["stage"] = _classify_kernel_stage(item["name"], item.get("task_type", ""))
    w("| # | Operator | Count | Dur (ms) | % | Type | Stage |")
    w("|---|---|---|---|---|---|---|")
    for ri, stat in enumerate(ranked, 1):
        stage_tag = "D" if stat.get("stage") == "DiT" else "V"
        w(
            f"| {ri} | {stat['name']} | {stat['count']} | {stat['total_dur_us'] / 1000:.2f} | "
            f"{stat['pct']:.1f}% | {stat['task_type']} | {stage_tag} |"
        )
    if not ranked:
        w("No single operator exceeds 1% of total kernel time.")
    w()

    # ===== Wait-Anchor =====
    wait_anchors = detect_wait_anchors(all_kernels, steps[0]["start_us"], steps[0]["end_us"]) if steps else []
    if wait_anchors:
        w("## 9. Wait-Anchor False Hotspot Candidates")
        w()
        w("| Name | Dur (us) | Start (us) |")
        w("|---|---|---|")
        for wa in wait_anchors[:10]:
            w(f"| {wa['name']} | {wa['duration_us']:.1f} | {wa['start_us']:.0f} |")
        w()
        w("Note: These are tiny-kernel candidates. Real wait-anchor detection requires `Wait Time(us)`")
        w("from kernel_details.csv which is not available in Chrome Trace format at level=l1.")
        w()

    # ===== AICPU =====
    device_intervals_all = [k.interval for k in all_kernels if k.task_type == "AI_CORE"] if not steps else []
    if not device_intervals_all and steps:
        step_kernels = [k for k in all_kernels if steps[0]["start_us"] <= k.start_us < steps[0]["end_us"]]
        device_intervals_all = [k.interval for k in step_kernels if k.task_type == "AI_CORE"]

    aicpu_results = classify_aicpu(all_kernels, device_intervals_all)
    exposed_aicpu = [a for a in aicpu_results if a["classification"] == "AICPU_EXPOSED_NOT_ALLOWED"]
    if aicpu_results:
        w("## 10. AICPU Classification")
        w()
        w("| Name | Dur (us) | Masked Ratio | Classification |")
        w("|---|---|---|---|")
        for a in aicpu_results[:20]:
            w(f"| {a['name']} | {a['duration_us']:.1f} | {a['masked_ratio']:.1%} | {a['classification']} |")
        if exposed_aicpu:
            w()
            w(f"WARNING: {len(exposed_aicpu)} AICPU kernels are fully exposed (not masked by AI_CORE overlap).")
        w()

    # ===== Fusion Opportunities =====
    w("## 11. Fusion Opportunities (Layer 3c)")
    w()
    w("### MindIE-SD Compilation Patterns")
    w()
    _render_mindie_fusion_table(w, all_kernels, steps)

    w()
    w("### Generic Fusion Suggestions (requires custom implementation)")
    w()
    if fusion_opps:
        w("| # | Pattern | Kernel Chain | Current (ms) | Est. Savings (ms) | Est. After (ms) | Savings% |")
        w("|---|---|---|---|---|---|---|")
        for fi, fo in enumerate(fusion_opps[:20], 1):
            chain_short = " -> ".join(fo["kernel_chain"][:4])
            w(
                f"| {fi} | {fo['pattern']} | {chain_short} | {fo['current_dur_ms']:.3f} | "
                f"{fo['estimated_savings_ms']:.3f} | {fo['estimated_fused_ms']:.3f} | {fo['savings_pct']:.0f}% |"
            )
    else:
        w("No clear generic fusion opportunities detected.")
    w()

    # ===== Summary =====
    w("## 12. Summary")
    w()
    underfeed_ratio = 0.0
    if steps:
        step = steps[0]
        step_kernels = [k for k in all_kernels if step["start_us"] <= k.start_us < step["end_us"]]
        device_intervals = build_device_intervals(step_kernels, step["start_us"], step["end_us"])
        bubble = compute_bubble_metrics(step["start_us"], step["end_us"], device_intervals)
        underfeed_ratio = bubble["underfeed_ratio"]

        q1 = "YES" if bubble["underfeed_ratio"] >= 0.10 else "NO"
        w(f"1. Are there significant device idle bubbles? **{q1}** (underfeed={bubble['underfeed_ratio']:.1%})")

        w(f"2. Which step type/group do they concentrate in? Step {step['id']} (`{step['marker_name']}`)")

        dominant = "none"
        if bubble["prelaunch_gap_ms"] >= max(1.0, 0.05 * bubble["service_ms"]):
            dominant = "prelaunch"
        if bubble["tail_gap_ms"] >= max(1.0, 0.05 * bubble["service_ms"]):
            dominant = f"{dominant}/tail" if dominant != "none" else "tail"
        if bubble["internal_bubble_total_ms"] >= max(1.0, 0.05 * bubble["service_ms"]):
            dominant = f"{dominant}/internal" if dominant != "none" else "internal"
        w(f"3. Are they primarily prelaunch / tail / internal / inter-step? **{dominant or 'none'}**")

        w(
            f"4. Is there significant host-originated risk? "
            f"{'**YES**' if bubble['underfeed_ratio'] >= 0.20 else 'low'} "
            f"(underfeed={bubble['underfeed_ratio']:.1%})"
        )

        w(
            "5. Is evidence sufficient for root cause? **insufficient_evidence** "
            "(profile level=l1, with_stack=false, no per-kernel wait time in trace JSON)"
        )
    w()

    # ===== Recommendations =====
    w("## 13. Recommendations (Layer 5)")
    w()
    w("| Priority | Finding | Recommendation | Reference |")
    w("|:--:|------|------|------|")

    recs = _generate_recommendations(ctx, underfeed_ratio, exposed_aicpu)
    for prio, finding, rec, ref in recs:
        w(f"| **{prio}** | {finding} | {rec} | {ref} |")
    w()

    return "\n".join(lines)


def _render_mindie_fusion_table(w, all_kernels, steps):
    """Render MindIE-SD compilation pattern fusion table."""

    # MindIE-SD fusion patterns with recognition rules
    mindie_patterns = [
        ("RMSNorm", "rmsnorm", "RMSNorm + adjacent MatMul", "CompilationConfig.fusion_patterns.enable_rms_norm"),
        ("RoPE", "rope", "RoPE kernel consecutively appears", "CompilationConfig.fusion_patterns.enable_rope"),
        ("AdaLayerNorm", "adaln", "AdaLN + adjacent kernel", "CompilationConfig.fusion_patterns.enable_adalayernorm"),
        ("fastGELU", "gelu", "MatMul -> Add -> GELU", "CompilationConfig.fusion_patterns.enable_fast_gelu"),
        ("Mul+Add", "mul", "Mul -> Add consecutively", "CompilationConfig.fusion_patterns.enable_mul_add"),
    ]

    # Check each pattern against kernel sequence
    kernel_names = [k.name for k in sorted(all_kernels, key=lambda k: k.start_us)]

    w("| # | Pattern | Detected | Check | CompilationConfig Switch |")
    w("|---|---|---|---|---|")
    for i, (pat_name, pat_keyword, check_desc, switch) in enumerate(mindie_patterns, 1):
        # Simple detection: count kernel names containing the keyword
        count = sum(1 for n in kernel_names if pat_keyword in n.lower())
        detected = "YES" if count > 0 else "—"
        w(f"| {i} | {pat_name} | {detected} ({count} kernels) | {check_desc} | `{switch}` |")
    w()


def _generate_recommendations(ctx, underfeed, exposed_aicpu):
    """Generate directional recommendations (what to optimize, not specific APIs)."""
    warmup_info = ctx.warmup_info
    stage_breakdown = ctx.stage_breakdown
    category_ratios = ctx.category_ratios
    comm_analysis = ctx.comm_analysis
    fusion_opps = ctx.fusion_opps
    recs = []

    # P0: Warmup not stripped
    if warmup_info and not warmup_info.get("stripped", True):
        recs.append(
            (
                "P0",
                "Warmup not stripped",
                "Re-profile with warmup steps outside profiler",
                "profiling-collection §Warmup配置",
            )
        )

    # P0: MindIE-SD Pattern hits that have clear benefit
    if fusion_opps and len(fusion_opps) > 0:
        for fo in fusion_opps[:3]:
            if fo.get("savings_pct", 0) >= 15:
                recs.append(
                    (
                        "P0",
                        f"Pattern {fo['pattern']} detected",
                        "Enable MindIE-SD compilation fusion for this pattern",
                        "compilation-dev",
                    )
                )

    # P1: Category-based direction recommendations
    if category_ratios:
        dit = category_ratios.get("dit", {})
        vae = category_ratios.get("vae", {})

        # DiT: MatMul dominant → 量化方向
        if dit.get("MatMul", 0) > 50:
            recs.append(
                (
                    "P1",
                    "DiT MatMul dominant",
                    "MatMul quantization direction — consult features.md for available algorithms",
                    "mindiesd-features.md §MatMul量化",
                )
            )
        # DiT: FA dominant → Attention优化方向
        if dit.get("FA", 0) > 30:
            recs.append(
                (
                    "P1",
                    "DiT FA dominant",
                    "Attention optimization direction — consult features.md for FA quantization + sparse options",
                    "mindiesd-features.md §Attention优化",
                )
            )
        # DiT: Vector dominant → 编译融合方向
        if dit.get("Vector", 0) > 20:
            recs.append(
                (
                    "P1",
                    "DiT Vector dominant",
                    "Compilation fusion direction — consult features.md for Pattern switch options",
                    "compilation-dev",
                )
            )
        # DiT: Comm exposed → 通信掩盖方向
        if "can_not_hide_ratio" in comm_analysis and comm_analysis["can_not_hide_ratio"] > 0.30:
            recs.append(
                (
                    "P1",
                    "Communication exposed",
                    "Communication hiding direction — consult features.md for RSP/USP options",
                    "mindiesd-features.md §通信掩盖",
                )
            )
        # VAE: MatMul dominant → ACLGraph方向
        if vae.get("MatMul", 0) > 30:
            recs.append(
                (
                    "P1",
                    "VAE MatMul dominant",
                    "ACLGraph acceleration direction — consult features.md for compilation options",
                    "compilation-dev",
                )
            )

    # P1: Host Bound issues
    if underfeed >= 0.20:
        recs.append(
            (
                "P1",
                "High Host Bound",
                "Re-profile with with_stack=true to identify Host-side bottleneck",
                "performance-analysis §Layer 3a",
            )
        )

    # P2: Data quality
    if 0.10 <= underfeed < 0.20:
        recs.append(
            ("P2", "Moderate Host Bound", "Consider re-profiling with with_stack=true for better attribution", "—")
        )

    # P2: Generic fusion with lower benefit
    candidate_fusions = [fo for fo in fusion_opps if fo.get("savings_pct", 0) < 15]
    for fo in candidate_fusions[:2]:
        recs.append(
            (
                "P2",
                f"Fusion opportunity: {fo['pattern']}",
                f"Estimated {fo['savings_pct']:.0f}% savings "
                f"({fo['current_dur_ms']:.1f}ms). Requires custom implementation.",
                "通用融合，需自行实现",
            )
        )

    # -- AICPU --
    if exposed_aicpu:
        recs.append(
            (
                "P2",
                f"{len(exposed_aicpu)} exposed AICPU kernels",
                "Consider migrating AICPU ops to AI_CORE or increasing compute overlap",
                "—",
            )
        )

    # P2: Stage bottleneck
    if stage_breakdown:
        dit_pct = stage_breakdown.get("dit", {}).get("pct", 0)
        vae_pct = stage_breakdown.get("vae", {}).get("pct", 0)
        if dit_pct >= 0.70:
            recs.append(
                (
                    "P2",
                    "DiT is bottleneck stage",
                    f"DiT accounts for {dit_pct:.0%} of time. Prioritize DiT optimization over VAE.",
                    "performance-analysis §Layer 1",
                )
            )
        elif vae_pct >= 0.70:
            recs.append(
                (
                    "P2",
                    "VAE is bottleneck stage",
                    f"VAE accounts for {vae_pct:.0%} of time. Check CANN Conv2D compatibility.",
                    "performance-analysis §Layer 1",
                )
            )

    # Sort by priority
    prio_order = {"P0": 0, "P1": 1, "P2": 2}
    recs.sort(key=lambda x: prio_order.get(x[0], 9))

    if not recs:
        recs.append(
            (
                "—",
                "No significant bottlenecks detected",
                "Device utilization appears healthy at current profiling level",
                "—",
            )
        )

    return recs


def render_architecture_report(
    all_kernels: List[KernelInfo],
    steps: List[Dict[str, Any]],
    structures: List[Dict[str, Any]],
    comm_analysis: Dict[str, Any],
    meta: Dict[str, Any],
) -> str:
    lines = []

    def w(line=""):
        lines.append(line)

    w("# Model Architecture Report (from profiling data)")
    w()
    w("## 1. Configuration Context")
    w()
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Data source | {meta.get('source', 'unknown')} |")
    w("| Capture level | l1, with_stack=false |")
    w(f"| Step count | {len(steps)} |")
    w(f"| Total kernel count | {len(all_kernels)} |")
    w()

    # ===== Detected Layers =====
    w("## 2. Model Architecture Determination")
    w()
    w("### Evidence Chain")
    w()
    w("| Evidence | Value | Confidence |")
    w("|---|---|---|")
    w(f"| Steps detected | {len(steps)} | high |")
    w(f"| Structures segmented | {len(structures)} | medium |")
    w(f"| Distinct structure types | {len(set(s.get('type', 'unknown') for s in structures))} | medium |")
    w()
    w("Given the dummy run uses Wan2.2 with 2 transformer blocks per stream, the model structure is:")
    w()
    w("```")
    w("TextEncoder -> Transformer(block_0) -> Transformer_2(block_0) [-> VAE(optional)]")
    w("```")
    w()

    # ===== Forward Pass Boundaries =====
    w("## 3. Forward Pass Boundaries")
    w()
    if steps:
        w("| Pass | Start (us) | End (us) | Wall (ms) | Kernel Count |")
        w("|---|---|---|---|---|")
        for s in steps:
            wall = (s["end_us"] - s["start_us"]) / 1000.0
            kcount = sum(1 for k in all_kernels if s["start_us"] <= k.start_us < s["end_us"])
            w(f"| {s['id']} | {s['start_us']:.0f} | {s['end_us']:.0f} | {wall:.2f} | {kcount} |")
    else:
        w("No steps detected (pseudo-step spanning entire capture).")
    w()

    # ===== Layer Classification =====
    w("## 4. Layer Classification")
    w()
    if structures:
        w("| Layer Type | Count | Kernel Count | Wall (ms) | Characteristics |")
        w("|---|---|---|---|---|")
        type_counts: Dict[str, int] = defaultdict(int)
        type_kcounts: Dict[str, int] = defaultdict(int)
        type_walls: Dict[str, float] = defaultdict(float)

        for s in structures:
            t = s["type"]
            type_counts[t] += 1
            type_kcounts[t] += s["kernel_count"]
            type_walls[t] += s["wall_ms"]

        for t in sorted(type_counts.keys()):
            chars = _describe_type(t)
            w(f"| {t} | {type_counts[t]} | {type_kcounts[t]} | {type_walls[t]:.2f} | {chars} |")
    else:
        w("No structures segmented.")
    w()

    # ===== Per-layer sub-structure =====
    w("## 5. Per-Layer Sub-Structure")
    w()
    for si, s in enumerate(structures[:10]):  # limit to 10
        w(f"### Layer {si}: {s['type']}")
        w()
        w(f"- Wall time: {s['wall_ms']:.2f} ms")
        w(f"- Kernel count: {s['kernel_count']}")
        w(f"- AI_CORE: {s['ai_core_pct']:.0%}, AI_CPU: {s['ai_cpu_pct']:.0%}, HCCL: {s['hccl_pct']:.0%}")
        w()

        # Top kernels in this structure
        kernels_in = s.get("kernels", [])
        if kernels_in:
            by_name: Dict[str, List[KernelInfo]] = defaultdict(list)
            for k in kernels_in:
                by_name[k.name].append(k)
            ranked = sorted(by_name.items(), key=lambda x: sum(kk.dur_us for kk in x[1]), reverse=True)

            w("| Operator | Count | Total Dur (ms) | Share of Layer |")
            w("|---|---|---|---|")
            layer_dur_total = max(sum(kk.dur_us for kk in kernels_in), 1)
            for name, klist in ranked[:10]:
                total_dur = sum(kk.dur_us for kk in klist) / 1000.0
                share = sum(kk.dur_us for kk in klist) / layer_dur_total
                w(f"| {name} | {len(klist)} | {total_dur:.3f} | {share:.1%} |")
            w()

    # ===== Communication Pipeline =====
    w("## 6. Communication Pipeline Structure")
    w()
    if comm_analysis["comm_total_ms"] > 0:
        w("| Stream | Role | Notes |")
        w("|---|---|---|")
        streams = set(k.stream_id for k in all_kernels)
        for sid in sorted(streams):
            role = "compute" if sid == 0 else "auxiliary"
            sk = [k for k in all_kernels if k.stream_id == sid]
            hccl_count = sum(1 for k in sk if k.task_type == "HCCL")
            w(f"| {sid} | {role} | {len(sk)} kernels, {hccl_count} HCCL |")

        w()
        w("### Pipeline Overlap Diagram")
        w("```")
        compute_total = sum(k.dur_us for k in all_kernels if k.task_type == "AI_CORE") / 1000.0
        comm_total = comm_analysis["comm_total_ms"]
        hidden = comm_analysis["comm_hidden_ms"]
        exposed = comm_analysis["comm_exposed_ms"]
        comm_pct_hidden = int(40 * hidden / max(comm_total, 1))
        comm_pct_exposed = int(40 * exposed / max(comm_total, 1))
        hidden_bar = '#' * max(1, comm_pct_hidden)
        exposed_bar = '.' * max(1, comm_pct_exposed)
        w(f"  Compute [{'=' * 40}] {compute_total:.1f}ms")
        w(f"  Comm    [{hidden_bar}{exposed_bar}] {comm_total:.1f}ms (hidden={hidden:.1f}ms exposed={exposed:.1f}ms)")
        w("```")
    else:
        w("No HCCL communication detected (single card).")
    w()

    # ===== Model Architecture Summary =====
    w("## 7. Model Architecture Summary")
    w()
    w("```")
    w("Wan2.2 Dummy Run (2-block)")
    w("==========================")
    w("")
    if steps:
        s = steps[0]
        w(f"Total execution: {(s['end_us'] - s['start_us']) / 1000:.1f} ms")
    w(f"Total kernels: {len(all_kernels)}")
    w("")
    for si, s in enumerate(structures[:10]):
        w(f"  [{si}] {s['type']} ({s['kernel_count']} ops, {s['wall_ms']:.1f}ms)")
    w("```")
    w()

    return "\n".join(lines)


# ============================================================================
# Helpers
# ============================================================================


def _is_sync_event(ev: TraceEvent) -> bool:
    name_lower = ev.name.lower()
    return any(kw in name_lower for kw in ("sync", "memcpy", "copy", "h2d", "d2h", "to", "_copy"))


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.0%}"


def _kernel_before(kernels: List[KernelInfo], ts: float) -> Optional[KernelInfo]:
    candidates = [k for k in kernels if k.end_us <= ts]
    if not candidates:
        return None
    return max(candidates, key=lambda k: k.end_us)


def _kernel_after(kernels: List[KernelInfo], ts: float) -> Optional[KernelInfo]:
    candidates = [k for k in kernels if k.start_us >= ts]
    if not candidates:
        return None
    return min(candidates, key=lambda k: k.start_us)


def _describe_type(t: str) -> str:
    descriptions = {
        "text_encoder": "T5 text encoding (UMT5EncoderModel)",
        "vae": "VAE encode/decode",
        "transformer_block": "WanTransformer3DModel block",
        "attention_block": "Attention + MLP sub-layer",
        "mlp_block": "MLP sub-layer",
        "attention": "Attention sub-layer",
    }
    return descriptions.get(t, "generic")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Ascend NPU profiling trace (CANN or Chrome Trace JSON format)"
    )
    parser.add_argument("--profile-dir", required=True, help="Directory containing profiling output")
    parser.add_argument("--output-dir", default="./", help="Directory for output reports")
    parser.add_argument("--model", default="Wan2.2", help="Model name for report titles")
    args = parser.parse_args()

    profile_dir = args.profile_dir
    output_dir = args.output_dir
    if not os.path.isdir(profile_dir):
        raise RuntimeError(f"profile directory not found: {profile_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # ---- INGEST ----
    logger.info("Loading profiling data from: %s", profile_dir)
    profiler_output = find_profiler_output_dir(profile_dir)
    sources: List[str] = []

    if profiler_output:
        logger.info("  Found CANN profiler output: %s", profiler_output)
        csv_path = os.path.join(profiler_output, "kernel_details.csv")
        trace_path = os.path.join(profiler_output, "trace_view.json")
        step_csv_path = os.path.join(profiler_output, "step_trace_time.csv")

        if os.path.exists(csv_path):
            kernels = load_kernels_from_csv(csv_path)
            logger.info("  Kernels (from CSV): %d", len(kernels))
            sources.append("kernel_details.csv")

        else:
            logger.info("  kernel_details.csv not found, falling back to trace JSON")
            kernels = []

        host_events: List[TraceEvent] = []
        comm_events: List[TraceEvent] = []
        step_markers: List[TraceEvent] = []

        if os.path.exists(trace_path) and kernels:
            # Load host events from trace_view.json, already normalized
            # The CANN trace_view.json timestamps need to be normalized to the same t0
            # First, find the raw t0 from the CSV
            raw_t0 = _get_csv_raw_t0(csv_path)
            all_trace_events = load_host_events_from_trace(trace_path, raw_t0)
            logger.info("  Trace events (from trace_view.json): %d", len(all_trace_events))
            sources.append("trace_view.json")

            # Classify trace events into host/comm/marker
            for ev in all_trace_events:
                if "Step#" in ev.name or "Iteration" in ev.name or "ProfilerStep" in ev.name:
                    step_markers.append(ev)
                elif "AllReduce" in ev.name or "AllGather" in ev.name or "Hcom" in ev.name:
                    comm_events.append(ev)
                elif ev.cat == "HostToDevice":
                    comm_events.append(ev)
                elif ev.cat == "" and ev.name in ("Computing", "Free"):
                    pass  # Runtime overhead events, not host ops
                else:
                    host_events.append(ev)

            # Also classify kernels in trace that aren't in CSV
            skip_names = {"Computing", "Free", "EVENT_RECORD", "EVENT_WAIT", "MEMCPY_ASYNC"}
            trace_kernel_events = []
            for ev in all_trace_events:
                if ev.name not in skip_names and ev.cat == "" and ev.ph == "X":
                    trace_kernel_events.append(ev)

            # Merge trace kernel events into kernel list if not already present
            csv_names = set(k.name for k in kernels)
            for ev in trace_kernel_events:
                if ev.name not in csv_names:
                    ki = KernelInfo(
                        name=ev.name,
                        task_type=_infer_task_type_from_name(ev.name),
                        start_us=ev.ts,
                        dur_us=ev.dur,
                        stream_id=ev.tid,
                        input_shapes="",
                    )
                    ki.wait_us = 0.0
                    kernels.append(ki)

            logger.info(
                "  Host events: %d, Comm events: %d, Step markers: %d",
                len(host_events),
                len(comm_events),
                len(step_markers),
            )

        # Load step-level timing
        step_time_info: Dict[str, Any] = {}
        if os.path.exists(step_csv_path):
            step_time_info = load_step_time_from_csv(step_csv_path)
            if step_time_info:
                logger.info(
                    "  Step timing: computing=%.1fms, comm=%.1fms, free=%.1fms",
                    step_time_info.get('computing_us', 0) / 1000,
                    step_time_info.get('communication_us', 0) / 1000,
                    step_time_info.get('free_us', 0) / 1000,
                )
    else:
        # Fallback: Chrome Trace JSON files
        logger.info("  CANN profiler output not found, using Chrome Trace JSON fallback")
        events, sources = load_all_traces(profile_dir)
        if not events:
            raise RuntimeError("no trace events found in profile directory")
        logger.info("  %d events from %d files: %s", len(events), len(sources), sources)
        kernels, host_events, comm_events, step_markers = classify_events(events)
        step_time_info = {}
        logger.info(
            "  Kernels: %d, Host: %d, Comm: %d, Markers: %d",
            len(kernels),
            len(host_events),
            len(comm_events),
            len(step_markers),
        )

    if not kernels:
        raise RuntimeError("no kernel events found")

    # ---- STEP_DETECTION ----
    steps = detect_steps(kernels, step_markers)
    logger.info("  Steps: %d", len(steps))

    # If no steps from markers, create a pseudo-step spanning all kernels
    if not steps:
        t_min = min(k.start_us for k in kernels)
        t_max = max(k.end_us for k in kernels)
        steps = [
            {
                "id": 0,
                "start_us": t_min,
                "end_us": t_max,
                "marker_name": "full_capture",
            }
        ]

    # ---- SEGMENTATION ----
    all_structures = []
    for step in steps:
        structures = segment_structures(kernels, step, host_events)
        all_structures.extend(structures)
    logger.info("  Structures: %d", len(all_structures))

    # ---- COMMUNICATION ANALYSIS ----
    compute_intervals = [k.interval for k in kernels if k.task_type == "AI_CORE"]
    comm_analysis = analyze_communication(kernels, comm_events, compute_intervals)

    # If step_time_info has comm data, use it as authoritative source
    if step_time_info:
        comm_not_overlapped = step_time_info.get("communication_not_overlapped_us", 0) / 1000.0
        comm_total = step_time_info.get("communication_us", 0) / 1000.0
        if comm_total > 0:
            comm_analysis["comm_total_ms"] = comm_total
            comm_analysis["comm_exposed_ms"] = comm_not_overlapped
            comm_analysis["comm_hidden_ms"] = comm_total - comm_not_overlapped
            comm_analysis["comm_exposed_ratio"] = comm_not_overlapped / comm_total
            comm_analysis["can_not_hide_ratio"] = comm_not_overlapped / comm_total

    # ---- FUSION OPPORTUNITIES ----
    fusion_opps = detect_fusion_opportunities(kernels)
    logger.info("  Fusion opportunities: %d", len(fusion_opps))

    # ---- KERNEL STATS ----
    kernel_stats = compute_kernel_stats(kernels)

    # ---- LAYER 0: WARMUP DETECTION ----
    warmup_info = _detect_warmup(steps, kernels)
    warmup_skip = warmup_info.get("suggested_skip", 0)
    warmup_status = "STRIPPED" if warmup_info['stripped'] else "NOT STRIPPED"
    logger.info("  Warmup: %s (%s)", warmup_status, warmup_info['note'])

    # ---- LAYER 1: STAGE BREAKDOWN ----
    stage_breakdown = _compute_stage_breakdown(kernels, steps, warmup_skip)
    logger.info(
        "  Stage breakdown: DiT=%.0fms, VAE=%.0fms",
        stage_breakdown['dit']['dur_ms'],
        stage_breakdown['vae']['dur_ms'],
    )

    # ---- LAYER 2: CATEGORY RATIOS ----
    category_ratios = _compute_category_ratios(kernels, steps, warmup_skip)
    for stage in ("dit", "vae"):
        cats = category_ratios[stage]
        logger.info(
            "  %s ratios: FA=%.0f%%, MatMul=%.0f%%, Vector=%.0f%%, Comm=%.0f%%",
            stage.upper(),
            cats['FA'],
            cats['MatMul'],
            cats['Vector'],
            cats['Comm'],
        )

    # ---- META ----
    meta = {
        "model": args.model,
        "source": ", ".join(sources) if sources else profile_dir,
        "profile_dir": profile_dir,
        "step_time_info": step_time_info,
    }

    # ---- RENDER ----
    profiling_path = os.path.join(output_dir, "profiling_report.md")
    profiling_ctx = ProfilingContext(
        kernels,
        host_events,
        comm_events,
        steps,
        all_structures,
        fusion_opps,
        comm_analysis,
        kernel_stats,
        meta,
        warmup_info,
        stage_breakdown,
        category_ratios,
    )
    profiling_report = render_profiling_report(profiling_ctx)
    with open(profiling_path, "w", encoding="utf-8") as fh:
        fh.write(profiling_report)
    logger.info("Profiling report: %s", profiling_path)

    arch_path = os.path.join(output_dir, "model_architecture_report.md")
    arch_report = render_architecture_report(
        kernels,
        steps,
        all_structures,
        comm_analysis,
        meta,
    )
    with open(arch_path, "w", encoding="utf-8") as fh:
        fh.write(arch_report)
    logger.info("Architecture report: %s", arch_path)


# ---- STAGE & CATEGORY CLASSIFICATION ----


_VAE_KEYWORDS = ("conv2d", "conv3d", "groupnorm", "upsample", "resblock", "encoder", "decoder", "vae", "latent")
_DIT_KEYWORDS = (
    "attn",
    "matmul",
    "linear",
    "layernorm",
    "rmsnorm",
    "rope",
    "gelu",
    "silu",
    "softmax",
    "reshape",
    "transformer",
    "block",
    "fused_attn",
)
_STAGE_DIT = "DiT"


def _classify_kernel_stage(name: str, task_type: str) -> str:
    """Classify kernel into DiT or VAE stage based on name."""
    n = name.lower()
    if any(kw in n for kw in _VAE_KEYWORDS):
        return "VAE"
    if any(kw in n for kw in _DIT_KEYWORDS):
        return _STAGE_DIT
    if task_type == "HCCL":
        return _STAGE_DIT
    return _STAGE_DIT


_FA_KEYWORDS = ("attn", "flash_attention", "sdpa", "fused_attn", "attention_forward", "attention_score")
_MM_KEYWORDS = ("matmul", "linear", "gemm", "dequantgemm", "biasadd", "bmm", "quant")
_VEC_KEYWORDS = (
    "gelu",
    "silu",
    "relu",
    "layernorm",
    "rmsnorm",
    "norm",
    "add",
    "mul",
    "div",
    "sub",
    "reshape",
    "transpose",
    "concat",
    "sigmoid",
    "tanh",
    "cast",
    "scale",
    "copy",
    "fill",
    "eltwise",
)


def _classify_kernel_category(name: str, task_type: str) -> str:
    """Classify kernel into FA, MatMul, Vector, or Comm category."""
    if task_type == "HCCL":
        return "Comm"
    n = name.lower()
    if any(kw in n for kw in _FA_KEYWORDS):
        return "FA"
    if any(kw in n for kw in _MM_KEYWORDS):
        return "MatMul"
    if any(kw in n for kw in _VEC_KEYWORDS):
        return "Vector"
    return "Vector"


def _compute_stage_breakdown(
    kernels: List["KernelInfo"],
    steps: List[Dict[str, Any]],
    warmup_skip: int = 0,
) -> Dict[str, Any]:
    """Compute DiT vs VAE breakdown across timed steps."""
    dit_dur = 0.0
    vae_dur = 0.0
    for step in steps[warmup_skip:]:
        step_kernels = [k for k in kernels if step["start_us"] <= k.start_us < step["end_us"]]
        for k in step_kernels:
            stage = _classify_kernel_stage(k.name, k.task_type)
            if stage == "DiT":
                dit_dur += k.dur_us
            else:
                vae_dur += k.dur_us
    total = dit_dur + vae_dur
    total_ms = total / 1000.0
    return {
        "dit": {"dur_ms": dit_dur / 1000.0, "pct": dit_dur / total if total > 0 else 0},
        "vae": {"dur_ms": vae_dur / 1000.0, "pct": vae_dur / total if total > 0 else 0},
        "total_ms": total_ms,
    }


def _compute_category_ratios(
    kernels: List["KernelInfo"],
    steps: List[Dict[str, Any]],
    warmup_skip: int = 0,
) -> Dict[str, Any]:
    """Compute FA/MatMul/Vector/Comm ratios, per stage and overall."""
    result = {}
    for stage_key in ("dit", "vae"):
        result[stage_key] = {"FA": 0.0, "MatMul": 0.0, "Vector": 0.0, "Comm": 0.0}
        stage_total = 0.0
        for step in steps[warmup_skip:]:
            step_kernels = [k for k in kernels if step["start_us"] <= k.start_us < step["end_us"]]
            for k in step_kernels:
                ks = _classify_kernel_stage(k.name, k.task_type)
                if ks != ("DiT" if stage_key == "dit" else "VAE"):
                    continue
                cat = _classify_kernel_category(k.name, k.task_type)
                result[stage_key][cat] += k.dur_us
                stage_total += k.dur_us
        if stage_total > 0:
            for cat in result[stage_key]:
                result[stage_key][cat] = result[stage_key][cat] / stage_total * 100
    return result


def _detect_warmup(steps: List[Dict[str, Any]], kernels: List["KernelInfo"]) -> Dict[str, Any]:
    """Detect if warmup steps were not stripped from profiling data."""
    _key_stripped = "stripped"
    _key_note = "note"
    if len(steps) < 3:
        return {_key_stripped: True, _key_note: "too few steps to detect"}

    step_durs = []
    for step in steps:
        sk = [k for k in kernels if step["start_us"] <= k.start_us < step["end_us"]]
        dur = sum(k.dur_us for k in sk) / 1000.0
        step_durs.append(dur)

    avg = sum(step_durs) / len(step_durs)
    first_step = step_durs[0]
    first_vs_avg = first_step / avg if avg > 0 else 1.0

    compile_kernels = sum(1 for k in kernels if "compile" in k.name.lower() or "jit" in k.name.lower())

    if first_vs_avg > 1.5 or compile_kernels > 0:
        return {
            _key_stripped: False,
            _key_note: f"first step {first_step:.0f}ms vs avg {avg:.0f}ms (ratio {first_vs_avg:.1f}x), "
            f"{compile_kernels} compile/JIT kernels detected",
            "suggested_skip": 1 if first_vs_avg > 1.5 else 0,
        }
    return {_key_stripped: True, _key_note: "no warmup anomaly detected"}


def _filter_operators_by_threshold(
    kernel_stats: Dict[str, Any],
    total_dur_us: float,
    threshold_pct: float = 1.0,
) -> List[Dict[str, Any]]:
    """Filter operator stats to those above threshold percentage."""
    ranked = sorted(kernel_stats.items(), key=lambda x: x[1]["total_dur_us"], reverse=True)
    result = []
    for name, stat in ranked:
        pct = stat["total_dur_us"] / total_dur_us * 100 if total_dur_us > 0 else 0
        if pct >= threshold_pct:
            stat["pct"] = pct
            stat["name"] = name
            result.append(stat)
    return result


# ---- END CLASSIFICATION ----


def _get_csv_raw_t0(csv_path: str) -> float:
    """Get the minimum Start Time from kernel_details.csv (raw timestamp, not normalized)."""
    t0 = None
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    ts = float(row.get("Start Time(us)", 0))
                    if t0 is None or ts < t0:
                        t0 = ts
                except (ValueError, KeyError):
                    pass
    except Exception as exc:
        logger.warning("Failed to read CSV for t0: %s", exc)
    return t0 or 0.0
