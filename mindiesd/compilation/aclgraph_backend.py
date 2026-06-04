#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import contextlib
import dataclasses
import importlib
from typing import Any

import torch

from ..utils.logs.logging import logger

DEBUG_LOG_LEVEL = 10

# ---------------------------------------------------------------------------
# NPU 可用性检测
# ---------------------------------------------------------------------------
try:
    if hasattr(torch.npu, "NPUGraph") and hasattr(torch.npu, "graph"):
        _npu_available = True
        npu_graph_available = True
    else:
        _npu_available = False
        npu_graph_available = False
except ImportError:
    _npu_available = False
    npu_graph_available = False

_global_graph_pool = None


def _get_global_graph_pool():
    global _global_graph_pool
    if _global_graph_pool is None:
        _global_graph_pool = torch.npu.graph_pool_handle()
    return _global_graph_pool


# ---------------------------------------------------------------------------
# ACLGraph 条目
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ACLGraphEntry:
    aclgraph: "torch.npu.NPUGraph"
    static_inputs: list
    output: Any
    input_addresses: list | None = None
    copy_stream: "torch.npu.Stream | None" = None

    def ensure_copy_stream(self):
        if self.copy_stream is None:
            self.copy_stream = torch.npu.Stream()


# ---------------------------------------------------------------------------
# 编译器工厂
# ---------------------------------------------------------------------------


def create_aclgraph_backend():
    """
    Create an ACLGraph backend function for NPU.

    Returns a callable ``aclgraph_backend(gm, example_inputs) -> compiled_fn``
    suitable for wrapping a ``torch.compile``-processed graph module.

    Implements P0/P1/P3 optimizations:
      A1 - synchronize before replay
      B1 - global graph memory pool
      C1 - skip copy_ on same data_ptr
      C2 - safe_output_mode to control clone
      C3 - async copy_ on dedicated stream with event pipeline
      C4 - GC disable during graph capture
      D1 - shape/dtype assert before copy_
      D2 - input address debug validation
    """
    from .compiliation_config import CompilationConfig

    entries: dict[tuple, _ACLGraphEntry] = {}

    def _get_input_shape(inputs):
        return tuple(arg.shape if isinstance(arg, torch.Tensor) else () for arg in inputs)

    def _evict_if_needed():
        max_entries = CompilationConfig.aclgraph_max_entries
        if 0 < max_entries <= len(entries):
            oldest_key = next(iter(entries))
            del entries[oldest_key]

    def _make_static_inputs(inputs):
        lazy = CompilationConfig.aclgraph_lazy_capture
        static_inputs = []
        for input_val in inputs:
            if isinstance(input_val, torch.Tensor):
                # Lazy capture uses fresh inference inputs that are never
                # modified by Dynamo, so detach() (shared storage) is safe.
                # Eager capture receives Dynamo example inputs whose storage
                # may be reused after the backend returns, so clone() is
                # required to keep stable, independent buffers.
                buf = input_val.detach() if lazy else input_val.detach().clone()
                if buf.device.type != "npu" and torch.npu.is_available():
                    buf = buf.npu()
                static_inputs.append(buf)
            else:
                static_inputs.append(input_val)
        return static_inputs

    def _capture_graph(gm, inputs):
        input_shape = _get_input_shape(inputs)
        if input_shape not in entries:
            _evict_if_needed()

            aclgraph = torch.npu.NPUGraph()
            pool = _get_global_graph_pool()

            static_inputs = _make_static_inputs(inputs)
            input_addresses = [x.data_ptr() for x in static_inputs if isinstance(x, torch.Tensor)]

            with contextlib.ExitStack() as stack:
                stack.enter_context(_patch_fn("gc.collect", lambda: None))
                if hasattr(torch.npu, "empty_cache"):
                    stack.enter_context(_patch_fn("torch.npu.empty_cache", lambda: None))

                with torch.npu.graph(npu_graph=aclgraph, pool=pool):
                    output = gm(*static_inputs)

            entries[input_shape] = _ACLGraphEntry(
                aclgraph=aclgraph,
                static_inputs=static_inputs,
                output=output,
                input_addresses=input_addresses,
                copy_stream=None,
            )
        return input_shape

    def aclgraph_backend(gm, example_inputs):
        if example_inputs and not CompilationConfig.aclgraph_lazy_capture:
            _capture_graph(gm, example_inputs)

        def compiled_fn(*args):
            input_shape = _get_input_shape(args)
            if input_shape not in entries:
                _capture_graph(gm, args)

            entry = entries[input_shape]

            # D2: input address debug validation
            if logger.isEnabledFor(DEBUG_LOG_LEVEL) and entry.input_addresses is not None:
                new_addrs = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
                for i, (old_addr, new_addr) in enumerate(zip(entry.input_addresses, new_addrs)):
                    if old_addr != new_addr:
                        logger.warning(
                            "[MindIE-SD/compilation] ACLGraph input address changed. "
                            "issue=input data_ptr differs from captured graph buffer, index=%d, "
                            "expected_data_ptr=%d, actual_data_ptr=%d. "
                            "possible_cause=caller reused the graph with a different tensor storage. "
                            "Troubleshooting: confirm input tensors are copied into static buffers before replay; "
                            "enable DEBUG graph logs to inspect the capture and replay sequence.",
                            i,
                            old_addr,
                            new_addr,
                        )

            # D1 + C1: shape/dtype validation and data_ptr skip for non-tensor
            # inputs; actual copy_ is deferred to the async copy stream below.
            needs_copy = []
            for i, (static_buf, new_inp) in enumerate(zip(entry.static_inputs, args)):
                if not isinstance(static_buf, torch.Tensor):
                    continue
                if not isinstance(new_inp, torch.Tensor):
                    continue
                if static_buf.data_ptr() == new_inp.data_ptr():
                    continue
                if static_buf.shape != new_inp.shape or static_buf.dtype != new_inp.dtype:
                    logger.error(
                        "[MindIE-SD/compilation] ACLGraph input validation failed. "
                        "issue=input shape or dtype differs from captured graph, index=%d, "
                        "expected_shape=%s, expected_dtype=%s, actual_shape=%s, actual_dtype=%s. "
                        "possible_cause=model graph was replayed with incompatible inputs. "
                        "Troubleshooting: use the same shape and dtype as graph capture or trigger a new graph capture.",
                        i,
                        tuple(static_buf.shape),
                        static_buf.dtype,
                        tuple(new_inp.shape),
                        new_inp.dtype,
                    )
                    raise RuntimeError(
                        f"ACLGraph input mismatch at position {i}: "
                        f"captured {tuple(static_buf.shape)}/{static_buf.dtype}, "
                        f"got {tuple(new_inp.shape)}/{new_inp.dtype}"
                    )
                needs_copy.append((static_buf, new_inp))

            # C3: issue async copy_ on a dedicated stream and record an event
            # for the default stream to wait on before replay.
            # A1: synchronize default stream only when copy_ is needed.
            if needs_copy:
                torch.npu.current_stream().synchronize()
                entry.ensure_copy_stream()
                with torch.npu.stream(entry.copy_stream):
                    for static_buf, new_inp in needs_copy:
                        static_buf.copy_(new_inp)
                copy_event = entry.copy_stream.record_event()
                torch.npu.current_stream().wait_event(copy_event)

            entry.aclgraph.replay()

            # C2: optional clone based on safe_output_mode.
            out = entry.output

            if isinstance(out, torch.Tensor):
                return out.clone() if CompilationConfig.safe_output_mode else out
            if isinstance(out, (list, tuple)):
                if CompilationConfig.safe_output_mode:
                    return type(out)(t.clone() if isinstance(t, torch.Tensor) else t for t in out)
                return out
            return out

        return compiled_fn

    return aclgraph_backend


# ---------------------------------------------------------------------------
# Utility: lightweight patching via ExitStack-compatible context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _patch_fn(qualified_name: str, replacement):
    """Patch a module-level function by qualified name, e.g. ``"gc.collect"``.

    This exists to avoid a hard dependency on ``unittest.mock`` at runtime.
    """
    parts = qualified_name.rsplit(".", 1)
    if len(parts) == 2:
        mod_name, attr = parts
        mod = importlib.import_module(mod_name)
    else:
        import builtins

        mod = builtins
        attr = parts[0]
    original = getattr(mod, attr)
    setattr(mod, attr, replacement)
    try:
        yield
    finally:
        setattr(mod, attr, original)
