#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may use this software according to the terms and conditions of the Mulan PSL v2.
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
    npu_graph_available = hasattr(torch.npu, "NPUGraph") and hasattr(torch.npu, "graph")
except ImportError:
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


def _copy_inputs_on_stream(entry: _ACLGraphEntry, needs_copy: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Copy dynamic inputs into graph buffers with device-side stream ordering."""
    current_stream = torch.npu.current_stream()
    entry.ensure_copy_stream()
    input_ready_event = current_stream.record_event()
    entry.copy_stream.wait_event(input_ready_event)
    with torch.npu.stream(entry.copy_stream):
        for static_buf, new_inp in needs_copy:
            static_buf.copy_(new_inp)
    copy_event = entry.copy_stream.record_event()
    current_stream.wait_event(copy_event)


# ---------------------------------------------------------------------------
# 模块级公共辅助函数（单段与分段后端共用）
# ---------------------------------------------------------------------------


def _symint_hint(value):
    """只读地取 SymInt 的具体值，取不到返回 None（不写 guard）。

    ``int(symint)`` 等价于 ``guard_int``：会往当前 ShapeEnv 追加 ``Eq(s, v)`` 并把符号
    特化成 singleton（实测日志 ``Eq(s13, 126)`` + ``set_replacement s13 = 126
    (range_refined_to_singleton)``），使同一份动态图按文本长度反复重编译。
    ``SymNode.hint`` 只读取 hint，不产生 guard；unbacked 符号取 hint 会抛异常。
    """
    node = getattr(value, "node", None)
    if node is None:
        return None
    try:
        hint = node.hint
    except Exception:  # noqa: BLE001
        return None
    return hint if isinstance(hint, int) else None


def _concrete_scalar_input(value):
    """回放期取标量输入的具体值：SymInt 先读只读 hint，取不到再 ``int()``。"""
    if not isinstance(value, torch.SymInt):
        return value
    hint = _symint_hint(value)
    return hint if hint is not None else int(value)


def _get_input_shape(inputs):
    """捕获键：张量取 shape，标量取具体值。

    标量必须进键——捕获时它会被固化进 kernel 参数（如 ``new_zeros(512 - L, ...)``），
    Dynamo 的动态形状编译会让同一个 compiled_fn 收到不同的 L，按 shape 分桶会让后一次
    静默复用前一次的捕获结果。
    """
    key = []
    for arg in inputs:
        if isinstance(arg, torch.Tensor):
            key.append(arg.shape)
        elif isinstance(arg, (bool, int, float, str)):
            key.append(arg)
        elif isinstance(arg, torch.SymInt):
            hint = _symint_hint(arg)
            # 无具体值的符号尺寸：交给 _materialize_scalar_input 报明确错误。
            key.append(() if hint is None else hint)
        else:
            key.append(())
    return tuple(key)


_symint_materialized_warned = False


def _materialize_scalar_input(value, index):
    """把捕获期的符号整数输入落地为具体 int。

    Dynamo/AOT 图会把符号尺寸作为图输入传入（例如 Wan2.2 文本 padding 的
    ``sub = 512 - arg10_1``），捕获时拿到的是 ``torch.SymInt`` 而不是 int。ACLGraph
    只能按具体值捕获（``SymIntArrayRef expected to contain only concrete integers``），
    且该值会固化在回放的 kernel 参数里，因此这里显式落地，并在回放时校验（见
    :func:`_validate_and_copy_inputs`）。

    优先走 :func:`_symint_hint`（只读 hint，不写 guard）；仅当取不到 hint 时退回
    ``int()``——后者会在 ShapeEnv 里追加 ``Eq(s, v)`` 并把符号特化，代价是同一份动态图
    按该标量值反复重编译，因此只作兜底。
    """
    global _symint_materialized_warned
    if not isinstance(value, torch.SymInt):
        return value
    concrete = _symint_hint(value)
    if concrete is not None:
        if not _symint_materialized_warned:
            _symint_materialized_warned = True
            logger.warning(
                "[MindIE-SD/compilation] symbolic integer graph input materialized for ACLGraph capture "
                "(input #%d -> %d, read from hint without a guard). issue=the compiled graph carries a "
                "SymInt input, which ACLGraph cannot replay dynamically. impact=the captured graph is "
                "fixed to this value; a different value at replay raises RuntimeError instead of silently "
                "reusing the captured one.",
                index,
                concrete,
            )
        return concrete
    try:
        concrete = int(value)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"ACLGraph capture requires concrete integer inputs, but graph input #{index} is a "
            "data-dependent symbolic size that has no concrete value. "
            "Troubleshooting: materialize the value before entering torch.compile (call int()/item() "
            "outside the compiled region, or use torch._dynamo.mark_static), or use "
            "backend_mode=aclgraph_only for this graph."
        ) from exc
    if not _symint_materialized_warned:
        _symint_materialized_warned = True
        logger.warning(
            "[MindIE-SD/compilation] symbolic integer graph input materialized for ACLGraph capture "
            "(input #%d -> %d, via int() fallback: the hint was unreadable, so this adds an Eq guard and "
            "specializes the symbol). issue=the compiled graph carries a SymInt input, which ACLGraph "
            "cannot replay dynamically. impact=the compiled graph is fixed to this value and will be "
            "recompiled for other values; at replay a different value raises RuntimeError.",
            index,
            concrete,
        )
    return concrete


def _make_static_inputs(inputs):
    """构造静态输入缓冲。

    lazy 捕获用 detach（共享存储，推理输入不被 Dynamo 改写）；eager 捕获用 detach().clone()
    （Dynamo example 输入存储可能被复用，需独立缓冲）。非张量输入中的 ``torch.SymInt``
    （符号尺寸）在此落地为具体 int，否则捕获期无法构造 kernel 参数。
    """
    from .compiliation_config import CompilationConfig

    lazy = CompilationConfig.aclgraph_lazy_capture
    static_inputs = []
    for index, input_val in enumerate(inputs):
        if isinstance(input_val, torch.Tensor):
            buf = input_val.detach() if lazy else input_val.detach().clone()
            if buf.device.type != "npu" and torch.npu.is_available():
                buf = buf.npu()
            static_inputs.append(buf)
        else:
            static_inputs.append(_materialize_scalar_input(input_val, index))
    return static_inputs


def _evict_entry_if_needed(entries: dict, max_entries: int):
    if 0 < max_entries <= len(entries):
        oldest_key = next(iter(entries))
        del entries[oldest_key]


def _capture_single_graph(gm, static_inputs):
    """在 NPUGraph 上下文内捕获 ``gm(*static_inputs)``，返回 (aclgraph, output)。

    共享全局 graph pool；capture 期间禁用 GC 与 empty_cache 以保证地址稳定。
    """
    aclgraph = torch.npu.NPUGraph()
    pool = _get_global_graph_pool()
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patch_fn("gc.collect", lambda: None))
        if hasattr(torch.npu, "empty_cache"):
            stack.enter_context(_patch_fn("torch.npu.empty_cache", lambda: None))
        with torch.npu.graph(npu_graph=aclgraph, pool=pool):
            output = gm(*static_inputs)
    return aclgraph, output


def _validate_and_copy_inputs(entry: "_ACLGraphEntry", args):
    """D2 地址 debug 校验 + D1/C1 shape/dtype 校验与 data_ptr 跳过 + C3 异步 copy。

    在默认流回放前，将新输入拷入静态缓冲；data_ptr 相同则跳过。
    """
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

    # D1 + C1: shape/dtype validation and data_ptr skip
    needs_copy = []
    for i, (static_buf, new_inp) in enumerate(zip(entry.static_inputs, args)):
        if not isinstance(static_buf, torch.Tensor):
            # 捕获期落地的标量输入（如序列长度）已固化进 kernel 参数，回放时值变了
            # 必须显式报错，否则会静默复用捕获时的旧值。
            if isinstance(static_buf, (bool, int, float)):
                actual = int(new_inp) if isinstance(new_inp, torch.SymInt) else new_inp
                if actual != static_buf:
                    raise RuntimeError(
                        f"ACLGraph scalar input mismatch at position {i}: "
                        f"captured {static_buf!r}, got {actual!r}. "
                        "The captured graph embeds this value in its kernel arguments; "
                        "re-capture with the new value."
                    )
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

    # C3: issue async copy_ on a dedicated stream and record events
    # for device-side producer/copy ordering before replay.
    if needs_copy:
        _copy_inputs_on_stream(entry, needs_copy)


def _safe_output(out, is_last: bool):
    """按 safe_output_mode 决定是否 clone。

    中间段（is_last=False）绝不 clone——其输出作为下一段 static input，地址必须稳定。
    末段（is_last=True）按 safe_output_mode 决定。
    """
    from .compiliation_config import CompilationConfig

    if not is_last:
        return out
    if isinstance(out, torch.Tensor):
        return out.clone() if CompilationConfig.safe_output_mode else out
    if isinstance(out, (list, tuple)):
        if CompilationConfig.safe_output_mode:
            return type(out)(t.clone() if isinstance(t, torch.Tensor) else t for t in out)
        return out
    return out


# ---------------------------------------------------------------------------
# 分段缓存（Breakable Graph）
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SegmentCache:
    """分段捕获的生命周期缓存。

    Attributes:
        seg_entries: 每段按输入形状分桶的捕获条目，``seg_entries[seg_id][shape_key]``。
        inter_buffers: 段间衔接缓冲，``inter_buffers[seg_id]`` 持有该段 output 引用，
            作为下一段 static input（地址稳定，回放时原地更新）。
    """

    seg_entries: list[dict] = dataclasses.field(default_factory=list)
    inter_buffers: list = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# 单段后端工厂（行为与原实现一致，内部改用模块级辅助）
# ---------------------------------------------------------------------------


def create_aclgraph_backend():
    """Create an ACLGraph backend function for NPU.

    Returns a callable ``aclgraph_backend(gm, example_inputs) -> compiled_fn``
    suitable for wrapping a ``torch.compile``-processed graph module.

    Implements P0/P1/P3 optimizations:
      A1 - device-side stream ordering before replay
      B1 - global graph memory pool
      C1 - skip copy_ on same data_ptr
      C2 - safe_output_mode to control clone
      C3 - async copy_ on a dedicated stream with event pipeline
      C4 - GC disable during graph capture
      D1 - shape/dtype assert before copy_
      D2 - input address debug validation
      E1 - lazy capture + LRU eviction (aclgraph_lazy_capture / aclgraph_max_entries)
    """
    from .compiliation_config import CompilationConfig

    entries: dict[tuple, _ACLGraphEntry] = {}

    def _capture_graph(gm, inputs):
        input_shape = _get_input_shape(inputs)
        if input_shape not in entries:
            _evict_entry_if_needed(entries, CompilationConfig.aclgraph_max_entries)
            static_inputs = _make_static_inputs(inputs)
            input_addresses = [x.data_ptr() for x in static_inputs if isinstance(x, torch.Tensor)]
            aclgraph, output = _capture_single_graph(gm, static_inputs)
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
            _validate_and_copy_inputs(entry, args)
            entry.aclgraph.replay()
            return _safe_output(entry.output, is_last=True)

        return compiled_fn

    return aclgraph_backend


# ---------------------------------------------------------------------------
# 分段后端工厂（Breakable Graph）
# ---------------------------------------------------------------------------


def create_segmented_aclgraph_backend(gm, seg_meta, out_spec=None):
    """分段捕获后端：每个子图段独立捕获为 NPUGraph，段间通过 static buffer 衔接。

    Args:
        gm: 原始 GraphModule，用于建立 placeholder 名 -> forward 参数索引的映射。
        seg_meta: :func:`graph_splitter.split_graph_by_breakpoints` 产出的段元数据列表。
        out_spec: 整图返回值结构描述（``graph_splitter.OutputSpec``）。为 None 时退回
            “只返回末段输出”的旧行为；提供时按原 output 顺序与容器类型跨段组装返回，
            并对叶子的来源、None 与元素个数做校验。

    段间衔接：前段 output buffer 直接绑定到后段对应 placeholder，地址不变，回放时
    前段原地更新、后段读取最新值，天然命中免拷贝；仅原始输入（orig）需 copy_。
    中间段输出绝不 clone，仅最终返回给调用方的值按 ``safe_output_mode`` 处理。

    限制（首版）：按首次捕获的输入形状固定，不支持运行时形状变化触发分段重捕获
    （静态 shape 场景够用；动态 shape 变化会显式报错，不会静默错算）。捕获期固化进
    kernel 参数的标量输入同理：回放时按值校验，取值不同直接报错而不是静默复用。
    """

    num_segments = len(seg_meta)
    # placeholder 名 -> forward 参数索引
    ph_name_to_idx = {node.name: idx for idx, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder")}

    cache = SegmentCache(
        seg_entries=[{} for _ in range(num_segments)],
        inter_buffers=[None] * num_segments,
    )
    captured = [False]

    def _build_segment_static_inputs(seg_id, orig_args):
        """构造第 seg_id 段 capture 用的静态输入列表。"""
        statics = []
        for spec in seg_meta[seg_id].inputs:
            if spec[0] == "orig":
                idx = ph_name_to_idx.get(spec[1])
                if idx is None:
                    raise RuntimeError(f"breakpoint split: orig input '{spec[1]}' not found in graph placeholders")
                val = orig_args[idx]
                if isinstance(val, torch.Tensor):
                    buf = val.detach().clone()
                    if buf.device.type != "npu" and torch.npu.is_available():
                        buf = buf.npu()
                    statics.append(buf)
                else:
                    statics.append(_materialize_scalar_input(val, idx))
            elif spec[0] == "prev":
                _, src_seg, out_idx = spec
                # 交给 _segment_output 而不是按 isinstance(list/tuple) 拆包：单输出段若导出
                # 的节点本身返回 tuple（如 npu_rms_norm 的 (y, rstd)），整个返回值就是一个
                # 衔接值，拆包会把 tuple 的第一个张量错传给下一段。
                statics.append(_segment_output(src_seg, out_idx))
        return statics

    def _capture_all(orig_args):
        for seg_id in range(num_segments):
            statics = _build_segment_static_inputs(seg_id, orig_args)
            seg_gm = seg_meta[seg_id].gm
            aclgraph, output = _capture_single_graph(seg_gm, statics)
            cache.seg_entries[seg_id][True] = _ACLGraphEntry(
                aclgraph=aclgraph,
                static_inputs=statics,
                output=output,
                input_addresses=None,
                copy_stream=None,
            )
            cache.inter_buffers[seg_id] = output

    def _replay_all(orig_args):
        for seg_id in range(num_segments):
            entry = cache.seg_entries[seg_id][True]
            spec_list = seg_meta[seg_id].inputs
            # 仅 orig 输入需要 copy_；prev 输入绑定前段 output，地址不变，跳过。
            needs_copy = []
            for static_buf, spec in zip(entry.static_inputs, spec_list):
                if spec[0] != "orig":
                    continue
                idx = ph_name_to_idx.get(spec[1])
                if idx is None:
                    raise RuntimeError(f"breakpoint split: orig input '{spec[1]}' not found in graph placeholders")
                new_inp = orig_args[idx]
                if not isinstance(static_buf, torch.Tensor):
                    # 捕获期固化进 kernel 参数的标量（SymInt 物化后的 int，或本来就是
                    # int/bool/float/str 的图输入）：回放时值变了必须显式报错，否则会静默
                    # 复用捕获时的旧值——与单段路径 _validate_and_copy_inputs 的标量校验一致。
                    expected = static_buf
                    actual = _concrete_scalar_input(new_inp)
                    if isinstance(expected, (bool, int, float, str)):
                        # actual 是张量时不比较（张量布尔值会触发 ambiguous 报错），直接判为不匹配。
                        if isinstance(actual, torch.Tensor) or actual != expected:
                            raise RuntimeError(
                                f"segmented aclgraph scalar input mismatch at segment {seg_id}, "
                                f"input '{spec[1]}': captured {expected!r}, got {actual!r}. "
                                "The captured graph embeds this value in its kernel arguments; "
                                "re-capture with the new value."
                            )
                    continue
                if not isinstance(new_inp, torch.Tensor):
                    continue
                if static_buf.data_ptr() == new_inp.data_ptr():
                    continue
                if static_buf.shape != new_inp.shape or static_buf.dtype != new_inp.dtype:
                    raise RuntimeError(
                        f"segmented aclgraph input mismatch at segment {seg_id}, "
                        f"captured {tuple(static_buf.shape)}/{static_buf.dtype}, "
                        f"got {tuple(new_inp.shape)}/{new_inp.dtype}"
                    )
                needs_copy.append((static_buf, new_inp))

            # C3: issue async copy_ on a dedicated stream and record events
            # for device-side producer/copy ordering before replay.
            if needs_copy:
                _copy_inputs_on_stream(entry, needs_copy)

            entry.aclgraph.replay()

    def _segment_output(seg_id, out_idx):
        """取第 seg_id 段的第 out_idx 个输出（段内顺序由 graph_splitter 固定）。"""
        raw = cache.inter_buffers[seg_id]
        if raw is None:
            raise RuntimeError(
                f"breakpoint split: segment {seg_id} produced no outputs, but output #{out_idx} "
                "is requested from it. check the breakpoint placement."
            )
        count = seg_meta[seg_id].num_outputs
        if count == 1:
            if out_idx != 0:
                raise RuntimeError(
                    f"breakpoint split: segment {seg_id} has a single output, requested index {out_idx}."
                )
            return raw
        if not isinstance(raw, (list, tuple)):
            raise RuntimeError(
                f"breakpoint split: segment {seg_id} declares {count} outputs but produced {type(raw).__name__}."
            )
        if out_idx >= len(raw):
            raise RuntimeError(
                f"breakpoint split: segment {seg_id} produced {len(raw)} outputs, requested index {out_idx}."
            )
        return raw[out_idx]

    def _assemble_output(values):
        """按 OutputSpec.layout 还原原 output 的容器结构（tuple/list/dict 递归）。"""

        def _make(layout):
            kind = layout[0]
            if kind == "leaf":
                return values[layout[1]]
            if kind == "tuple":
                return tuple(_make(child) for child in layout[1])
            if kind == "list":
                return [_make(child) for child in layout[1]]
            if kind == "dict":
                return {key: _make(child) for key, child in layout[1].items()}
            return layout[1]

        return _make(out_spec.layout)

    def compiled_fn(*args):
        if not captured[0]:
            _capture_all(args)
            captured[0] = True
        _replay_all(args)
        if out_spec is None:
            # 兼容旧调用方：无返回值结构描述时退回“只返回末段输出”。
            return _safe_output(cache.inter_buffers[num_segments - 1], is_last=True)
        values = [_segment_output(seg_id, out_idx) for seg_id, out_idx in out_spec.leaves]
        if values and any(value is None for value in values):
            raise RuntimeError(
                "breakpoint split: graph output assembly produced None; "
                "check that the breakpoints do not separate an output node from its producer."
            )
        # 返回给调用方的值可能在下次 replay 时被原地覆盖（非末段叶子尤其如此），
        # 因此统一按末段语义过 safe_output_mode。
        return _assemble_output([_safe_output(value, is_last=True) for value in values])

    return compiled_fn


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
