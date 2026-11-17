# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Load real parallel modules and stock functions without importing the NPU package."""

import ast
import importlib.util
import math
import sys
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[3]
PARALLEL_DIR = REPO_ROOT / "mindiesd/parallel"
RF_SOURCE = REPO_ROOT / "mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py"


def new_package(name, monkeypatch=None, directory=PARALLEL_DIR):
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    register_module(name, package, monkeypatch)
    return package


def register_module(name, module, monkeypatch=None):
    if monkeypatch is None:
        sys.modules[name] = module
    else:
        monkeypatch.setitem(sys.modules, name, module)


def load_parallel_module(package, relative, monkeypatch=None):
    name = package.__name__ + "." + relative
    return load_module(name, PARALLEL_DIR / (relative + ".py"), monkeypatch, package, relative)


def load_module(name, path, monkeypatch, package=None, relative=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    register_module(name, module, monkeypatch)
    if package is not None:
        setattr(package, relative, module)
    spec.loader.exec_module(module)
    return module


def load_functions(source, names, namespace=None):
    """Execute selected original AST nodes in a fresh caller-owned namespace."""
    namespace = {"torch": torch} if namespace is None else namespace
    nodes = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names), "missing source function"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)  # pylint: disable=exec-used
    return namespace


def native_avgpool():
    return load_functions(RF_SOURCE, {"avgpool"})["avgpool"]


def native_rf_functions():
    names = {
        "avgpool",
        "rearrange_with_remaining",
        "_sequence_length",
        "_index_sequence",
        "_normalize_video_spans",
        "_normalize_image_spans",
        "_span_rearranged_indices",
        "_multi_span_permutation",
        "do_multi_span_tensor_rearrange_pooling",
        "get_multi_span_blockwise_mask",
        "get_mask_index",
    }
    return load_functions(
        RF_SOURCE,
        names,
        dict(
            torch=torch,
            math=math,
            Optional=Optional,
            Mapping=Mapping,
            Sequence=Sequence,
            rearrange=rearrange,
            ParametersInvalid=ValueError,
        ),
    )


class DeviceTensor(torch.Tensor):
    """CPU storage presented to the orchestration layer as NPU storage."""

    @staticmethod
    def wrap(tensor):
        return tensor.as_subclass(DeviceTensor)

    @property
    def device(self):
        return types.SimpleNamespace(type="npu", index=0)

    def record_stream(self, s):
        return None


class _TorchProxy:
    Tensor = torch.Tensor

    def __init__(self, npu):
        self.npu = npu

    def __getattr__(self, name):
        function = getattr(torch, name)
        if name not in {"tensor", "zeros", "empty"}:
            return function

        def allocate(*args, **kwargs):
            if isinstance(kwargs.get("device"), types.SimpleNamespace):
                kwargs["device"] = "cpu"
            return DeviceTensor.wrap(function(*args, **kwargs))

        return allocate


class NpuSimulation:
    """One stream/event model shared by overlap and head-major CPU tests."""

    def __init__(self):
        self.timeline, self.current, self.streams, self.events = [], [], [], []
        self.current.append(self.stream("compute0"))
        self.npu = types.SimpleNamespace(
            Stream=self._create_stream,
            current_stream=lambda device: self.current[0],
            stream=self._context,
            Event=self._event,
        )

    def stream(self, name):
        value = types.SimpleNamespace(name=name)
        value.wait_event = lambda event: self.timeline.append(("wait", name, event.name))
        return value

    def _create_stream(self, **kwargs):
        value = self.stream("comm")
        self.streams.append(value)
        return value

    @contextmanager
    def _context(self, selected):
        previous, self.current[0] = self.current[0], selected
        try:
            yield
        finally:
            self.current[0] = previous

    def _event(self):
        value = types.SimpleNamespace(name="event%d" % len(self.events))
        value.record = lambda selected: self.timeline.append(("record", selected.name, value.name))
        self.events.append(value)
        return value

    def install(self, pipeline, monkeypatch, *, allocation_proxy=False, trace_collectives=False):
        monkeypatch.setattr(pipeline, "torch", _TorchProxy(self.npu) if allocation_proxy else torch)
        monkeypatch.setattr(torch, "npu", self.npu, raising=False)
        monkeypatch.setattr(pipeline, "_record_stream", lambda tensors, selected: None, raising=False)
        if trace_collectives:
            original = pipeline.dist.all_to_all_single

            def exchange(recv, send, **kwargs):
                self.timeline.append(("a2a", self.current[0].name, send.dtype))
                return original(recv, send, **kwargs)

            monkeypatch.setattr(pipeline.dist, "all_to_all_single", exchange)
        return self
