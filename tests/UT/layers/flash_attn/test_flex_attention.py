#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Host-side CATLASS FlexAttention contracts."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

flex = importlib.import_module("mindiesd.layers.flash_attn.flex_attention")


def test_910_is_rejected_before_importing_dsl(monkeypatch):
    name = "Ascend910B"
    monkeypatch.setattr(flex, "is_a5_device", lambda: False)
    monkeypatch.setattr(flex.torch.npu, "get_device_name", lambda: name, raising=False)
    monkeypatch.setattr(flex.importlib, "import_module", lambda name: pytest.fail(f"unexpected import {name}"))

    with pytest.raises(NotImplementedError, match=f"Ascend 950.*{name}"):
        flex._load_dsl(SimpleNamespace(type="npu"))


@pytest.mark.parametrize("dependency", ["catlass", "unexpected_transitive_dependency"])
def test_dsl_import_distinguishes_missing_dependencies(monkeypatch, dependency):
    monkeypatch.setattr(flex, "is_a5_device", lambda: True)
    missing = ModuleNotFoundError(name=dependency)
    monkeypatch.setattr(flex.importlib, "import_module", Mock(side_effect=missing))

    with pytest.raises(ImportError) as error:
        flex._load_dsl(SimpleNamespace(type="npu"))
    if dependency == "catlass":
        assert "requires ascend-catlass-dsl" in str(error.value)
        assert error.value.__cause__ is missing
    else:
        assert error.value is missing


def test_dsl_adapter_passes_block_mask_to_backend(monkeypatch):
    class Index(int):
        def to(self, dtype):
            return dtype(self)

    class Mod:
        def __init__(self, callback, **options):
            self.callback = callback
            self.options = options

    qkv = torch.zeros(1, 3, 1, 128, dtype=torch.bfloat16)
    lse = torch.zeros(1, 1, 3)
    sparsity, attention = Mock(return_value=object()), Mock(return_value=(qkv, lse))
    tla = SimpleNamespace(jit=lambda fn: fn, const_expr=bool, Int32=int, Int64=int, Bool=bool)
    monkeypatch.setitem(sys.modules, "catlass", SimpleNamespace(tla=tla))
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_npu_4.catlass_dsl",
        SimpleNamespace(compute_block_sparsity=sparsity, flash_attn_func=attention),
    )
    monkeypatch.setitem(sys.modules, "flash_attn_npu_4.catlass_dsl.mods", SimpleNamespace(MaskMod=Mod, Mod=Mod))
    # Isolate the real adapter module so these stand-ins never leak into a later import.
    spec = importlib.util.spec_from_file_location(
        "_flex_dsl_ut", Path(flex.__file__).with_name("flex_attention_dsl.py")
    )
    dsl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dsl)

    mask = flex.make_block_causal_mask([(0, 2)], 3, device="cpu")
    prepared = dsl.prepare(mask, torch.arange(3), torch.arange(3), torch.tensor([0]), 1)
    assert sparsity.call_args.args[:6] == (128, 128, None, None, 3, 3)
    mod = prepared.mask_mod
    assert mod.options == {"broadcast_batch": True, "broadcast_head": True}
    aux = ([Index(0), Index(-1)], [Index(1), Index(2)], *prepared.aux_tensors[2:])
    assert mod.callback(1, 0, 0, 0, None, aux, ())
    assert not mod.callback(1, 0, 0, 1, None, aux, ())
    assert not mod.callback(1, 0, 1, 0, None, aux, ())
    output, actual_lse = dsl.attention(qkv, qkv, qkv, prepared, 0.125, True)
    assert output is qkv and actual_lse is lse
    assert attention.call_args.kwargs["mask_mod"] is mod
    assert attention.call_args.kwargs["block_sparse_tensors"] is sparsity.return_value


def test_block_causal_mask_retains_actual_sequence_length():
    valid = torch.ones((1, 133), dtype=torch.bool)
    valid[0, 1] = False
    mask = flex.make_block_causal_mask([(3, 16), (129, 133)], 133, device="cpu", valid_tokens=valid)
    block_ids, keep = mask.aux_tensors
    assert block_ids.shape == (133,)
    assert block_ids[0] == -1
    assert block_ids[3] == 0 and block_ids[132] == 1
    assert keep.shape == (1, 133)
    assert not keep[0, 1]
    assert mask.head_shared


@pytest.mark.parametrize("spans", [[(2, 5), (4, 6)], [(0, 134)], [(False, 3)]])
def test_invalid_spans_are_rejected(spans):
    with pytest.raises(ValueError, match="span"):
        flex.make_block_causal_mask(spans, 133, device="cpu")


@pytest.mark.parametrize("seq_len", [True, 0])
def test_invalid_sequence_lengths_are_rejected(seq_len):
    with pytest.raises(ValueError, match="positive integer"):
        flex.make_block_causal_mask([], seq_len, device="cpu")


def test_cpu_device_is_rejected_before_distributed_initialization():
    with pytest.raises(ValueError, match="requires NPU tensors"):
        flex.check_flex_attention_device(torch.device("cpu"))


@pytest.mark.parametrize(
    "case, message",
    [
        ("layout", "layout"),
        ("rank", "four-dimensional"),
        ("empty", "non-empty"),
        ("dtype", "same device and dtype"),
        ("precision", "FP16 and BF16"),
        ("grad", "inference-only"),
    ],
)
def test_local_tensor_validation_rejects_invalid_contracts(case, message):
    q = torch.zeros(1, 3, 2, 128, dtype=torch.bfloat16)
    k, v, layout = q, q, "BSND"
    if case == "layout":
        layout = "SBND"
    elif case == "rank":
        k = object()
    elif case == "empty":
        k = q[:, :0]
    elif case == "dtype":
        k = q.float()
    elif case == "precision":
        q = k = v = q.float()
    with torch.set_grad_enabled(case == "grad"), pytest.raises(ValueError, match=message):
        flex._validate_inputs(q, k, v, layout)


@pytest.mark.parametrize("metadata", [object(), torch.ones(3), torch.ones(3, dtype=torch.int64, device="meta")])
def test_position_metadata_rejects_non_tensor_dtype_or_device(metadata):
    with pytest.raises(ValueError, match="positions must be"):
        flex._validate_map("positions", metadata, 3, torch.device("cpu"))
