# pylint: disable=R0401
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Public distributed import and sparse-attention compatibility contracts."""

# pylint: disable=cyclic-import,no-name-in-module,redefined-outer-name

import ast
from typing import Optional

import pytest

torch = pytest.importorskip("torch")

from cpu_support import REPO_ROOT, load_functions, load_module


def call(producer, state):
    shape = (1, 68, 4, 2)
    query = (torch.arange(torch.tensor(shape).prod()).reshape(shape) / 64).to(torch.bfloat16)
    return producer.entry(
        query,
        query * 2,
        query * 0.5,
        [{"start": 4, "latent_shape": [1, 8, 16]}],
        chunking=2,
        group=producer.group,
        used_len=132,
        state=state,
        return_stats=True,
    )


@pytest.fixture
def public_sparse():
    calls = []

    def backend(q, k, v, **kwargs):
        calls.append(kwargs)
        return q, None

    namespace = {
        "torch": torch,
        "Optional": Optional,
        "ParametersInvalid": ValueError,
        "check_params": lambda *args: None,
        "_resolve_sparse_type_for_a5": lambda sparse_type, inner: (
            "rf_v3" if sparse_type == "rf_v2" else sparse_type,
            inner,
        ),
        "bsa_sparse_attention_v3": backend,
    }
    function = load_functions(
        REPO_ROOT / "mindiesd/layers/flash_attn/sparse_flash_attn.py",
        {"sparse_attention"},
        namespace,
    )["sparse_attention"]
    return function, calls


def test_sparse_public_options_validate_and_forward_without_renaming(public_sparse):
    sparse, calls = public_sparse
    query = torch.ones(1, 4, 1, 2, dtype=torch.bfloat16)
    spans = [{"start": 0, "latent_shape": [1, 2, 2]}]
    assert (
        sparse(
            query,
            query,
            query,
            sparse_type="rf_v3",
            precision="bf16",
            video_spans=spans,
            bsa_backend="eagle",
            pre_rearranged=True,
        )
        is query
    )
    assert calls[-1]["bsa_backend"] == "eagle" and calls[-1]["pre_rearranged"] is True
    sparse(
        query,
        query,
        query,
        sparse_type="rf_v3",
        precision="bf16",
        video_spans=spans,
        exec_kernel="eagle_bsa",
    )
    assert calls[-1]["bsa_backend"] == "eagle"
    sparse(query, query, query, sparse_type="rf_v2", precision="bf16", video_spans=spans, bsa_backend="eagle")
    assert calls[-1]["bsa_backend"] == "eagle"
    for options, match in (
        ({"bsa_backend": "bad"}, "bsa_backend"),
        ({"exec_kernel": "bad"}, "exec_kernel"),
        ({"exec_kernel": "eagle_bsa", "bsa_backend": "cann"}, "conflicts"),
        ({"bsa_backend": "eagle", "precision": "mix"}, "requires precision"),
        ({"pre_rearranged": 1}, "must be a bool"),
        ({"pre_rearranged": True}, "requires video_spans"),
    ):
        with pytest.raises(ValueError, match=match):
            sparse(query, query, query, **{"sparse_type": "rf_v3", "precision": "bf16", **options})


def test_parallel_public_package_and_root_exports_are_explicit(producer, monkeypatch):
    public = load_module(
        producer.package.__name__,
        REPO_ROOT / "mindiesd/parallel/__init__.py",
        monkeypatch,
    )
    assert public.__all__ == [
        "distributed_sparse_attention",
        "ParallelState",
        "QKVLayoutPlan",
        "plan_qkv_layout",
    ]
    assert public.distributed_sparse_attention is producer.entry
    assert public.ParallelState is producer.modules["state"].ParallelState
    root = ast.parse((REPO_ROOT / "mindiesd/__init__.py").read_text(encoding="utf-8"))
    exported = next(
        node.value
        for node in root.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if getattr(target, "id", None) == "__all__"
    )
    names = {element.value for element in exported.elts}
    assert {"RFMultiSpanPlan", "build_rf_attention_plan", "rf_multi_span_permutation"} <= names


def test_requested_head_major_mixed_fallback_is_reported(producer):
    state = producer.modules["state"].ParallelState(group=producer.group, head_major_qkv=True)
    _output, stats = call(producer, state)
    assert stats["tally"]["consumer"] == "quantized"
    assert state.last_call_stats["actual"]["head_major_qkv"] is False
    assert state.last_call_stats["actual"]["qkv_layout_fallback_reason"] == "attention_mode_unsupported"
