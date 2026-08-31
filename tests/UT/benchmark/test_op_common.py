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

"""Unit tests for op_defs._common pure accounting helpers.

These functions (tensor_bytes / quant_flops / mx_scale_bytes /
attention_valid_parts) carry the MFU/MBU byte and FLOPs accounting used by all
four ops; they must not regress independently of the NPU runtime.
"""

import sys
import types

import pytest
from op_defs._common import (
    DTYPE_BYTES,
    MX_BLOCK_SIZE,
    QUANT_FLOPS_PER_ELEM,
    MfuMbuSummaryMixin,
    attention_valid_parts,
    mx_scale_bytes,
    quant_flops,
    tensor_bytes,
)


def _stub_xpu_perf(monkeypatch):
    """Minimal xpu_perf stub so op_defs modules import without the NPU runtime.

    op_defs/fa.py and op_defs/bsa.py import BasicOp / ProviderRegistry at
    module top level; the real xpu_perf package needs third-party deps
    (jsonlines etc.) that are not installed on dev machines. The stub is only
    used to reach the flops_calc accounting methods (no kernel execution).
    monkeypatch-scoped so sys.modules is restored after each test.
    """
    if "xpu_perf.micro_perf.core.op" in sys.modules:
        return

    class BasicOp:
        def __init__(self, args_dict, backend, *args, **kwargs):
            self.args_dict = args_dict
            self.backend = backend

        def summary(self, latency_us, kernel_mapping=None):
            return {}

    class ProviderRegistry:
        @staticmethod
        def register_base_impl(name, engine):
            return lambda cls: cls

    op_mod = types.ModuleType("xpu_perf.micro_perf.core.op")
    op_mod.BasicOp = BasicOp
    op_mod.ProviderRegistry = ProviderRegistry
    for mod_name, mod in (
        ("xpu_perf", types.ModuleType("xpu_perf")),
        ("xpu_perf.micro_perf", types.ModuleType("xpu_perf.micro_perf")),
        ("xpu_perf.micro_perf.core", types.ModuleType("xpu_perf.micro_perf.core")),
        ("xpu_perf.micro_perf.core.op", op_mod),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod)


@pytest.fixture(autouse=True)
def _xpu_perf_stub(monkeypatch):
    _stub_xpu_perf(monkeypatch)


# --- tensor_bytes -----------------------------------------------------------
@pytest.mark.parametrize(
    ("dtype", "per_elem", "has_scale"),
    [("bf16", 2.0, False), ("fp8", 1.0, False), ("mxfp8", 1.0, True), ("mxfp4", 0.5, True)],
)
def test_tensor_bytes(dtype, per_elem, has_scale):
    numel = 100
    expected = numel * per_elem
    if has_scale:
        expected += (numel + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE
    assert tensor_bytes(numel, dtype) == expected


def test_tensor_bytes_unknown_dtype():
    with pytest.raises(ValueError):
        tensor_bytes(10, "int8")


def test_dtype_bytes_match_rfc_convention():
    assert DTYPE_BYTES == {"bf16": 2.0, "fp8": 1.0, "mxfp8": 1.0, "mxfp4": 0.5}


# --- quant_flops ------------------------------------------------------------
def test_quant_flops_default_per_elem():
    assert quant_flops(10) == 10 * QUANT_FLOPS_PER_ELEM
    assert QUANT_FLOPS_PER_ELEM == 2.0


def test_quant_flops_zero():
    assert quant_flops(0) == 0.0


def test_quant_flops_negative_raises():
    with pytest.raises(ValueError):
        quant_flops(-1)


# --- mx_scale_bytes ---------------------------------------------------------
def test_mx_scale_bytes_block_boundary():
    assert mx_scale_bytes(32) == 1.0
    assert mx_scale_bytes(33) == 2.0
    assert mx_scale_bytes(64) == 2.0


def test_mx_scale_bytes_zero_elements():
    assert mx_scale_bytes(0) == 0.0


def test_mx_scale_bytes_bad_block_size():
    with pytest.raises(ValueError):
        mx_scale_bytes(10, block_size=0)


# --- attention_valid_parts --------------------------------------------------
def test_full_attention_no_sparsity():
    assert attention_valid_parts(100, 100, causal=False, sparsity=0.0) == 10_000


def test_sparsity_discount():
    assert attention_valid_parts(100, 100, causal=False, sparsity=0.8) == pytest.approx(2_000)


def test_causal_square_uses_lower_triangle():
    # q_len == kv_len: q*(q+1)/2 = 100*101/2 = 5050
    assert attention_valid_parts(100, 100, causal=True, sparsity=0.0) == 5050


def test_causal_rectangular_half():
    assert attention_valid_parts(100, 200, causal=True, sparsity=0.0) == 10_000


def test_causal_rectangular_with_sparsity():
    assert attention_valid_parts(100, 200, causal=True, sparsity=0.5) == 5_000


# --- flops_calc accounting (batch/quant scaling) ----------------------------
def _fa_op(batch_size=1, q_len=1024, kv_len=1024, causal=False, sparsity=0.0, dtype="bf16"):
    from op_defs.fa import FlashAttentionOp

    op = object.__new__(FlashAttentionOp)
    op.batch_size = batch_size
    op.num_heads = 8
    op.head_dim = 64
    op.q_len = q_len
    op.kv_len = kv_len
    op.causal = causal
    op.sparsity = sparsity
    op.dtype = dtype
    return op


def test_fa_flops_calc_batch_and_sparsity():
    # dense, batch=1: 4*B*H*D*S*S = 4*1*8*64*1024*1024
    op = _fa_op()
    op.flops_calc()
    assert op.calc_flops == 4 * 8 * 64 * 1024 * 1024
    # batch=2 must double the FLOPs (bytes/quant accounting already scale by batch)
    op = _fa_op(batch_size=2)
    op.flops_calc()
    assert op.calc_flops == 4 * 2 * 8 * 64 * 1024 * 1024
    # sparsity discounts the kept fraction only
    op = _fa_op(sparsity=0.8)
    op.flops_calc()
    assert op.calc_flops == pytest.approx(4 * 8 * 64 * 1024 * 1024 * 0.2)


def test_fa_flops_calc_quant_adds_elementwise():
    op = _fa_op(dtype="mxfp8")
    op.flops_calc()
    dense = 4 * 8 * 64 * 1024 * 1024
    quant_elems = 8 * 1024 * 64 + 2 * (8 * 1024 * 64)  # q + 2*kv
    assert op.calc_flops == pytest.approx(dense + quant_elems * 2.0)


def test_bsa_flops_calc_scales_with_batch():
    from op_defs.bsa import BlockSparseAttentionOp

    op = object.__new__(BlockSparseAttentionOp)
    op.batch_size = 4
    op.num_heads = 8
    op.head_dim = 64
    op.q_len = 1024
    op.kv_len = 1024
    op.causal = False
    op.sparsity = 0.75
    op.mask_type = "rf_v3"
    op.flops_calc()
    assert op.calc_flops == pytest.approx(4 * 4 * 8 * 64 * 1024 * 1024 * 0.25)


def test_bsa_flops_calc_ada_bsa_not_discounted():
    # ada_bsa receives a dense mask in the vendor impl -> FLOPs stay full.
    from op_defs.bsa import BlockSparseAttentionOp

    op = object.__new__(BlockSparseAttentionOp)
    op.batch_size = 1
    op.num_heads = 8
    op.head_dim = 64
    op.q_len = 1024
    op.kv_len = 1024
    op.causal = False
    op.sparsity = 0.9
    op.mask_type = "ada_bsa"
    op.flops_calc()
    assert op.calc_flops == pytest.approx(4 * 8 * 64 * 1024 * 1024)


def test_bsa_rejects_non_bf16_dtype():
    # vendor kernel is bf16-only; quantized dtypes must fail loudly instead of
    # measuring bf16 while accounting at the quantized rate.
    from op_defs.bsa import BlockSparseAttentionOp

    op = object.__new__(BlockSparseAttentionOp)
    op.num_heads = 8
    op.head_dim = 64
    op.q_len = 1024
    op.kv_len = 1024
    op.sparsity = 0.8
    op.dtype = "mxfp8"
    op.args_dict = {}
    with pytest.raises(ValueError):
        op._validate_args()


def _gmm_op(quant_algo="NO_QUANT", num_tokens=1024, hidden_size=1536, moe_inter=3200):
    from op_defs.gmm import GroupedMatMulOp

    op = object.__new__(GroupedMatMulOp)
    op.num_tokens = num_tokens
    op.hidden_size = hidden_size
    op.moe_inter = moe_inter
    op.experts = 128
    op.top_k = 16
    op.quant_algo = quant_algo
    return op


def test_gmm_flops_calc_dense_full_shape():
    # gate_up [M,C]@[C,2*inter] (4*M*C*inter) + w2 [M,inter]@[inter,C] (2*M*C*inter)
    op = _gmm_op()
    op.flops_calc()
    assert op.calc_flops == 6 * 1024 * 1536 * 3200


def test_gmm_flops_calc_quant_adds_elementwise():
    op = _gmm_op(quant_algo="W8A8_DYNAMIC")
    op.flops_calc()
    dense = 6 * 1024 * 1536 * 3200
    elems = 1024 * 1536 + 2 * 3200 * 1536 + 1536 * 3200  # x + w13 + w2
    assert op.calc_flops == pytest.approx(dense + elems * 2.0)


# --- MfuMbuSummaryMixin -----------------------------------------------------
class _FakeBackend:
    peak_flops = 560.0
    peak_bw = 1275.0


class _FakeBaseOp:
    def summary(self, latency_us, kernel_mapping=None):
        return {
            "latency(us)": latency_us,
            "calc_flops_power(tflops)": 280.0,
            "mem_bw(GB/s)": 637.5,
        }


class _FakeOp(MfuMbuSummaryMixin, _FakeBaseOp):
    def __init__(self):
        self.backend = _FakeBackend()
        self.args_dict = {"peak_flops": 560.0, "peak_bw": 1275.0}


def test_mixin_injects_mfu_mbu():
    out = _FakeOp().summary(100.0)
    assert out["MFU"] == 0.5
    assert out["MBU"] == 0.5


def test_mixin_mfu_none_without_peaks():
    op = _FakeOp()
    op.args_dict = {}
    out = op.summary(100.0)
    assert "MFU" not in out
    assert "MBU" not in out


def test_mixin_omits_executed_path_when_unset():
    out = _FakeOp().summary(100.0)
    assert "executed_path" not in out


def test_mixin_injects_executed_path_when_set():
    op = _FakeOp()
    op.executed_path = "bf16_fallback"
    out = op.summary(100.0)
    assert out["executed_path"] == "bf16_fallback"


def test_mixin_injects_executed_path_real_quant():
    op = _FakeOp()
    op.executed_path = "mxfp8"
    out = op.summary(100.0)
    assert out["executed_path"] == "mxfp8"
