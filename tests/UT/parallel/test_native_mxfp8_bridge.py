# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from types import SimpleNamespace
import pytest
import torch
from cpu_support import REPO_ROOT, load_functions


def test_native_mxfp8_single_head_pages_have_kernel_strides():
    functions = load_functions(
        REPO_ROOT / "mindiesd/parallel/_native_mxfp8.py",
        {"paged_kv"},
        {"torch": torch, "BLOCK": 128},
    )
    key, scale = torch.ones(1, 131, 1, 132, dtype=torch.uint8).split((128, 4), dim=-1)
    for tensor in (key, scale):
        pages = functions["paged_kv"](tensor, length=131)
        width = tensor.shape[-1]
        assert pages.shape == (2, 1, 128, width)
        assert pages.stride() == (128 * width, 128 * width, width, 1)
        rows = pages.reshape(256, width)
        assert torch.equal(rows[:131], tensor.view(131, width))
        assert not rows[131:].any()


@pytest.mark.parametrize("q_len,kv_len", [(3, 7), (128, 128), (131, 257)])
def test_native_mxfp8_adapter_passes_e4m3_e8m0_without_requantizing(monkeypatch, q_len, kv_len):
    calls = []
    metadata_options = []
    q_blocks, kv_blocks = (q_len + 127) // 128, (kv_len + 127) // 128

    def metadata(counts, heads, key_heads, head_dim, **options):
        assert (heads, key_heads, head_dim) == (1, 1, 128)
        assert options["quant_mode"] == 2 and options["layout_q"] == "TND"
        assert options["layout_kv"] == "PA_BNBD"
        assert options["cu_seqlens_q"].dtype == torch.int32
        assert options["cu_seqlens_q"].tolist() == [0, q_len]
        assert options["seqused_kv"].tolist() == [kv_len]
        assert options["seqused_q"] is None and options["cu_seqlens_kv"] is None
        assert torch.equal(counts, torch.full((1, 1, q_blocks), kv_blocks, dtype=torch.int32))
        metadata_options.append(options)
        return "metadata"

    def attention(q, k, v, qs, ks, vs, *args, **options):
        calls.append((q, k, v, qs, ks, vs))
        assert options["metadata"] == "metadata"
        assert options["layout_q"] == options["layout_out"] == "TND"
        assert q.shape == (q_len, 1, 128) and qs.shape == (q_len, 1, 2, 2)
        assert options["cu_seqlens_q"] is metadata_options[0]["cu_seqlens_q"]
        assert options["seqused_kv"] is metadata_options[0]["seqused_kv"]
        return torch.arange(q.numel(), dtype=torch.float32).reshape(q.shape).bfloat16(), None

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_quant_block_sparse_attn_metadata=metadata,
            npu_quant_block_sparse_attn=attention,
        ),
    )
    functions = load_functions(
        REPO_ROOT / "mindiesd/parallel/_native_mxfp8.py",
        {"e8m0_scales", "indexed_mask", "prequantized_mxfp8_bsa"},
        {"torch": torch, "BLOCK": 128},
    )
    q = torch.arange(q_len * 128).remainder(127).byte().reshape(1, q_len, 1, 128)
    q = q.view(torch.float8_e4m3fn)
    k = torch.empty(kv_blocks, 1, 128, 128, dtype=q.dtype)
    v = torch.empty_like(k)
    shapes = ((1, q_len, 1, 2, 2), (kv_blocks, 1, 128, 2, 2), (kv_blocks, 1, 2, 128, 2))
    scales = [torch.ones(shape, dtype=torch.uint8) for shape in shapes]
    output = functions["prequantized_mxfp8_bsa"](
        query=q,
        key=k,
        value=v,
        query_scale=scales[0],
        key_scale=scales[1],
        value_scale=scales[2],
        block_sparse_mask=torch.ones(1, 1, q_blocks, kv_blocks, dtype=torch.int8),
        scale_value=0.125,
        actual_seq_lengths=[q_len],
        actual_seq_lengths_kv=[kv_len],
    )
    assert output.shape == q.shape and output.dtype == torch.bfloat16
    assert torch.equal(output, torch.arange(q.numel(), dtype=torch.float32).reshape(q.shape).bfloat16())
    assert len(calls) == 1
    assert calls[0][0].dtype == q.dtype and calls[0][0].data_ptr() == q.data_ptr()
    assert torch.equal(calls[0][0].view(torch.uint8), q.squeeze(0).view(torch.uint8))
    for actual, expected in zip(calls[0][1:3], (k, v)):
        assert actual is expected
    for actual, expected in zip(calls[0][3:], (scales[0].squeeze(0), *scales[1:])):
        assert actual.dtype == torch.float8_e8m0fnu
        assert torch.equal(actual.view(torch.uint8), expected)


@pytest.mark.parametrize(
    "version,ops,valid",
    [
        ("9.2.0b2", True, True),
        ("9.1.0", True, False),
        (None, True, False),
        ("bad", True, False),
        ("9.2.0b2", False, False),
    ],
)
def test_native_runtime_contract(monkeypatch, version, ops, valid):
    from packaging.version import InvalidVersion, Version

    extensions = {name: object() for name in ("npu_quant_block_sparse_attn", "npu_quant_block_sparse_attn_metadata")}
    monkeypatch.setattr(torch.ops, "custom", SimpleNamespace(**(extensions if ops else {})))
    require = load_functions(
        REPO_ROOT / "mindiesd/parallel/_native_mxfp8.py",
        {"require_native_mxfp8_runtime"},
        dict(torch=torch, Version=Version, InvalidVersion=InvalidVersion, get_cann_version=lambda: version),
    )["require_native_mxfp8_runtime"]
    if valid:
        require()
    else:
        with pytest.raises(RuntimeError, match="requires"):
            require()


def test_bf16_native_consumer_quantizes_qk_d32_v_s32():
    calls = []

    def quant(tensor, **kw):
        calls.append((tensor.shape, kw))
        return tensor, torch.ones((*tensor.shape, 1, 1), dtype=torch.uint8)

    def consume(**kw):
        assert all(kw[name].dtype == torch.float8_e8m0fnu for name in ("query_scale", "key_scale", "value_scale"))
        return kw["query"]

    funcs = load_functions(
        REPO_ROOT / "mindiesd/parallel/_native_mxfp8.py",
        {"paged_kv", "e8m0_scales", "quantize_mx", "bf16_mxfp8_bsa"},
        dict(
            torch=torch,
            BLOCK=128,
            torch_npu=SimpleNamespace(npu_dynamic_mx_quant=quant),
            prequantized_mxfp8_bsa=consume,
        ),
    )
    q = torch.ones(1, 131, 2, 128, dtype=torch.bfloat16)
    options = dict(
        actual_seq_lengths=[131],
        actual_seq_lengths_kv=[131],
        block_sparse_mask=torch.ones(1, 2, 2, 2),
        scale_value=0.125,
    )
    assert funcs["bf16_mxfp8_bsa"]((q, q, q), options) is q
    assert [c[1]["axis"] for c in calls] == [-1, -1, 2]
    assert calls[1][0] == calls[2][0] == (2, 2, 128, 128)
    with pytest.raises(TypeError, match="E8M0"):
        funcs["e8m0_scales"](torch.ones(1))


@pytest.mark.parametrize("empty,optimized", [(False, False), (False, True), (True, True)])
def test_native_wire_padding_scales_and_consumer_layout(producer, monkeypatch, empty, optimized):
    import importlib
    import sys
    from types import ModuleType

    native = ModuleType(producer.package.__name__ + "._native_mxfp8")

    def quant(value, *, axis):
        scale_shape = (*value.shape[:-1], 1, 1) if value.shape[-1] == 1 else (*value.shape[:-1], 2, 2)
        return value.to(torch.uint8), torch.ones(scale_shape, dtype=torch.uint8)

    native.quantize_mx = quant
    native.BLOCK = 128
    native.paged_kv = load_functions(
        REPO_ROOT / "mindiesd/parallel/_native_mxfp8.py", {"paged_kv"}, dict(torch=torch, BLOCK=128)
    )["paged_kv"]

    def consume(**kw):
        assert kw["query"].shape == (1, 7, 1, 128)
        assert kw["key_scale"].shape == (1, 1, 128, 2, 2)
        assert kw["value_scale"].shape == (1, 1, 2, 128, 2)
        return kw["query"].view(torch.uint8).bfloat16()

    native.prequantized_mxfp8_bsa = consume
    monkeypatch.setitem(sys.modules, producer.package.__name__ + "._native_mxfp8", native)
    wire = importlib.import_module(producer.package.__name__ + ".quantized_attention_wire")
    mapping = SimpleNamespace(world=2, rank=1 if empty else 0, used_len=7, token_to_reordered_position=torch.arange(7))
    chain = SimpleNamespace(
        reorder_mapping=mapping,
        precision="mxfp8",
        optimize_attention_pipeline=optimized,
        pre_rearranged=optimized,
        group=producer.group,
    )
    q = torch.ones(1, 8, 4, 128, dtype=torch.bfloat16)
    counts = {"allreduce": 0}
    packets, pools, scales = wire.prepare_quantized_wire(chain, q, q, q, kv_group=None, phase=0, counts=counts)
    assert counts["allreduce"] == 2 and packets[0].shape == (1, 8, 4, 132)
    assert not packets[0][:, 7:].any()
    if empty:
        assert not packets[0].any() and not pools[0].any()
    tensors = tuple(t[:, :7, :1].contiguous() for t in packets)
    out = wire.run_quantized_bsa(
        "mxfp8",
        tensors,
        scales,
        0,
        dict(start=0, end=7),
        dict(
            actual_seq_lengths=[7],
            actual_seq_lengths_kv=[7],
            block_sparse_mask=torch.ones(1, 1, 1, 1),
            scale_value=0.125,
        ),
    )
    assert torch.equal(out, tensors[0][..., :128].bfloat16())
