# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU contracts for native BF16 preparation, RF plans and operator adapters."""

# pylint: disable=redefined-outer-name,no-member,no-name-in-module

import sys
import types
import math
import inspect

import pytest

torch = pytest.importorskip("torch")

from cpu_support import PARALLEL_DIR, REPO_ROOT, load_functions, load_module, native_rf_functions, new_package


@pytest.fixture
def facade(monkeypatch):
    package = new_package("_mindiesd_fused_facade", monkeypatch, REPO_ROOT / "mindiesd")
    return load_module(package.__name__ + ".fused_ops", REPO_ROOT / "mindiesd/fused_ops.py", monkeypatch)


@pytest.fixture
def rf_plan(monkeypatch):
    package = new_package("_mindiesd_rf_plan", monkeypatch, REPO_ROOT / "mindiesd")
    layers = new_package(package.__name__ + ".layers", monkeypatch, REPO_ROOT / "mindiesd/layers")
    flash = new_package(layers.__name__ + ".flash_attn", monkeypatch, REPO_ROOT / "mindiesd/layers/flash_attn")
    stock = native_rf_functions()
    rf_v2 = types.ModuleType(flash.__name__ + ".sparse_flash_attn_rf_v2")
    rf_v2._normalize_video_spans = stock["_normalize_video_spans"]
    rf_v2._multi_span_permutation = stock["_multi_span_permutation"]
    monkeypatch.setitem(sys.modules, rf_v2.__name__, rf_v2)
    return load_module(package.__name__ + ".rf_attention", REPO_ROOT / "mindiesd/rf_attention.py", monkeypatch)


@pytest.fixture
def layer_boundary(monkeypatch):
    sparse = types.ModuleType("mindiesd.layers.flash_attn.sparse_flash_attn")
    sparse.A5_V3_INNER_PRECISE = 4
    rf_v2 = types.ModuleType("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2")
    rf_v2.avgpool = lambda value, *args: value
    rf_v2.pool_qk_separately = lambda q, k, *args: torch.cat((q, k))
    rf_v2.do_multi_span_tensor_rearrange_pooling = lambda *args: args
    rf_v2.get_multi_span_blockwise_mask = lambda *args, **kwargs: None
    rf_v2.is_a5_device = lambda: True
    rf_v2.rain_fusion_attention = lambda q, *args, **kwargs: q
    rf_v3 = types.ModuleType("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3")
    calls = []

    def attention_v3(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q

    rf_v3.rain_fusion_attention_v3 = attention_v3
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.float8_e4m3fn = torch.float8_e4m3fn
    for module in (sparse, rf_v2, rf_v3, torch_npu):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    boundary = load_module("_parallel_layers_boundary", PARALLEL_DIR / "_layers.py", monkeypatch)
    return types.SimpleNamespace(module=boundary, calls=calls)


def test_fused_facade_loads_only_requested_native_symbol(facade, monkeypatch):
    layers = new_package(facade.__package__ + ".layers", monkeypatch, REPO_ROOT / "mindiesd/layers")
    custom = types.ModuleType(layers.__name__ + "._custom_ops")
    indexed = types.ModuleType(layers.__name__ + ".index_modulation")
    sentinel = object()
    custom.eagle_split_norm_rope_concat = lambda **kwargs: (sentinel, kwargs)
    custom.eagle_indexed_gate_norm_modulate = lambda **kwargs: (sentinel, kwargs)
    custom.eagle_quant_batch_matmul_swiglu = lambda **kwargs: (sentinel, kwargs)
    indexed.rms_norm_indexed_scale_shift = lambda **kwargs: (sentinel, kwargs)
    indexed.indexed_gate = lambda **kwargs: (sentinel, kwargs)
    monkeypatch.setitem(sys.modules, custom.__name__, custom)
    monkeypatch.setitem(sys.modules, indexed.__name__, indexed)
    for name in (
        "eagle_split_norm_rope_concat",
        "eagle_indexed_gate_norm_modulate",
        "eagle_quant_batch_matmul_swiglu",
        "rms_norm_indexed_scale_shift",
        "indexed_gate",
    ):
        function = getattr(facade, name)
        arguments = {
            key: sentinel
            for key, param in inspect.signature(function).parameters.items()
            if param.default is inspect.Parameter.empty
        }
        result, kwargs = function(**arguments)
        aliases = {
            "eagle_split_norm_rope_concat": facade._QKV_KEYWORDS,
            "eagle_indexed_gate_norm_modulate": facade._MODULATION_KEYWORDS,
            "eagle_quant_batch_matmul_swiglu": facade._SWIGLU_KEYWORDS,
        }.get(name, {})
        assert result is sentinel
        assert all(kwargs[aliases.get(key, key)] is sentinel for key in arguments)
        assert all(
            kwargs[aliases.get(key, key)] is param.default
            for key, param in inspect.signature(function).parameters.items()
            if key not in arguments
        )


def test_public_rf_plan_preserves_stock_permutation_and_validates_inputs(rf_plan):
    spans = [{"start": 4, "latent_shape": [1, 8, 16]}]
    plan = rf_plan.build_rf_attention_plan(spans, 132, 128, "BSND", "cpu")
    rows = torch.arange(132)
    assert torch.equal(rows.index_select(0, plan.permutation).index_select(0, plan.inverse), rows)
    assert plan.video_spans == ((4, (1, 8, 16)),)
    assert plan.sequence_len == 132 and plan.pool_size == 128
    as_tuple = rf_plan.rf_multi_span_permutation(spans, 132)
    assert torch.equal(as_tuple[0], plan.permutation) and torch.equal(as_tuple[1], plan.inverse)
    with pytest.raises(ValueError, match="input_layout"):
        rf_plan.build_rf_attention_plan(spans, 132, input_layout="TND")
    with pytest.raises(ValueError, match="sequence_len"):
        rf_plan.build_rf_attention_plan(spans, 0)
    with pytest.raises(ValueError, match="pool_size"):
        rf_plan.build_rf_attention_plan(spans, 132, pool_size=0)


def test_layer_boundary_forwards_bf16_and_quantized_operator_contracts(layer_boundary, monkeypatch):
    boundary = layer_boundary.module
    query = torch.ones(1, 4, 1, 2, dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 1, 1, dtype=torch.int8)
    assert (
        boundary.bf16_block_sparse_attention(
            query=query,
            key=query * 2,
            value=query * 3,
            block_sparse_mask=mask,
            scale_value=0.25,
            inner_precise=4,
            actual_seq_lengths=[4],
            actual_seq_lengths_kv=[4],
            bsa_backend="eagle",
        )
        is query
    )
    kwargs = layer_boundary.calls[0][3]
    assert kwargs["input_layout"] == "BSND" and kwargs["bsa_backend"] == "eagle"
    assert kwargs["block_size_q"] == kwargs["block_size_kv"] == 128

    native_calls = []

    def quantized(**native_kwargs):
        native_calls.append(native_kwargs)
        return query, torch.tensor(0)

    monkeypatch.setattr(
        boundary.torch.ops, "mindiesd", types.SimpleNamespace(eagle_quant_block_sparse_attention=quantized)
    )
    assert (
        boundary.eagle_quant_block_sparse_attention(
            query=query.transpose(1, 2).contiguous().to(torch.int8),
            key=query.transpose(1, 2).contiguous().to(torch.int8),
            value=query.transpose(1, 2).contiguous().to(torch.float8_e4m3fn),
            block_sparse_mask=mask,
            scale_value=0.25,
            inner_precise=4,
            actual_seq_lengths=[4],
            actual_seq_lengths_kv=[4],
            query_scale=torch.ones(1),
            key_scale=torch.ones(1),
            value_scale=torch.ones(1),
        )
        is query
    )
    assert native_calls[0]["q_input_layout"] == "BNSD"
    assert native_calls[0]["block_shape"] == [128, 128]


def test_bf16_preparation_caches_indices_pooling_and_mask_slices(producer):
    chain = producer.build(precision="bf16")
    assert "bf16_indices" in chain.retained_blocks and "bf16_protected_mask" in chain.retained_blocks
    tensors = [torch.arange(132 * 2, dtype=torch.float32).reshape(1, 132, 1, 2) + offset for offset in range(3)]
    prepared, pooled = chain._prepare_bf16_tensors(tensors)
    assert len(prepared) == 3 and tuple(pooled.shape[:2]) == (2, 2)
    mask = chain._make_bf16_sparse_mask(pooled, scale=0.25, sparsity=0.5)
    assert mask.dtype == torch.int8
    kwargs = producer.layers.calls[-1][1]
    assert kwargs["pool_parts"] == 2 and kwargs["retained_idx"] is chain.retained_blocks["bf16_indices"]
    assert tuple(chain._slice_bf16_sparse_mask(mask, 0, 1).shape)[1] == 1

    legacy = producer.build(precision="bf16", sparse_type="rf_v2")
    legacy_prepared, legacy_pool = legacy._prepare_bf16_tensors(tensors)
    indices, selected = legacy._make_bf16_sparse_mask(legacy_pool, scale=0.25, sparsity=0.5)
    sliced = legacy._slice_bf16_sparse_mask((indices, selected), 0, 1)
    assert len(legacy_prepared) == 3 and sliced[0].shape[1] == sliced[1].shape[1] == 1


@pytest.mark.parametrize("sparse_type", ["rf_v2", "rf_v3"])
def test_bf16_compute_restores_token_order_and_prearranged_copy(producer, sparse_type):
    chain = producer.build(precision="bf16", sparse_type=sparse_type)
    mapping = chain.reorder_mapping
    tensors = [torch.arange(132 * 2, dtype=torch.float32).reshape(1, 132, 1, 2) for _ in range(3)]
    reordered_to_token = torch.arange(132).roll(7)
    token_to_reordered = torch.empty_like(reordered_to_token)
    token_to_reordered[reordered_to_token] = torch.arange(132)
    mapping.token_block_mapping["reordered_position_to_token"] = reordered_to_token
    mapping.token_block_mapping["token_to_reordered_position"] = token_to_reordered
    rf_tensors = [tensor.index_select(1, reordered_to_token) for tensor in tensors]
    pooled = chain._prepare_bf16_tensors(tensors)[1]
    mask = chain._make_bf16_sparse_mask(pooled, scale=0.25, sparsity=0.5)
    output = torch.empty_like(tensors[0])
    got = chain._compute_bf16_head(rf_tensors, mask, output, scale=0.25, inner_precise=4)
    assert got is output
    assert torch.equal(got, tensors[0])

    direct = producer.build(precision="bf16", sparse_type="rf_v3", pre_rearranged=True)
    direct_output = torch.empty_like(tensors[0])
    direct_mask = direct._make_bf16_sparse_mask(direct._prepare_bf16_tensors(tensors)[1], scale=0.25, sparsity=0.5)
    direct._compute_bf16_head(tensors, direct_mask, direct_output, scale=0.25, inner_precise=4)
    assert torch.equal(direct_output, tensors[0])


@pytest.mark.parametrize("backend", ["cann", "eagle"])
@pytest.mark.parametrize("optimize", [False, True])
def test_distributed_mix_reaches_real_adapter(producer, layer_boundary, monkeypatch, backend, optimize):
    calls = []
    monkeypatch.setattr(
        sys.modules["mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2"],
        "get_multi_span_blockwise_mask",
        producer.layers.get_multi_span_blockwise_mask,
    )

    def native(**kwargs):
        calls.append(kwargs)
        assert kwargs["key"].shape[1:3] == (1, 132)
        return kwargs["query"].to(torch.bfloat16), None

    monkeypatch.setattr(torch.ops, "mindiesd", types.SimpleNamespace(eagle_quant_block_sparse_attention=native))
    monkeypatch.setattr(
        producer.layers, "eagle_quant_block_sparse_attention", layer_boundary.module.eagle_quant_block_sparse_attention
    )
    state = producer.modules["state"].ParallelState(
        group=producer.group, bsa_backend=backend, optimize_attention_pipeline=optimize
    )
    query = torch.ones(1, 68, 4, 2, dtype=torch.bfloat16)
    spans = [{"start": 4, "latent_shape": [1, 8, 16]}]
    options = dict(group=producer.group, state=state, used_len=132, chunking=2, return_stats=True)
    output, stats = producer.entry(query, query, query, spans, **options)
    assert output.shape == query.shape and len(calls) == 2
    assert stats["tally"]["allreduce"] == 1 and stats["tally"]["a2a_calls"] == 8
    assert state.last_call_stats["actual"]["bsa_backend"] == "eagle_quant"
    with pytest.raises(ValueError, match="one KV head"):
        arguments = {k: v for k, v in calls[0].items() if k not in layer_boundary.module._DEVICE_OP_DEFAULTS}
        layer_boundary.module.eagle_quant_block_sparse_attention(**{**arguments, "key": torch.ones(1, 2, 132, 2)})

    events = len(producer.events)
    for mode in ({"precision": "bf16", "inner_precise": 0}, {"precision": "bf16", "sparse_type": "rf_v2"}):
        with pytest.raises(ValueError):
            producer.entry(query, query, query, spans, **options, **mode)
        assert len(producer.events) == events and state.last_call_stats is None


@pytest.mark.parametrize("precision", ["bf16", "mix"])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("pre_rearranged", [False, True])
def test_serial_rf_order_cache_and_backend(rf_plan, monkeypatch, precision, layout, pre_rearranged):
    namespace = dict(native_rf_functions(), build_rf_attention_plan=rf_plan.build_rf_attention_plan)

    def quant(q, k, v, **kwargs):
        values = [x.transpose(1, 2) if layout == "BSND" else x for x in (q, k, v)]
        return *values, None, None, None

    namespace.update(
        math=math,
        torch_npu=types.SimpleNamespace(float8_e4m3fn=torch.float8_e4m3fn),
        _eagle_qbsa_quant_qkv=quant,
        rain_fusion_attention_v3=lambda q, *args, **kwargs: q,
    )
    namespace = load_functions(
        REPO_ROOT / "mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py", {"bsa_sparse_attention_v3"}, namespace
    )
    function = namespace["bsa_sparse_attention_v3"]
    monkeypatch.setattr(
        torch.ops,
        "mindiesd",
        types.SimpleNamespace(eagle_quant_block_sparse_attention=lambda **kw: (kw["query"], None)),
    )
    query = torch.arange(132 * 2).reshape(1, 132, 1, 2).to(torch.bfloat16)
    query = query if layout == "BSND" else query.transpose(1, 2)
    options = dict(
        latent_shape_q=None,
        video_spans=[{"start": 4, "latent_shape": [1, 8, 16]}],
        precision=precision,
        input_layout=layout,
        bsa_backend="eagle",
        pre_rearranged=pre_rearranged,
    )
    if precision == "mix" and pre_rearranged:
        with pytest.raises(ValueError, match="pre_rearranged"):
            function(query, query, query, **options)
        return
    output, mask = function(query, query, query, **options)
    cached_output, cached = function(query, query, query, cached_mask=mask, **options)
    assert torch.equal(output, query) and torch.equal(cached_output, query) and cached is mask
    with pytest.raises(ValueError, match="video_spans"):
        function(query, query, query, **{**options, "txt_len": 1})
    with pytest.raises(ValueError, match="input_layout"):
        function(query, query, query, **{**options, "input_layout": "bad"})
    with pytest.raises(ValueError, match="pool_parts"):
        namespace["get_multi_span_blockwise_mask"](torch.ones(3, 2, 1, 2), 0.5, 1, set(), set(), "BSND", pool_parts=2)


@pytest.mark.parametrize("sparsity", [0.0, 0.8, 1.0])
def test_cached_bf16_mask_keeps_protection_without_index_rewrites(producer, monkeypatch, sparsity):
    mask_helper = native_rf_functions()["get_multi_span_blockwise_mask"]
    monkeypatch.setattr(producer.layers, "get_multi_span_blockwise_mask", mask_helper)
    primitive = producer.modules["bf16_primitives"].BF16Primitives()
    primitive.reorder_mapping = types.SimpleNamespace(
        bs=1,
        s_local=256,
        hc=4,
        hs=2,
        chunking=2,
        world=2,
        used_len=512,
        dense_blocks={0},
        first_frame_blocks={1},
        reordered_position_to_token=torch.arange(512),
    )
    primitive.optimize_attention_pipeline = True
    primitive.sparse_type = "rf_v3"
    primitive.retained_blocks = {}
    primitive._init_bf16()
    pool = torch.arange(16).reshape(2, 4, 1, 2).float() / 16
    expected = mask_helper(pool, sparsity, 0.25, {0}, {1}, "BSND", return_binary=True, pool_parts=2)
    writes = []
    original = torch.Tensor.__setitem__

    def tracked(tensor, key, value):
        writes.append(key)
        return original(tensor, key, value)

    monkeypatch.setattr(torch.Tensor, "__setitem__", tracked)
    actual = primitive._make_bf16_sparse_mask(pool, scale=0.25, sparsity=sparsity)
    assert torch.equal(actual, expected) and not writes
