# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared CPU loader for the staged packet, BF16 and execution contracts."""

# pylint: disable=no-member,no-name-in-module

import sys
import types

import pytest
import torch.distributed as dist

torch = pytest.importorskip("torch")

from cpu_support import (
    NpuSimulation,
    PARALLEL_DIR,
    load_parallel_module,
    native_avgpool,
    native_rf_functions,
    new_package,
)


@pytest.fixture
def fake_npu():
    return NpuSimulation()


@pytest.fixture
def producer(monkeypatch):
    package = new_package("_parallel_pipeline_contracts", monkeypatch)
    layers = types.ModuleType(package.__name__ + "._layers")
    stock = native_rf_functions()
    layers.do_multi_span_tensor_rearrange_pooling = stock["do_multi_span_tensor_rearrange_pooling"]
    layers.avgpool = native_avgpool()
    layers.pool_qk_separately = lambda q, k, block, layout: layers.avgpool(torch.cat((q, k)), block, layout)
    layers.DEVICE_OP_BLOCK_SHAPE = [128, 128]
    layers.A5_INNER_PRECISE = 4
    layers.is_a5_device = lambda: True
    layers.calls = []

    def mask(pool, *args, return_binary=False, **kwargs):
        layers.calls.append(("mask", kwargs))
        cells, heads = pool.shape[1:3]
        if return_binary:
            return torch.ones((1, heads, cells, cells), dtype=torch.int8)
        return torch.zeros(cells, heads, cells, dtype=torch.long), torch.ones(cells, heads, dtype=torch.long)

    def bf16_attention(*, query, key, value, **kwargs):
        layers.calls.append(("bf16", kwargs))
        return query.clone()

    def rf_v2_attention(query, key, value, **kwargs):
        layers.calls.append(("rf_v2", kwargs))
        return query.clone()

    def quant_attention(*, query, key, value, **kwargs):
        layers.calls.append(("mix", kwargs))
        return (query.float() + key.float() * 0.5 + value.view(torch.float8_e4m3fn).float()).to(torch.bfloat16)

    layers.get_multi_span_blockwise_mask = mask
    stock_layer = types.ModuleType("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2")
    stock_layer.get_multi_span_blockwise_mask = mask
    monkeypatch.setitem(sys.modules, stock_layer.__name__, stock_layer)
    layers.bf16_block_sparse_attention = bf16_attention
    layers.rain_fusion_attention = rf_v2_attention
    layers.eagle_quant_block_sparse_attention = quant_attention
    monkeypatch.setitem(sys.modules, layers.__name__, layers)
    package._layers = layers
    loaded = {}
    for relative in (
        "config",
        "topology",
        "qkv_layout",
        "state",
        "attention_reorder_mapping",
        "distributed_quant",
        "packet",
        "mixed_producer",
    ):
        loaded[relative] = load_parallel_module(package, relative, monkeypatch)

    for relative in ("bf16_primitives", "pipeline", "distributed_sparse_attention"):
        if (PARALLEL_DIR / (relative + ".py")).exists():
            loaded[relative] = load_parallel_module(package, relative, monkeypatch)

    group = types.SimpleNamespace(world=2, rank=0)
    events = []

    def all_reduce(value, *, op, group):
        events.append(("reduce", value.clone(), group))

    def all_to_all(recv, send, *, group):
        events.append(("a2a", send.clone(), group))
        recv.copy_(send)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda selected: selected.world)
    monkeypatch.setattr(dist, "get_rank", lambda selected: selected.rank)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(dist, "all_to_all_single", all_to_all)

    def build(
        rank=0,
        *,
        precision="mix",
        sparse_type="rf_v3",
        pre_rearranged=False,
        optimize=True,
        backend="cann",
    ):
        config = loaded["config"].derive_parallel_config(
            world=2,
            rank=rank,
            hc=4,
            hs=2,
            bs=1,
            s_local=68,
            used_len=132,
            chunking=2,
            spans=[{"start": 4, "latent_shape": [1, 8, 16]}],
        )
        mapping = loaded["attention_reorder_mapping"].get_attention_reorder_mapping(config, torch.device("cpu"))
        state = loaded["state"].ParallelState(group=group, optimize_attention_pipeline=optimize, bsa_backend=backend)
        chain_type = loaded["pipeline"].Chain if "pipeline" in loaded else loaded["mixed_producer"].MixedProducer
        return chain_type(
            mapping,
            state=state,
            precision=precision,
            sparse_type=sparse_type,
            pre_rearranged=pre_rearranged,
        )

    entry = loaded.get("distributed_sparse_attention")
    return types.SimpleNamespace(
        modules=loaded,
        package=package,
        layers=layers,
        build=build,
        group=group,
        events=events,
        entry=None if entry is None else entry.distributed_sparse_attention,
    )
