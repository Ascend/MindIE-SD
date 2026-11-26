# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# SPDX-License-Identifier: MulanPSL-2.0

import importlib
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize("phase,rank", [(0, 0), (1, 1)])
def test_global_mix_packets_refresh_values_and_reuse_geometry(producer, phase, rank):
    wire = importlib.import_module(producer.package.__name__ + ".quantized_attention_wire")
    mapping = SimpleNamespace(world=2, rank=rank, hc=4, hs=128, chunking=2, used_len=264)

    def chain(enabled):
        return SimpleNamespace(
            reorder_mapping=mapping,
            pre_rearranged=True,
            precision="mix",
            group=producer.group,
            optimize_attention_pipeline=enabled,
            retained_blocks={},
        )

    actual, reference = chain(True), chain(False)
    torch.manual_seed(72)
    values = [torch.randn(1, 68, 4, 128).bfloat16() for _ in range(3)]
    pointer = None
    for repeat in range(2):
        values_now = [v * (repeat + 1) for v in values]
        counts = {"allreduce": 0}
        got = wire.prepare_quantized_wire(actual, *values_now, kv_group=None, phase=phase, counts=counts)
        wanted = wire.prepare_quantized_wire(
            reference, *values_now, kv_group=None, phase=phase, counts={"allreduce": 0}
        )
        assert counts["allreduce"] == 2
        assert all(torch.equal(x, y) for aa, bb in zip(got, wanted) for x, y in zip(aa, bb))
        positions = next(iter(actual.retained_blocks["mix_wire_geometry"].values()))["positions"]
        if pointer is not None:
            assert positions.data_ptr() == pointer
        pointer = positions.data_ptr()
