#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Small physical shapes: direct NPU QFA against CPU kernel-stage simulation."""

import pytest

from tests.ops.quant_flash_attn.qfa_accuracy_common import (
    Case,
    bind,
    check_outputs,
    compare,
    cpu_golden,
    initialize,
    make_inputs,
)


@pytest.mark.parametrize("layout", ("BNSD", "BSND"))
@pytest.mark.parametrize(
    "q_seq,kv_seq,q_valid,kv_valid",
    (
        pytest.param(128, 256, 128, 256, id="native-q128-kv256"),
        pytest.param(129, 257, 129, 257, id="native-q129-kv257"),
        pytest.param(129, 513, 129, 513, id="native-q129-kv513"),
        pytest.param(512, 512, 128, 256, id="pad512-q128-kv256"),
        pytest.param(512, 512, 129, 257, id="pad512-q129-kv257"),
        pytest.param(512, 512, 511, 511, id="pad512-q511-kv511"),
        pytest.param(512, 512, 512, 512, id="pad512-q512-kv512"),
        pytest.param(1024, 1024, 513, 513, id="pad512-q513-kv513"),
    ),
)
@pytest.mark.parametrize("length_mode", ("used",))
@pytest.mark.parametrize("distribution", ("uniform", "normal"))
def test_qfa_small_accuracy(layout, q_seq, kv_seq, q_valid, kv_valid, length_mode, distribution):
    from tests.ops.quant_flash_attn.qfa_tolerance import BUDGET

    # Mixed per-batch lengths exercise the shared scheduler's cost accounting.
    q_used = (q_valid, 65)
    kv_used = (kv_valid, 129)
    case = Case(
        batch=2,
        q_heads=4,
        kv_heads=2,
        q_seq=q_seq,
        kv_seq=kv_seq,
        layout=layout,
        length_mode=length_mode,
        q_used=q_used,
        kv_used=kv_used,
    )
    torch, torch_npu, device = initialize(0)
    with torch.inference_mode():
        native, encodings = make_inputs(case, distribution, 21, torch, device)
        invoke, _ = bind(case, native, torch, torch_npu, device)
        reference = cpu_golden(case, encodings, torch)
        output = invoke()
        torch.npu.synchronize()
        check_outputs(output, case, torch, device)
        # Check valid rows separately so cleared padding cannot dilute the metrics.
        for batch, valid in enumerate(case.q_used):
            actual = output[0][batch, :, :valid, :] if layout == "BNSD" else output[0][batch, :valid, :, :]
            golden = reference[batch, :, :valid, :] if layout == "BNSD" else reference[batch, :valid, :, :]
            padding = output[0][batch, :, valid:, :] if layout == "BNSD" else output[0][batch, valid:, :, :]
            print(f"valid rows: batch={batch}, q_used={valid}, kv_used={case.kv_used[batch]}", flush=True)
            compare(actual, golden, BUDGET, small=True)
            assert torch.count_nonzero(padding).item() == 0, f"Nonzero query padding in batch {batch}"
        compare(output[0], reference, BUDGET, small=True)
