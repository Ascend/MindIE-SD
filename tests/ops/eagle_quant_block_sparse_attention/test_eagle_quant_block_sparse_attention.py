# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Small real-operator tests. CPU arithmetic is only the golden, never the DUT."""

import json
import pytest
from tests.ops.device_support import init_test_device, require_support
from tests.ops.eagle_quant_block_sparse_attention.eagle_quant_block_sparse_attention_accuracy_common import (
    OP,
    attributes,
    bit_difference,
    compare,
    compare_output,
    contract,
    dut,
    fingerprint,
    golden,
    input_hashes,
    lengths,
    make_inputs,
    metadata,
    name,
    preview,
    scenario,
    to_device,
)
from tests.ops.eagle_quant_block_sparse_attention.eqbsa_state import save, small_device_identity

# Twelve random cases, uniform/normal split 6:6; cross-select instead of a product.
CASES = [
    scenario(
        batch_size=1,
        num_query_heads=2,
        num_kv_heads=2,
        query_seq_len=256 + (i % 2),
        kv_seq_len=256 + (i % 2),
        head_dim=128 if i < 4 else 64,
        layout="BNSD" if i % 4 < 2 else "TND",
        output_dtype="bf16" if i % 2 == 0 else "fp16",
        distribution="uniform" if i % 2 == 0 else "normal",
        seed=20260811 + i,
    )
    for i in range(8)
]
CASES += [
    scenario(
        batch_size=2,
        num_query_heads=4,
        num_kv_heads=1,
        query_seq_len=257,
        kv_seq_len=513,
        head_dim=128,
        layout="BNSD" if i < 2 else "TND",
        output_dtype="fp16" if i % 2 else "bf16",
        distribution="normal" if i % 2 else "uniform",
        seed=20260900 + i,
        **({"query_lengths": [257, 256], "kv_lengths": [513, 512]} if i >= 2 else {}),
    )
    for i in range(4)
]
# Dense 512/513 keys exercise the fused 512-column softmax and online max rescaling.
CASES += [
    scenario(
        batch_size=1,
        num_query_heads=2,
        num_kv_heads=1,
        query_seq_len=256,
        kv_seq_len=513,
        head_dim=128,
        layout=layout,
        output_dtype="bf16",
        special=special,
        distribution="uniform" if layout == "BNSD" else "normal",
    )
    for layout in ("BNSD", "TND")
    for special in ("dense", "zero", "empty", "partial_empty")
]

# Sparse Q blocks can start inside a 64-token quantization group or straddle its boundary.
CASES += [
    scenario(
        batch_size=1,
        num_query_heads=2,
        num_kv_heads=1,
        query_seq_len=q_len,
        kv_seq_len=385,
        head_dim=64,
        layout=layout,
        output_dtype="fp16",
        distribution="uniform",
        special="q_scale_groups",
        q_block_size=q_block,
    )
    for layout in ("BNSD", "TND")
    for q_block, q_len in ((32, 128), (96, 225), (65, 137))
]

# Exercise both sides of every AIV split boundary, with real and empty final tiles.
CASES += [
    scenario(
        batch_size=1,
        num_query_heads=2,
        num_kv_heads=2,
        query_seq_len=256 + tail,
        kv_seq_len=513,
        head_dim=64 if i % 2 == 0 else 128,
        layout=layout,
        output_dtype="fp16" if i % 2 == 0 else "bf16",
        special="empty" if i % 3 == 0 else "dense",
        distribution="uniform" if layout == "BNSD" else "normal",
        seed=20261000 + i,
    )
    for layout in ("BNSD", "TND")
    for i, tail in enumerate((1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128))
]
# Four gathered KV tiles wrap the three-slot pipeline and exercise middle rescaling at D=64.
CASES += [
    scenario(
        batch_size=1,
        num_query_heads=2,
        num_kv_heads=1,
        query_seq_len=129,
        kv_seq_len=1537,
        head_dim=64,
        layout=layout,
        output_dtype="bf16",
        special="dense",
        distribution="uniform" if layout == "BNSD" else "normal",
    )
    for layout in ("BNSD", "TND")
]

# More than one task per core, with a three-tile period mixing empty and nonempty rows.
CASES += [
    scenario(
        batch_size=1,
        num_query_heads=64,
        num_kv_heads=1,
        query_seq_len=257,
        kv_seq_len=129,
        head_dim=64,
        layout=layout,
        output_dtype="bf16",
        special="partial_empty",
        distribution="uniform" if layout == "BNSD" else "normal",
    )
    for layout in ("BNSD", "TND")
]


# Padded nRound=512 must mask dequantized logits in both DN32 query blocks.
# Reuse the failing negative-logit seed; zero logits independently check valid-key counts.
CASES += [
    scenario(
        batch_size=1,
        num_query_heads=2,
        num_kv_heads=2,
        query_seq_len=288,
        kv_seq_len=497 if special == "dense" else 511,
        head_dim=64 if special == "dense" else 128,
        layout=layout,
        output_dtype="fp16" if special == "dense" else "bf16",
        special=special,
        distribution="normal",
        seed=20261002,
    )
    for layout in ("BNSD", "TND")
    for special in ("dense", "zero")
]


@pytest.fixture(scope="module", name="device")
def npu_device(request):
    device = init_test_device().index
    require_support(OP)
    import mindiesd  # Register the real operator and its FakeTensor implementation.

    batch = request.config._eqbsa_batch
    if device != batch.plan["parameters"]["device"]:
        raise RuntimeError("small-shape logical device changed after collection")
    path = batch.path / "device.json"
    if not path.exists():
        save(path, small_device_identity(device))
    return device


@pytest.mark.parametrize("s", CASES, ids=name)
def test_small_cpu_golden_and_lse_identity(s, device, request):
    import torch

    cpu, parameters = make_inputs(s)
    inputs = to_device(cpu, device)
    detail = metadata(inputs, s, parameters)
    print(json.dumps(detail), flush=True)
    request.node.user_properties.append(("inputs", detail))
    off = dict(s, softmax_lse_flag=0)
    original = dut(inputs, off)
    enabled = dut(inputs, s)
    torch.npu.synchronize()
    contract(original, off)
    contract(enabled, s)
    identity_pass = fingerprint(original[0]) == fingerprint(enabled[0])
    request.node.user_properties.append(
        ("O_bitwise_identity", dict(passed=identity_pass, **bit_difference(enabled[0], original[0])))
    )
    expected = golden(cpu, s, staged=True)
    metrics = compare(enabled, expected, s)
    metrics["input_unchanged"] = input_hashes(inputs) == input_hashes(cpu)
    metrics["passed"] &= metrics["input_unchanged"]
    request.node.user_properties.append(("metrics", metrics))
    assert identity_pass, "LSE flag changed O bits (including signed zero)"
    assert metrics["passed"], "O/LSE gates failed; inspect per-output report"
    if s["special"] == "zero":
        # Independent analytical oracle; does not reuse the staged softmax implementation.
        lses = []
        for b, (qlen, klen) in enumerate(zip(lengths(s, "query"), lengths(s, "kv"))):
            q_block, k_block = detail["attributes"]["block_shape"]
            mask = cpu["block_sparse_mask"][b].repeat_interleave(q_block, -2).repeat_interleave(k_block, -1)
            lses.append(mask[:, :qlen, :klen].sum(-1).float().log().unsqueeze(-1))
        analytical = (
            torch.cat([v.transpose(0, 1) for v in lses]).contiguous() if s["layout"] == "TND" else torch.stack(lses)
        )
        report = compare_output(
            enabled[1], analytical, "LSE", "fp32", basis="zero logits: FP32 log of exact integer valid-key count"
        )
        request.node.user_properties.append(("zero_logits_analytic", report))
        assert report["passed"]


@pytest.mark.parametrize("flag", [-1, 2])
def test_reject_invalid_lse_flag(flag, device, request):
    s = scenario(batch_size=1, num_query_heads=1, num_kv_heads=1, query_seq_len=128, kv_seq_len=128)
    cpu, parameters = make_inputs(s)
    inputs = to_device(cpu, device)
    invalid = dict(s, softmax_lse_flag=flag)
    detail = metadata(inputs, invalid, parameters)
    print(json.dumps(detail), flush=True)
    request.node.user_properties.append(("inputs", detail))
    import torch

    # Establish that the same inputs and backend work before testing flag rejection.
    valid = dut(inputs, s)
    torch.npu.synchronize()
    contract(valid, s)
    with pytest.raises(RuntimeError):
        dut(inputs, invalid)
        torch.npu.synchronize()


@pytest.mark.parametrize("layout", ["BNSD", "TND"])
@pytest.mark.parametrize("flag", [-1, 0, 1, 2])
def test_fake_lse_flag_contract(layout, flag, device):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    s = scenario(
        batch_size=1,
        num_query_heads=2,
        num_kv_heads=1,
        query_seq_len=128,
        kv_seq_len=256,
        layout=layout,
        output_dtype="bf16" if layout == "BNSD" else "fp16",
    )
    spec = preview(s)
    attrs = dict(attributes(s), softmax_lse_flag=flag)
    if layout == "TND":
        attrs["output_dtype"] = torch.float16
    with FakeTensorMode():
        keys = ("query", "key", "value", "block_sparse_mask", "query_scale", "key_scale", "value_scale")
        inputs = {
            key: torch.empty(
                spec[key]["shape"],
                dtype=torch.float32 if key.endswith("_scale") else torch.int8,
                device=f"npu:{device}",
            )
            for key in keys
        }
        op = torch.ops.mindiesd.eagle_quant_block_sparse_attention
        if flag not in (0, 1):
            with pytest.raises(RuntimeError, match="softmax_lse_flag must be 0 or 1"):
                op(**inputs, **attrs)
        else:
            outputs = op(**inputs, **attrs)
            contract(outputs, s, finite=False)
            assert all(output.device == inputs["query"].device for output in outputs)
