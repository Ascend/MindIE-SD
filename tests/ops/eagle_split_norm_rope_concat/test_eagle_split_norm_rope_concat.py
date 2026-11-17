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

import os

import pytest
import torch

from tests.ops.device_support import init_test_device, require_support

from tests.ops.eagle_split_norm_rope_concat.eagle_split_norm_rope_concat_accuracy_common import (
    call_dut,
    compare,
    golden,
    inputs_to_cpu,
    make_inputs,
)


@pytest.fixture(scope="module", name="device")
def device_fixture():
    if os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU":
        pytest.skip("eagle_split_norm_rope_concat requires an NPU")
    selected = init_test_device()
    import mindiesd  # noqa: F401

    return selected


@pytest.fixture(scope="module", name="supported_soc")
def supported_soc_fixture(device):
    require_support("eagle_split_norm_rope_concat", kind="positive")


@pytest.fixture(scope="module", name="registered_op")
def registered_op_fixture(device):
    return torch.ops.mindiesd.eagle_split_norm_rope_concat.default


# Explicit crossed cases keep accuracy and interface checks within a 100-case budget.
# See README for the coverage/count breakdown; do not stack distribution decorators.
@pytest.mark.parametrize(
    "distribution,dtype,rotary_dim,rope_type,norm_type,output_bnsd,rank,rope_rank,shared",
    [
        ("uniform", torch.float16, 32, 0, 2, False, 3, 2, True),
        ("normal", torch.bfloat16, 32, 1, 3, True, 4, 2, True),
        ("normal", torch.float16, 64, 0, 3, True, 4, 3, False),
        ("uniform", torch.bfloat16, 64, 1, 2, False, 4, 3, True),
        ("uniform", torch.bfloat16, 96, 0, 3, False, 4, 4, False),
        ("normal", torch.float16, 96, 1, 2, True, 4, 4, True),
        ("normal", torch.bfloat16, 128, 0, 2, True, 4, 3, True),
        ("uniform", torch.float16, 128, 1, 3, False, 4, 4, False),
    ],
)
@pytest.mark.usefixtures("supported_soc")
def test_compact_cache_matches_expanded(
    device,
    distribution,
    dtype,
    rotary_dim,
    rope_type,
    norm_type,
    output_bnsd,
    rank,
    rope_rank,
    shared,
):
    shape = (3, 12, 128) if rank == 3 else (2, 3, 12, 128)
    compact = list(
        make_inputs(shape, (8, 2, 2), rotary_dim, dtype, device, distribution, rope_rank=rope_rank, compact=True)
    )
    if shared and rope_rank > 2:
        compact[3:5] = [cache[:1].contiguous() for cache in compact[3:5]]
    if norm_type == 2:
        compact[1] = compact[2] = None
    attrs = dict(rotary_dim=rotary_dim, rope_type=rope_type, norm_type=norm_type, output_bnsd=output_bnsd)
    expected = golden(inputs_to_cpu(compact), **attrs)
    actual = tuple(x.cpu() for x in call_dut(compact, **attrs))
    compare(actual, expected)
    # Construct the explicitly equivalent full-width cache, not an input-format workaround.
    expanded = list(compact)
    for index in (3, 4):
        cache = compact[index].cpu()
        full = torch.cat((cache, cache), dim=-1) if rope_type == 1 else cache.repeat_interleave(2, dim=-1)
        expanded[index] = full.to(device)
    full_outputs = tuple(x.cpu() for x in call_dut(expanded, **attrs))
    compare(full_outputs, expected)
    for half_out, full_out in zip(actual, full_outputs):
        assert torch.equal(half_out.view(torch.int16), full_out.view(torch.int16)), "Cache encodings must agree bitwise"


@pytest.mark.parametrize(
    "rotary_dim,rope_type,compact,backend",
    [
        (32, 0, True, "fake"),
        (64, 1, False, "meta"),
        (96, 0, True, "meta"),
        (128, 1, False, "fake"),
    ],
)
def test_cache_width_fake_and_meta(registered_op, rotary_dim, rope_type, compact, backend):
    from contextlib import nullcontext
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() if backend == "fake" else nullcontext():
        target = "npu:0" if backend == "fake" else "meta"
        qkv = torch.empty((2, 3, 12, 128), dtype=torch.bfloat16, device=target)
        width = rotary_dim // 2 if compact else rotary_dim
        cache = torch.empty((2, 3, 1, width), dtype=qkv.dtype, device=target)
        outputs = registered_op(
            qkv, None, None, cache, cache, [8, 2, 2], norm_type=2, rope_type=rope_type, rotary_dim=rotary_dim
        )
        for output, heads in zip(outputs, (8, 2, 2)):
            assert output.shape == (2, heads, 3, 128)
            assert output.dtype == qkv.dtype and output.device == qkv.device
            assert output.is_contiguous()


@pytest.mark.parametrize(
    "backend,width,sin_width",
    [
        ("real", 47, 47),
        ("fake", 49, 49),
        ("meta", 64, 64),
        ("real", 48, 96),
        ("fake", 96, 48),
    ],
)
def test_rejects_invalid_cache_width(registered_op, backend, width, sin_width):
    from contextlib import nullcontext
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() if backend == "fake" else nullcontext():
        target = "meta" if backend == "meta" else "npu:0"
        qkv = torch.empty((1, 6, 128), dtype=torch.bfloat16, device=target)
        cos = torch.empty((1, width), dtype=qkv.dtype, device=target)
        sin = torch.empty((1, sin_width), dtype=qkv.dtype, device=target)
        with pytest.raises(RuntimeError):
            registered_op(qkv, None, None, cos, sin, [2, 2, 2], norm_type=2, rope_type=0, rotary_dim=96)


@pytest.mark.parametrize(
    "distribution,norm_type,dtype,heads,batch,seq,rank,rotary_dim,rope_type,output_bnsd",
    [
        ("uniform", 2, torch.bfloat16, (2, 2, 2), 1, 1, 3, 96, 1, False),
        ("normal", 3, torch.float16, (8, 2, 2), 1, 16, 3, 128, 0, True),
        ("uniform", 3, torch.bfloat16, (8, 1, 1), 2, 17, 4, 64, 0, False),
        ("normal", 2, torch.float16, (2, 2, 2), 2, 17, 4, 32, 1, True),
    ],
)
@pytest.mark.usefixtures("supported_soc")
def test_accuracy(
    device,
    distribution,
    norm_type,
    dtype,
    heads,
    batch,
    seq,
    rank,
    rotary_dim,
    rope_type,
    output_bnsd,
):
    shape = (seq, sum(heads), 128) if rank == 3 else (batch, seq, sum(heads), 128)
    inputs = list(make_inputs(shape, heads, rotary_dim, dtype, device, distribution, rope_rank=rank))
    if norm_type == 2:
        inputs[1] = inputs[2] = None
    attrs = dict(norm_type=norm_type, rotary_dim=rotary_dim, rope_type=rope_type, output_bnsd=output_bnsd)
    expected = golden(inputs_to_cpu(inputs), **attrs)
    actual = call_dut(inputs, **attrs)
    compare(tuple(x.cpu() for x in actual), expected)


@pytest.mark.parametrize(
    "dim",
    [
        64,
        129,
    ],
)
def test_rejects_unsupported_head_dim(device, dim):
    inputs = make_inputs((1, 6, dim), (2, 2, 2), 32, torch.bfloat16, device, "uniform")
    with pytest.raises(RuntimeError, match="head_dim must be 128"):
        call_dut(inputs, rotary_dim=32)


@pytest.mark.parametrize(
    "distribution,dtype,heads,output_bnsd,compact",
    [
        ("uniform", torch.float16, 15, False, True),
        ("normal", torch.bfloat16, 16, True, False),
        ("normal", torch.float16, 17, False, False),
        ("uniform", torch.bfloat16, 63, True, True),
        ("uniform", torch.float16, 64, True, False),
        ("normal", torch.bfloat16, 65, False, True),
    ],
)
@pytest.mark.usefixtures("supported_soc")
def test_interleave_r96_gather_boundaries(device, distribution, dtype, heads, output_bnsd, compact):
    inputs = make_inputs((1, 3 * heads, 128), (heads, heads, heads), 96, dtype, device, distribution, compact=compact)
    attrs = dict(rope_type=0, rotary_dim=96, output_bnsd=output_bnsd)
    actual = tuple(x.cpu() for x in call_dut(inputs, **attrs))
    expected = golden(inputs_to_cpu(inputs), **attrs)
    compare(actual, expected)
    # Check the suffix separately so a prefix error cannot obscure the R=96 boundary.
    compare(tuple(x[..., 96:].contiguous() for x in actual), tuple(x[..., 96:].contiguous() for x in expected))


@pytest.mark.parametrize(
    "rope_type,norm_type,compact",
    [
        (0, 2, True),
        (1, 3, False),
    ],
)
@pytest.mark.usefixtures("supported_soc")
def test_fp32_norm_flows_directly_into_rope(device, rope_type, norm_type, compact):
    inputs = make_inputs((1, 6, 128), (2, 2, 2), 96, torch.bfloat16, device, "uniform", compact=compact)
    inputs[0].fill_(1)
    inputs[1].fill_(1)
    inputs[2].fill_(1)
    inputs[3].fill_(1.375)
    inputs[4].zero_()
    attrs = dict(rope_type=rope_type, norm_type=norm_type, rotary_dim=96, eps=2.0)
    actual = tuple(x.cpu() for x in call_dut(inputs, **attrs))
    expected = golden(inputs_to_cpu(inputs), **attrs)
    compare(actual, expected)
    # BF16(1.375 / sqrt(3)) = 0.79296875; rounding norm first produces 0.796875.
    for output in actual[:2]:
        assert torch.equal(output[..., :96], torch.full_like(output[..., :96], 0.79296875)), (
            "RMSNorm must remain FP32 until after RoPE"
        )


@pytest.mark.parametrize(
    "rope_type,rope_rank,distribution,compact",
    [
        (0, 2, "uniform", False),
        (0, 4, "normal", True),
        (1, 4, "uniform", False),
        (1, 2, "normal", True),
    ],
)
@pytest.mark.usefixtures("supported_soc")
def test_resident_rope_reused_across_head_tiles(device, rope_type, rope_rank, distribution, compact):
    inputs = make_inputs(
        (2, 1, 4098, 128), (4096, 1, 1), 96, torch.bfloat16, device, distribution, rope_rank=rope_rank, compact=compact
    )
    attrs = dict(rope_type=rope_type, rotary_dim=96, output_bnsd=True)
    actual = tuple(x.cpu() for x in call_dut(inputs, **attrs))
    compare(actual, golden(inputs_to_cpu(inputs), **attrs))


@pytest.mark.usefixtures("supported_soc")
def test_zero_input(device):
    inputs = make_inputs((1, 6, 128), (2, 2, 2), 96, torch.bfloat16, device, "uniform")
    inputs[0].zero_()
    compare(tuple(x.cpu() for x in call_dut(inputs)), golden(inputs_to_cpu(inputs)))


@pytest.mark.parametrize(
    "distribution,rope_type,dtype,seq,heads,rotary_dim,output_bnsd,compact",
    [
        pytest.param("uniform", 0, torch.float16, 1, (130, 65, 65), 128, False, False, id="fp16-idle-cores-r128-snd"),
        pytest.param("normal", 1, torch.bfloat16, 129, (56, 56, 56), 96, True, True, id="bf16-h56-r96-bnsd"),
        pytest.param("uniform", 1, torch.bfloat16, 129, (130, 65, 65), 32, False, False, id="bf16-gqa-r32-snd"),
        pytest.param("normal", 0, torch.float16, 81, (512, 1, 1), 64, True, True, id="fp16-mqa-r64-bnsd"),
    ],
)
@pytest.mark.usefixtures("supported_soc")
def test_pipeline_reuse(device, distribution, rope_type, dtype, seq, heads, rotary_dim, output_bnsd, compact):
    # Include idle K/V cores, repeated buffer reuse, and a one-head final tile.
    attrs = dict(rotary_dim=rotary_dim, rope_type=rope_type, output_bnsd=output_bnsd)
    for seed in (20261017, 20261018):
        inputs = make_inputs(
            (seq, sum(heads), 128), heads, rotary_dim, dtype, device, distribution, seed=seed, compact=compact
        )
        expected = golden(inputs_to_cpu(inputs), **attrs)
        actual = tuple(x.cpu() for x in call_dut(inputs, **attrs))
        compare(actual, expected)
        assert torch.equal(actual[2].view(torch.int16), expected[2].view(torch.int16)), "V bits must be preserved"
        repeated = tuple(x.cpu() for x in call_dut(inputs, **attrs))
        for first, second in zip(actual, repeated):
            assert torch.equal(first.view(torch.int16), second.view(torch.int16)), (
                "Repeated outputs must be bitwise equal"
            )


@pytest.mark.parametrize("output_bnsd", [False, True])
@pytest.mark.usefixtures("supported_soc")
def test_value_signed_zero_bits(device, output_bnsd):
    inputs = make_inputs((1, 6, 128), (2, 2, 2), 96, torch.bfloat16, device, "uniform")
    inputs[0][:, 4:, :].fill_(-0.0)
    expected = inputs[0][:, 4:, :].cpu().contiguous()
    # Inspect storage on CPU so an NPU signbit implementation cannot affect this copy test.
    assert bool((expected.view(torch.int16) == -32768).all()), "Input must contain BF16 negative zero (0x8000)"
    if output_bnsd:
        expected = expected.unsqueeze(0).transpose(1, 2).contiguous()
    value = call_dut(inputs, output_bnsd=output_bnsd)[2].cpu()
    assert value.dtype == expected.dtype and value.shape == expected.shape
    assert value.is_contiguous()
    assert torch.equal(value.view(torch.int16), expected.view(torch.int16)), "V must preserve negative-zero bits"


@pytest.mark.parametrize("rope_type", [0, 1])
@pytest.mark.usefixtures("supported_soc")
def test_tail_is_normalized_but_not_rotated(device, rope_type):
    inputs = list(make_inputs((17, 12, 128), (8, 2, 2), 96, torch.bfloat16, device, "uniform"))
    first = call_dut(inputs, rope_type=rope_type)
    compare(tuple(x.cpu() for x in first), golden(inputs_to_cpu(inputs), rope_type=rope_type))
    inputs[3].fill_(3)
    inputs[4].fill_(-2)
    second = call_dut(inputs, rope_type=rope_type)
    compare(tuple(x.cpu() for x in second), golden(inputs_to_cpu(inputs), rope_type=rope_type))
    for before, after in zip(first[:2], second[:2]):
        assert torch.equal(before[..., 96:], after[..., 96:]), "RoPE must not alter the normalized suffix"
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize(
    "norm_type,dim,rope_type,batched,compact",
    [
        (2, 128, 0, False, True),
        (3, 128, 1, True, False),
    ],
)
@pytest.mark.usefixtures("supported_soc")
def test_full_rope_and_bnsd_defaults(device, norm_type, dim, rope_type, batched, compact):
    shape = (2, 17, 12, dim) if batched else (17, 12, dim)
    inputs = list(make_inputs(shape, (8, 2, 2), dim, torch.bfloat16, device, "normal", compact=compact))
    if norm_type == 2:
        inputs[1] = inputs[2] = None
    attrs = dict(norm_type=norm_type, rope_type=rope_type)
    # Call the public schema directly so fixture defaults cannot mask API defaults.
    actual = torch.ops.mindiesd.eagle_split_norm_rope_concat(*inputs, **attrs)
    expected = golden(inputs_to_cpu(inputs), rotary_dim=None, output_bnsd=True, **attrs)
    compare(tuple(x.cpu() for x in actual), expected)
    explicit = torch.ops.mindiesd.eagle_split_norm_rope_concat(*inputs, rotary_dim=dim, output_bnsd=True, **attrs)
    for default, specified in zip(actual, explicit):
        assert torch.equal(default.cpu().view(torch.int16), specified.cpu().view(torch.int16))


@pytest.mark.parametrize(
    "dim,rope_type,missing_weights",
    [
        (128, 0, (1,)),
        (128, 1, (2,)),
        (128, 0, (1, 2)),
    ],
)
@pytest.mark.usefixtures("supported_soc")
def test_unweighted_rmsnorm_ignores_supplied_weights(device, dim, rope_type, missing_weights):
    inputs = list(make_inputs((17, 12, dim), (8, 2, 2), 32, torch.bfloat16, device, "normal"))
    attrs = dict(norm_type=2, rope_type=rope_type, rotary_dim=32, output_bnsd=False)
    with_weights = call_dut(inputs, **attrs)
    for index in missing_weights:
        inputs[index] = None
    without_weights = call_dut(inputs, **attrs)
    expected = golden(inputs_to_cpu(inputs), **attrs)
    compare(tuple(x.cpu() for x in without_weights), expected)
    for first, second in zip(with_weights, without_weights):
        assert torch.equal(first.cpu().view(torch.int16), second.cpu().view(torch.int16))


@pytest.mark.parametrize(
    "norm_type",
    [
        0,
        9,
    ],
)
def test_unsupported_norm_type(device, norm_type):
    inputs = make_inputs((1, 6, 128), (2, 2, 2), 96, torch.bfloat16, device, "uniform")
    with pytest.raises(RuntimeError, match="norm_type"):
        call_dut(inputs, norm_type=norm_type)


@pytest.mark.parametrize("weight_index", [1, 2])
def test_affine_rmsnorm_requires_weights(device, weight_index):
    inputs = list(make_inputs((1, 6, 128), (2, 2, 2), 96, torch.bfloat16, device, "uniform"))
    inputs[weight_index] = None
    with pytest.raises(RuntimeError, match="required for norm_type=3"):
        call_dut(inputs, norm_type=3)


@pytest.mark.parametrize(
    "case,message",
    [
        ("head_empty", "three"),
        ("head_zero", "each head count"),
        ("head_large", "each head count"),
        ("head_sum", "sum to Htotal"),
        ("head_gqa", "Hk=Hv"),
        ("head_divisibility", "divisible"),
        ("qkv_rank", "qkv must be"),
        ("qkv_empty", "numel"),
        ("qkv_dtype", "float16 or bfloat16"),
        ("qkv_noncontiguous", "contiguous"),
        ("dim_alignment", "head_dim"),
        ("weight_shape", "shape"),
        ("weight_dtype", "same dtype"),
        ("weight_device", "same NPU"),
        ("rope_shape", "identical shapes"),
        ("rope_sequence", "sequence"),
        ("rope_batch", "batch"),
        ("rope_head", "head dimension"),
        ("rope_width", "last dimension"),
        ("rotary_small", "rotary_dim"),
        ("rotary_large", "rotary_dim"),
        ("rotary_alignment", "rotary_dim"),
        ("rope_type", "rope_type"),
        ("eps_zero", "eps"),
        ("eps_nan", "eps"),
        ("eps_underflow", "eps"),
    ],
)
def test_invalid_parameters(device, case, message):
    inputs = list(make_inputs((17, 12, 128), (8, 2, 2), 96, torch.bfloat16, device, "uniform"))
    attrs = {}
    if case.startswith("head_"):
        inputs[5] = {
            "head_empty": [],
            "head_zero": [12, 0, 0],
            "head_large": [13, 1, 1],
            "head_sum": [7, 2, 2],
            "head_gqa": [7, 2, 3],
            "head_divisibility": [2, 5, 5],
        }[case]
    elif case == "qkv_rank":
        inputs[0] = torch.empty((12, 128), dtype=torch.bfloat16, device=device)
    elif case == "qkv_empty":
        inputs[0] = torch.empty((0, 12, 128), dtype=torch.bfloat16, device=device)
    elif case == "qkv_dtype":
        inputs[0] = torch.empty((17, 12, 128), dtype=torch.float32, device=device)
    elif case == "qkv_noncontiguous":
        inputs[0] = torch.empty((17, 12, 256), dtype=torch.bfloat16, device=device)[..., ::2]
    elif case == "dim_alignment":
        inputs[0] = torch.empty((17, 12, 127), dtype=torch.bfloat16, device=device)
    elif case == "weight_shape":
        inputs[1] = torch.empty((1, 128), dtype=torch.bfloat16, device=device)
    elif case == "weight_dtype":
        inputs[1] = torch.empty(128, dtype=torch.float32, device=device)
    elif case == "weight_device":
        inputs[1] = torch.empty(128, dtype=torch.bfloat16, device="cpu")
    elif case == "rope_shape":
        inputs[4] = torch.empty((1, 17, 96), dtype=torch.bfloat16, device=device)
    elif case.startswith("rope_") and case != "rope_type":
        shape = {
            "rope_sequence": (16, 96),
            "rope_batch": (2, 17, 96),
            "rope_head": (1, 17, 2, 96),
            "rope_width": (17, 128),
        }[case]
        inputs[3] = torch.empty(shape, dtype=torch.bfloat16, device=device)
        inputs[4] = torch.empty(shape, dtype=torch.bfloat16, device=device)
    elif case.startswith("rotary_"):
        attrs["rotary_dim"] = {"rotary_small": 0, "rotary_large": 160, "rotary_alignment": 94}[case]
    elif case == "rope_type":
        attrs["rope_type"] = 2
    else:
        attrs["eps"] = {"eps_zero": 0, "eps_nan": float("nan"), "eps_underflow": 1e-100}[case]
    with pytest.raises(RuntimeError, match=message):
        call_dut(inputs, **attrs)


def test_dispatcher_contract(registered_op):
    schema = registered_op._schema
    assert [arg.name for arg in schema.arguments] == [
        "qkv",
        "norm_query_weight",
        "norm_key_weight",
        "rope_cos",
        "rope_sin",
        "head_counts",
        "norm_type",
        "rope_type",
        "rotary_dim",
        "eps",
        "output_bnsd",
    ]
    assert not schema.arguments[6].has_default_value()
    assert not schema.arguments[7].has_default_value()
    assert [arg.default_value for arg in schema.arguments[8:]] == [None, 1e-5, True]
    assert len(schema.returns) == 3
    assert str(schema.arguments[5].type) == "List[int]"
    assert torch.Tag.dynamic_output_shape not in registered_op.tags


@pytest.mark.parametrize(
    "dim",
    [
        32,
        256,
    ],
)
def test_fake_rejects_unsupported_head_dim(registered_op, dim):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        qkv = torch.empty((1, 6, dim), dtype=torch.bfloat16, device="npu:0")
        cache = torch.empty((1, 32), dtype=qkv.dtype, device=qkv.device)
        with pytest.raises(RuntimeError, match="head_dim must be 128"):
            registered_op(qkv, None, None, cache, cache, [2, 2, 2], norm_type=2, rope_type=0, rotary_dim=32)


@pytest.mark.parametrize(
    "eps",
    [
        1e-100,
        1e100,
        -1e-5,
    ],
)
def test_fake_rejects_invalid_float32_eps(registered_op, eps):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        qkv = torch.empty((17, 12, 128), dtype=torch.bfloat16, device="npu:0")
        heads = [8, 2, 2]
        weight = torch.empty(128, dtype=qkv.dtype, device=qkv.device)
        cache = torch.empty((17, 96), dtype=qkv.dtype, device=qkv.device)
        with pytest.raises(RuntimeError, match="eps must be finite and positive after conversion to float32"):
            registered_op(qkv, weight, weight, cache, cache, heads, norm_type=3, rope_type=1, rotary_dim=96, eps=eps)


@pytest.mark.parametrize(
    "batched,output_bnsd,eps,heads",
    [
        (False, False, 1e-5, [2, 2, 2]),
        (False, True, 2.0**-149, [8, 2, 2]),
        (True, False, float.fromhex("0x1.fffffep127"), [8, 1, 1]),
        (True, True, 1e-5, [4, 4, 4]),
    ],
)
def test_fake_explicit_heads(registered_op, batched, output_bnsd, eps, heads):
    from torch._subclasses.fake_tensor import FakeTensorMode

    # No ShapeEnv or data-dependent output-shape opt-in is needed for integer heads.
    with FakeTensorMode():
        shape = (2, 17, sum(heads), 128) if batched else (17, sum(heads), 128)
        qkv = torch.empty(shape, dtype=torch.bfloat16, device="npu:0")
        weight = torch.empty(128, dtype=qkv.dtype, device=qkv.device)
        cache = torch.empty((17, 96), dtype=qkv.dtype, device=qkv.device)
        query, key, value = registered_op(
            qkv,
            weight,
            weight,
            cache,
            cache,
            heads,
            norm_type=3,
            rope_type=1,
            rotary_dim=96,
            eps=eps,
            output_bnsd=output_bnsd,
        )
        head_axis = 1 if output_bnsd else -2
        assert [output.shape[head_axis] for output in (query, key, value)] == heads
        for output in (query, key, value):
            assert output.dtype == qkv.dtype and output.device == qkv.device
            assert output.is_contiguous()
            assert output.shape[-1] == 128
            assert output.shape[-2 if output_bnsd else -3] == 17
            assert output.ndim == (4 if batched or output_bnsd else 3)
            if output_bnsd:
                assert output.shape[0] == (2 if batched else 1)


@pytest.mark.parametrize(
    "norm_type,rope_type,compact",
    [
        (2, 0, False),
        (3, 1, True),
    ],
)
def test_fake_omitted_rotary_dim_uses_full_head(registered_op, norm_type, rope_type, compact):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        qkv = torch.empty((17, 12, 128), dtype=torch.bfloat16, device="npu:0")
        heads = [8, 2, 2]
        weight = torch.empty(128, dtype=qkv.dtype, device=qkv.device) if norm_type == 3 else None
        cache = torch.empty((17, 64 if compact else 128), dtype=qkv.dtype, device=qkv.device)
        outputs = registered_op(qkv, weight, weight, cache, cache, heads, norm_type=norm_type, rope_type=rope_type)
        assert all(output.shape[-1] == 128 for output in outputs)
        assert all(output.ndim == 4 and output.shape[0] == 1 and output.shape[2] == 17 for output in outputs)
        short_cache = torch.empty((17, 96), dtype=qkv.dtype, device=qkv.device)
        with pytest.raises(RuntimeError):
            registered_op(
                qkv, weight, weight, short_cache, short_cache, heads, norm_type=norm_type, rope_type=rope_type
            )


@pytest.mark.parametrize("missing", ["norm_type", "rope_type"])
def test_modes_are_required(registered_op, missing):
    tensor = torch.empty(0)
    attrs = {"norm_type": 3, "rope_type": 1}
    del attrs[missing]
    with pytest.raises(RuntimeError, match=missing):
        registered_op(tensor, tensor, tensor, tensor, tensor, [8, 2, 2], **attrs)


@pytest.mark.parametrize(
    "norm_type",
    [
        4,
    ],
)
def test_fake_rejects_unsupported_norm_type(registered_op, norm_type):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        qkv = torch.empty((17, 12, 128), dtype=torch.bfloat16, device="npu:0")
        heads = [8, 2, 2]
        cache = torch.empty((17, 128), dtype=qkv.dtype, device=qkv.device)
        with pytest.raises(RuntimeError, match="norm_type"):
            registered_op(qkv, None, None, cache, cache, heads, norm_type=norm_type, rope_type=1)


@pytest.mark.parametrize(
    "rope_type",
    [
        -1,
    ],
)
def test_fake_rejects_invalid_rope_type(registered_op, rope_type):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        qkv = torch.empty((17, 12, 128), dtype=torch.bfloat16, device="npu:0")
        heads = [8, 2, 2]
        cache = torch.empty((17, 128), dtype=qkv.dtype, device=qkv.device)
        with pytest.raises(RuntimeError, match="rope_type"):
            registered_op(qkv, None, None, cache, cache, heads, norm_type=2, rope_type=rope_type)


@pytest.mark.parametrize(
    "heads,message",
    [
        ([], "three"),
        ([2, 5, 5], "divisible"),
    ],
)
def test_fake_rejects_invalid_heads(registered_op, heads, message):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        qkv = torch.empty((17, 12, 128), dtype=torch.bfloat16, device="npu:0")
        cache = torch.empty((17, 128), dtype=qkv.dtype, device=qkv.device)
        with pytest.raises(RuntimeError, match=message):
            registered_op(qkv, None, None, cache, cache, heads, norm_type=2, rope_type=1)


@pytest.mark.parametrize("output_bnsd", [False, True])
def test_meta_output_shapes(registered_op, output_bnsd):
    qkv = torch.empty((2, 17, 12, 128), dtype=torch.bfloat16, device="meta")
    cache = torch.empty((17, 128), dtype=qkv.dtype, device="meta")
    outputs = registered_op(qkv, None, None, cache, cache, [8, 2, 2], norm_type=2, rope_type=1, output_bnsd=output_bnsd)
    for output, heads in zip(outputs, (8, 2, 2)):
        assert output.shape == ((2, heads, 17, 128) if output_bnsd else (2, 17, heads, 128))
        assert output.device.type == "meta"
        assert output.is_contiguous()


def test_fake_preserves_symbolic_input_dimensions(registered_op):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    mode = FakeTensorMode(shape_env=ShapeEnv(allow_dynamic_output_shape_ops=False))
    qkv = mode.from_tensor(torch.empty((2, 17, 12, 128), dtype=torch.bfloat16), static_shapes=False)
    cache = mode.from_tensor(torch.empty((17, 128), dtype=torch.bfloat16), static_shapes=False)
    with mode:
        outputs = registered_op(qkv, None, None, cache, cache, [8, 2, 2], norm_type=2, rope_type=1)
    for output, heads in zip(outputs, (8, 2, 2)):
        assert output.shape[1] == heads
        for out_axis, in_axis in ((0, 0), (2, 1)):
            assert isinstance(output.shape[out_axis], torch.SymInt)
            assert output.shape[out_axis].node.expr == qkv.shape[in_axis].node.expr
