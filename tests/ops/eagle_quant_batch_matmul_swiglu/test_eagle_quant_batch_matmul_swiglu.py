import json
import pytest
from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_common import (
    compare,
    cpu_golden,
    dut,
    initialize,
    make_inputs,
    metadata,
    npu_golden,
    scenario,
    to_device,
)


CASES = [
    scenario(
        m=128 + (i & 1),
        n=4096,
        k=256 + ((i >> 2) & 1),
        transpose_x1=bool(i & 2),
        transpose_x2=bool(i & 4),
        distribution="normal" if i % 2 else "uniform",
        seed=20261030 + i,
        bias=None,
    )
    for i in range(8)
]
CASES += [
    scenario(
        m=129,
        k=257,
        n=4096,
        distribution="normal" if i % 2 else "uniform",
        seed=20261100 + i,
        transpose_x1=bool(i & 1),
        transpose_x2=bool(i & 2),
        output_dtype="fp16",
        bias=None,
    )
    for i in range(4)
]
# Full channel tiles exercise the CV 1:2 specialization with every transpose combination.
CASES += [
    scenario(
        m=129,
        k=257,
        n=4096,
        seed=20261340 + i,
        transpose_x1=bool(i & 1),
        transpose_x2=bool(i & 2),
        output_dtype="bf16" if i < 4 else "fp16",
    )
    for i in range(8)
]
# Half-N tails cover FP32/FP16 row padding and partial 64-element vector masks.
CASES += [
    scenario(
        m=129,
        k=33,
        n=2 * half_n,
        seed=20261400 + i,
        transpose_x1=bool(i & 1),
        transpose_x2=bool(i & 2),
        output_dtype="bf16" if i < 4 else "fp16",
    )
    for half_n in (1, 7, 8, 9, 15, 16, 17, 63, 65, 80, 96, 129, 193, 257, 513)
    for i in range(8)
]


@pytest.fixture(name="device", scope="module")
def fixture_device(request):
    from tests.ops.device_support import init_test_device, require_support
    from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_perf import identity
    from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_state import read, save

    path = request.config._qbmm_batch.path / "device.json"
    index = request.config._qbmm_batch.plan["parameters"]["device"]
    initialize(index)
    init_test_device()
    require_support("eagle_quant_batch_matmul_swiglu")
    current = identity(index)
    if path.exists():
        if read(path) != current:
            raise RuntimeError("small-shape resume cannot change physical NPU or host")
    else:
        save(path, current)
    return index


@pytest.mark.parametrize(
    "s",
    CASES,
    ids=lambda s: "_".join(
        str(s[k])
        for k in ("m", "k", "n", "output_dtype", "distribution", "transpose_x1", "transpose_x2", "bias", "batch")
    ),
)
def test_small_cpu_golden(s, device, request):
    import torch

    cpu, parameters = make_inputs(s)
    if s["output_dtype"] == "fp16":
        # Bound the final product while retaining positive/negative activation inputs.
        cpu["scale"] *= 0.01
        parameters["channel_scale_multiplier"] = 0.01
    inputs = to_device(cpu, device)
    print(json.dumps(metadata(inputs, s, parameters)), flush=True)
    request.node.user_properties.append(("inputs", metadata(inputs, s, parameters)))
    actual = dut(inputs, s)
    torch.npu.synchronize()
    metrics = compare(actual, cpu_golden(cpu, s), s)
    request.node.user_properties.append(("metrics", metrics))
    assert metrics["passed"]


@pytest.mark.parametrize("batch_kind", ["a", "b", "both"])
def test_paired_batch_shared_token_scale(batch_kind, device, request):
    import torch

    s = scenario(m=129, k=257, n=4096, seed=20261320)
    cpu, parameters = make_inputs(s)
    for name, kind in (("x1", "a"), ("x2", "b")):
        if batch_kind in (kind, "both"):
            cpu[name] = torch.stack([cpu[name].roll(i, dims=-1) for i in range(3)])
    # The existing QBMM contract shares a one-dimensional [M] token scale across batches.
    s = dict(s, batch=[3])
    inputs = to_device(cpu, device)
    request.node.user_properties.append(("inputs", metadata(inputs, s, parameters)))
    actual = dut(inputs, s)
    torch.npu.synchronize()
    metrics = compare(actual, cpu_golden(cpu, s), s)
    request.node.user_properties.append(("metrics", metrics))
    assert metrics["passed"]


@pytest.mark.parametrize("output_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("n", [4096, 8192])
@pytest.mark.parametrize("m", [1, 21, 22, 23, 43, 44, 45, 65, 66, 67, 127, 128, 129, 2049])
def test_paired_dequant_distinct_scales(output_dtype, n, m, device, request):
    import torch

    # Whole 22-row tiles, idle-AIV tails, multiple pairs and distinct left/right scales.
    s = scenario(m=m, k=33, n=n, output_dtype=output_dtype, seed=20261330)
    cpu, parameters = make_inputs(s)
    half_n = s["n"] // 2
    cpu["scale"][:half_n] = torch.linspace(0.0001, 0.0002, half_n)
    cpu["scale"][half_n:] = torch.linspace(0.0007, 0.0009, half_n)
    cpu["pertoken_scale"] = torch.linspace(0.01, 0.05, s["m"])
    parameters["paired_scales"] = dict(left=[0.0001, 0.0002], right=[0.0007, 0.0009], token=[0.01, 0.05])
    inputs = to_device(cpu, device)
    request.node.user_properties.append(("inputs", metadata(inputs, s, parameters)))
    actual = dut(inputs, s)
    torch.npu.synchronize()
    metrics = compare(actual, cpu_golden(cpu, s), s)
    request.node.user_properties.append(("metrics", metrics))
    assert metrics["passed"]


@pytest.mark.parametrize("bias_dtype", ["int32", "fp32", "bf16", "fp16"])
@pytest.mark.parametrize("token,n", [(True, 4096)])
def test_reject_bias_after_no_bias_call(bias_dtype, token, n, device):
    import torch

    s = scenario(
        m=129,
        k=257,
        n=n,
        token=token,
        bias=bias_dtype,
        output_dtype="fp16" if bias_dtype == "fp16" and token else "bf16",
    )
    cpu, _ = make_inputs(s)
    inputs = to_device(cpu, device)
    bias = inputs["bias"]
    inputs["bias"] = None
    dut(inputs, s)
    torch.npu.synchronize()
    # A prior successful tiling/cache lookup must not bypass the bias rejection.
    inputs["bias"] = bias
    with pytest.raises(RuntimeError):
        dut(inputs, s)
        torch.npu.synchronize()


@pytest.mark.parametrize(
    "invalid",
    [
        "input_dtype",
        "k",
        "group_size",
        "output_dtype",
        "noncontiguous",
        "batch",
        "offset",
        "token_missing",
        "token_dtype",
        "scale_dtype",
        "odd_n",
        "int8_output",
    ],
)
def test_reject_invalid(invalid, device):
    import torch

    s = scenario(m=128, n=4096, k=256)
    cpu, _ = make_inputs(s)
    inputs = to_device(cpu, device)
    extra = {}
    if invalid == "input_dtype":
        inputs["x1"] = inputs["x1"].float()
    elif invalid == "k":
        inputs["x2"] = inputs["x2"][:-1].contiguous()
    elif invalid == "group_size":
        extra["group_size"] = 1
    elif invalid == "output_dtype":
        extra["output_dtype"] = torch.float32
    elif invalid == "noncontiguous":
        inputs["x1"] = inputs["x1"].t()
    elif invalid == "offset":
        inputs["offset"] = torch.zeros(128, device="npu:0")
    elif invalid == "token_missing":
        inputs["pertoken_scale"] = None
    elif invalid == "token_dtype":
        inputs["pertoken_scale"] = inputs["pertoken_scale"].half()
    elif invalid == "scale_dtype":
        inputs["scale"] = inputs["scale"].bfloat16()
    elif invalid == "odd_n":
        inputs["x2"] = inputs["x2"][:, :-1].contiguous()
        inputs["scale"] = inputs["scale"][:-1].contiguous()
    elif invalid == "int8_output":
        extra["output_dtype"] = torch.int8
    else:
        inputs["x1"] = inputs["x1"].expand(2, -1, -1).contiguous()
        inputs["x2"] = inputs["x2"].expand(3, -1, -1).contiguous()
    with pytest.raises(RuntimeError):
        torch.ops.mindiesd.eagle_quant_batch_matmul_swiglu(**inputs, **extra)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_npu_golden_against_cpu(dtype, device):
    import torch

    s = scenario(m=22, k=33, n=4096, output_dtype=dtype)
    cpu, _ = make_inputs(s)
    cpu["scale"] *= 0.01
    reference = npu_golden(to_device(cpu, device), s).cpu()
    torch.testing.assert_close(reference, cpu_golden(cpu, s), rtol=0.01, atol=0.001)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_swiglu_without_intermediate_rounding(dtype, device):
    import torch

    s = scenario(m=128, k=256, n=4096, output_dtype=dtype)
    cpu, _ = make_inputs(s)
    cpu["x1"].zero_()
    cpu["x1"][:, 0] = 1
    cpu["x2"].fill_(1)
    cpu["scale"][:2048] = torch.linspace(-4.123, 4.789, 2048)
    cpu["scale"][2048:] = torch.linspace(0.117, 2.319, 2048)
    cpu["pertoken_scale"].fill_(1)
    golden = cpu_golden(cpu, s)
    out_dtype = golden.dtype
    a = cpu["scale"][:2048].to(out_dtype).float()
    b = cpu["scale"][2048:].to(out_dtype).float()
    rounded = (a / (1 + torch.exp(-a)) * b).to(out_dtype).expand_as(golden)
    staged_error = (rounded.float() - golden.float()).square().sum()
    assert staged_error > 0, "fixture must expose premature BF16/FP16 rounding"
    actual = dut(to_device(cpu, device), s).cpu()
    error = (actual.float() - golden.float()).square().sum()
    assert error < staged_error * 0.1


@pytest.mark.parametrize("n", [2, 258, 4096])
def test_fake_swiglu_output(n, device):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        target = f"npu:{device}"
        x1 = torch.empty((2, 128, 256), dtype=torch.int8, device=target)
        x2 = torch.empty((n, 256), dtype=torch.int8, device=target)
        scale = torch.empty(n, device=target)
        token = torch.empty(128, device=target)
        out = torch.ops.mindiesd.eagle_quant_batch_matmul_swiglu(x1, x2, scale, pertoken_scale=token, transpose_x2=True)
        assert out.shape == (2, 128, n // 2)
        assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("trans", range(4))
@pytest.mark.parametrize("output_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "m,k,n,batch_kind", [(1, 128, 576, "none"), (129, 257, 640, "both"), (65, 33, 192, "b"), (23, 64, 4096, "a")]
)
def test_nz_weight_cpu_golden(trans, output_dtype, m, k, n, batch_kind, device, request):
    import torch
    import torch_npu

    if n == 192 and trans & 1:
        n = 96  # Transposed NZ accepts half-N=48, aligned to 16 but not 32.
    s = scenario(
        m=m,
        k=k,
        n=n,
        output_dtype=output_dtype,
        transpose_x1=bool(trans & 2),
        transpose_x2=bool(trans & 1),
        seed=20261500 + trans,
    )
    cpu, parameters = make_inputs(s)
    cpu["scale"] *= 0.01
    parameters["channel_scale_multiplier"] = 0.01
    for name, kind in (("x1", "a"), ("x2", "b")):
        if batch_kind in (kind, "both"):
            cpu[name] = torch.stack([cpu[name].roll(i, dims=-1) for i in range(3)])
    if batch_kind != "none":
        s = dict(s, batch=[3])
    inputs = to_device(cpu, device)
    # Same logical weights: separate arithmetic failures from NZ descriptor/layout failures.
    expected = cpu_golden(cpu, s)
    nd_actual = dut(inputs, s)
    torch.npu.synchronize()
    nd_metrics = compare(nd_actual, expected, s)
    request.node.user_properties.append(("nd_metrics", nd_metrics))
    assert nd_metrics["passed"], "ND control failed before NZ conversion"
    inputs["x2"] = torch_npu.npu_format_cast(inputs["x2"], 29)
    assert torch_npu.get_npu_format(inputs["x2"]) == 29
    details = metadata(inputs, s, parameters)
    details["inputs"]["x2"]["layout"] = "FRACTAL_NZ"
    request.node.user_properties.append(("inputs", details))
    actual = dut(inputs, s)
    torch.npu.synchronize()
    metrics = compare(actual, expected, s)
    request.node.user_properties.append(("metrics", metrics))
    assert metrics["passed"]
    format_metrics = compare(actual, nd_actual, s)
    request.node.user_properties.append(("nz_vs_nd_metrics", format_metrics))
    assert format_metrics["passed"]


@pytest.mark.parametrize("transpose_x2", [False, True])
def test_reject_nz_half_inside_fractal(transpose_x2, device):
    import torch_npu

    s = scenario(m=23, k=33, n=66, transpose_x2=transpose_x2)
    cpu, _ = make_inputs(s)
    inputs = to_device(cpu, device)
    inputs["x2"] = torch_npu.npu_format_cast(inputs["x2"], 29)
    with pytest.raises(RuntimeError, match="NZ weights require"):
        dut(inputs, s)
