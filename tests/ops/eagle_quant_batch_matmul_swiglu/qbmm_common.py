"""Direct QBMM inputs and references. Importing this module does not initialize NPU."""

import hashlib
import json
import math
import random

OP = "eagle_quant_batch_matmul_swiglu"


def scenario(
    m=5672,
    k=5376,
    n=28672,
    distribution="normal",
    seed=20261030,
    transpose_x1=False,
    transpose_x2=False,
    batch=(),
    token=True,
    bias=None,
    output_dtype="bf16",
):
    return dict(
        m=m,
        k=k,
        n=n,
        distribution=distribution,
        seed=seed,
        transpose_x1=transpose_x1,
        transpose_x2=transpose_x2,
        batch=list(batch),
        token=token,
        bias=bias,
        output_dtype=output_dtype,
    )


def preview(s):
    m, k, n = s["m"], s["k"], s["n"]
    return dict(
        scenario=s,
        layout="ND",
        x1=dict(shape=[m, k], dtype="INT8"),
        x2=dict(shape=[k, n], dtype="INT8"),
        scale=dict(shape=[n], dtype="FP32"),
        pertoken_scale=dict(shape=[m], dtype="FP32"),
        output=dict(shape=[m, n // 2], dtype="BF16"),
        offset=None,
        bias=None,
        group_size=0,
    )


def initialize(device):
    import os

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("unset ASCEND_RT_VISIBLE_DEVICES; use physical --device-id")
    import torch
    import torch_npu

    torch.npu.set_device(device)
    import mindiesd  # registration must follow set_device

    return torch, torch_npu


def make_inputs(s):
    import torch

    rng = random.Random(s["seed"])
    gen = torch.Generator(device="cpu").manual_seed(s["seed"])
    parameters = {}

    def sample(name, shape):
        if s["distribution"] == "uniform":
            parameters[name] = dict(distribution="uniform", low=-5, high=5)
            return torch.rand(shape, generator=gen) * 10 - 5
        mean, std = rng.uniform(-5, 5), rng.uniform(0.1, 2)
        parameters[name] = dict(distribution="normal", mean=mean, std=std)
        return torch.randn(shape, generator=gen) * std + mean

    m, k, n = s["m"], s["k"], s["n"]
    # Generate the physical contiguous layouts accepted by the operator.
    a = sample("x1", (*s["batch"], k, m) if s["transpose_x1"] else (*s["batch"], m, k))
    b = sample("x2", (n, k) if s["transpose_x2"] else (k, n))
    a_axis = -2 if s["transpose_x1"] else -1
    b_axis = -1 if s["transpose_x2"] else -2
    a_scale = (a.abs().amax(dim=a_axis, keepdim=True) if s["token"] else a.abs().max()) / 127
    b_scale = b.abs().amax(dim=b_axis, keepdim=True) / 127
    a_scale = a_scale.clamp_min(torch.finfo(torch.float32).tiny)
    b_scale = b_scale.clamp_min(torch.finfo(torch.float32).tiny)
    x1, x2 = (a / a_scale).round().to(torch.int8), (b / b_scale).round().to(torch.int8)
    scale = b_scale.reshape(n).contiguous()
    pertoken = a_scale.reshape(*s["batch"], m).contiguous() if s["token"] else None
    if not s["token"]:
        scale = scale * a_scale
    bias = None
    if s["bias"]:
        bias = sample("bias", (n,))
        bias = (
            bias.round().to(torch.int32)
            if s["bias"] == "int32"
            else bias.to({"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[s["bias"]])
        )
    data = dict(x1=x1, x2=x2, scale=scale, offset=None, bias=bias, pertoken_scale=pertoken)
    parameters["quantization"] = dict(
        method="symmetric absmax / 127; round to nearest even",
        x1_scale_shape=list(a_scale.shape),
        x2_scale_shape=list(b_scale.shape),
        x1_scale_min=a_scale.min().item(),
        x1_scale_max=a_scale.max().item(),
        x2_scale_min=b_scale.min().item(),
        x2_scale_max=b_scale.max().item(),
    )
    return data, parameters


def resolve_output_dtype(s):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16}[s["output_dtype"]]


def to_device(data, device):
    return {k: v.to(f"npu:{device}") if v is not None else None for k, v in data.items()}


def metadata(data, s, parameters):
    return dict(
        scenario=s,
        generation=parameters,
        dut=f"torch.ops.mindiesd.{OP}",
        inputs={
            k: None
            if v is None
            else dict(
                shape=list(v.shape),
                dtype=str(v.dtype),
                device=str(v.device),
                stride=list(v.stride()),
                contiguous=v.is_contiguous(),
                layout="ND",
            )
            for k, v in data.items()
        },
    )


def dut(data, s):
    import torch

    return getattr(torch.ops.mindiesd, OP)(
        **data,
        output_dtype=resolve_output_dtype(s),
        transpose_x1=s["transpose_x1"],
        transpose_x2=s["transpose_x2"],
        group_size=0,
    )


def cpu_golden(data, s):
    import torch

    a, b = data["x1"].to(torch.int32), data["x2"].to(torch.int32)
    if s["transpose_x1"]:
        a = a.transpose(-1, -2)
    if s["transpose_x2"]:
        b = b.transpose(-1, -2)
    # INT32 accumulation, FP32 dequantization and SwiGLU, one final rounding.
    acc = a @ b
    bias = data["bias"]
    if bias is not None and bias.dtype == torch.int32:
        acc = acc + bias
    out = acc.float() * data["scale"].float()
    if data["pertoken_scale"] is not None:
        out = out * data["pertoken_scale"].unsqueeze(-1)
    if bias is not None and bias.dtype != torch.int32:
        out = out + bias.float()
    left, right = out.chunk(2, dim=-1)
    return (left / (1 + torch.exp(-left)) * right).to(resolve_output_dtype(s))


def npu_golden(data, s):
    import torch
    import torch_npu

    a, b = data["x1"], data["x2"]
    if s["transpose_x1"]:
        a = a.transpose(-1, -2)
    if s["transpose_x2"]:
        b = b.transpose(-1, -2)
    # Request raw integer accumulation; BF16/FP16 QBMM would round before SwiGLU.
    acc = torch_npu.npu_quant_matmul(a, b, torch.ones_like(data["scale"]), output_dtype=torch.int32)
    if acc.dtype != torch.int32:
        raise RuntimeError("FP32 SwiGLU golden requires npu_quant_matmul INT32 output support")
    out = acc.float() * data["scale"]
    out = out * data["pertoken_scale"].unsqueeze(-1)
    left, right = out.chunk(2, dim=-1)
    return (left / (1 + torch.exp(-left)) * right).to(resolve_output_dtype(s))


def contract(out, s, finite=True):
    import torch

    expected = (*s["batch"], s["m"], s["n"] // 2)
    if tuple(out.shape) != expected or out.dtype != resolve_output_dtype(s) or not out.is_contiguous():
        raise AssertionError(
            f"output contract: {out.shape}, {out.dtype}, contiguous={out.is_contiguous()}; expected {expected}"
        )
    if finite and not torch.isfinite(out).all().item():
        raise AssertionError("output contains NaN/Inf")


def fingerprint(tensor):
    import torch

    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def input_hashes(data):
    return {k: fingerprint(v) if v is not None else None for k, v in data.items()}


def bit_difference(actual, baseline):
    import torch

    a, b = actual.cpu().contiguous(), baseline.cpu().contiguous()
    byte_diff = a.view(torch.uint8).reshape(-1, a.element_size()) != b.view(torch.uint8).reshape(-1, b.element_size())
    changed = byte_diff.any(dim=1)
    count = changed.sum().item()
    max_rel = 0.0
    af, bf = a.flatten(), b.flatten()
    for start in range(0, a.numel(), 1 << 20):
        av, bv = af[start : start + (1 << 20)].double(), bf[start : start + (1 << 20)].double()
        max_rel = max(max_rel, ((av - bv).abs() / bv.abs().clamp_min(1e-300)).max().item())
    return dict(
        changed_elements=count,
        changed_ratio=count / a.numel(),
        max_rel=max_rel if math.isfinite(max_rel) else str(max_rel),
        first_changed_flat_index=changed.nonzero()[0].item() if count else None,
    )


def compare(actual, golden, s, large=False):
    import torch

    contract(actual, s)
    contract(golden, s)
    # Chunk FP64 statistics to keep production comparisons bounded in memory.
    a, g = actual.detach().cpu().flatten(), golden.detach().cpu().flatten()
    atol = rtol = 2 ** (-7 if s["output_dtype"] == "bf16" else -10)
    dot = an = gn = matched = 0
    max_abs, max_index = -1.0, 0
    for start in range(0, a.numel(), 1 << 20):
        av, gv = a[start : start + (1 << 20)].double(), g[start : start + (1 << 20)].double()
        error = (av - gv).abs()
        value, pos = error.max(dim=0)
        if value.item() > max_abs:
            max_abs, max_index = value.item(), start + pos.item()
        matched += (error <= atol + rtol * gv.abs()).sum().item()
        dot += (av * gv).sum().item()
        an += (av * av).sum().item()
        gn += (gv * gv).sum().item()
    cosine = dot / math.sqrt(an * gn) if an and gn else float(an == gn)
    ratio = math.sqrt(an / gn) if gn else (1.0 if not an else float("inf"))
    anchor = g[max_index].abs()
    ulp = (torch.nextafter(anchor, torch.full_like(anchor, float("inf"))) - anchor).item()
    limit = max(0.1, 32 * ulp)
    match_ratio = matched / a.numel()
    mixed_pass = match_ratio >= 0.99 and max_abs <= limit
    hard_pass = cosine >= 0.99 and 0.9 <= ratio <= 1.1 and (large or mixed_pass)
    coord, index = [], max_index
    for size in reversed(golden.shape):
        coord.append(index % size)
        index //= size
    result = dict(
        output="y",
        tolerance_dtype=s["output_dtype"],
        output_dtype=str(actual.dtype),
        basis="shared INT8 inputs; INT32 accumulation; FP32 dequantization and SwiGLU; final output rounding",
        atol=atol,
        rtol=rtol,
        required_matched_ratio=0.99,
        fallback=0.1,
        matched_ratio=match_ratio,
        max_abs=max_abs,
        max_error_coordinate=list(reversed(coord)),
        golden_at_max=g[max_index].item(),
        actual_at_max=a[max_index].item(),
        ulp=ulp,
        ulp32=32 * ulp,
        max_abs_limit=limit,
        cosine=cosine,
        norm_ratio=ratio if math.isfinite(ratio) else str(ratio),
        cosine_pass=cosine >= 0.99,
        norm_ratio_pass=0.9 <= ratio <= 1.1,
        mixed_pass=mixed_pass,
        mixed_role="soft" if large else "hard",
        passed=hard_pass,
    )
    print(json.dumps(result), flush=True)
    return result
