# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Numerical contract and matched workloads for the Ascend 950 gate and MX quantizer."""

import math
import random

import torch


SHAPE = (8068, 1536)
SEED = 20261123


def make_inputs(shape, distribution, seed=SEED):
    rng = random.Random(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = []
    for name in ("x", "gate"):
        sample = torch.empty(shape, dtype=torch.float32, device="cpu")
        if distribution == "normal":
            mean, std = rng.uniform(-5, 5), rng.uniform(0.1, 2)
            sample.normal_(mean, std, generator=generator)
            print(f"{name} normal mean={mean:.6f} std={std:.6f} seed={seed}", flush=True)
        else:
            sample.uniform_(-5, 5, generator=generator)
        inputs.append(sample.bfloat16())
    return inputs[0], inputs[1]


def fused(x, gate, mode, scale_alg=0):
    return torch.ops.mindiesd.sigmoid_mul_dynamic_mx_quant(x, gate, mode, scale_alg=scale_alg)


def product(x, gate, mode):
    if mode == 0:
        return x.float() * torch.sigmoid(gate.float())
    return x * torch.sigmoid(gate)


def quantize(value, scale_alg):
    import torch_npu

    return torch_npu.npu_dynamic_mx_quant(value, dst_type=torch.float8_e4m3fn, block_size=32, scale_alg=scale_alg)


def reference(x, gate, mode, scale_alg=0):
    return quantize(product(x, gate, mode), scale_alg)


def cpu_reference(x, gate, mode, scale_alg=0):
    """Shared exponent for finite inputs. Quantize each row independently."""
    value = product(x, gate, mode)
    rows, cols = value.shape
    groups = (cols + 63) // 64 * 2
    padded = torch.zeros(rows, groups * 32, dtype=value.dtype)
    padded[:, :cols] = value
    blocks = padded.reshape(rows, groups, 32)
    maximum = blocks.abs().float().amax(dim=-1)
    if scale_alg == 1:
        # The native E4M3FN path multiplies by the FP32 reciprocal of 448 before rounding the scale upward.
        scaled = maximum * (1.0 / 448.0)
        mantissa, exponent = torch.frexp(scaled)
        biased = (exponent + 127 - (mantissa == 0.5).int()).clamp(0, 254)
        biased = torch.where(scaled < torch.finfo(torch.float32).tiny, (scaled > 2.0**-127).int(), biased)
    else:
        _, exponent = torch.frexp(maximum)
        biased = (exponent + 118).clamp(0, 254)
    biased = torch.where(maximum == 0, 0, biased)
    reciprocal = torch.exp2((127 - biased).float())
    zero_recip = maximum == 0 if scale_alg == 1 else biased == 0
    reciprocal = torch.where(zero_recip, 0.0, reciprocal).to(value.dtype)
    encoded = (blocks * reciprocal.unsqueeze(-1)).float().clamp(-448, 448).to(torch.float8_e4m3fn)
    return encoded.reshape(rows, -1)[:, :cols].contiguous(), biased.byte().reshape(rows, groups // 2, 2)


def metadata(actual, shape, device):
    rows, cols = shape
    expected = [(shape, torch.float8_e4m3fn), ((rows, (cols + 63) // 64, 2), torch.uint8)]
    if len(actual) != len(expected):
        raise AssertionError("expected two outputs")
    for tensor, (size, dtype) in zip(actual, expected):
        if tuple(tensor.shape) != tuple(size) or tensor.dtype != dtype or tensor.device != device:
            raise AssertionError(f"unexpected output {tensor.shape} {tensor.dtype} {tensor.device}")
        if not tensor.is_contiguous():
            raise AssertionError("expected contiguous outputs")


def compare(actual, expected, *, small):
    if not torch.equal(actual[1], expected[1]):
        raise AssertionError(f"E8M0 mismatches: {(actual[1] != expected[1]).sum().item()}")
    x, y = actual[0].float().flatten(), expected[0].float().flatten()
    if not torch.isfinite(x).all().item() or not torch.isfinite(y).all().item():
        raise AssertionError("nonfinite FP8 output for finite ordinary inputs")
    diff = (x - y).abs()
    index = int(diff.argmax().item())
    maximum, ref = diff[index].item(), y[index].item()
    spacing = 2**-9 if ref == 0 else max(2**-9, 2 ** (math.frexp(abs(ref))[1] - 4))
    match = (diff <= 0.125 + 0.125 * y.abs()).float().mean().item()
    xn, yn = torch.linalg.vector_norm(x).item(), torch.linalg.vector_norm(y).item()
    cosine = torch.dot(x / xn, y / yn).item() if xn and yn else float(xn == yn)
    ratio = xn / yn if yn else float(xn == yn)
    exact = (actual[0].view(torch.uint8) == expected[0].view(torch.uint8)).float().mean().item()
    print(
        f"E8M0 exact; E4M3FN exact={exact:.9f} match={match:.9f} max_abs={maximum} "
        f"index={index} golden={ref} ulp={spacing} cap={max(1, 32 * spacing)} "
        f"cosine={cosine:.9f} norm_ratio={ratio:.9f} mixed_tolerance_hard={small}",
        flush=True,
    )
    if cosine < 0.99 or not 0.9 <= ratio <= 1.1:
        raise AssertionError("FP8 cosine or norm ratio outside tolerance")
    if small and (match < 0.99 or maximum > max(1.0, 32 * spacing)):
        raise AssertionError("FP8 mixed tolerance outside limits")
