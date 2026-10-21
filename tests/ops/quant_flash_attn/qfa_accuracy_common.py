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

"""Direct MXFP4 QFA inputs, CPU stage reference and NPU attention reference."""

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from qfa_reference import attention_rows, bf16_round, decode_mxfp4, quantize_mxfp4

REPO_ROOT = Path(__file__).resolve().parents[3]
DISTRIBUTIONS = ("uniform", "normal")


@dataclass(frozen=True)
class Case:
    batch: int = 1
    q_heads: int = 32
    kv_heads: int = 4
    q_seq: int = 2048
    kv_seq: int = 32768
    dim: int = 128
    layout: str = "BNSD"
    length_mode: str = "used"
    q_used: tuple = ()
    kv_used: tuple = ()

    def __post_init__(self):
        if min(self.batch, self.q_heads, self.kv_heads, self.q_seq, self.kv_seq) <= 0:
            raise ValueError("QFA dimensions must be positive")
        if self.dim != 128 or self.q_heads % self.kv_heads or self.layout not in ("BNSD", "BSND"):
            raise ValueError("Require D=128, integral GQA and BNSD/BSND")
        if self.length_mode not in ("used", "max"):
            raise ValueError("length_mode must be used or max")
        for name, length in (("q_used", self.q_seq), ("kv_used", self.kv_seq)):
            values = tuple(getattr(self, name)) or (length,) * self.batch
            if len(values) != self.batch or any(n <= 0 or n > length for n in values):
                raise ValueError(f"{name} must contain B positive lengths no greater than {length}")
            if self.length_mode == "max" and len(set(values)) != 1:
                raise ValueError("max sequence length requires equal effective lengths across batches")
            object.__setattr__(self, name, values)

    @property
    def physical_q(self):
        return self.q_seq

    @property
    def physical_kv(self):
        return self.kv_seq

    @property
    def scale(self):
        return self.dim**-0.5

    def shape(self, heads, length, dim=128):
        return (self.batch, heads, length, dim) if self.layout == "BNSD" else (self.batch, length, heads, dim)

    @property
    def label(self):
        return f"{self.layout}_B{self.batch}_N{self.q_heads}-{self.kv_heads}_S{self.q_seq}-{self.kv_seq}_D128_MXFP4"


def initialize(device_id):
    if device_id < 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("Use a nonnegative physical --device-id and unset ASCEND_RT_VISIBLE_DEVICES")
    import torch
    import torch_npu

    torch.npu.set_device(device_id)
    sys.path.insert(0, str(REPO_ROOT))
    import mindiesd  # Register the plugin only after selecting the device.

    if REPO_ROOT not in Path(mindiesd.__file__).resolve().parents:
        raise RuntimeError(f"Expected mindiesd from {REPO_ROOT}, got {mindiesd.__file__}")
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    return torch, torch_npu, f"npu:{device_id}"


def make_inputs(case, distribution, seed, torch, device):
    """Generate the physical layout directly; packing/scales are input encoding."""
    rng = np.random.default_rng(seed)
    native, encodings, parameters = {}, {}, {}
    seq_axis = 2 if case.layout == "BNSD" else 1
    for name, heads, length in (
        ("q", case.q_heads, case.q_seq),
        ("k", case.kv_heads, case.kv_seq),
        ("v", case.kv_heads, case.kv_seq),
    ):
        shape = case.shape(heads, length)
        if distribution == "uniform":
            values = rng.uniform(-5, 5, shape).astype(np.float32)
            parameters[name] = {"low": -5, "high": 5}
        elif distribution == "normal":
            mean, std = float(rng.uniform(-5, 5)), float(rng.uniform(0.1, 2))
            values = rng.normal(mean, std, shape).astype(np.float32)
            parameters[name] = {"mean": mean, "std": std}
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        encoded = quantize_mxfp4(values, seq_axis if name == "v" else -1)
        encodings[name] = encoded
        native[name] = torch.from_numpy(encoded.packed).to(device)
        scales = encoded.scales
        if name == "q" and case.layout == "BSND":
            # Q scales have the independent BNGSD contract even for BSND data.
            scales = scales.swapaxes(1, 2)
        elif name == "v":
            padding = [(0, 0)] * 4
            padding[seq_axis] = (0, scales.shape[seq_axis] % 2)
            scales = np.pad(scales, padding, constant_values=127)
            if case.layout == "BNSD":
                scales = scales.reshape(case.batch, heads, -1, 2, 128).transpose(0, 1, 2, 4, 3)
            else:
                scales = scales.reshape(case.batch, -1, 2, heads, 128).transpose(0, 1, 3, 4, 2)
        native[name + "_scale"] = torch.from_numpy(np.ascontiguousarray(scales)).to(device)
    print(json.dumps({"distribution": distribution, "seed": seed, "generation": parameters}), flush=True)
    return native, encodings


def tensor_info(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
    }


def bind(case, native, torch, torch_npu, device):
    options = dict(
        q_quant_mode=3,
        k_quant_mode=3,
        v_quant_mode=3,
        q_dtype=torch_npu.float4_e2m1fn_x2,
        k_dtype=torch_npu.float4_e2m1fn_x2,
        v_dtype=torch_npu.float4_e2m1fn_x2,
        mask_mode=0,
        win_left=2147483647,
        win_right=2147483647,
        layout_q=case.layout,
        layout_kv=case.layout,
        layout_out=case.layout,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
    )
    if case.length_mode == "used":
        options["seqused_q"] = torch.tensor(case.q_used, dtype=torch.int32, device=device)
        options["seqused_kv"] = torch.tensor(case.kv_used, dtype=torch.int32, device=device)
    else:
        options.update(max_seqlen_q=case.q_used[0], max_seqlen_kv=case.kv_used[0])
    metadata = torch.ops.mindiesd.quant_flash_attn_metadata(
        num_heads_q=case.q_heads, num_heads_kv=case.kv_heads, head_dim=128, batch_size=case.batch, **options
    )
    inputs = dict(native, metadata=metadata)
    inputs.update({name: value for name, value in options.items() if isinstance(value, torch.Tensor)})
    dut_options = dict(
        softmax_scale=case.scale,
        return_softmax_lse=0,
        softmax_precision=0,
        quant_block_size_qs=128,
        quant_block_size_ks=256,
        quant_block_size_vs=256,
        q_descale_dtype=torch_npu.float8_e8m0fnu,
        k_descale_dtype=torch_npu.float8_e8m0fnu,
        v_descale_dtype=torch_npu.float8_e8m0fnu,
    )
    print(
        json.dumps(
            {
                "case": asdict(case),
                "inputs": {n: tensor_info(t) for n, t in inputs.items()},
                "attributes": {n: v for n, v in options.items() if not isinstance(v, torch.Tensor)},
                "dut_attributes": dut_options,
                "scale_layout": {
                    "q": "BNS4",
                    "k": case.layout[:-1] + "4",
                    "v": "BN(S/64)D2" if case.layout == "BNSD" else "B(S/64)ND2",
                },
                "softmax_scale": case.scale,
                "mx_block_size": 32,
                "logical_qkv_dtype": "E2M1",
                "logical_scale_dtype": "E8M0",
            }
        ),
        flush=True,
    )
    torch.npu.synchronize()

    def invoke():
        return torch.ops.mindiesd.quant_flash_attn(
            *(native[n] for n in ("q", "k", "v", "q_scale", "k_scale", "v_scale")),
            metadata=metadata,
            **dut_options,
            **options,
        )

    return invoke, inputs


def check_outputs(outputs, case, torch, device):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise AssertionError("QFA must return (attention_out, softmax_lse)")
    for name, value, shape, dtype in (
        ("attention_out", outputs[0], case.shape(case.q_heads, case.q_seq), torch.bfloat16),
        ("softmax_lse", outputs[1], (0,), torch.float32),
    ):
        if tuple(value.shape) != shape or value.dtype != dtype or str(value.device) != device:
            raise AssertionError(f"{name}: expected {shape}/{dtype}/{device}, got {tensor_info(value)}")
        if not value.is_contiguous() or not torch.isfinite(value).all().item():
            raise AssertionError(f"{name}: expected contiguous finite output, got {tensor_info(value)}")


def cpu_golden(case, encodings, torch):
    decoded = {name: decode_mxfp4(value) for name, value in encodings.items()}
    if case.layout == "BSND":
        decoded = {name: value.swapaxes(1, 2) for name, value in decoded.items()}
    rows = [(b, h, s) for b in range(case.batch) for h in range(case.q_heads) for s in range(case.q_used[b])]
    values = attention_rows(case, decoded, rows)
    output = np.zeros(case.shape(case.q_heads, case.q_seq), dtype=np.float32)
    for index, (b, h, s) in enumerate(rows):
        output[(b, h, s) if case.layout == "BNSD" else (b, s, h)] = values[index]
    return torch.from_numpy(output).to(torch.bfloat16)


def npu_decode(case, native, name, torch):
    """Decode the actual device bytes; no CPU attention or CPU dequantization."""
    packed = native[name].to(torch.int32)
    shape = list(packed.shape)
    shape[-1] *= 2
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(shape)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=packed.device)
    values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    scales = native[name + "_scale"]
    if name == "q" and case.layout == "BSND":
        scales = scales.permute(0, 2, 1, 3)
    if name == "v":
        if case.layout == "BNSD":
            scales = scales.permute(0, 1, 2, 4, 3).reshape(case.batch, case.kv_heads, -1, 128)
            scales = scales.repeat_interleave(32, dim=2)[:, :, : case.kv_seq]
        else:
            scales = scales.permute(0, 1, 4, 2, 3).reshape(case.batch, -1, case.kv_heads, 128)
            scales = scales.repeat_interleave(32, dim=1)[:, : case.kv_seq]
    else:
        scales = scales.repeat_interleave(32, dim=-1)
    values *= torch.exp2(scales.float() - 127)
    return values if case.layout == "BNSD" else values.permute(0, 2, 1, 3)


def npu_golden(case, native, torch):
    """FP32 online softmax + matmul, bounded Q/KV blocks on the selected NPU."""
    q, k, v = (npu_decode(case, native, n, torch) for n in ("q", "k", "v"))
    output = torch.zeros(case.shape(case.q_heads, case.q_seq), dtype=torch.bfloat16, device=q.device)
    for b in range(case.batch):
        for h in range(case.q_heads):
            kh = h // (case.q_heads // case.kv_heads)
            for qs in range(0, case.q_used[b], 128):
                qe = min(qs + 128, case.q_used[b])
                maximum = torch.full((qe - qs, 1), -float("inf"), device=q.device)
                denom = torch.zeros_like(maximum)
                acc = torch.zeros((qe - qs, 128), device=q.device)
                for ks in range(0, case.kv_used[b], 2048):
                    ke = min(ks + 2048, case.kv_used[b])
                    score = (q[b, h, qs:qe] @ k[b, kh, ks:ke].T) * case.scale
                    new_max = torch.maximum(maximum, score.amax(dim=-1, keepdim=True))
                    correction = torch.exp(maximum - new_max)
                    prob = torch.exp(score - new_max)
                    acc = acc * correction + prob @ v[b, kh, ks:ke]
                    denom = denom * correction + prob.sum(dim=-1, keepdim=True)
                    maximum = new_max
                result = (acc / denom).to(torch.bfloat16)
                if case.layout == "BNSD":
                    output[b, h, qs:qe] = result
                else:
                    output[b, qs:qe, h] = result
    return output


def fingerprints(inputs, torch):
    return {
        name: hashlib.sha256(
            tensor.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        for name, tensor in inputs.items()
    }


def compare(actual, reference, budget, *, small):
    """Compare original BF16 outputs using an explicit operator-specific budget."""
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise AssertionError(
            f"Output contract mismatch: {actual.shape}/{actual.dtype} vs {reference.shape}/{reference.dtype}"
        )
    a = actual.detach().float().cpu().numpy().astype(np.float64)
    g = reference.detach().float().cpu().numpy().astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(g).all():
        raise AssertionError("attention_out: nonfinite DUT or golden value")
    delta = np.abs(a - g)
    position = tuple(int(n) for n in np.unravel_index(int(delta.argmax()), delta.shape))
    g_low = float(bf16_round(np.array([g[position]], dtype=np.float32))[0])
    # BF16 subnormals have a fixed 2^-133 spacing.
    ulp = 2.0 ** max(-133, int(np.floor(np.log2(abs(g_low)))) - 7) if g_low else 2.0**-133
    maximum_limit = max(budget["fallback"], budget["ulp_factor"] * ulp)
    matched = float(np.mean(delta <= budget["atol"] + budget["rtol"] * np.abs(g)))
    an, gn = float(np.linalg.norm(a)), float(np.linalg.norm(g))
    cosine = float(np.sum(a * g) / (an * gn)) if an and gn else float(an == gn)
    ratio = an / gn if gn else (1.0 if an == 0 else float("inf"))
    gates = {
        "matched_ratio": matched >= budget["matched_ratio"],
        "max_abs": float(delta.max()) <= maximum_limit,
        "cosine": cosine >= 0.99,
        "norm_ratio": 0.9 <= ratio <= 1.1,
    }
    report = {
        "output": "attention_out",
        "output_dtype": str(actual.dtype),
        "tolerance_dtype": "FLOAT4 E2M1 (local provisional tier)",
        "budget": budget,
        "tolerance_reason": "contract includes internal E2M1 probability quantization",
        "golden": "CPU MXFP4 stage simulation" if small else "NPU FP32 online attention on decoded MXFP4",
        "mixed_tolerance_role": "hard" if small else "soft",
        "matched_ratio": matched,
        "max_abs": float(delta.max()),
        "max_position": position,
        "golden_at_max": float(g[position]),
        "golden_low_at_max": g_low,
        "output_ulp": ulp,
        "ulp_term": budget["ulp_factor"] * ulp,
        "max_abs_limit": maximum_limit,
        "cosine": cosine,
        "norm_ratio": ratio,
        "gates": gates,
    }
    print(json.dumps(report), flush=True)
    failed = [name for name, passed in gates.items() if not passed and (small or name in ("cosine", "norm_ratio"))]
    if failed:
        raise AssertionError(f"attention_out failed gates: {', '.join(failed)}")
    return report


def stress(invoke, inputs, case, runs, torch, device):
    """Check every output bit, with an independent CPU baseline and fixed inputs."""
    before = fingerprints(inputs, torch)
    outputs = invoke()
    torch.npu.synchronize()
    check_outputs(outputs, case, torch, device)
    baseline = tuple(value.detach().cpu().clone() for value in outputs)
    del outputs
    if fingerprints(inputs, torch) != before:
        raise AssertionError("QFA modified its inputs during baseline invocation")
    for index in range(1, runs + 1):
        outputs = invoke()
        torch.npu.synchronize()
        try:
            check_outputs(outputs, case, torch, device)
            for name, value, reference in zip(("attention_out", "softmax_lse"), outputs, baseline):
                current = value.detach().cpu()
                # Integer byte views preserve signed zero and the original dtype.
                if not torch.equal(current.view(torch.uint8), reference.view(torch.uint8)):
                    changed = current.view(torch.int16) != reference.view(torch.int16)
                    delta = (current.double() - reference.double()).abs()
                    flat_index = int(delta.argmax())
                    position = tuple(int(n) for n in np.unravel_index(flat_index, tuple(current.shape)))
                    details = {
                        "case": case.label,
                        "iteration": index,
                        "output": name,
                        "different_elements": int(changed.sum()),
                        "different_ratio": float(changed.float().mean()),
                        "max_abs": float(delta.max()),
                        "max_rel": float((delta / reference.double().abs().clamp_min(1e-300)).max()),
                        "position": position,
                        "baseline": float(reference[position]),
                        "current": float(current[position]),
                    }
                    raise AssertionError("Non-deterministic output: " + json.dumps(details))
        except Exception as error:
            if fingerprints(inputs, torch) != before:
                raise AssertionError(f"{case.label}: QFA input bytes changed at stress iteration {index}") from error
            raise AssertionError(f"{case.label}: stress iteration {index}: {error}") from error
        del outputs
        if index % 100 == 0 or index == runs:
            print(f"stress {case.label}: completed {index}/{runs}", flush=True)
    if fingerprints(inputs, torch) != before:
        raise AssertionError("QFA input bytes changed during stress")
