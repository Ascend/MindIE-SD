# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Direct qmode1 inputs, staged CPU reference, NPU reference and per-output gates.

Imports are deliberately device-free; the scheduler and --help need only Python.
"""

import hashlib
import json
import math
import random
import struct

OP = "eagle_quant_block_sparse_attention"
SEED = 20260811
BLOCK_SHAPE = (128, 128)
QUANT_BLOCK = 64
PRODUCTION = {
    "batch_size": 1,
    "num_query_heads": 32,
    "num_kv_heads": 4,
    "query_seq_len": 2304,
    "kv_seq_len": 30757,
    "head_dim": 128,
    "layout": "BNSD",
    "output_dtype": "bf16",
    "seed": SEED,
}
TOLERANCES = {
    "fp16": (2**-10, 2**-10, 0.1),
    "bf16": (2**-7, 2**-7, 0.1),
    "fp32": (2**-13, 2**-13, 0.01),
    "fp8_e4m3": (2**-3, 2**-3, 1.0),
}


def scenario(**overrides):
    result = dict(PRODUCTION, distribution="normal", softmax_lse_flag=1, special="random")
    result.update(overrides)
    return result


def lengths(s, which):
    return s.get(which + "_lengths", [s[which + "_seq_len"]] * s["batch_size"])


def validate(s):
    if s.get("q_block_size", BLOCK_SHAPE[0]) <= 0:
        raise ValueError("sparse query block size must be positive")
    if not 0 <= s["seed"] < 2**63:
        raise ValueError("seed must be a nonnegative signed 64-bit integer")
    if min(s[k] for k in ("batch_size", "num_query_heads", "num_kv_heads", "query_seq_len", "kv_seq_len")) <= 0:
        raise ValueError("batch, heads and sequence lengths must be positive")
    if s["num_query_heads"] % s["num_kv_heads"] or s["head_dim"] not in (64, 128):
        raise ValueError("query heads must be divisible by KV heads; D must be 64 or 128")
    if s["layout"] not in ("BNSD", "TND") or s["output_dtype"] not in ("bf16", "fp16"):
        raise ValueError("qmode1 supports BNSD/TND and BF16/FP16 output")
    if s["distribution"] not in ("uniform", "normal") or s["softmax_lse_flag"] not in (0, 1):
        raise ValueError("invalid distribution or softmax-lse-flag")
    for kind in ("query", "kv"):
        sizes = lengths(s, kind)
        if len(sizes) != s["batch_size"] or min(sizes) <= 0 or max(sizes) != s[kind + "_seq_len"]:
            raise ValueError("length lists must match batch and maximum sequence length")
        if s["layout"] == "BNSD" and len(set(sizes)) != 1:
            raise ValueError("this suite uses full physical BNSD sequences; variable lengths use TND")


def preview(s):
    validate(s)
    b, nq, nk, d = (s[k] for k in ("batch_size", "num_query_heads", "num_kv_heads", "head_dim"))
    q, k = lengths(s, "query"), lengths(s, "kv")
    tnd = s["layout"] == "TND"
    q_block = s.get("q_block_size", BLOCK_SHAPE[0])
    qshape = [sum(q), nq, d] if tnd else [b, nq, q[0], d]
    kshape = [sum(k), nk, d] if tnd else [b, nk, k[0], d]
    return {
        "scenario": s,
        "query": {"shape": qshape, "dtype": "int8"},
        "key": {"shape": kshape, "dtype": "int8"},
        "value": {"shape": kshape, "dtype": "int8 (FP8 E4M3 bits)"},
        "query_scale": {
            "shape": [sum((v + 63) // 64 for v in q), nq] if tnd else [b, nq, (q[0] + 63) // 64],
            "dtype": "float32",
        },
        "key_scale": {
            "shape": [sum((v + 63) // 64 for v in k), nk] if tnd else [b, nk, (k[0] + 63) // 64],
            "dtype": "float32",
        },
        "value_scale": {"shape": [b, nk, d], "dtype": "float32"},
        "block_sparse_mask": {
            "shape": [b, nq, (max(q) + q_block - 1) // q_block, (max(k) + 127) // 128],
            "dtype": "int8",
        },
        "attention_out": {"shape": qshape, "dtype": s["output_dtype"]},
        "softmax_lse": {"shape": qshape[:-1] + [1], "dtype": "float32", "defined": bool(s["softmax_lse_flag"])},
    }


def name(s):
    block = f"_block{s['q_block_size']}x{BLOCK_SHAPE[1]}" if "q_block_size" in s else ""
    return (
        f"{s['layout']}_B{s['batch_size']}_N{s['num_query_heads']}-{s['num_kv_heads']}_"
        f"Q{s['query_seq_len']}_KV{s['kv_seq_len']}_D{s['head_dim']}_{s['output_dtype']}_"
        f"lse{s['softmax_lse_flag']}_{s['distribution']}_{s['special']}{block}"
    )


def initialize(device):
    import os

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("unset ASCEND_RT_VISIBLE_DEVICES; use physical --device-id")
    import torch
    import torch_npu

    torch.npu.set_device(device)
    from tests.ops.device_support import require_support

    require_support(OP)
    import mindiesd  # Registration follows physical device selection.

    return torch, torch_npu


def output_dtype(s):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16}[s["output_dtype"]]


def make_source_inputs(s):
    """Sample the BF16 inputs and sparse mask shared by the performance implementations."""
    import torch

    spec = preview(s)
    rng = random.Random(s["seed"])
    gen = torch.Generator(device="cpu").manual_seed(s["seed"])
    parameters, src = {}, {}
    for key in ("query", "key", "value"):
        shape = spec[key]["shape"]
        if s["distribution"] == "uniform":
            params = {"distribution": "uniform", "low": -5, "high": 5, "seed": s["seed"]}
            src[key] = (torch.rand(shape, generator=gen) * 10 - 5).bfloat16()
        else:
            mean, std = rng.uniform(-5, 5), rng.uniform(0.1, 2)
            params = {"distribution": "normal", "mean": mean, "std": std, "seed": s["seed"]}
            src[key] = (torch.randn(shape, generator=gen) * std + mean).bfloat16()
        parameters[key] = params
    if s["special"] == "zero":
        src["query"].zero_()
        parameters["query"] = {"distribution": "constant", "value": 0}
    elif s["special"] == "q_scale_groups":
        # Deliberately separate adjacent absmax scales, including a partial last group.
        seq_offset = 0
        for batch, size in enumerate(lengths(s, "query")):
            q = (
                src["query"][seq_offset : seq_offset + size].transpose(0, 1)
                if s["layout"] == "TND"
                else src["query"][batch]
            )
            for start in range(0, size, 2 * QUANT_BLOCK):
                q[:, start : start + QUANT_BLOCK].mul_(0.125)
            seq_offset += size
        parameters["query"]["even_group_multiplier"] = 0.125
    mask = (torch.rand(spec["block_sparse_mask"]["shape"], generator=gen) > 0.5).to(torch.int8)
    q_block = s.get("q_block_size", BLOCK_SHAPE[0])
    for batch, (qlen, klen) in enumerate(zip(lengths(s, "query"), lengths(s, "kv"))):
        qblocks, kblocks = (qlen + q_block - 1) // q_block, (klen + 127) // 128
        valid_mask = mask[batch, :, :qblocks, :kblocks]
        valid_mask[..., 0] = 1
        valid_mask[..., -1] = 1
        valid_mask[:, -1, :] = 1
        if s["special"] == "dense":
            valid_mask.fill_(1)
        elif s["special"] == "empty":
            valid_mask.zero_()
        elif s["special"] == "partial_empty":
            valid_mask[:, 0, :] = 0
        # Mask2Idx scans the physical mask extent, including ragged-batch padding.
        mask[batch, :, qblocks:, :] = 0
        mask[batch, :, :, kblocks:] = 0
    parameters["mask"] = (
        "Bernoulli(0.5), per-batch first/last valid KV and last valid Q blocks connected; "
        "padding blocks zero; special=" + s["special"]
    )
    return src, mask, parameters


def make_inputs(s):
    """Encode Q/K per 64 actual tokens and V per channel on the CPU."""
    import torch

    src, mask, parameters = make_source_inputs(s)
    spec = preview(s)
    data = {"block_sparse_mask": mask}
    tnd = s["layout"] == "TND"
    for key, kind in (("query", "query"), ("key", "kv"), ("value", "kv")):
        encoded = torch.empty_like(src[key], dtype=torch.int8)
        scales = torch.empty(spec[key + "_scale"]["shape"], dtype=torch.float32)
        seq_offset = scale_offset = 0
        for batch, size in enumerate(lengths(s, kind)):
            # This is quantization-axis access to the already chosen physical layout.
            raw = src[key][seq_offset : seq_offset + size].transpose(0, 1) if tnd else src[key][batch]
            target = encoded[seq_offset : seq_offset + size].transpose(0, 1) if tnd else encoded[batch]
            if key == "value":
                maximum = raw.float().abs().amax(1)
                scale = torch.where(maximum > 0, maximum / 448, torch.ones_like(maximum))
                fp8 = (raw.float() / scale[:, None, :]).clamp(-448, 448).to(torch.float8_e4m3fn)
                target.copy_(fp8.view(torch.int8))
                scales[batch] = scale
            else:
                for block, start in enumerate(range(0, size, QUANT_BLOCK)):
                    tile = raw[:, start : start + QUANT_BLOCK].float()
                    maximum = tile.abs().amax((1, 2))
                    scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
                    target[:, start : start + QUANT_BLOCK] = (
                        (tile / scale[:, None, None]).round().clamp(-127, 127).to(torch.int8)
                    )
                    if tnd:
                        scales[scale_offset + block] = scale
                    else:
                        scales[batch, :, block] = scale
            seq_offset += size
            scale_offset += (size + 63) // 64
        data[key], data[key + "_scale"] = encoded, scales
    parameters["encoding"] = "CPU absmax; Q/K INT8 RNE per64 tokens; V E4M3 RNE per channel; scales FP32"
    return data, parameters


def to_device(data, device):
    return {k: v.to(f"npu:{device}") for k, v in data.items()}


def attributes(s):
    return {
        "block_shape": [s.get("q_block_size", BLOCK_SHAPE[0]), BLOCK_SHAPE[1]],
        "q_input_layout": s["layout"],
        "kv_input_layout": s["layout"],
        "num_key_value_heads": s["num_kv_heads"],
        "scale_value": s["head_dim"] ** -0.5,
        "inner_precise": 4,
        "actual_seq_lengths": lengths(s, "query"),
        "actual_seq_lengths_kv": lengths(s, "kv"),
        "softmax_lse_flag": s["softmax_lse_flag"],
    }


def metadata(data, s, parameters):
    return {
        "scenario": name(s),
        "generation": parameters,
        "attributes": attributes(s),
        "inputs": {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "device": str(v.device),
                "stride": list(v.stride()),
                "contiguous": v.is_contiguous(),
                "layout": s["layout"] if k in ("query", "key", "value") else "ND",
            }
            for k, v in data.items()
        },
    }


def dut(data, s):
    import torch

    return torch.ops.mindiesd.eagle_quant_block_sparse_attention(**data, **attributes(s), output_dtype=output_dtype(s))


def contract(outputs, s, finite=True):
    import torch

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise AssertionError("EQBSA must return (O, LSE)")
    spec = preview(s)
    for out, key, dtype in zip(outputs, ("attention_out", "softmax_lse"), (output_dtype(s), torch.float32)):
        if list(out.shape) != spec[key]["shape"] or out.dtype != dtype or not out.is_contiguous():
            raise AssertionError(f"{key} raw output contract failed: {out.shape}/{out.dtype}/{out.stride()}")
    if finite and not torch.isfinite(outputs[0]).all().item():
        raise AssertionError("O contains NaN/Inf")
    if finite and s["softmax_lse_flag"]:
        lse = outputs[1]
        valid = (
            torch.isfinite(lse) | torch.isneginf(lse)
            if s["special"] in ("empty", "partial_empty")
            else torch.isfinite(lse)
        )
        if not valid.all().item():
            raise AssertionError("LSE contains unexpected NaN/Inf")


def defined_outputs(outputs, s):
    # flag=0 returns an allocated but unwritten LSE tensor; only its contract exists.
    return {"O": outputs[0], **({"LSE": outputs[1]} if s["softmax_lse_flag"] else {})}


def batch_views(data, s, batch):
    """Logical views for references only; never used to transform DUT inputs."""
    result = {}
    for key, kind in (("query", "query"), ("key", "kv"), ("value", "kv")):
        sizes = lengths(s, kind)
        if s["layout"] == "TND":
            start = sum(sizes[:batch])
            result[key] = data[key][start : start + sizes[batch]].transpose(0, 1)
            start_scale = sum((v + 63) // 64 for v in sizes[:batch])
            scale = (
                data[key + "_scale"][start_scale : start_scale + (sizes[batch] + 63) // 64].t()
                if key != "value"
                else data["value_scale"][batch]
            )
        else:
            result[key], scale = data[key][batch], data[key + "_scale"][batch]
        result[key + "_scale"] = scale
    return result


def _sum_fp16_dn32(probs):
    """Four registers, four K lanes per register, then the DN32 lane reduction."""
    import torch

    n = (probs.shape[-1] + 15) // 16 * 16
    padded = torch.nn.functional.pad(probs, (0, n - probs.shape[-1]))
    acc = torch.zeros((*probs.shape[:-1], 4, 4), dtype=torch.float16)
    for tile in padded.split(16, dim=-1):
        acc = (acc + tile.reshape(*probs.shape[:-1], 4, 4)).half()
    lanes = ((acc[..., 0, :] + acc[..., 1, :]).half() + (acc[..., 2, :] + acc[..., 3, :]).half()).half()
    return ((lanes[..., 0] + lanes[..., 2]).half() + (lanes[..., 1] + lanes[..., 3]).half()).half().float()


def _fixpipe_deq_scale(scale):
    """Arch3510 QUANT_PRE M1 uses FP32 bits 13..31; bits 0..12 are ignored."""
    bits = struct.unpack("<I", struct.pack("<f", scale))[0]
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFE000))[0]


def _kernel_tile(q, k, v, qs, ks, vs, selected, scale):
    """INT32 QK -> FP16 Fixpipe/dequant/exp -> FP8 P/V -> FP32 online O/LSE."""
    import torch

    scale = _fixpipe_deq_scale(scale)
    gm = torch.full((q.shape[0],), -60000.0, dtype=torch.float16)
    gl = torch.zeros(q.shape[0], dtype=torch.float32)
    go = torch.zeros((q.shape[0], q.shape[1]), dtype=torch.float32)
    for start in range(0, len(selected), 4):
        indices = torch.cat(
            [torch.arange(i * 128, min((i + 1) * 128, k.shape[0])) for i in selected[start : start + 4]]
        )
        # Fixpipe DEQF16 is saturating (it does not support INF_NAN mode).
        # See the Ascend 950 asc_copy_l0c2ub quant_pre contract linked in README.
        scores = ((q.int() @ k[indices].int().t()).float() * scale).clamp(-65504, 65504).half()
        dequant = (qs * ks[indices // 64]).half()
        scores = (scores * dequant[None, :]).half()
        new_max = torch.maximum(gm, scores.amax(-1))
        alpha = (gm - new_max).half().exp().half().float()
        p = (scores - new_max[:, None]).half().exp().half()
        gl = alpha * gl + _sum_fp16_dn32(p)
        # The kernel's positive FP16->FP8 RNA bit encoding (including its minimum code).
        bits = (((p * 448).half().view(torch.int16).int().clamp_min(8256) - 8128) >> 7).to(torch.uint8)
        pv = bits.view(torch.float8_e4m3fn).float() @ v[indices].float()
        go = go * alpha[:, None] + pv
        gm = new_max
    if not selected:
        return go, torch.full_like(gl, -math.inf)
    return (go * (vs / 448)[None, :]) / gl[:, None], gm.float() + gl.log()


def golden(data, s, staged=False):
    """Small: CPU staged kernel model. Large: FP32 composition on the DUT NPU."""
    import torch

    if staged and data["query"].device.type != "cpu":
        raise ValueError("staged golden must run on CPU")
    if not staged and data["query"].device.type != "npu":
        raise ValueError("large golden must run on the same NPU as DUT")
    batch_out, batch_lse = [], []
    q_block = s.get("q_block_size", BLOCK_SHAPE[0])
    for b in range(s["batch_size"]):
        views = batch_views(data, s, b)
        q, k = views["query"], views["key"]
        v = views["value"].contiguous().view(torch.float8_e4m3fn).float()
        out = torch.zeros_like(q, dtype=torch.float32)
        lse = torch.full((*q.shape[:-1], 1), -math.inf, dtype=torch.float32, device=q.device)
        for h in range(s["num_query_heads"]):
            kh = h // (s["num_query_heads"] // s["num_kv_heads"])
            row = 0
            while row < q.shape[1]:
                end = min((row // QUANT_BLOCK + 1) * QUANT_BLOCK, (row // q_block + 1) * q_block, q.shape[1])
                mask = data["block_sparse_mask"][b, h, row // q_block]
                if staged:
                    selected = [i for i in range((k.shape[1] + 127) // 128) if mask[i].item()]
                    tile_o, tile_lse = _kernel_tile(
                        q[h, row:end],
                        k[kh],
                        v[kh],
                        views["query_scale"][h, row // 64],
                        views["key_scale"][kh],
                        views["value_scale"][kh],
                        selected,
                        s["head_dim"] ** -0.5,
                    )
                else:
                    # All arithmetic, including dequantization, remains on NPU.
                    qf, kf = q[h, row:end].float(), k[kh].float()
                    vf = v[kh] * views["value_scale"][kh][None, :]
                    scale = _fixpipe_deq_scale(s["head_dim"] ** -0.5)
                    scores = ((qf @ kf.t()) * scale).clamp(-65504, 65504).half()
                    dequant = (views["query_scale"][h, row // 64] * views["key_scale"][kh]).half()
                    scores = (scores * dequant.repeat_interleave(64)[: k.shape[1]][None, :]).half().float()
                    keep = mask.repeat_interleave(128)[: k.shape[1]].bool()
                    scores = scores.masked_fill(~keep[None, :], -math.inf)
                    tile_lse = scores.logsumexp(-1)
                    p = torch.nan_to_num(scores.softmax(-1), nan=0.0)
                    tile_o = p @ vf
                out[h, row:end], lse[h, row:end, 0] = tile_o, tile_lse
                row = end
        batch_out.append(out.to(output_dtype(s)))
        batch_lse.append(lse)
    if s["layout"] == "TND":
        return tuple(torch.cat([x.transpose(0, 1) for x in group]).contiguous() for group in (batch_out, batch_lse))
    return torch.stack(batch_out), torch.stack(batch_lse)


def fingerprint(tensor):
    import torch

    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def input_hashes(data):
    return {k: fingerprint(v) for k, v in data.items()}


def bit_difference(actual, expected):
    import torch

    a, g = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    changed = (
        a.view(torch.uint8).reshape(-1, a.element_size()) != g.view(torch.uint8).reshape(-1, g.element_size())
    ).any(-1)
    delta = (a.double() - g.double()).abs()
    finite = torch.isfinite(delta)
    return {
        "changed_elements": changed.sum().item(),
        "changed_ratio": changed.double().mean().item(),
        "first_changed_flat_index": changed.nonzero()[0].item() if changed.any() else None,
        "max_abs": delta[finite].max().item() if finite.any() else None,
        "max_relative": (delta[finite] / g.double()[finite].abs().clamp_min(1e-300)).max().item()
        if finite.any()
        else None,
    }


def compare_output(actual, reference, key, tolerance, large=False, basis=None):
    import torch

    a, g = actual.detach().cpu().flatten(), reference.detach().cpu().flatten()
    atol, rtol, fallback = TOLERANCES[tolerance]
    finite = torch.isfinite(a) & torch.isfinite(g)
    # NaN never passes. Same-sign inf, including legal narrowing overflow, is explicit.
    narrowed = g.to(a.dtype)
    allowed = finite | (torch.isinf(a) & (a == narrowed))
    special_pass = allowed.all().item()
    av, gv = a[finite].double(), g[finite].double()
    error = (av - gv).abs()
    if error.numel():
        worst = error.argmax().item()
        index = finite.nonzero().flatten()[worst].item()
        max_abs = error[worst].item()
        anchor = g[index].to(a.dtype).abs()
        following = torch.nextafter(anchor, torch.full_like(anchor, math.inf))
        # At max-finite, use the finite spacing below the value.
        ulp = (
            (following - anchor).item()
            if torch.isfinite(following)
            else (anchor - torch.nextafter(anchor, torch.zeros_like(anchor))).item()
        )
        matched = (error <= atol + rtol * gv.abs()).sum().item() + ((~finite) & allowed).sum().item()
        an, gn = av.square().sum().item(), gv.square().sum().item()
        cosine = max(-1.0, min(1.0, (av * gv).sum().item() / math.sqrt(an * gn))) if an and gn else float(an == gn)
        ratio = math.sqrt(an / gn) if gn else (1.0 if not an else math.inf)
    else:
        index, max_abs, ulp, matched, cosine, ratio = 0, 0.0, 0.0, allowed.sum().item(), 1.0, 1.0
    limit, matched_ratio = max(fallback, 32 * ulp), matched / max(1, a.numel())
    coord, pos = [], index
    for dim in reversed(actual.shape):
        coord.append(pos % dim)
        pos //= dim
    result = {
        "output": key,
        "tolerance_dtype": tolerance,
        "output_dtype": str(actual.dtype),
        "atol": atol,
        "rtol": rtol,
        "required_matched_ratio": 0.99,
        "fallback": fallback,
        "matched_ratio": matched_ratio,
        "max_abs": max_abs,
        "max_error_coordinate": list(reversed(coord)),
        "golden_at_max": g[index].item(),
        "actual_at_max": a[index].item(),
        "ulp": ulp,
        "ulp32": 32 * ulp,
        "max_abs_limit": limit,
        "cosine": cosine,
        "norm_ratio": ratio,
        "matched_pass": matched_ratio >= 0.99,
        "max_abs_pass": max_abs <= limit,
        "cosine_pass": cosine >= 0.99,
        "norm_ratio_pass": 0.9 <= ratio <= 1.1,
        "nonfinite_pass": special_pass,
        "mixed_role": "soft" if large else "hard",
        "basis": basis
        or (
            "FP8 P requantization and FP8 PV"
            if key == "O"
            else "FP16 logits/exp/reduction; FP32 running sum/log; no P/V quantization"
        ),
    }
    result["passed"] = (
        special_pass
        and result["cosine_pass"]
        and result["norm_ratio_pass"]
        and (large or (result["matched_pass"] and result["max_abs_pass"]))
    )
    print(json.dumps(result), flush=True)
    return result


def compare(actual, reference, s, large=False):
    contract(actual, s)
    contract(reference, s)
    actual_named, reference_named = defined_outputs(actual, s), defined_outputs(reference, s)
    metrics = {
        key: compare_output(value, reference_named[key], key, "fp8_e4m3" if key == "O" else "fp16", large)
        for key, value in actual_named.items()
    }
    return {"outputs": metrics, "passed": all(v["passed"] for v in metrics.values())}
