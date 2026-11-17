#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2 that you can use this software in accordance
# with the terms and conditions of the Mulan PSL v2. You may obtain a copy of the
# Mulan PSL at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""P2/P4 融合 pass 的 NPU 数值验证脚本（需 torch + torch_npu，仅在 NPU 机运行）。

    python tests/UT/compilation/verify_gelu_quant_and_rope.py

对应两个默认关闭的融合开关（FusionPatterns.enable_rope_cast_fusion /
enable_gelu_quant_fusion），全 [PASS] 才适合开图改写：

  1. npu_rotary_mul bf16 全参 vs「fp32 计算后转 bf16」——P2 的数值一致性基线。
     已验证形态 [1,8,128] + Wan 形态 [1,512,24,128]（cos/sin [512,64]）；若 Wan 形态
     fp32 基线本身不可运行，标 [SKIP]（形状语义需按 fx dump 实测链调整后重测）。
  2. npu_gelu_quant(quant_mode="dynamic") 与「gelu → npu_dynamic_mx_quant(dst_type=fp8)」
     的 payload/scale 兼容性——P4 生死判定：
     a) dst_type 接受 dtype 对象还是整数（schema 印的是 int?）；
     b) payload/scale 形状、dtype 与基线一致（scale 形状不一致 = 下游
        npu_quant_matmul(pertoken_scale=...) 直接不兼容，P4 判死）；
     c) 反量化相对误差（in-kernel gelu 与 aten.gelu 的近似差异）。
  3. （仅信息）npu_fast_gelu 与 gelu none/tanh 的基线差异，辅助选 gelu_quant_approximate。
"""

import sys

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    tag = "[PASS]" if ok else "[FAIL]" if ok is False else "[SKIP]"
    line = f"{tag} {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)


def _f32(t):
    return t.to(torch.float32)


def _dequant(payload, scale):
    """按 MX 组粒度反量化：scale 末维是 payload 末维的 1/g 时逐组广播。"""
    p = _f32(payload)
    s = _f32(scale)
    if s.dim() == p.dim() and s.shape[-1] and p.shape[-1] % s.shape[-1] == 0:
        g = p.shape[-1] // s.shape[-1]
        s = s.repeat_interleave(g, dim=-1)
    return p * s


def check_rotary():
    print("== P2: npu_rotary_mul bf16 vs fp32→bf16 ==")
    torch.manual_seed(0)
    combos = [
        ("verified_1_8_128", torch.randn(1, 8, 128, device="npu"), None),
        (
            "wan_1_512_24_128",
            torch.randn(1, 512, 24, 128, device="npu"),
            torch.randn(512, 64, device="npu"),
        ),
    ]
    for name, x, r in combos:
        cos = r if r is not None else x
        sin = r if r is not None else x
        try:
            # 基线 = 原链语义：bf16 x 上转 fp32 计算，再舍入回 bf16。
            ref = torch_npu.npu_rotary_mul(x.to(torch.bfloat16).float(), cos.float(), sin.float(), "interleave").to(
                torch.bfloat16
            )
        except Exception as exc:  # noqa: BLE001
            report(f"rotary[{name}]", None, f"fp32 基线不可运行: {exc!r}"[:160])
            continue
        try:
            got = torch_npu.npu_rotary_mul(
                x.to(torch.bfloat16), cos.to(torch.bfloat16), sin.to(torch.bfloat16), "interleave"
            )
            diff = (got.float() - ref.float()).abs().max().item()
            ok = torch.allclose(got.float(), ref.float(), rtol=5e-2, atol=5e-2)
            report(f"rotary[{name}]", ok, f"bf16 直入 maxdiff={diff:.4g} (rtol/atol=5e-2)")
        except Exception as exc:  # noqa: BLE001
            report(f"rotary[{name}]", False, f"bf16 直入报错: {exc!r}"[:160])


def check_gelu_quant():
    print("== P4: npu_gelu_quant vs gelu→npu_dynamic_mx_quant(dst_type=fp8) ==")
    torch.manual_seed(42)
    x = (torch.randn(256, 512, device="npu") * 2).to(torch.bfloat16)
    fp8 = torch_npu.float8_e4m3fn
    for approx in ("none", "tanh"):
        ref_act = torch.nn.functional.gelu(x, approximate=approx)
        try:
            ref_payload, ref_scale = torch_npu.npu_dynamic_mx_quant(ref_act, dst_type=fp8)
        except Exception as exc:  # noqa: BLE001
            report(f"gelu_quant[{approx}]", False, f"基线 dynamic_mx_quant 报错: {exc!r}"[:160])
            continue

        new_pair = None
        used_form = None
        errors = []
        for form, dst in (("dtype", fp8), ("int", 292)):
            try:
                new_pair = torch_npu.npu_gelu_quant(x, approximate=approx, quant_mode="dynamic", dst_type=dst)
                used_form = form
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{form}: {exc!r}"[:120])
        if new_pair is None:
            report(f"gelu_quant[{approx}]", False, "dst_type 两种形态均失败: " + " | ".join(errors))
            continue
        new_payload, new_scale = new_pair[0], new_pair[1]

        shape_ok = (
            new_payload.shape == ref_payload.shape
            and new_payload.dtype == ref_payload.dtype
            and new_scale.shape == ref_scale.shape
            and new_scale.dtype == ref_scale.dtype
        )
        report(
            f"gelu_quant[{approx}].dst_type_form",
            True,
            f"dst_type={used_form} 形态可用",
        )
        if not shape_ok:
            report(
                f"gelu_quant[{approx}].compat",
                False,
                f"payload/scale 不一致: new=({tuple(new_payload.shape)},{new_payload.dtype},"
                f"{tuple(new_scale.shape)},{new_scale.dtype}) ref=({tuple(ref_payload.shape)},"
                f"{ref_payload.dtype},{tuple(ref_scale.shape)},{ref_scale.dtype})",
            )
            continue

        dq_ref = _dequant(ref_payload, ref_scale)
        dq_new = _dequant(new_payload, new_scale)
        err = (dq_new - dq_ref).abs().max().item()
        mag = max(dq_ref.abs().max().item(), 1e-6)
        report(
            f"gelu_quant[{approx}].numerics",
            err <= 0.05 * mag,
            f"反量化 maxdiff={err:.4g}, 基线幅值={mag:.4g} (阈值 5%)",
        )

    # 仅信息：近似模式选择参考。
    fast = torch_npu.npu_fast_gelu(x)
    for approx in ("none", "tanh"):
        ref = torch.nn.functional.gelu(x, approximate=approx)
        diff = (fast.float() - ref.float()).abs().max().item()
        report(f"info.fast_gelu_vs_gelu[{approx}]", None, f"maxdiff={diff:.4g}")


def main():
    if torch_npu is None or not torch.npu.is_available():
        print("[SKIP] 本脚本验证 torch_npu 算子（npu_rotary_mul / npu_gelu_quant），需 NPU 环境，当前环境不可用。")
        return 0
    torch.npu.set_device(0)
    check_rotary()
    check_gelu_quant()
    fails = sum(1 for ok in RESULTS if ok is False)
    print(f"\n== summary: {len(RESULTS)} checks, {fails} FAIL ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
