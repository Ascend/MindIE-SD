#!/usr/bin/env python3
"""闸①：CPU 等价性对拍 —— 整段解码 vs 分段解码必须逐位一致。

用法
  1) 自证（内置记忆块网络，验证骨架与判据本身可用）：
       python shard_equivalence_check.py --demo
  2) 检查你自己的实现：
       python shard_equivalence_check.py --model-spec my_pkg.my_mod:build --pieces 2 4
     其中 `build()` 需返回一个对象/字典，提供两个可调用：
       whole(x)            -> 整段解码输出（对照基准）
       sliced(x, pieces)   -> 分段解码输出（被测实现）
     x 形状默认 (1, T, C, H, W)，用 --shape 覆盖；两者必须在同一 dtype/device 上。

判据
  * `max|d| == 0`  => PASS（逐位等价；这是分片可宣称“无损”的唯一硬证据）
  * `0 < max|d| <= rtol*absmax` => WARN（数值等价，但不可宣称逐字节无损）
  * 否则 => FAIL（先查 state 语义 / 切分点对齐 / 首段 zeros pad，见 references/sliced-state-decode.md §3）

退出码：0=PASS，1=FAIL（可直接用作 CI / 门禁）。
"""

from __future__ import annotations

import argparse
import importlib
import sys

try:  # 让中文/符号输出在 GBK 控制台也不炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    print("需要 torch（CPU 即可）：pip install torch")
    raise SystemExit(2)


def _load_spec(spec: str):
    if ":" not in spec:
        raise SystemExit("--model-spec 需要形如 'module.path:build_function'")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise SystemExit(f"模块 {mod_name} 中找不到 {fn_name}")
    built = fn()
    if isinstance(built, dict):
        whole, sliced = built.get("whole"), built.get("sliced")
    else:
        whole, sliced = getattr(built, "whole", None), getattr(built, "sliced", None)
    if not callable(whole) or not callable(sliced):
        raise SystemExit("build() 必须提供可调用的 whole(x) 与 sliced(x, pieces)")
    return whole, sliced


def _demo_pair(shape):
    """内置：真实网络同构的小型“记忆块 + 时间上采样”解码器。"""
    sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0] + "/scripts")
    try:
        import sliced_state_decode as sd
    except ModuleNotFoundError:
        import os

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import sliced_state_decode as sd
    from torch import nn

    torch.manual_seed(0)
    act = nn.ReLU(inplace=True)
    model = nn.Sequential(
        nn.Conv2d(shape[2], 16, 3, padding=1),
        act,
        sd._MemBlock(16, 16, act),
        sd._MemBlock(16, 16, act),
        nn.Upsample(scale_factor=2, mode="nearest"),
        sd._TGrow(16, 2),
        nn.Conv2d(16, 8, 3, padding=1),
        sd._MemBlock(8, 8, act),
        nn.Conv2d(8, 12, 3, padding=1),
    ).eval()

    def post(flat, n, t):
        flat = torch.nn.functional.pixel_shuffle(flat, 2)
        bt, c, h, w = flat.shape
        return flat.view(n, bt // n, c, h, w)

    is_mem = lambda b: isinstance(b, sd._MemBlock)

    def whole(x):
        with torch.no_grad():
            return sd.apply_whole(model, x, post, is_mem)

    def sliced(x, pieces):
        with torch.no_grad():
            return sd.apply_sliced(model, x, post, is_mem, pieces)

    return whole, sliced


def check(whole, sliced, shape, pieces_list, dtype, seed, rtol):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=dtype)
    ref = whole(x)
    print(f"整段输出: shape={tuple(ref.shape)} dtype={ref.dtype} absmax={ref.abs().max().item():.4g}")
    bad = 0
    for p in pieces_list:
        got = sliced(x, p)
        if tuple(got.shape) != tuple(ref.shape):
            print(f"  {p} 段: **shape 不一致** {tuple(got.shape)} vs {tuple(ref.shape)} => FAIL")
            bad += 1
            continue
        d = (got - ref).abs().max().item()
        absmax = ref.abs().max().item() or 1.0
        if d == 0:
            verdict = "PASS（逐位等价）"
        elif d <= rtol * absmax:
            verdict = "WARN（数值等价 %.3e，不可宣称逐字节无损）" % (d / absmax)
            bad += 1
        else:
            verdict = "**FAIL**（相对 %.3e）" % (d / absmax)
            bad += 1
        print(f"  {p} 段: max|d| = {d:.6e}  {verdict}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="整段 vs 分段解码的逐位等价性对拍")
    ap.add_argument("--model-spec", help="module:build —— build() 返回提供 whole(x)/sliced(x,pieces) 的对象")
    ap.add_argument("--demo", action="store_true", help="用内置记忆块网络自证判据可用")
    ap.add_argument("--pieces", type=int, nargs="+", default=[2, 4], help="待测段数（默认 2 4）")
    ap.add_argument("--shape", type=int, nargs=5, default=[1, 12, 4, 8, 8], metavar=("N", "T", "C", "H", "W"))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-6, help="数值等价阈值（相对 absmax）")
    args = ap.parse_args()

    if args.demo:
        whole, sliced = _demo_pair(args.shape)
    elif args.model_spec:
        whole, sliced = _load_spec(args.model_spec)
    else:
        ap.print_help()
        return 2

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    bad = check(whole, sliced, args.shape, args.pieces, dtype, args.seed, args.rtol)
    print(
        "\n结论: %s" % ("全部逐位等价 [PASS]" if bad == 0 else "存在不一致 [FAIL]（先查 state 语义/切分点/首段 pad）")
    )
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
