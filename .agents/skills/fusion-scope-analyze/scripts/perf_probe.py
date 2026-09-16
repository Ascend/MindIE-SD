#!/usr/bin/env python3
"""设备性能探针：为 L1 理论列提供**实测峰值口径**与对抵/地板参数。

构造最小算子集合实测四件事（不依赖任何外部工具）：

1. **有效带宽**（GB/s）：大张量 device 间拷贝 / 读写，取多档尺寸的**上包络**（避免小尺寸受启动开销污染）；
2. **峰值算力**（TFLOPs）：bf16 / fp16 大 GEMM（M=N=K 多档）取上包络；
3. **固定开销**（µs）：同形状 kernel 的**空跑/最小 kernel 启动**开销 → 作为 `fixed_overhead_us` 的对抵基准；
4. **噪声地板**（%）：同配置重复 N 次的离散度（max−min 相对中位数）→ "效应上限低于地板即关项"。

**计时口径**：统一用「逐次 synchronize + 墙钟」，**不把 `torch.npu.Event` 当测量手段**——event 窗口在
部分平台上只覆盖 launch 阶段（对拷贝 / GEMM 这类算子尤为明显），会把峰值抬成非物理值。
因此本探针**不内置任何设备代际的绝对量级**，改用两条与设备无关的判据：

1. **规模不变性（硬门禁）**：字节相差 ≥4× 的两个规模档，耗时若几乎不涨 ⇒ 测到的是固定窗口而不是 workload；
2. **双方法交叉（平台能力辨识，不否决）**：同一次 workload 的墙钟与 event 读数比值超过上限 ⇒ 记为该平台
   **event 计时不可作测量口径**，出警告并按墙钟出数（含 launch 开销的墙钟只会偏保守，不会漏算执行）。

规模不变性不通过 ⇒ fail-closed（退出码 3，样本仍落盘供排查，但不得喂给 `theory_columns`）。
若确知本代设备的量级，可用 `--max-bw-gbps` / `--max-tflops` **追加**一条绝对上限（默认不启用）。

用法（在目标设备上跑）：

```bash
python perf_probe.py --out probe.json                  # 全量
python perf_probe.py --selftest                        # 不碰设备：校验收敛/退化/公式/自检判据
```

产物 `probe.json` 直接喂给：

```bash
python theory_columns.py --csv <op_summary.csv> \
  --peak-bw-gbps <probe 的 bw_gbps> --peak-tflops <probe 的 tflops_bf16> -o theory.csv
```

退出码：0 = 成功；2 = 前置条件缺失（未装 torch / 未装 torch_npu 且未改 `--device cpu`）；
3 = 口径自检不通过（`selfcheck` 全 false 的项见产物，不得把峰值喂给 `theory_columns`）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

CONFIGS_BW = ((64, 1024, 1024), (256, 1024, 1024), (1024, 1024, 1024))  # (MB, rows, cols) 量级档
CONFIGS_GEMM = (2048, 4096, 8192)
REPEATS = 5

# 口径自检阈值：只比「两个规模档」与「两种计时口径」之间的**比值**，与设备绝对量级无关。
SCALE_BYTES_RATIO_MIN = 4.0  # 规模档字节差小于该倍数时，规模不变性判据不成立（跳过）
SCALE_TIME_RATIO_MIN = 1.5  # 字节差 ≥4× 时，耗时至少应涨该倍数，否则判测到固定窗口
METHOD_RATIO_MAX = 3.0  # 墙钟 / event 两种口径的比值上限（超出 ⇒ 该平台 event 不可作测量口径）


def _sync(torch, device: str) -> None:
    """同步设备：墙钟计时前后各一次，排除异步入队造成的失真。"""
    if device.startswith("npu"):
        torch.npu.synchronize()


def _time_once(torch, device: str, fn) -> float:
    """单次调用的墙钟计时（ms）。"""
    _sync(torch, device)
    start = time.perf_counter()
    fn()
    _sync(torch, device)
    return (time.perf_counter() - start) * 1e3


def _time_once_event(torch, device: str, fn) -> float:
    """单次调用的 event 计时（ms）：**只用于双方法交叉自检**，不作为测量口径。"""
    _sync(torch, device)
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    _sync(torch, device)
    return start.elapsed_time(end)


def _scale_ok(samples: list[dict], size_key: str, time_key: str) -> bool:
    """规模不变性自检：字节差够大的档位之间，耗时必须跟着涨。"""
    usable = [s for s in samples if s.get(time_key, 0.0) > 0.0]
    if len(usable) < 2:
        return True  # 档位不足，该判据不适用（不据此否决）
    lo, hi = min(usable, key=lambda s: s[size_key]), max(usable, key=lambda s: s[size_key])
    if hi[size_key] < lo[size_key] * SCALE_BYTES_RATIO_MIN:
        return True  # 规模差不够，判据不成立
    return hi[time_key] >= lo[time_key] * SCALE_TIME_RATIO_MIN


def _method_ratio_ok(wall_ms: float, event_ms: float) -> bool:
    """双方法交叉自检：墙钟与 event 读数不应差出数量级。"""
    lo = min(wall_ms, event_ms)
    if lo <= 0.0:
        return True  # 缺一侧读数，该判据不适用
    return max(wall_ms, event_ms) / lo <= METHOD_RATIO_MAX


def _upper_envelope(values: list[float]) -> float:
    """上包络 = 各档最大值：**峰值口径**（服务于"理论最小耗时"的下界估计）。

    保留各档样本供核对；若某档明显高于其余档，说明该档受缓存 / 启动开销影响，
    引用峰值做下界估计时可一并说明档位差异。
    """
    return max(values) if values else 0.0


def probe_bandwidth(torch, device: str) -> dict:
    """有效带宽：d2d 拷贝 bytes = 2 × 张量字节（读+写）。"""
    out = []
    for mb, rows, cols in CONFIGS_BW:
        n = max(rows * cols, mb * 1024 * 1024 // 4)
        src = torch.empty(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)
        times = [_time_once(torch, device, lambda src=src, dst=dst: dst.copy_(src)) for _ in range(REPEATS)]
        best = min(times)
        out.append({"bytes": n * 4 * 2, "ms": best, "gbps": (n * 4 * 2 / 1e9) / (best / 1e3)})
    best = max(out, key=lambda item: item["gbps"])
    return {"bw_gbps": _upper_envelope([item["gbps"] for item in out]), "bandwidth_samples": out, "picked": best}


def probe_gemm(torch, device: str) -> dict:
    """峰值算力：FLOPs = 2·M·N·K（bf16）。"""
    out = []
    for size in CONFIGS_GEMM:
        a = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        b = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        times = [_time_once(torch, device, lambda a=a, b=b: a @ b) for _ in range(REPEATS)]
        best = min(times)
        flops = 2.0 * size**3
        out.append({"flops": flops, "ms": best, "tflops": flops / (best / 1e3) / 1e12})
    return {"tflops_bf16": _upper_envelope([item["tflops"] for item in out]), "gemm_samples": out}


def probe_fixed_overhead(torch, device: str) -> dict:
    """固定开销：极小 kernel（1 元素 add）的启动耗时中位数。"""
    x = torch.zeros(1, device=device)
    y = torch.ones(1, device=device)
    times = [_time_once(torch, device, lambda: x + y) for _ in range(20)]
    return {"fixed_overhead_us": statistics.median(times) * 1e3, "samples_us": [t * 1e3 for t in times]}


def probe_noise_floor(torch, device: str) -> dict:
    """噪声地板：同一中等 kernel 重复 2N 次，取前后两半中位数差（相对值）。"""
    size = 4096
    a = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    times = [_time_once(torch, device, lambda a=a, b=b: a @ b) for _ in range(REPEATS * 2)]
    half = len(times) // 2
    drift = abs(statistics.median(times[half:]) - statistics.median(times[:half])) / statistics.median(times)
    spread = (max(times) - min(times)) / statistics.median(times)
    return {"noise_floor_pct": max(drift, spread) * 100, "samples_ms": times}


def probe_event_timing(torch, device: str) -> dict:
    """平台能力辨识：同一次 workload 分别用墙钟与 event 计时，只比两者比值。

    不一致**不否决**本次测量：含 launch 开销的墙钟口径只会偏保守、不会漏算执行；
    该结论用于判定「本平台能否用 event 当测量口径」，供后续采集/分析引用。
    """
    if not device.startswith("npu"):
        return {"applicable": False, "ratio": 0.0, "aligned": True}
    size = CONFIGS_GEMM[-1]
    a = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    wall_ms = _time_once(torch, device, lambda a=a, b=b: a @ b)
    event_ms = _time_once_event(torch, device, lambda a=a, b=b: a @ b)
    ratio = max(wall_ms, event_ms) / min(wall_ms, event_ms) if min(wall_ms, event_ms) > 0 else 0.0
    return {
        "applicable": True,
        "wall_ms": wall_ms,
        "event_ms": event_ms,
        "ratio": ratio,
        "ratio_max": METHOD_RATIO_MAX,
        "aligned": _method_ratio_ok(wall_ms, event_ms),
    }


def _selftest(base=None) -> int:
    """不碰设备：校验上包络、空数据退化、公式方向与两条口径自检判据。"""
    failures: list[str] = []
    if _upper_envelope([1.0, 9.0, 3.0]) != 9.0:
        failures.append("上包络取的不是最大值")
    if _upper_envelope([]) != 0.0:
        failures.append("空输入未退化为 0")
    # 带宽公式方向：bytes/(ms/1e3)/1e9 → GB/s（1 GB 用 10 ms ⇒ 100 GB/s）
    gbps = (1e9 / (10 / 1e3)) / 1e9
    if abs(gbps - 100.0) > 1e-6:
        failures.append(f"带宽公式方向错：{gbps}")
    # FLOPs：2·M·N·K；M=N=K=1024 ⇒ 2.1e9，若 1 ms 完成 ⇒ 2.1 TFLOPs
    tflops = 2.0 * 1024**3 / (1 / 1e3) / 1e12
    if abs(tflops - 2.147) > 0.01:
        failures.append(f"GEMM FLOPs 公式错：{tflops}")
    # 规模不变性：字节差 16×、耗时几乎不变 ⇒ 必须判可疑（固定窗口签名）
    fixed_window = [{"bytes": 1.34e8, "ms": 0.0140}, {"bytes": 5.37e8, "ms": 0.0139}, {"bytes": 2.15e9, "ms": 0.0140}]
    if _scale_ok(fixed_window, "bytes", "ms"):
        failures.append("固定窗口签名（字节 16× / 耗时 1.0×）未被规模不变性判据拦下")
    # 正常缩放：字节 16×、耗时 15× ⇒ 必须放行
    healthy = [{"bytes": 1.34e8, "ms": 0.10}, {"bytes": 2.15e9, "ms": 1.55}]
    if not _scale_ok(healthy, "bytes", "ms"):
        failures.append("正常缩放的样本被规模不变性判据误拦")
    # 规模差不足（<4×）时判据不适用，不得否决
    if not _scale_ok([{"bytes": 1.0e8, "ms": 0.02}, {"bytes": 2.0e8, "ms": 0.02}], "bytes", "ms"):
        failures.append("规模差不足时应判「不适用」而非否决")
    # 双方法交叉：比值 15.6 判可疑，比值 1.1 放行
    if _method_ratio_ok(2.97, 0.19):
        failures.append("两种计时口径差 15× 未被交叉自检拦下")
    if not _method_ratio_ok(2.25, 2.50):
        failures.append("两种计时口径接近时被交叉自检误拦")
    if failures:
        print("perf_probe --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("perf_probe --selftest: PASS（上包络/退化/带宽与 FLOPs 公式/规模不变性/双方法交叉）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="设备性能探针（L1 峰值口径）")
    parser.add_argument("--out", help="结果 JSON 路径（默认 probe.json）")
    parser.add_argument("--device", default="npu:0", help="设备串（默认 npu:0）")
    parser.add_argument("--max-bw-gbps", type=float, default=0.0, help="可选绝对上限（GB/s；默认 0 = 不启用）")
    parser.add_argument("--max-tflops", type=float, default=0.0, help="可选绝对上限（TFLOPs；默认 0 = 不启用）")
    parser.add_argument("--selftest", action="store_true", help="不碰设备，仅自测公式与自检判据")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    try:
        import torch
    except ImportError:
        print("perf_probe: 前置条件缺失：未安装 torch（本探针需在目标设备环境运行）", file=sys.stderr)
        return 2
    if args.device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            print("perf_probe: 前置条件缺失：未安装 torch_npu（或改用 --device cpu）", file=sys.stderr)
            return 2

    result = {
        "device": args.device,
        **probe_bandwidth(torch, args.device),
        **probe_gemm(torch, args.device),
        **probe_fixed_overhead(torch, args.device),
        **probe_noise_floor(torch, args.device),
    }
    result["selfcheck"] = {
        "bandwidth_scale": _scale_ok(result["bandwidth_samples"], "bytes", "ms"),
        "gemm_scale": _scale_ok(result["gemm_samples"], "flops", "ms"),
        "event_timing": probe_event_timing(torch, args.device),
    }
    checks = [result["selfcheck"]["bandwidth_scale"], result["selfcheck"]["gemm_scale"]]
    if args.max_bw_gbps > 0:
        checks.append(result["bw_gbps"] <= args.max_bw_gbps)
    if args.max_tflops > 0:
        checks.append(result["tflops_bf16"] <= args.max_tflops)
    result["plausible"] = all(checks)

    out = args.out or "probe.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    if not result["plausible"]:
        print(
            f"perf_probe: 规模不变性自检不通过（selfcheck={result['selfcheck']}）⇒ 拒绝输出峰值口径；"
            f"原始样本留在 {out}（plausible=false），不得喂给 theory_columns",
            file=sys.stderr,
        )
        return 3
    event_timing = result["selfcheck"]["event_timing"]
    if event_timing.get("applicable") and not event_timing["aligned"]:
        print(
            f"perf_probe: 注意——本平台 event 计时与墙钟差 {event_timing['ratio']:.1f}×"
            f"（墙钟 {event_timing['wall_ms']:.3f}ms / event {event_timing['event_ms']:.3f}ms）"
            f"⇒ event 不可作测量口径，本次峰值按墙钟口径出数",
            file=sys.stderr,
        )
    print(
        f"perf_probe: bw={result['bw_gbps']:.1f} GB/s  tflops_bf16={result['tflops_bf16']:.1f}  "
        f"fixed_overhead={result['fixed_overhead_us']:.1f}us  floor={result['noise_floor_pct']:.1f}%  -> {out}"
    )
    return 0
    print(
        f"perf_probe: bw={result['bw_gbps']:.1f} GB/s  tflops_bf16={result['tflops_bf16']:.1f}  "
        f"fixed_overhead={result['fixed_overhead_us']:.1f}us  floor={result['noise_floor_pct']:.1f}%  -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
