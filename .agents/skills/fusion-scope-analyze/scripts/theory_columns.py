#!/usr/bin/env python3
"""逐算子理论列（L1 收益口径的锚）：字节 ÷ 实测有效带宽、FLOPs ÷ 实测峰值算力。

用途：把「可回收量」从经验系数改成物理锚定——`可回收量 = duration − theoretical_operator_time_us`；
`duration_over_theoretical ≈ 1` 表示该算子已在物理下限上（即便 `vec < mte` 也无空间）。

**峰值口径必须由使用者给出**（`--peak-bw-gbps` / `--peak-tflops`），缺参数即拒算——不编造默认值。

用法：
    python theory_columns.py --csv <op_summary.csv> --peak-bw-gbps 1200 --peak-tflops 400 -o theory.csv
    python theory_columns.py --selftest

退出码：0 = 成功；1 = 数据不满足前置（无 shape/dtype 列等）；2 = 前置条件缺失（文件不存在 / 缺峰值参数）。
输出列：theoretical_memory_time_us / theoretical_compute_time_us / theoretical_operator_time_us /
        fixed_overhead_us / bound_type / duration_over_theoretical / theory_supported
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

DTYPE_BYTES = {
    "DT_BF16": 2,
    "DT_FLOAT16": 2,
    "DT_INT16": 2,
    "DT_FLOAT": 4,
    "DT_INT32": 4,
    "DT_UINT32": 4,
    "DT_FLOAT8_E4M3": 1,
    "DT_FLOAT8_E5M2": 1,
    "DT_FLOAT8_E8M0": 1,
    "DT_INT8": 1,
    "DT_UINT8": 1,
    "DT_BOOL": 1,
    "DT_INT64": 8,
    "DT_UINT64": 8,
    "DT_DOUBLE": 8,
}
MATMUL_HINT = ("MatMul", "Matmul", "Addmm", "GroupedMatmul", "GroupedMatMul", "Gemm")


def _num(value: str) -> float:
    try:
        return float((value or "").strip())
    except ValueError:
        return 0.0


def _shapes(field: str) -> list[list[int]]:
    """解析 `"a,b;c"` 形式的多张量 shape；出现动态维（≤0/非数字）该张量返回 []。"""
    out: list[list[int]] = []
    for chunk in (field or "").replace('"', "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            out.append([])
            continue
        # 动态/符号维（含 '-' 或字母标记）⇒ 视为未知，不参与理论计算（不得编造）
        if "-" in chunk or re.search(r"[A-Za-z]", chunk):
            out.append([])
            continue
        out.append([int(x) for x in re.findall(r"\d+", chunk)])
    return out


def _dtypes(field: str) -> list[int]:
    return [DTYPE_BYTES.get(tok, 0) for tok in re.findall(r"DT_[A-Z0-9_]+", field or "")]


def _bytes(shape_field: str, dtype_field: str) -> int:
    sizes, dts = _shapes(shape_field), _dtypes(dtype_field)
    total = 0
    for idx, dims in enumerate(sizes):
        if not dims:
            return -1  # 动态/未知 shape ⇒ 不支持
        elems = 1
        for dim in dims:
            elems *= dim
        size = dts[idx] if idx < len(dts) else (dts[0] if dts else 0)
        if size == 0:
            return -1
        total += elems * size
    return total


def theory_for(row: dict, bw_gbps: float, tflops: float) -> dict:
    """单算子理论列；不支持时 theory_supported=False 且不编造数值。"""
    name = row.get("Op Name") or row.get("OP Type") or ""
    dur = _num(row.get("Task Duration(us)") or row.get("Duration(us)"))
    in_b = _bytes(row.get("Input Shapes", ""), row.get("Input Data Types", ""))
    out_b = _bytes(row.get("Output Shapes", ""), row.get("Output Data Types", ""))
    unsupported = "0.0" == f"{0.0:.1f}" and False  # 占位，保持函数纯计算
    res = {
        "theoretical_memory_time_us": "",
        "theoretical_compute_time_us": "",
        "theoretical_operator_time_us": "",
        "fixed_overhead_us": "",
        "bound_type": "unknown",
        "duration_over_theoretical": "",
        "theory_supported": False,
    }
    if unsupported or in_b < 0 or out_b < 0 or bw_gbps <= 0:
        return res
    total_bytes = in_b + out_b
    mem_us = total_bytes / (bw_gbps * 1e3)  # bytes / (GB/s) → µs
    flops = 0.0
    if any(hint in name for hint in MATMUL_HINT):
        shapes_out = _shapes(row.get("Output Shapes", ""))
        if shapes_out and len(shapes_out[0]) >= 2:
            m, n = shapes_out[0][-2], shapes_out[0][-1]
            k = 0
            for dims in _shapes(row.get("Input Shapes", "")):
                if len(dims) >= 2:
                    k = dims[-1] if dims[-1] != n else (dims[-2] if len(dims) >= 2 else 0)
            if k:
                flops = 2.0 * m * n * k
    comp_us = flops / (tflops * 1e6) if (flops and tflops > 0) else 0.0  # FLOPs / (TFLOPs) → µs
    supported = mem_us > 0 or comp_us > 0
    if not supported:
        return res
    theo = max(mem_us, comp_us)
    # fail-closed：实测快于理论（ratio < 1）说明估算本身不可信（如 matmul 收缩维取错），
    # 此时不得输出数值，判 theory_supported=False，交由人/代码复核形状与峰值口径。
    if dur > 0 and theo > 0 and dur / theo < 1.0:
        return res
    res.update(
        theoretical_memory_time_us=f"{mem_us:.3f}",
        theoretical_compute_time_us=f"{comp_us:.3f}" if comp_us else "",
        theoretical_operator_time_us=f"{theo:.3f}",
        fixed_overhead_us=f"{max(dur - theo, 0.0):.3f}" if dur > 0 else "",
        bound_type="memory" if mem_us >= comp_us else "compute",
        duration_over_theoretical=f"{dur / theo:.3f}" if (dur > 0 and theo > 0) else "",
        theory_supported=True,
    )
    return res


def run(csv_path: Path, bw: float, tflops: float, out_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        print("theory_columns: 前置不满足：CSV 无数据行", file=sys.stderr)
        return 1
    if not any("Shape" in f for f in fields) or not any("Data Type" in f for f in fields):
        print("theory_columns: 前置不满足：CSV 缺少 shape / dtype 列（无法算字节）", file=sys.stderr)
        return 1
    extra = [
        "theoretical_memory_time_us",
        "theoretical_compute_time_us",
        "theoretical_operator_time_us",
        "fixed_overhead_us",
        "bound_type",
        "duration_over_theoretical",
        "theory_supported",
    ]
    supported = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields + extra)
        writer.writeheader()
        for row in rows:
            res = theory_for(row, bw, tflops)
            supported += bool(res["theory_supported"])
            writer.writerow({**row, **res})
    print(f"theory_columns: {len(rows)} 行，支持理论列 {supported} 行 → {out_path}")
    return 0


def _selftest(base: Path | None = None) -> int:
    """负样本自测：数值正确性 + 三类前置/口径拒绝。"""
    failures: list[str] = []
    root = (base or Path(__file__).resolve().parents[1]) / ".theory_columns_selftest"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    # ① 数值：1000 万字节 / 1000 GB/s = 10 µs；matmul 2*128*128*128 FLOPs / 1 TFLOPs = 4.19 µs
    row = {
        "Op Name": "aclnnMatMul_x_MatMulV3",
        "Task Duration(us)": "20",
        "Input Shapes": '"128,128;128,128"',
        "Input Data Types": '"DT_BF16;DT_BF16"',
        "Output Shapes": '"128,128"',
        "Output Data Types": '"DT_BF16"',
    }
    res = theory_for(row, 1000.0, 1.0)
    if not res["theory_supported"]:
        failures.append("数值样例被判不支持")
    if abs(_num(res["theoretical_memory_time_us"]) - 0.098304) > 1e-3:
        failures.append(f"memory 理论值不符：{res['theoretical_memory_time_us']}（期望 0.098304µs）")
    if abs(_num(res["theoretical_compute_time_us"]) - 4.194304) > 1e-3:
        failures.append(f"compute 理论值不符：{res['theoretical_compute_time_us']}（期望 4.194304µs）")
    if res["bound_type"] != "compute":
        failures.append(f"bound_type 应为 compute，实为 {res['bound_type']}")
    if abs(_num(res["duration_over_theoretical"]) - 4.768) > 0.01:
        failures.append(f"duration/theory 不符：{res['duration_over_theoretical']}")

    # ② 动态 shape ⇒ 不支持（不得编造）
    dyn = dict(row, **{"Input Shapes": '"-1,128;128,128"'})
    if theory_for(dyn, 1000.0, 1.0)["theory_supported"]:
        failures.append("动态 shape 未被拒（不得编造理论值）")

    # ③ 缺峰值参数 ⇒ 不支持
    if theory_for(row, 0.0, 1.0)["theory_supported"]:
        failures.append("缺带宽参数时仍给出理论值")

    # ④ CLI：缺参数 => 退出码 2
    csv_path = root / "in.csv"
    csv_path.write_text(
        "Op Name,Task Duration(us),Input Shapes,Input Data Types,Output Shapes,Output Data Types\n"
        'k,"20","128,128","DT_BF16","128,128","DT_BF16"\n',
        encoding="utf-8",
    )
    if main(["--csv", str(csv_path), "-o", str(root / "out.csv")]) != 2:
        failures.append("缺峰值参数时 CLI 未返回 2")
    code = main(["--csv", str(csv_path), "--peak-bw-gbps", "1000", "--peak-tflops", "1", "-o", str(root / "out.csv")])
    if code != 0 or "theoretical_operator_time_us" not in (root / "out.csv").read_text(encoding="utf-8"):
        failures.append("正常路径未产出理论列")
    shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("theory_columns --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("theory_columns --selftest: PASS（数值 3 例 + 三类拒绝 + CLI 两路径）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="逐算子理论列（L1 收益锚）")
    parser.add_argument("--csv", help="输入 op_summary / kernel_details CSV")
    parser.add_argument("--peak-bw-gbps", type=float, default=0.0, help="实测有效带宽（GB/s）")
    parser.add_argument("--peak-tflops", type=float, default=0.0, help="实测峰值算力（TFLOPs，bf16）")
    parser.add_argument("-o", "--out", help="输出 CSV（含理论列）")
    parser.add_argument("--selftest", action="store_true", help="负样本自测并退出")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.csv or not args.out:
        print("theory_columns: 需要 --csv 与 -o（或 --selftest）", file=sys.stderr)
        return 2
    if args.peak_bw_gbps <= 0 or args.peak_tflops <= 0:
        print("theory_columns: 前置条件缺失：必须给出 --peak-bw-gbps 与 --peak-tflops（不编造默认值）", file=sys.stderr)
        return 2
    path = Path(args.csv)
    if not path.is_file():
        print(f"theory_columns: 前置条件缺失：CSV 不存在（{path}）", file=sys.stderr)
        return 2
    return run(path, args.peak_bw_gbps, args.peak_tflops, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
