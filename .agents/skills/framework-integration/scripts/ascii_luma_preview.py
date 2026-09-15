#!/usr/bin/env python3
"""ascii_luma_preview.py —— 解码输出亮度预览与对拍（零依赖、零 NPU、零网络）。

做：读入一目录的 8bit PNG/PGM 帧 → ① 逐帧聚合指标（均值 / 标准差 / 钳位占比）；
② 降采样成 **ASCII 亮度图**（可人眼读、可贴进存证文件）；③ 两目录**同尺寸帧对拍**：
逐帧灰度相关系数 + 两侧指标并排（判「替换解码器是否接对」，判据见
`accuracy-gate/references/quality-gate.md`）。
不做：视频容器解码、色彩空间变换、质量门禁终判（门禁结论在 quality-gate.md）。

为什么需要：执行侧常常拿不到图像输入 → 视觉判卷只能记 `inconclusive`；本工具补上
「可机读 + 可人眼读」的存证，避免把「无法看图」当成质量通过的等价物。相关系数对全局
尺度不变，故它判**接对与否**；画质好坏另看 SSIM/视觉门。

用法：
    python ascii_luma_preview.py --input <帧目录> --out preview.txt
    python ascii_luma_preview.py --input <A> --compare <B> --out compare.txt
    python ascii_luma_preview.py --input <A> --compare <B> --json compare.json
退出码：0 = 完成；1 = 输入不可读 / 帧数或尺寸不匹配。
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path

ASCII_RAMP = " .:-=+*#%@"
DEFAULT_COLS = 56
DEFAULT_ROWS = 24
DEFAULT_MAX_MAPS = 4
CLIP_LO = 1
CLIP_HI = 254
FRAME_SUFFIXES = (".png", ".pgm")


def _paeth(a: int, b: int, c: int) -> int:
    """PNG Paeth 预测器（filter type 4）。"""
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw: bytes, height: int, bpp: int, stride: int) -> list[bytes]:
    """逐行还原 PNG 扫描线（filter 0-4）。"""
    rows: list[bytes] = []
    prev = bytearray(stride)
    pos = 0
    for _ in range(height):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if ftype == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                up_left = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(left, prev[i], up_left)) & 0xFF
        elif ftype != 0:
            raise ValueError(f"不支持的 PNG filter type {ftype}")
        rows.append(bytes(line))
        prev = line
    return rows


def load_png(path: Path) -> tuple[int, int, bytes]:
    """读 8bit 非隔行 PNG → (width, height, 灰度字节)。支持灰/RGB/带 alpha。"""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name}：不是 PNG 文件")
    pos = 8
    width = height = 0
    bit_depth = color_type = 0
    idat = bytearray()
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
            width, height, bit_depth, color_type = head[0], head[1], head[2], head[3]
            if head[6] != 0:
                raise ValueError(f"{path.name}：不支持隔行 PNG")
            if bit_depth != 8:
                raise ValueError(f"{path.name}：仅支持 8bit（实际 {bit_depth}）")
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"{path.name}：不支持的色彩类型 {color_type}")
    rows = _unfilter(zlib.decompress(bytes(idat)), height, channels, width * channels)
    if len(rows) * width != width * height:
        raise ValueError(f"{path.name}：像素数据不足")
    gray = bytearray(width * height)
    for y, row in enumerate(rows):
        base = y * width
        for x in range(width):
            i = x * channels
            if channels >= 3:
                r, g, b = row[i], row[i + 1], row[i + 2]
                gray[base + x] = (299 * r + 587 * g + 114 * b) // 1000
            else:
                gray[base + x] = row[i]
    return width, height, bytes(gray)


def _read_pgm_tokens(data: bytes) -> tuple[list[bytes], int]:
    """读 PGM 头部 token（容忍 `#` 注释），返回 (tokens, 像素起点)。"""
    tokens: list[bytes] = []
    idx = 0
    while len(tokens) < 4 and idx < len(data):
        ch = data[idx:idx + 1]
        if ch.isspace():
            idx += 1
        elif ch == b"#":
            while idx < len(data) and data[idx:idx + 1] not in (b"\n", b"\r"):
                idx += 1
        else:
            start = idx
            while idx < len(data) and not data[idx:idx + 1].isspace():
                idx += 1
            tokens.append(data[start:idx])
    return tokens, idx + 1


def load_pgm(path: Path) -> tuple[int, int, bytes]:
    """读二进制 PGM(P5) → (width, height, 灰度字节)。"""
    data = path.read_bytes()
    tokens, start = _read_pgm_tokens(data)
    if len(tokens) < 4 or tokens[0] != b"P5":
        raise ValueError(f"{path.name}：仅支持二进制 PGM（P5）")
    width, height, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    if maxval != 255:
        raise ValueError(f"{path.name}：仅支持 maxval=255（实际 {maxval}）")
    gray = data[start:start + width * height]
    if len(gray) != width * height:
        raise ValueError(f"{path.name}：像素数据不足")
    return width, height, bytes(gray)


def load_frame(path: Path) -> tuple[int, int, bytes]:
    """按扩展名分派解码器。"""
    suffix = path.suffix.lower()
    if suffix == ".png":
        return load_png(path)
    if suffix == ".pgm":
        return load_pgm(path)
    raise ValueError(f"{path.name}：不支持的扩展名（仅 {'/'.join(FRAME_SUFFIXES)}）")


def list_frames(directory: Path) -> list[Path]:
    """按文件名序收集目录下的帧文件（非递归）。"""
    if not directory.is_dir():
        raise ValueError(f"{directory}：不是目录")
    frames = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES
    )
    if not frames:
        raise ValueError(f"{directory}：未找到 {'/'.join(FRAME_SUFFIXES)} 帧")
    return frames


def frame_metrics(gray: bytes) -> dict[str, float]:
    """逐帧聚合指标：均值 / 标准差 / 极值 / 钳位占比。"""
    n = len(gray)
    mean = sum(gray) / n
    var = sum((v - mean) ** 2 for v in gray) / n
    low = sum(1 for v in gray if v <= CLIP_LO)
    high = sum(1 for v in gray if v >= CLIP_HI)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": float(min(gray)),
        "max": float(max(gray)),
        "clipped_frac": (low + high) / n,
        "clipped_low_frac": low / n,
        "clipped_high_frac": high / n,
    }


def ascii_map(gray: bytes, width: int, height: int, cols: int, rows: int) -> list[str]:
    """块均值降采样 → ASCII 亮度图（每行 cols 字符，共 rows 行）。"""
    last = len(ASCII_RAMP) - 1
    out: list[str] = []
    for ry in range(rows):
        y0 = ry * height // rows
        y1 = max(y0 + 1, (ry + 1) * height // rows)
        line: list[str] = []
        for rx in range(cols):
            x0 = rx * width // cols
            x1 = max(x0 + 1, (rx + 1) * width // cols)
            acc = 0
            count = 0
            for y in range(y0, y1):
                base = y * width
                for x in range(x0, x1):
                    acc += gray[base + x]
                    count += 1
            line.append(ASCII_RAMP[round(acc / count / 255 * last)])
        out.append("".join(line))
    return out


def gray_correlation(left: bytes, right: bytes) -> float:
    """两帧灰度值的 Pearson 相关系数；任一为常量时返回 nan。"""
    n = min(len(left), len(right))
    if n == 0:
        return math.nan
    mean_l = sum(left[:n]) / n
    mean_r = sum(right[:n]) / n
    num = 0.0
    dev_l = 0.0
    dev_r = 0.0
    for i in range(n):
        dl = left[i] - mean_l
        dr = right[i] - mean_r
        num += dl * dr
        dev_l += dl * dl
        dev_r += dr * dr
    if dev_l <= 0 or dev_r <= 0:
        return math.nan
    return num / math.sqrt(dev_l * dev_r)


def _fmt_metrics(tag: str, m: dict[str, float]) -> str:
    return (
        f"{tag}: mean={m['mean']:.2f} std={m['std']:.2f} "
        f"min={m['min']:.0f} max={m['max']:.0f} "
        f"clipped={m['clipped_frac'] * 100:.2f}%"
    )


def render_single(
    frames: list[Path], cols: int, rows: int, max_maps: int
) -> tuple[list[str], dict]:
    """单目录预览：指标表 + ASCII 图。"""
    lines: list[str] = []
    report: dict = {"frames": [], "metrics": {}}
    stats: list[dict[str, float]] = []
    for path in frames:
        width, height, gray = load_frame(path)
        m = frame_metrics(gray)
        stats.append(m)
        report["frames"].append(path.name)
        report["metrics"][path.name] = m
        lines.append(f"{path.name}  {width}x{height}  {_fmt_metrics('gray', m)}")
    report["size"] = [width, height]
    report["summary"] = {
        "mean_of_means": sum(s["mean"] for s in stats) / len(stats),
        "mean_std": sum(s["std"] for s in stats) / len(stats),
        "max_clipped_frac": max(s["clipped_frac"] for s in stats),
    }
    extremes = {
        "mean": report["summary"]["mean_of_means"],
        "std": report["summary"]["mean_std"],
        "min": min(s["min"] for s in stats),
        "max": max(s["max"] for s in stats),
        "clipped_frac": report["summary"]["max_clipped_frac"],
    }
    lines.append("")
    lines.append(_fmt_metrics("summary(all; clipped=max)", extremes))
    lines.append("")
    for path in frames[:max_maps]:
        _, _, gray = load_frame(path)
        lines.append(f"--- {path.name} (ASCII {cols}x{rows}) ---")
        lines.extend(ascii_map(gray, width, height, cols, rows))
        lines.append("")
    return lines, report


def render_compare(left_dir: Path, right_dir: Path, cols: int, rows: int, max_maps: int):
    """两目录对拍：逐帧灰度相关 + 两侧指标并排。"""
    left_frames = list_frames(left_dir)
    right_frames = list_frames(right_dir)
    if len(left_frames) != len(right_frames):
        raise ValueError(f"帧数不匹配：{len(left_frames)} vs {len(right_frames)}")
    lines: list[str] = []
    report: dict = {"pairs": [], "correlation": {}}
    corrs: list[float] = []
    for lf, rf in zip(left_frames, right_frames):
        lw, lh, lg = load_frame(lf)
        rw, rh, rg = load_frame(rf)
        if (lw, lh) != (rw, rh):
            raise ValueError(f"尺寸不匹配 {lf.name}({lw}x{lh}) vs {rf.name}({rw}x{rh})")
        corr = gray_correlation(lg, rg)
        corrs.append(corr)
        lm, rm = frame_metrics(lg), frame_metrics(rg)
        report["pairs"].append({"left": lf.name, "right": rf.name, "corr": corr})
        report["correlation"][lf.name] = corr
        lines.append(f"{lf.name} vs {rf.name}  corr={corr:.4f}")
        lines.append(f"    {_fmt_metrics('native', lm)}")
        lines.append(f"    {_fmt_metrics('replace', rm)}")
    valid = [c for c in corrs if not math.isnan(c)]
    summary = {
        "frames": len(corrs),
        "mean_corr": sum(valid) / len(valid) if valid else math.nan,
        "min_corr": min(valid) if valid else math.nan,
        "max_corr": max(valid) if valid else math.nan,
        "nan_frames": len(corrs) - len(valid),
    }
    report["summary"] = summary
    lines.append("")
    lines.append(
        f"summary: frames={summary['frames']} mean_corr={summary['mean_corr']:.4f} "
        f"min_corr={summary['min_corr']:.4f} max_corr={summary['max_corr']:.4f} "
        f"nan={summary['nan_frames']}"
    )
    lines.append("")
    lines.append("判据（quality-gate.md）：corr 高 ⇒ 接口接对（对全局尺度不变）；")
    lines.append("画质档位另看 SSIM/视觉门；clipped 明显高于 native ⇒ 后处理/归一化约定有误。")
    lines.append("")
    for lf, rf in list(zip(left_frames, right_frames))[:max_maps]:
        _, _, lg = load_frame(lf)
        _, _, rg = load_frame(rf)
        lines.append(f"--- {lf.name} | native ---")
        lines.extend(ascii_map(lg, lw, lh, cols, rows))
        lines.append(f"--- {rf.name} | replace ---")
        lines.extend(ascii_map(rg, rw, rh, cols, rows))
        lines.append("")
    return lines, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="解码输出亮度预览与对拍（零依赖；出 ASCII 亮度图 + 聚合指标）",
    )
    parser.add_argument("--input", required=True, type=Path, help="帧目录（PNG/PGM）")
    parser.add_argument("--compare", type=Path, help="对拍目录（与 --input 逐帧同尺寸）")
    parser.add_argument("--out", type=Path, help="存证文本落点（默认打印到 stdout）")
    parser.add_argument("--json", type=Path, help="指标 JSON 落点（可选）")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="ASCII 图列数")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="ASCII 图行数")
    parser.add_argument("--max-maps", type=int, default=DEFAULT_MAX_MAPS, help="最多几帧 ASCII 图")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.compare is not None:
            lines, report = render_compare(
                args.input, args.compare, args.cols, args.rows, args.max_maps
            )
        else:
            frames = list_frames(args.input)
            lines, report = render_single(frames, args.cols, args.rows, args.max_maps)
    except (ValueError, OSError, struct.error, zlib.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = "\n".join(lines) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.json is not None:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
