#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""
从 MindIE-SD docs 自动生成 mindiesd-features.md。

用法:
    python scripts/refresh_features.py \
        --docs-dir <docs-directory> \
        --output references/mindiesd-features.md
"""

import argparse
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_table_lines(lines, start_idx):
    """Parse a markdown table from lines, returning headers and rows.
    Returns (headers, rows) or (None, None) if no table found at start_idx.
    """
    if start_idx >= len(lines) or not lines[start_idx].startswith("|"):
        return None, None

    header_line = lines[start_idx].strip()
    sep_line = lines[start_idx + 1].strip() if start_idx + 1 < len(lines) else ""
    if not re.match(r"^\|[\s\-:|]+\|$", sep_line):
        return None, None

    headers = [h.strip() for h in header_line.split("|")[1:-1]]
    rows = []
    for i in range(start_idx + 2, len(lines)):
        line = lines[i].strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        rows.append(cells)
    return headers, rows


def find_section(lines, heading_pattern):
    """Find the start index of a section by heading regex."""
    for i, line in enumerate(lines):
        if re.match(heading_pattern, line.strip()):
            return i
    return None


def find_tables_in_section(lines, section_heading_pat):
    """Find all tables within a given section."""
    start = find_section(lines, section_heading_pat)
    if start is None:
        return [], []
    tables = []
    section_lines = []
    i = start
    while i < len(lines):
        section_lines.append(lines[i])
        if lines[i].startswith("|") and i > 0 and not lines[i - 1].startswith("|"):
            headers, rows = parse_table_lines(lines, i)
            if headers:
                tables.append((headers, rows))
        i += 1
        if i < len(lines) and lines[i].startswith("## ") and not lines[i].startswith("### "):
            break
    return tables, section_lines


def extract_quantization(docs_dir, output_lines, rel_path):
    """Extract quantization info from quantization.md."""
    path = docs_dir / "quantization.md"
    if not path.exists():
        output_lines.append(f"\n> ⚠️ 未找到 {path}，MatMul 量化/Attention 优化章节未更新。\n")
        return

    lines = path.read_text(encoding="utf-8").splitlines()

    output_lines.append("## MatMul 量化\n")
    output_lines.append("MatMul 本身不通过融合优化，而是通过低比特量化减少显存带宽和计算量。\n")
    output_lines.append("| 瓶颈指标 | MindIE-SD 方案 | 接口 | 硬件约束 | 模型约束 |")
    output_lines.append("|---------|---------------|------|---------|---------|")

    found_algos = set()

    for _, line in enumerate(lines):
        if line.startswith("| W") or line.startswith("| **W") or line.startswith("| `W"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) >= 3:
                algo = cells[0].strip("**`")
                if algo in found_algos:
                    continue
                found_algos.add(algo)

                if "MXFP8" in algo or "mxfp8" in algo.lower():
                    strategy = "W8A8_MXFP8（MX 格式）"
                elif "MXFP4" in algo or "mxfp4" in algo.lower():
                    strategy = f"{algo}"
                elif "INT8" in algo or "W8A8" in algo:
                    strategy = f"{algo}（INT8 权重激活量化）"
                elif "INT4" in algo or "W4A" in algo:
                    strategy = f"{algo}"
                elif "FP8" in algo:
                    strategy = f"{algo}"
                else:
                    strategy = algo

                bottleneck = "MatMul 占比 >50%"
                if "W4A4" in algo:
                    bottleneck += "，高压缩"
                if "DYNAMIC" in algo:
                    bottleneck += "，动态量化"
                if "TIMESTEP" in algo:
                    bottleneck = "MatMul 占比 >50%，时间步动态"
                if "W4A16" in algo or "W8A16" in algo:
                    bottleneck = "仅权重量化需求"

                interface = f'`quantize(model, "quant_desc_{algo.lower()}_0.json")`'
                hardware = "Atlas 800I A2"
                model_constraint = "—"

                output_lines.append(f"| {bottleneck} | {strategy} | {interface} | {hardware} | {model_constraint} |")

    output_lines.append("")
    output_lines.append(
        f"> 量化描述符和权重文件由 msmodelslim 工具预导出。详见 [quantization.md]({rel_path}/quantization.md)。\n"
    )

    # FA quantization (Attention)
    fa_start = find_section(lines, r"^##\s+FA量化")
    if fa_start:
        output_lines.append("## Attention 优化\n")
        output_lines.append("Attention 本身不可通过算子融合加速。优化手段为 FA 量化（FP8块量化 Q/K/V）和稀疏注意力。\n")
        output_lines.append("| 瓶颈指标 | MindIE-SD 方案 | 接口 | 硬件约束 | 模型约束 |")
        output_lines.append("|---------|---------------|------|---------|---------|")

        output_lines.append(
            "| Attention 占比 >30%，头间显存带宽瓶颈 | FA 量化 (FP8) "
            "| `quantize(model, ...)` 自动注入 `FP8RotateQuantFA` "
            "| **仅** Atlas 800I A2 | Q/K/V 布局支持 BNSD/BSND |"
        )

    output_lines.append("")
    output_lines.append(f"> 详见 [quantization.md]({rel_path}/quantization.md) §FA量化。\n")


def extract_sparse(docs_dir, output_lines, rel_path):
    """Extract sparse attention info from sparse.md."""
    path = docs_dir / "sparse.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()

    rf_v2_desc = []
    in_rf_v2 = False
    for line in lines:
        if "### rf_v2" in line:
            in_rf_v2 = True
        elif in_rf_v2 and line.startswith("### "):
            in_rf_v2 = False
        elif in_rf_v2:
            rf_v2_desc.append(line.strip())

    output_lines.append(
        "| Attention 占比 >30%，视频模型 | 稀疏 rf_v2 (RainFusion2.0) "
        "| `sparse_attention(q,k,v, sparse_type=\"rf_v2\", sparsity=0.8, latent_shape_q=[t,h,w])` "
        "| Atlas 800I A2 | 需 `latent_shape_q/k` |"
    )
    output_lines.append(
        "| Attention 占比 >30%，图像模型 | 稀疏 rf_v2 "
        "| `sparse_attention(q,k,v, sparse_type=\"rf_v2\", sparsity=0.6)` "
        "| Atlas 800I A2 | — |"
    )
    output_lines.append(
        "| Attention 占比 >30%，rf_v2 不兼容 | 稀疏 ada_bsa "
        "| `sparse_attention(q,k,v, sparse_type=\"ada_bsa\", cdf_threshold=...)` "
        "| Atlas 800I A2 | — |"
    )

    output_lines.append("")
    output_lines.append("> rf_v2 80% 稀疏率下端到端加速 1.5–1.8×。图像 sparsity 建议 0.6 起步，视频 0.8 起步。")
    output_lines.append(f"> 详见 [sparse.md]({rel_path}/sparse.md)。\n")


def extract_parallelism(docs_dir, output_lines, rel_path):
    """Extract parallelism info from parallelism.md."""
    path = docs_dir / "parallelism.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()

    output_lines.append("## 通信掩盖\n")
    output_lines.append("多卡场景中不可避免的通信开销可通过计算掩盖。\n")
    output_lines.append("| 场景 | MindIE-SD 方案 | 原理 | 硬件约束 |")
    output_lines.append("|------|---------------|------|---------|")

    sections = {
        "Tensor Parallel": (r"^##\s+Tensor\s+Parallel", "TP"),
        "Ring Sequence Parallel": (r"^##\s+Ring\s+Sequence\s+Parallel", "RSP"),
        "Ulysses Sequence Parallel": (r"^##\s+Ulysses\s+Sequence\s+Parallel", "USP"),
        "CFG Parallel": (r"^##\s+CFG\s+Parallel", "CFG"),
    }

    for _, (pat, abbr) in sections.items():
        sec = find_section(lines, pat)
        if sec:
            if abbr == "TP":
                output_lines.append(
                    "| 序列较长，hidden_size 大 | 张量并行 (TP) | 按行/按列切分权重，减少单卡显存 | 单机多卡 HCCS |"
                )
            elif abbr == "RSP":
                output_lines.append(
                    "| 序列较长，head_dim 大 | 环状序列并行 (RSP) | P2P 环形传递 KV，计算耗时掩盖通信 | 同机 NPU HCCS |"
                )
            elif abbr == "USP":
                output_lines.append(
                    "| head 数多，AlltoAll 带宽充裕 | Ulysses 序列并行 (USP) "
                    "| AlltoAll 在头维度重组，通信量恒定 | 并行度需整除 head_num |"
                )
            elif abbr == "CFG":
                output_lines.append("| CFG > 1 | CFG 并行 | 正负样本分卡并行，通信量极小 | ≥ 2 卡 |")

    output_lines.append("")
    output_lines.append(f"> 详见 [parallelism.md]({rel_path}/parallelism.md)。\n")


def extract_offload(docs_dir, output_lines, rel_path):
    """Extract CPU offload info from cpu_offload.md."""
    path = docs_dir / "cpu_offload.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()

    output_lines.append("## 显存优化\n")
    output_lines.append("| 瓶颈指标 | MindIE-SD 方案 | 接口 | 硬件约束 | 模型约束 |")
    output_lines.append("|---------|---------------|------|---------|---------|")

    params_found = False
    for _, line in enumerate(lines):
        if "enable_offload" in line and "(" in line and not line.startswith("```"):
            output_lines.append(
                f"| 峰值≈物理显存，block 数多 | 异步 CPU Offload | `{line.strip()}` | — | 需指定 blocks 列表 |"
            )
            params_found = True
            break

    if not params_found:
        output_lines.append(
            "| 峰值≈物理显存，block 数多 | 异步 CPU Offload "
            "| `enable_offload(model, blocks, min_reserved_blocks_count=2)` "
            "| — | 需指定 blocks 列表 |"
        )

    output_lines.append(
        "| 峰值≈物理显存，hidden_size 大 | 张量并行 (TP) | 按行/按列切分权重 | 单机多卡 HCCS | TP degree ≤ 卡数 |"
    )
    output_lines.append("| 激活值占比高 | Activation Checkpoint | PyTorch 原生 | — | 换计算时间 |")
    output_lines.append("")
    output_lines.append(
        f"> 详见 [cpu_offload.md]({rel_path}/cpu_offload.md) 和 [parallelism.md]({rel_path}/parallelism.md)。\n"
    )


def extract_compilation(docs_dir, output_lines, rel_path):
    """Extract compilation info from compilation.md."""
    path = docs_dir / "compilation.md"
    if not path.exists():
        return

    output_lines.append("## 编译路径优化（融合 + 图捕获）\n")
    output_lines.append(
        "通过 `torch.compile(backend=MindieSDBackend())` 触发，同时启用 Pattern 融合和 ACLGraph 加速。\n"
    )
    output_lines.append("| 瓶颈指标 | MindIE-SD 方案 | 接口 | 硬件约束 |")
    output_lines.append("|---------|---------------|------|---------|")

    output_lines.append(
        "| 未触发 MindieSDBackend（eager fallback） | 启用编译后端 "
        "| `torch.compile(model, backend=MindieSDBackend())` | — |"
    )

    fusion_patterns = {
        "enable_rms_norm": ("RMSNorm", "Norm 层大量小 kernel launch"),
        "enable_rope": ("RoPE", "RoPE 独立 kernel 开销"),
        "enable_adalayernorm": ("AdaLayerNorm", "AdaLayerNorm 独立调度"),
        "enable_fast_gelu": ("fastGELU", "GELU/SiLU 激活独立 kernel"),
        "enable_mul_add": ("Mul+Add", "element-wise Mul+Add 开销"),
    }

    for key, (name, bottleneck) in fusion_patterns.items():
        output_lines.append(f"| {bottleneck} | {name} 融合 | `CompilationConfig.fusion_patterns.{key} = True` | — |")

    output_lines.append("| 每步动态图调度开销 | ACLGraph 静态图捕获 | 自动启用（`MindieSDBackend()` 内） | — |")

    output_lines.append("")
    output_lines.append("> Pattern 融合作用于 Norm/激活/元素级操作，不作用于 MatMul 和 Attention 本身。")
    output_lines.append("> 首次推理有 JIT 编译开销（最多 8 次尝试）。Benchmark 时需 ≥5 步 warmup 排除编译耗时。")
    output_lines.append(f"> 详见 [compilation.md]({rel_path}/compilation.md)。\n")


def extract_cache(docs_dir, output_lines, rel_path):
    """Extract cache info from cache.md."""
    path = docs_dir / "cache.md"
    if not path.exists():
        return

    output_lines.append("## 缓存加速（以存代算）\n")
    output_lines.append("扩散模型相邻时间步存在冗余计算，通过缓存中间结果跳过重复计算。\n")
    output_lines.append("| 场景 | MindIE-SD 方案 | 接口 | 适用条件 |")
    output_lines.append("|------|---------------|------|---------|")

    output_lines.append(
        "| block 数多 | DiTCache | `CacheConfig(method=\"dit_block_cache\", ...)` + `CacheAgent` | 通用 |"
    )
    output_lines.append(
        "| Attention 占比高 | AttentionCache "
        "| `CacheConfig(method=\"attention_cache\", ...)` + `CacheAgent` | Attention 密集型 |"
    )
    output_lines.append("| 辅助任何缓存方案 | 时间步优化 | 减少/跳过扩散步数 | 需质量容忍 |")

    output_lines.append("")
    output_lines.append("> DiTCache 优先尝试，AttentionCache 备选。")
    output_lines.append(f"> 详见 [cache.md]({rel_path}/cache.md)。\n")


def extract_supported_matrix(docs_dir, output_lines, rel_path):
    """Extract model/hardware support matrix from supported_matrix.md."""
    path = docs_dir / "supported_matrix.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()

    output_lines.append("## 模型/硬件支持矩阵（速查）\n")

    target_models = ["FLUX.1-dev", "Wan2.2", "HunyuanVideo-1.5", "Qwen-Image"]
    output_lines.append("| 模型 | 并行 | 稀疏FA | 量化 | Cache | 融合算子 |")
    output_lines.append("|------|:----:|:-----:|:----:|:-----:|:-------:|")

    for model in target_models:
        row = [model]
        found = False
        for _, line in enumerate(lines):
            if model.replace("-", "\\-") in line or model in line:
                if line.startswith("|"):
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    if len(cells) >= 6:
                        row = [model] + cells[-5:]
                        found = True
                        break
        if not found:
            row = [model, "—", "—", "—", "—", "—"]
        output_lines.append(f"| {' | '.join(row)} |")

    output_lines.append("")
    output_lines.append(f"> 完整矩阵见 [supported_matrix.md]({rel_path}/supported_matrix.md)。\n")


def generate_features(docs_dir, output_path):
    """Generate mindiesd-features.md from docs."""
    docs_dir = Path(docs_dir)
    output_path = Path(output_path)

    output_dir = output_path.parent
    try:
        rel_path = os.path.relpath(docs_dir, output_dir).replace("\\", "/")
    except ValueError:
        rel_path = str(docs_dir)

    if not docs_dir.exists():
        raise RuntimeError(f"docs directory not found: {docs_dir}")

    output_lines = []
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    output_lines.append("# MindIE-SD 优化特性映射\n")
    output_lines.append(
        f"> 瓶颈 → MindIE-SD 方案的唯一映射表。版本升级时只需更新此文件。\n"
        f">\n"
        f"> **最后同步**: {utc_now} (由 scripts/refresh_features.py 自动生成)\n"
        f">\n"
        f"> **源文件** (当内容可疑时查阅):\n"
        f">\n"
        f"> - [quantization.md]({rel_path}/quantization.md)\n"
        f"> - [sparse.md]({rel_path}/sparse.md)\n"
        f"> - [parallelism.md]({rel_path}/parallelism.md)\n"
        f"> - [cpu_offload.md]({rel_path}/cpu_offload.md)\n"
        f"> - [compilation.md]({rel_path}/compilation.md)\n"
        f"> - [cache.md]({rel_path}/cache.md)\n"
        f"> - [supported_matrix.md]({rel_path}/supported_matrix.md)\n"
    )

    extract_quantization(docs_dir, output_lines, rel_path)
    extract_sparse(docs_dir, output_lines, rel_path)
    extract_compilation(docs_dir, output_lines, rel_path)
    extract_offload(docs_dir, output_lines, rel_path)
    extract_parallelism(docs_dir, output_lines, rel_path)
    extract_cache(docs_dir, output_lines, rel_path)
    extract_supported_matrix(docs_dir, output_lines, rel_path)

    output_lines.append("---\n")
    output_lines.append("## 维护说明\n")
    output_lines.append("- **更新触发**: MindIE-SD 发版新增/废弃算法时\n")
    output_lines.append("- **更新方式**: 运行 `python scripts/refresh_features.py --docs-dir <path>`\n")
    output_lines.append("- **手动更新**: 运行脚本后可在输出文件中手动补充表格行\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    logger.info("Generated: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="从 MindIE-SD docs 自动生成 mindiesd-features.md")
    parser.add_argument(
        "--docs-dir",
        required=True,
        help="MindIE-SD docs/zh/features 目录路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 mindiesd-features.md 的路径",
    )
    args = parser.parse_args()
    generate_features(args.docs_dir, args.output)


if __name__ == "__main__":
    main()
