#!/bin/bash
# MindIE-SD 核心算子（FA / MM / GMM / BSA）benchmark 样例脚本
#
# 覆盖两种负载：
#   1) 模型负载（默认注释）：按真实模型 packed 序列规格启用（Wan2.2 / MiniMax-H3 / 图片）
#   2) 序列扫描（默认启用）：一条命令覆盖四算子，1k → 256k，共享 seqlen 扫描轴
#
# 运行后自动生成合并 HTML 报告（每 op 一张 MFU 图 + 数据表）。
#
# 用法（仓库根目录执行）：
#   bash benchmarks/example/run_benchmark.sh             # 序列扫描 + HTML
#   bash benchmarks/example/run_benchmark.sh /tmp/rep    # 自定义报告目录
#
# 说明：
#   - 峰值必填：--config 内必须给出 peak_flops / peak_bw（代码不内置峰值）。
#     按设备实测填写（A310 类 CUBE 峰值 = 425/9*8 ≈ 377.78 TFLOPS）；
#     peak_bw 未知时可先留示例值，缺失时 GMM 的 MBU 显示 n/a
#   - 长序列必配 timeout：--config 保留 timeout: 300（默认 5s 会超时跳过大档位）
#   - --op/--config 值必须用双引号包裹（bash / PowerShell 均需）：
#     不带引号时 bash 会对含逗号的 {...} 做花括号展开/分词，把参数拆碎导致 argparse 报错
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPORT_DIR="${1:-$ROOT/benchmarks/reports_example}"

# ============================================================
# 1) 模型负载（默认注释，按真实模型 packed 序列规格启用）
#    启用方法：去掉下方命令行的 # 注释
# ============================================================
# Wan2.2 720P（heads=40, ffn=13824, packed 76k/148k/220k）
# python "$ROOT/benchmarks/scripts/mindie_bench.py" run \
#     --op "{fa: {num_heads: 40}}" \
#     --config "{seqlen: [75600, 147600, 219600], dtype: [bf16, fp8, mxfp8, mxfp4], timeout: 300, peak_flops: 377.78, peak_bw: 2000}" \
#     --report-dir "$REPORT_DIR/models_wan"
#
# MiniMax-H3 768P + 2K（heads=56, packed 75k~772k）
# python "$ROOT/benchmarks/scripts/mindie_bench.py" run \
#     --op "{fa: {num_heads: 56}}" \
#     --config "{seqlen: [75520, 146496, 217472, 267328, 519744, 772160], dtype: [bf16, fp8, mxfp8, mxfp4], timeout: 300, peak_flops: 377.78, peak_bw: 2000}" \
#     --report-dir "$REPORT_DIR/models_minimax"
#
# 图片模型（Qwen-Image / FLUX heads=24；HunyuanImage3 heads=32，packed 4.6k/16.9k）
# python "$ROOT/benchmarks/scripts/mindie_bench.py" run \
#     --op "{fa: {num_heads: 24}}" \
#     --config "{seqlen: [4608, 16896], dtype: [bf16, fp8, mxfp8, mxfp4], peak_flops: 377.78, peak_bw: 2000}" \
#     --report-dir "$REPORT_DIR/models_images"

# ============================================================
# 2) 序列扫描（默认启用）：一条命令覆盖 FA / MM / GMM / BSA 四算子
#    seqlen 为共享扫描轴（fa/bsa→q_len、gmm→num_tokens、mm→M）
#    各算子其余参数用内置默认规格（--op {<op>: {}}）
# ============================================================
python "$ROOT/benchmarks/scripts/mindie_bench.py" run \
    --op "{fa: {}, mm: {}, gmm: {}, bsa: {}}" \
    --config "{seqlen: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], timeout: 300, peak_flops: 377.78, peak_bw: 2000}" \
    --report-dir "$REPORT_DIR/seqlen_all"

# ============================================================
# 3) HTML 报告（合并 seqlen_all 下所有 run，每 op 一张 MFU 图 + 数据表，
#    含 Command 与 Peak config 段；产物统一在 reports/）
# ============================================================
python "$ROOT/benchmarks/scripts/mindie_bench.py" report \
    --report-dir "$REPORT_DIR/seqlen_all" \
    --baseline-dir "$REPORT_DIR/baselines"

echo ""
echo "HTML report: $(ls -t "$ROOT/benchmarks/reports/benchmark-report_"*.html 2>/dev/null | head -1)"
