# MindIE-SD Benchmarks

MindIE-SD 核心算子（FA / BSA / GMM / MM）微基准，基于 [xpu-perf](https://gitcode.com/) micro_perf 框架，在昇腾 NPU 上度量 MFU / MBU / 时延。

## 目录结构

```text
benchmarks/
├── common/                     # 运行时与离线工具共享（benchmarks/ 加入 sys.path 后可用）
│   ├── env_util.py             # env.json peak_flops/peak_bw 统一解析
│   ├── metrics.py              # MFU/MBU 统一公式（util_metrics）
│   └── schema.py               # op slot / seq 轴 / series 轴常量（单份维护）
├── workloads/                    # 默认 workload（与 baselines/ 对应）
│   ├── fa.json / bsa.json / gmm.json / mm.json
│   └── seqlen/                   # 序列长度扫描 workload（2^k，1024→1M）
│       ├── fa.json / bsa.json / gmm.json / mm.json
├── xpu-perf-plugin/              # NPU 后端 shim + op 定义
│   ├── npu_launch.py             # benchmark 运行入口
│   ├── backend_npu.py            # BackendNPU（xpu-perf Backend 实现，单一墙钟计时）
│   ├── op_defs/                  # fa/bsa/gmm/mm 基础实现（schema + FLOPs 记账）
│   └── vendor_ops/NPU/           # NPU vendor 实现 + env.json（peak_flops/peak_bw）
│       └── _quant.py             # 量化 kernel 不可用判定（统一 fallback 判定）
├── baselines/                    # baseline JSON（每 op 一个文件，唯一数据源）
├── reports*/                     # 运行产出（report_dir 生成，含 jsonl/csv/info.json）
│   └── benchmark-report_<time>.{json,html}   # baseline 生成的时间戳报告快照 + HTML
├── scripts/
│   └── benchmark_report.py       # baseline 导出 / 报告快照 / HTML 渲染 / drift 对比（统一入口）
└── README.md
```

## 核心脚本

### 1. 运行 benchmark — `xpu-perf-plugin/npu_launch.py`

```bash
cd benchmarks/xpu-perf-plugin
python npu_launch.py --task_dir ../workloads/seqlen --task all \
    --device 8 --report_dir ../reports_seqlen_v2
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--task_dir` | workload 目录（含各 op 的 json） | `../workloads` |
| `--task` | 指定 op（逗号分隔）或 `all` | `all` |
| `--device` | NPU 逻辑设备 ID（逗号分隔） | 全部 |
| `--report_dir` | 报告输出目录 | `../reports` |

产物：`<report_dir>/NPU/<device>/<op>/NPU/<op>-NPU.{jsonl,csv}` 与 `info.json`。

### 2. 报告工具 — `scripts/benchmark_report.py`

统一入口，三个子命令：

```bash
# 默认执行 == baseline：导出 baseline JSON + 时间戳报告快照 + 渲染 HTML
python scripts/benchmark_report.py \
    --report_dir ../reports_seqlen_v2 --baseline_dir ../baselines \
    --env ../xpu-perf-plugin/vendor_ops/NPU/env.json

# 显式 baseline（等价于默认；--no-html 跳过 HTML）
python scripts/benchmark_report.py baseline --report_dir <reports> \
    --baseline_dir <baselines> --env <env.json> [--no-html]

# 从报告快照重渲染 HTML（默认取 report_dir 下最新快照）
python scripts/benchmark_report.py render --report_dir <reports>

# drift 对比（exit 1 表示越限）
python scripts/benchmark_report.py compare --report_dir <reports> \
    --baseline_dir <baselines> --env <env.json> [--threshold 0.03]
```

| 子命令 | 职责 |
|---|---|
| **baseline**（默认） | 读最新 reports jsonl → ① 写 `baselines/{op}.json`（MFU/MBU 按 `env.json` 的 peak_flops/peak_bw **离线重算**）② 写时间戳报告快照 `reports/benchmark-report_<time>.json`（含 `ops` 与**内嵌 CSV 内容** `csv` 段，不生成 CSV 文件）③ 默认用同一份内存数据渲染同名 `.html` |
| **render** | 从 `--report-json`（默认最新快照）重渲染 HTML；HTML 与所选快照同目录同名 |
| **compare** | 读 baseline 目录 + 最新 reports → 输出 MFU/MBU 相对漂移，超阈值 exit 1（CI 门禁） |

**数据流**：`baseline` 读取最新运行 jsonl 后，同一份内存数据既写 baseline JSON、又写报告快照、又渲染 HTML——数据复用，且 `baselines/*.json` 是**唯一数据源**，报告快照与 HTML 均为其派生物。报告产物带时间戳、直接落在 `report_dir/`（已被 `.gitignore` 忽略），多次运行互不覆盖。

## Workload 参数

序列长度扫描（`workloads/seqlen/`）均为 2^k，共 11 档：`1024 … 1048576`。

| op | 扫描维度 | 固定参数 | dtype / quant | case 数 |
|---|---|---|---|---|
| fa | `q_len=kv_len` | batch=1, heads=24, head_dim=128, causal=false | bf16 | 11 |
| bsa | `q_len=kv_len` | batch=1, heads=16, head_dim=128, mask=rf_v3 | bf16, fp8 | 110 |
| gmm | `num_tokens` | **hidden_size(K)=1536, moe_inter(N)=3200**, experts=128, top_k=16 | NO_QUANT, W8A8_MXFP8 | 22 |
| mm | `M` | **K=1536, N=3200**, group_size=32 | NO_QUANT, W8A8, W8A8_MXFP8, W4A4_MXFP4 | 44 |

> 默认 `workloads/*.json` 为与 `baselines/` 对齐的单档配置。
> gmm `W8A8_DYNAMIC` 因 `vendor_ops/NPU/gmm.py` 权重形状不匹配（`x_k_dim 4096 ≠ weight_k_dim 6144`）暂时跳过。

## 指标口径

- **MFU** = `calc_flops_power(tflops) / peak_flops`；**MBU** = `mem_bw(GB/s) / peak_bw`
- peak 来自 `vendor_ops/NPU/env.json`（`Ascend910_9382`：`peak_flops=560.0`，`peak_bw=1275.0`，对应 Atlas 800 A3）；解析统一在 `common/env_util.py`（`common` + 设备条目，设备名缺失时回退首个非 common 项）
- **MFU/MBU 公式单点维护**：`common/metrics.py: util_metrics`，运行时 summary 与离线重算共用，口径变更只需改一处
- **slot 键规范化**：slot 键省略取默认值的参数（`kv_len` 未设置或等于 `q_len` 时省略），保证同一 op 在不同 workload（显式/隐式设置 kv_len）下 baseline 键一致
- **量化 FLOPs 已计入**：量化 kernel 在计时区内执行（时延已含量化开销），FLOPs 记账同步补计每元素 2 FLOPs（`_common.py: QUANT_FLOPS_PER_ELEM`）——fa 的 q/k/v、mm 的 x/w；gmm 量化档当前走 bf16 fallback、bsa 无量化 kernel，不计
- **BSA 稀疏度已计入 FLOPs**：`op_defs/bsa.py:40` 用 `(1 - sparsity)` 折扣（`fa.py:77` 同理）
- **量化 fallback**：`ascend910_93` 平台缺 `DynamicMxQuant`，量化 FA/MM 自动回退 bf16（字节记账仍按量化 dtype）
- **BSA MBU 口径**：read_bytes 未按稀疏度折扣（稀疏只降 FLOPs 不降 Q/K/V 读字节），高稀疏档 MBU 偏大属预期

## 计时口径

- **单一墙钟路径**：warmup（2 iter）后整段运行 `prefer_iterations`（≥5）iter，末尾一次 `synchronize`，时延取均值（`backend_npu.py: core_perf`）；不再使用 profiler 路径（原逐 iter 同步导致时延虚高，已移除）

## 典型流程

```bash
# 1. 运行 benchmark（全部算子，单卡）
python xpu-perf-plugin/npu_launch.py --task_dir workloads/seqlen --task all \
    --device 8 --report_dir reports_seqlen_v2

# 2. 导出 baseline + 时间戳报告快照 + HTML（默认执行）
python scripts/benchmark_report.py --report_dir reports_seqlen_v2 \
    --baseline_dir baselines --env xpu-perf-plugin/vendor_ops/NPU/env.json
#   产物：baselines/{fa,bsa,gmm,mm}.json + reports_seqlen_v2/benchmark-report_<time>.{json,html}

# 2b. 按需重渲染 HTML（默认取最新快照）
python scripts/benchmark_report.py render --report_dir reports_seqlen_v2

# 3. drift 门禁（CI）
python scripts/benchmark_report.py compare --report_dir reports_seqlen_v2 \
    --baseline_dir baselines --env xpu-perf-plugin/vendor_ops/NPU/env.json
```
