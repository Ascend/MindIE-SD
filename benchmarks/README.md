# MindIE-SD Benchmarks

核心算子（FA / BSA / GMM / MM）性能基准工具，在昇腾 NPU 上度量 MFU / MBU / 时延。

## 快速开始

```bash
# 一键：序列扫描（FA/MM/GMM/BSA 四算子，1k→256k）+ 生成 HTML 报告
bash benchmarks/example/run_benchmark.sh

# 手动等价命令（peak_flops/peak_bw 按设备实测填写，见下）：
python benchmarks/scripts/mindie_bench.py run \
    --op "{fa: {}, mm: {}, gmm: {}, bsa: {}}" \
    --config "{seqlen: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], timeout: 300, peak_flops: 377.78, peak_bw: 2000}" \
    --report-dir reports/seqlen_all

python benchmarks/scripts/mindie_bench.py report \
    --report-dir reports/seqlen_all --baseline-dir baselines
```

产物：`reports/benchmark-report_<时间>.html` + `reports/<op>.csv`（浏览器打开 HTML）。

## CLI 参考（`benchmarks/scripts/mindie_bench.py`）

| 子命令 | 作用 |
|---|---|
| `run` | 运行基准（参数由 `--op` / `--config` 内联表达） |
| `report` | 生成 baseline + HTML 报告（`--render` 重渲染） |
| `compare` | 与 baseline 对比，超阈值 exit 1（CI 门禁，`--threshold 0.03`） |

| `run` 参数 | 说明 |
|---|---|
| `--op` | 算子选择 + 结构参数（fa/bsa: num_heads/head_dim、gmm: hidden_size/moe_inter/experts/top_k、mm: K/N + 保留键 `func`=内核来源标签）；值 `{}`（默认规格）或对象；缺省 = 全部算子。注意：**内核实现按 dtype/quant_algo 自动分发**（如 fa bf16→npu_fusion_attention、mxfp4→torch.ops.mindiesd.quant_flash_attn），`func` 仅作为报告系列标签（fn=）记录，不切换内核 |
| `--config` | 扫描矩阵 + 峰值：`seqlen` / `dtype` / `sparse` / `quant_algo` / `seed` / `timeout` / **`peak_flops`** / **`peak_bw`**；列表=笛卡尔积，标量=固定 |
| `--devices` | NPU 设备：默认 `0`；`0,2` 多卡并行；`auto` 自动检测空闲卡 |
| `--report-dir` | run 的结果目录（缺省自动 `reports/run_<时间戳>`）；`report` 的产物统一输出到 `reports/` |

- **`--op` / `--config` 值必须加引号**（bash / PowerShell 均需）：`--op "{fa: {num_heads: 32}}"`——不带引号时 bash 会对含逗号的 `{...}` 做花括号展开/分词，把参数拆碎导致 argparse 报错
- **峰值必填**：`--config {..., peak_flops: <实测>, peak_bw: <实测>}`——代码不内置任何峰值，MFU/MBU 都以你输入的峰值为分母（A310 类 CUBE 峰值 = 425/9×8 ≈ 377.78 TFLOPS）
- **长序列必配 `timeout`**：`--config {..., timeout: 300}`（默认 5s 会跳过大档位）
- **复现**：`--config {..., seed: 42}` 固定输入张量

## 负载

| 类型 | 用法 |
|---|---|
| 序列扫描 | `run_benchmark.sh` 默认段（一条命令四算子），seqlen 共享扫描轴（fa/bsa→q_len、gmm→num_tokens、mm→M），1k→256k（序列档位按需调整） |
| 模型负载 | `run_benchmark.sh` 模型段（默认注释）：按真实模型 packed 序列（Wan2.2 / MiniMax-H3 / 图片）启用 |

各算子默认规格（`--op {<op>: {}}`）：fa heads=32/head_dim=128、bsa heads=32/sparsity=0.8、gmm hidden_size=1536/moe_inter=3200/experts=128/top_k=16、mm K=5120/N=13824。更多变体见 `benchmarks/example/README.md`。

## 指标

- **MFU**（计算利用率）= 实测 TFLOPS / `peak_flops`；**MBU**（带宽利用率）= 实测带宽 / `peak_bw`
- 展示为**百分比**（如 59.74%）；FA/BSA/MM 仅展示 MFU，GMM 同时展示 MFU+MBU
- **峰值不内置**：`peak_flops` / `peak_bw` 必须由使用者通过 `--config` 输入（代码中无任何硬编码峰值）；`peak_bw` 缺失时 GMM 的 MBU 显示 n/a

## 报告（HTML）

- 产物统一在 `reports/`：`benchmark-report_<时间>.{json,html}` + 每 op 一个 CSV（`reports/<op>.csv`）
- **CSV 是数据源**：每行含 `peak_flops` / `peak_bw`；在 CSV 中更新 peak 后重跑 `report`，会用新峰值重算各 op 的 MFU/MBU，并同步更新「Peak config」段（未改 CSV 时以 run 的 `--config` 峰值为准）
- 每个算子一个章节：综合图（全部系列）+ 图下按系列的数据表；BSA 为稀疏度透视表（metric × sparsity，无 latency 列），其余 op 为 seq len / latency / MFU... 表
- 「Command」段：生成本报告的确切命令（可复现）
- 「Peak config (CUBE flops / bandwidth)」段：本次 run 通过 `--config` 输入的 `peak_flops` / `peak_bw`
- **多 run 合并**：多次 `run` 的结果放同一 report 父目录，`report --report-dir <父目录>` 合并为单个 HTML

## 常见场景

```bash
# 单算子参数扫描（FA 序列 × dtype）
python benchmarks/scripts/mindie_bench.py run --op "{fa: {}}" \
    --config "{seqlen: [1024, 4096, 16384, 65536], dtype: [bf16, mxfp8], timeout: 300, peak_flops: 377.78}"

# BSA 稀疏度扫描
python benchmarks/scripts/mindie_bench.py run --op "{bsa: {}}" \
    --config "{seqlen: [8192, 32768, 131072], dtype: [bf16], sparse: [0.6, 0.8, 0.95, 0.99], timeout: 300, peak_flops: 377.78}"

# MM 量化档位
python benchmarks/scripts/mindie_bench.py run --op "{mm: {}}" \
    --config "{seqlen: [4096, 65536], quant: [NO_QUANT, W8A8, W8A8_MXFP8, W4A4_MXFP4], timeout: 300, peak_flops: 377.78}"

# 模型调优中对比不同 FA 实现（func），选性能最优
python benchmarks/scripts/mindie_bench.py run --op "{fa: {func: torch_npu.npu_fusion_attention}}" \
    --config "{seqlen: [4096, 8192], dtype: [bf16], timeout: 300, peak_flops: 377.78}"
```

## 离线单测

```bash
python -m pytest tests/UT/benchmark -q
```

## 目录结构

```text
benchmarks/
├── common/              # 共享口径（MFU/MBU 公式 / schema；峰值不内置，由 --config 输入）
├── scripts/             # mindie_bench.py（CLI）+ benchmark_report.py（报告）
├── xpu-perf-plugin/     # 运行时（npu_launch.py 旧入口 / backend_npu.py / op_defs / vendor_ops）
├── example/             # 样例脚本 run_benchmark.sh + 使用说明（见 example/README.md）
└── tests/UT/benchmark/ # 离线单测
```

工具链的开发与排障见 skill：`.agents/skills/benchmark-dev/SKILL.md`。
