# 产物目录规范

每次模型自动优化闭环（采集→分析→方案→复验）的产物目录结构：

```text
runs/YYYYMMDD_{model_slug}_optimization/
├── manifest.txt                    # 环境版本：CANN / PyTorch / TorchNPU / MindIE-SD
├── step1_profile/
│   ├── profile_l1.tar.gz           # 原始 profiling 数据
│   └── collect_profile.log         # 采集日志
├── step2_analysis/
│   ├── profiling_report.md         # 5 层分析报告
│   ├── model_architecture_report.md
│   └── compare_traces.log          # 对比分析日志（Step 4）
├── step3_patches/
│   ├── patch_001_quantization.diff
│   ├── patch_002_compilation.diff
│   └── patch_003_sparse.diff
├── step4_validation/
│   ├── profile_l1.tar.gz           # 复验 profiling 数据
│   └── profiling_report.md         # 复验分析报告
├── overview_report.md              # 优化总览报表（强制；基线=TP 多卡未优化，见
│                                   #   model-auto-optimization/references/overview-report.md）
├── detail_report.md                # 优化细分报表（强制，紧随总览；融合算子/并行/稀疏/量化/cache 与
│                                   #   采样步数/组合 的构成口径，见 references/detail-report.md）
├── final_report.md                 # 闭环报告（含总览/细分报表引用）
└── compare/                        # 对比产物
    ├── comparison_report.md
    └── kernel_diff.csv
```

## 运行状态文件（run-state.md，阶段推进锚点）

上面是**终态产物**结构；流程**进行中**的推进状态（当前阶段/验收记录/决策与轮次）不放在上树，
由 `references/run-state.md` 规范管理，默认 `{工作目录}/agentic/run-state.md` + `evidence/{stage}/`
（不入 git）。与终态文件的分工：

| 文件 | 管什么 | 何时写 |
|------|--------|--------|
| run-state.md（references/run-state.md） | 进行中流程状态（推进表单一真相源） | 任务开始建，每阶段收尾即时回写 |
| manifest.txt / manifest.toml | 任务 scope 与口径声明 | 启动确认后一次成型 |
| overview/detail/final_report + evidence.json（上树） | 终态归档 | 闭环收尾 |

阶段收尾按推进表回写后跑 `scripts/stage_gate.py --stage {Sn}` 门禁（error=0 才推进/宣称闭环）。

## manifest.txt 格式

```text
device: ATLAS_800_A2_376T_64G
cann: 8.0.0
pytorch: 2.6.0
TorchNPU: 2.6.0
mindiesd: {git_commit}
model: Wan2.2-T2V-14B
framework: Cache DiT + diffusers
precision: bfloat16
topology: 4xNPU（TP1/CP2, rank2/4）      # 并行拓扑与 rank 口径（同窗口同卡组）
run_cmd: {可复现启动命令}
feature_flags: <特性开关 env/参数清单>
timing_excludes: model_load, compile_prime, warmup   # 计时排除项
```

## final_report.md 模板

```markdown
# 优化闭环报告

## 环境
- 模型: {model}
- 硬件: {device}
- 框架: {framework}

## 口径与声明（claim limits）
- 对比分母: {vs-naive / 同拓扑同卡组 baseline / 上一版本}
- 计时口径: 起止 {边界}；排除 {model_load/compile_prime/warmup}；{N} 次 hot 取中位数
- 证据分级: {CPU 检查 / smoke / formal}（smoke ≠ formal ≠ 质量门）
- claim limits: 匹配基线 {有/无}；加速声明 {…/无——只报绝对值}；质量门禁 {run=是/否, 结论}

## 基线
- DiT: xx ms, VAE: xx ms, 总计: xx ms
- 显存峰值: xx GB

## 优化方案（按实施顺序）
1. {P0}: {方案描述} → 改进: xx%
2. {P1}: {方案描述} → 改进: xx%

## 复验结果
- DiT: xx ms ({delta}%), VAE: xx ms ({delta}%), 总计: xx ms ({delta}%)
- 显存峰值: xx GB ({delta}%)

## 结论
{通过/部分通过/未通过}
- 目标达成 / 噪声范围 / 外部瓶颈 / 硬件瓶颈
```

## 机器可读证据 evidence.json（可选最小字段）

final_report 面向人读；需要机器可读伴生块时（跨会话交接/归档/门禁），在闭环目录根放
`evidence.json`。最小字段集（改编自同族方案栈 BENCHMARK_REFERENCE 结构，仅取与本仓匹配子集）：

```json
{
  "schema_version": 1,
  "benchmark_id": "{model}_<方案slug>_{YYYYMMDD}",
  "hardware": {"device": "ATLAS_800_A2_376T_64G", "npu_count": 1, "topology": "TP1/CP1"},
  "profile": {"model": "{model}", "framework": "{framework}", "precision": "bfloat16",
              "features": ["量化", "稀疏", "缓存"]},
  "sampling": {"hot_samples": 5, "excluded": ["model_load", "compile_prime", "warmup"]},
  "latency_median_seconds": {"diT": 0.0, "total": 0.0},
  "timing_boundary": {"start": "...", "end": "...", "rank": "rank0"},
  "validation": {"technique_counters": {"cache_reuse": 0, "sparse_kernel_calls": 0}},
  "source_binding": {"mindiesd_commit": "{git}", "framework_repo": "{repo}",
                     "framework_commit": "{hash}", "framework_version": "{ver}",
                     "manifest_file": "<manifest.toml 指针>", "profile": "evals/profiles/{model}.toml",
                     "run_dir": "{远端产物目录指针}"},
  "claim_limits": {"matched_baseline": true, "speedup_claim": null,
                   "quality_gate_run": true, "quality_gate_result": "pass"}
}
```

> **仓库零数据**：`source_binding` 只存标识与指针（commit/hash/版本/manifest/profile/远端目录）；
> 帧/媒体/原始 profile/日志等数据只保留在远端 `runs/…` 产物目录，不入 git。
> claim_limits 语义：`matched_baseline=false` 时 `speedup_claim` 必须为 null（只报绝对值）；
> `quality_gate_run=false` 时不得宣称质量通过（质量门禁见 performance-optimization
> `references/quality-gate.md` + 仓库 `evals/`）。

## 维护与更新

当优化闭环产出物结构变化时，按 dev-workflow 的复盘流程更新本文件。
