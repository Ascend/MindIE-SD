# MindIE-SD Evals — 生成质量门禁

有损优化（S4 量化/稀疏/缓存/步数裁剪等）的**端到端生成质量**评估承载目录，与
`benchmarks/`（算子级速度基准，mindie_bench）分工互补：

| 目录 | 对象 | 判什么 |
|---|---|---|
| `benchmarks/` | 核心算子 FA/BSA/GMM/MM | 速度：MFU/MBU/时延、实现级选型 |
| `evals/` | 扩散模型端到端产物（图像/视频帧） | 质量：定量指标 + 视觉伪影 + 归因核验 |

> 命名消歧：本目录是产品仓的**质量门禁**；`.agents/skills/*/evals/` 是技能**触发测试**
> （evals.json，skill-creator 语义），两者无关。

## 门禁契约（有损优化验收 = 性能门 + 质量门）

| 门 | 判什么 | 谁提供 | 说明 |
|---|---|---|---|
| artifact | 运行产物齐备（帧/报告/技术计数） | 运行方 | 缺产物不进入质量判定 |
| official_config | 与可比 target 基线同设置 | 运行方 | 分辨率/步数/seed/提示词对齐，见 `profiles/` |
| performance | 比选定 baseline 快（阈值自定） | `benchmarks/` + profiling 回路 | 速度侧，本目录不判 |
| off_identity | 关闭该技术应恢复 baseline 行为 | 本目录归因 | 伪影/收益归因证据 |
| quantitative_quality | 简单视觉指标在容差内 | `scripts/quality_compare.py` | PSNR/SSIM 内置；LPIPS 可用时启用 |
| visual_artifact | 视觉判卷未见新材料级伪影 | `rubrics/visual-artifact-gate.md` | VLM 有则用，否则人工并排 |

最小晋升规则（改编自跨栈质量门禁方法论，仅借纪律与结构）：

```text
artifact == pass
official_config == pass
performance == pass
off_identity == pass 或 not_applicable
quantitative_quality == pass 或 explicitly_deferred（须说明原因 + 并排产物）
visual_artifact == pass
```

> 校准经验（首个真实案例 minimax-h3，2026-09-05）：扩散视频同 prompt 异配置的帧级相似度天然偏低
> （混沌轨迹），定量指标（psnr/ssim）只作**基线登记与回归对比，不设绝对门槛**；
> 有损档主判据是 visual_artifact + off_identity。首个校准 profile：`profiles/minimax-h3.toml`。
>
> Profile 演进（2026-09 重构）：`profiles/{model}.toml` 只存**契约 + 判定结论（[domain]/
> [decisions]）+ 指针（[baseline]/[last_verified]）**；质量数值由 `scripts/quality_compare.py`
> 对 `runs/` 冻结产物**现算**存 `{run}/quality.json`（不入 git），并**强制随报表逐行登记**
> （overview 质量列 + detail-report §E 每运行标杆对照）——细则见
> `profiles/README.md`「数值不入库」约定。

## 使用（S4 / 闭环复验）

方法文档（何时用、流程、与相邻边界）见
`.agents/skills/performance-optimization/references/quality-gate.md`；
本目录只放契约、判定标准、基线 profile 约定与工具。

```bash
# 帧级定量对照（baseline vs config，同帧索引，CPU 可跑；阈值可选 fail-closed）
python evals/scripts/quality_compare.py --baseline {baseline_frames_dir} \
    --config {config_frames_dir} --metric ssim --output evals_out/quality.json

# 视觉伪影判卷：按 rubrics/visual-artifact-gate.md 组织 VLM 或人工并排对照
```

产物建议落在闭环产物目录（见
`.agents/skills/model-auto-optimization/references/artifact-layout.md`
的 `runs/YYYYMMDD_{model}_optimization/` 布局）。

## 目录结构

```text
evals/
├── README.md                        # 本文件：门禁契约
├── rubrics/
│   └── visual-artifact-gate.md      # 视觉伪影判定标准（类别/时序/pass 规则）
├── profiles/
│   ├── README.md                    # baseline 对照 profile 约定（契约/判定/指针，数值不入库）
│   └── _template.toml               # 仓库唯一 profile 形态（具体模型 profile 由流程生成，不入库）
└── scripts/
    ├── quality_compare.py           # 定量对照（psnr/ssim 内置，lpips 可选）→ runs/{id}/quality.json
    ├── gen_profile.py               # S0 冻结基线后生成具体模型 profile → runs/{task}/profiles/
    └── check_profile.py             # close 前置强校验（profile 完整性 + 与 evidence 一致 + 不入库约束）
```

## 维护与更新

- 具体模型 profile 一律由流程生成：S0 冻结基线后跑 `scripts/gen_profile.py` →
  `runs/{task_id}/profiles/{model}.toml`（**不提交仓库**）；close 前由 `scripts/check_profile.py`
  强校验（profile 完整、frozen_hash 非空、quality_json 指针、evidence 一致、仓库无具体 profile）。
- 仓库只维护 `profiles/_template.toml`（契约骨架）与 `profiles/README.md`（约定）；
  首个真实 NPU 有损案例回填时若需新增字段 → 先改 _template + gen_profile/check_profile 再生成。
- 门禁契约/结构变化 → 同步 `.agents/skills/performance-optimization/references/quality-gate.md`
  与 `.agents/README.md` §4 槽位（S4-1）。
