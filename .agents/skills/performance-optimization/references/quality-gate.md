# 有损优化质量门禁（S4 / 闭环端到端精度判定）

> 方法文档。判定标准细则与工具在仓库 `evals/`（契约 `evals/README.md`、伪影标准
> `evals/rubrics/visual-artifact-gate.md`、定量工具 `evals/scripts/quality_compare.py`、
> 基线 profile `evals/profiles/README.md`）。
> 来源改编：跨栈质量门禁方法论——方法与硬件无关，仅借纪律与结构。

术语对照（避免门禁词族歧义）：本文件 = 有损优化的**质量门禁总方法**；`evals/rubrics/visual-artifact-gate.md`
= 其中的**视觉伪影单门**判定标准；`evals/README.md` = 六门**契约与晋升规则**；`artifact-layout.md`
的「口径与声明 / claim-limits」= 收益**宣称边界**（不属于质量门本身）。

## 何时用（触发）

- S4 任一有损项（量化/稀疏/缓存/步数裁剪等）验收；
- 闭环复验：有损组合的最终判定；
- 无损阶段（S1–S3）不需要：以 kernel diff + 墙钟验收，输出应与 baseline 一致或按口径核对。

## 流程（五步）

1. **冻结 baseline 产物 + 生成 profile**：官方/同设置 baseline 的采样帧（同 seed/提示词/分辨率/
   帧数/步数）冻结后，用 `evals/scripts/gen_profile.py` 生成该模型的 profile 到
   `runs/{task_id}/profiles/{model}.toml`（**不入库**；仓库只维护 `evals/profiles/_template.toml`
   契约模板）；基线帧冻结后不再改动，close 前由 `evals/scripts/check_profile.py` 强校验
   （仓库无具体 profile / 生成物完整 / 与 evidence 一致）。
2. **定量门**：`evals/scripts/quality_compare.py`（psnr/ssim 内置，lpips 可选）→ quality.json；
   分辨率/帧索引不一致即失败（fail-closed），不允许混比。
3. **视觉门**：按 `evals/rubrics/visual-artifact-gate.md` 判卷——VLM 有则用，无则人工并排，
   输出 pass/fail/inconclusive。
4. **off-identity 归因**：关闭该技术复跑 → 应恢复 baseline 行为；证明伪影/收益归因于技术本身
   （与 model-auto-optimization「无损项叠加复核」配套）。
5. **结论**：performance + quantitative + visual 三门同过才可宣称「该有损档可用」；
   任一 fail → 回退或调档，带证据记录。

## 判定要点

- 定量失败可 `explicitly_deferred`，但必须说明原因并附并排产物（不许静默跳过）。
- 视觉门：medium/high 级新伪影即 fail；时序类（闪烁/跳变/块边界 popping）低严重度也不放行。
- **校准经验（首个案例 minimax-h3，2026-09-05）**：扩散视频同 prompt 异配置的帧级相似度天然偏低
  （混沌轨迹），定量指标（psnr/ssim）用于**基线登记与回归对比，不设绝对门槛**；
  **主判据 = 视觉判卷 + off-identity**。VLM 不可用时输出 `inconclusive` + 并排产物存证即可
  （首个案例已实践：montage_*.png 并排存证），不许把「无 VLM」当质量通过的等价物。
- **校准经验（图像第二案例 qwen-image-2512，2026-09-05/06，V3）**：图像无帧轴 → 取样 = **固定
  21-seed 同 seed 像素对**（生成确定性 → 每 seed 单图；21 seed 均化，文件名 sNNNN.png 对齐）；
  图像同 seed 像素质量域显著高于视频（同量化为 w8a8：图像 PSNR 37.5/SSIM 0.986 vs 视频 19.6/0.679
  ——无轨迹混沌）→ **数值域与阈值不可跨任务类型迁移**；方法/校准值见 evals/profiles/README.md
  （运行 profile 由 gen_profile.py 生成到 runs/{task_id}/profiles/，不入库）。
- 组合试验中单特性先过门、再组合过门（与 model-auto-optimization S4 纪律一致）；
  换分辨率/步数 = 新对照。
- 阈值以真实案例校准后固化进运行时 profile（gen_profile.py 生成到
  `runs/{task_id}/profiles/{model}.toml`，不入库；首个校准：MiniMax-H3 × vllm-omni）；
  **阈值未校准前不得宣称「质量通过」**。
- **绝对口径**：所有有损/组合档的质量数值 = vs 同构 lossless 基线（psnr/ssim）；「有损 × 并行」等
  两有损档间的交叉对比只作次级参考，不得替代绝对值（首个案例：USP4 档曾以 vs USP2-int8 交叉
  20.8/0.750 展示，已改为 vs lossless 绝对值 19.43/0.692）。
- **存量产物补测**：跑过但未算质量的档（远端 mp4/帧仍在）→ 抽帧（`select=not(mod(n,N))`）后交给
  quality_compare 后处理即可补算，不必重跑占卡；先盘点产物目录（各档 mp4/quality json/帧目录）再决定是否新跑。

## 产物落点

判卷结论与 quality.json 落入闭环产物目录（model-auto-optimization 的
`references/artifact-layout.md` 布局的 `step4_validation/`），并在 final_report 的
「口径与声明」节登记：质量门禁 run=是/否、结论、defer 说明。

## 与相邻能力边界

- 单算子数值偏差/精度（cosine 等）→ compilation-dev / operator-dev 的算子级测试
  （本文件不管算子级数值）。
- 速度/收益证据 → `benchmarks/`（mindie_bench）+ profiling 回路（本文件只判质量）。
- 技能触发类 evals（`.agents/skills/*/evals/evals.json`）与本质量门禁无关。

## 维护与更新

`evals/` 契约/工具变化时同步本文件；本文件被 performance-optimization（Step 4/5）与
model-auto-optimization（声明纪律与真实性核验）引用。
