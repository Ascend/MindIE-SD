# 有损优化质量门禁（精度验收标准的第三级）

> 方法文档。**本文件是精度验收标准 `../SKILL.md`（`accuracy-gate`）三级验收的第三级**：
> 一级"同配置重跑逐位"、二级"跨配置数值门 + 产物 md5 不变"只回答"结果有没有被改变"；
> 本文件回答"改变之后还能不能用"，**只在有损档启用**（量化 / 稀疏 / 缓存 / 步数裁剪 /
> 换入外部训练权重等），无损档不启用本门。
> 判定标准细则与工具在仓库 `evals/`（契约 `evals/README.md`、伪影标准
> `evals/rubrics/visual-artifact-gate.md`、定量工具 `evals/scripts/quality_compare.py`、
> 基线 profile `evals/profiles/README.md`）。
> 来源改编：跨栈质量门禁方法论——方法与硬件无关，仅借纪律与结构。

术语对照（避免门禁词族歧义）：本文件 = 有损优化的**质量门禁总方法**；`evals/rubrics/visual-artifact-gate.md`
= 其中的**视觉伪影单门**判定标准；`evals/README.md` = 六门**契约与晋升规则**；`model-auto-optimization/references/artifact-layout.md`
的「口径与声明 / claim-limits」= 收益**宣称边界**（不属于质量门本身）。

## 何时用（触发）

- 精度验收标准 `../SKILL.md` 三级验收走到**第三级**时（该档为有损档：量化 / 稀疏 / 缓存 /
  步数裁剪 / 换入外部训练权重）；
- S4 任一有损项验收、闭环复验的有损组合最终判定；
- 无损档（等价替换 / 融合 / 并行切分 / 编译下发）**不需要**本门：一级逐位 + 二级 md5 已判定，
  质量列按"输出一致"口径写。

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
  **质量变化度量级**（vs 同构 lossless）：单点有损档 **SSIM 降幅约三成（差值约 0.3）**，
  叠加档降幅更大（约四成半，差值约 0.46 量级）——故定量只作登记与回归对比（读数见归档
  `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md` §4）。
- **校准经验（图像第二案例 qwen-image-2512，2026-09-05/06，V3）**：图像无帧轴 → 取样 = **固定
  21-seed 同 seed 像素对**（生成确定性 → 每 seed 单图；21 seed 均化，文件名 sNNNN.png 对齐）；
  图像同 seed 像素质量域显著高于视频（**同量化为 w8a8 的质量变化度**：图像 **SSIM 降幅约 0.01
  （质量基本无感）**、视频同档 **降幅约三成（差值约 0.3）**——相差一个数量级以上，无轨迹混沌）
  → **数值域与阈值不可跨任务类型迁移**（读数见归档
  `{run_results_dir}/archive/vllm-omni-qwen-image-case.md` §4 与
  `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md` §4）；方法/校准值见 evals/profiles/README.md
  （运行 profile 由 gen_profile.py 生成到 runs/{task_id}/profiles/，不入库）。
- 组合试验中单特性先过门、再组合过门（与 model-auto-optimization S4 纪律一致）；
  换分辨率/步数 = 新对照。
- **校准经验（训练感知档，2026-09，V4）**：**换入外部训练权重**类优化（`少步蒸馏` / `VAE解码替换`）
  的质量口径分两层，**顺序不可颠倒**（方法全文见 framework-integration
  `framework-integration/references/train-aware-lossy-method.md` §4.2）——
  ① **接口正确性**（必须先证）：与**原生组件同 latent 对拍**，判据 = 逐帧**灰度相关系数**
  （对全局尺度不变 ⇒ 只反映内容是否对上，**与画质无关**，故不能拿它判画质）+ 输出
  **std/mean/被钳位像素占比**（与原生同量级才判「接对」）。
  **PSNR 在此层不可用**——预览级解码器天然丢高频，PSNR 明显偏低是**预期**而非故障；
  用它判「接错了」会把排障引向错误方向（本例曾穷举十余档输入缩放 / 三种上采样 /
  十余组激活与后处理组合，与原生相关性始终极低；真因是**架构级选错 checkpoint**，扫参不可能收敛）。
  ② **画质档位**（接口对了才谈）：与原生组件对拍 SSIM + 视觉门，结论**必须落成档位声明**
  （「预览级」/「保画质」），报表须给**两条并列部署建议**（只报最高加速比 = 交付缺失）。
  本例：相关系数 0.97–0.98（内容对上）+ SSIM 0.75 量级 → 判「接口正确 + 预览级」，与保原生组件档并列。
- **校准经验（数值敏感度地板，2026-09，V5）**：把某个输出差异判成「错误」**之前**，先用**已知无害的改动**
  标定该模型的**数值敏感度地板**（4 步蒸馏扩散模型实测锚点，0–255 尺度上的**平均 |diff|**）：
  **同配置跑两遍 = 0.000**；**1-ULP 级归约序变化 = 22.0**；**已验收的等效优化 = 48.0**；
  **未证实的改动 = 32.4** → 该模型 **20–50/255 属正常敏感度**，故 **32.4 不构成「出错」的证据**
  （拿它当 bug 会白耗迭代轮次）。**方向必须分清**：本条给的是**噪声地板**（多大差异**算正常**），
  本文件的 SSIM/质量阈值给的是**容忍上限**（多大差异**可接受**）——**两者方向相反、不可互换**。
  校准 SOP 见 `../../perf-gate/references/measurement-discipline.md` §3。
- **无图像输入时的视觉存证替代**：执行侧拿不到图像 → 视觉门记 `inconclusive`，并**另附可机读的
  亮度简图 + 聚合指标**（`framework-integration/scripts/ascii_luma_preview.py`：ASCII 亮度图 +
  逐帧均值/标准差/钳位占比）——**不得**把「无图像输入」当质量通过的等价物，也不得因无法看图而省略存证。
- **`inconclusive` 的三种成因必须分开标注**（写法不同；混用会让审阅者把「**没判**」误读成
  「**判过但不明确**」，两者对后续动作的含义完全不同）：
  - ① **无图像输入能力**：执行侧不具备图像输入（看图不了）——记 `inconclusive`，并附并排产物 +
    机读亮度存证（见上一条）；
  - ② **用户未要求判别**：视觉判卷**按需触发**，本仓**默认不主动发起**；未要求时记
    `视觉门=待判（用户未要求判别）`，判卷材料**备用就绪**即可，**不催促判卷、不暗示应当现在判**；
  - ③ **已判但结论不明确**：VLM/人已判却无法定论——须记录判卷者、依据与不确定点。
  - 三者**均不得**当作质量通过；未判视觉门时，报表质量列仍给定量结论与变化度，
  但**不得写「质量通过」**，且须与「说明」列的待判标注**同时出现**（禁止静默留空）。
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
`model-auto-optimization/references/artifact-layout.md` 布局的 `step4_validation/`），并在 final_report 的
「口径与声明」节登记：质量门禁 run=是/否、结论、defer 说明。

- **视觉门判卷材料的生成方式与待判卷提示**：视觉门的**并排对比图**（同 seed/同帧号一行、
  配置各一列、**基线在最左**）/ 抽帧蒙太奇 / ASCII 亮度存证，一律落在**会话产物目录**
  `{run_results_dir}/…`（如 `{run_results_dir}/s41_visual_gate/`）——**不入库**（产物体积大且
  属会话证据，仓库只留方法与指针）。
- 生成材料的同时须随材料留一份**待判卷提示**（判什么类别、判哪些文件与对照关系、**基线是谁**、
  已测定量变化度、以及**未做/做不到**的缺口），并写明**须由具备图像输入的人或工具**按
  `evals/rubrics/visual-artifact-gate.md` 判定后**回填结论**
  （`pass` / `fail` / `inconclusive`）。
- **无图像输入时**：按本文件规则记 `inconclusive`（成因标注见「判定要点」）+ **并排存证**，
  不得省略存证；机读存证手段 = `framework-integration/scripts/ascii_luma_preview.py`
  （ASCII 亮度图 + 逐帧均值/标准差/钳位占比，支持两目录**逐帧对拍**出灰度相关系数）
  ——它**不构成视觉门结论**，只证「材料齐备 + 可机读复核」。
- **触发纪律**：视觉判卷**按需触发**——用户明确要求判别时才执行；未要求时只把材料备好并标注
  `视觉门=待判（用户未要求判别）`，**不主动判、不催促判**。

## 与相邻能力边界

- 单算子数值偏差/精度（cosine 等）→ pattern-dev / operator-dev 的算子级测试
  （本文件不管算子级数值）。
- 速度/收益证据 → `benchmarks/`（mindie_bench）+ profiling 回路（本文件只判质量）。
- 技能触发类 evals（`.agents/skills/*/evals/evals.json`）与本质量门禁无关。

## 维护与更新

- **归属**：本文件是精度验收标准 `../SKILL.md` 的第三级判据（有损档专用）；`evals/` 契约/工具
  变化时同步本文件。
- **引用方**：`model-auto-optimization`（S4 / 闭环验收的声明纪律与真实性核验）、
  `framework-integration`（训练感知档质量分层与各框架 enablement 文档）、
  `dit-perf-opt`（有损档实施后的复验）。
- **迁移记录（2026-09-14 阶段 2a·A4）**：本文件自 `performance-optimization/references/` 迁入
  本目录（`git mv`，全库只此一份）。**旧路径引用必须改指**
  `lossless-op-replacement/references/quality-gate.md`（改名批次后为
  `accuracy-gate/references/quality-gate.md`）——旧路径引用清单见 A4 批次报告。
