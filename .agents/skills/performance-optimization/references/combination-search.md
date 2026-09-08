# 有损组合试验设计（S4-2 · 组合搜索协议）

> 定位：S4 多特性组合（量化/稀疏/缓存/步数等）的**试验协议**——把「全对组合矩阵盲试」
> 升级为「seam 静态判定 + 单变量叠加 + frontier 保留 + 层回退」。方法改编自跨栈组合搜索方法论（compose/search 思路）：能力契约与互斥 seam 思想与硬件
> 无关，仅借结构；seam 表取值以 `references/mindiesd-features.md` 支持矩阵与首次真实组合
> 试验校准为准。

## 何时用

- S4 需同时开启 ≥2 个有损维度时；
- 单特性有损判定直接用 `references/quality-gate.md`，不进入本协议。

## 纪律（先于一切步骤）

1. **一次一变量**：只在前一已验收组合上加 1 个特性；多变量同时上无法归因收益。
2. **单特性先过门**：进入组合的每个维度先单独过 quality gate（quality-gate.md + 仓库 `evals/`）。
3. **跨 seam 组合优先**：作用在不同 seam 的维度收益近似可加（见 seam 表），先做这类组合。
4. **frontier 保留**：组合仅在「质量不降（gate pass）且首步耗时下降」时保留；其余记录原因后丢弃。
5. **无损项叠加复核**：有损叠加后复核既有无损项（融合/并行）是否仍有效，被稀释即回退
   （呼应 model-auto-optimization 的 S4 纪律）。

## 必测组合覆盖集（S4-2 强制 · 防静默缺行）

> 背景驱动：组合闭环中曾出现"组合矩阵只含已选候选链、漏掉跨 seam 合理两两与全维度三元"
> （如量化×稀疏、稀疏×Cache、Cache+量化+稀疏 缺行）——**frontier 收敛 ≠ 组合空间被覆盖**。
> 本节把「每个有损组合必须有裁决」显式化为强制清单：不允许静默缺行。

### 触发与必测集

- **触发**：特性覆盖清单判定 `量化` / `稀疏` / `Cache` 中 **≥2 个为「做」且各自有通过 quality
  gate 的单点 frontier** → 进入 S4-2 前先建「组合覆盖清单」。
- **必测集**（三个维度单点 frontier 齐备时，全部必须裁决）：

| # | 必测组合 | seam 判读 | 叠加说明 |
|---|---------|-----------|---------|
| 1 | 量化 × 稀疏 | 跨 seam（precision × attention_backend） | 直接叠加，无窗口冲突 |
| 2 | 稀疏 × Cache | 同 seam（step_decision，窗口可分） | seam_check 报 warning → 先静态确认窗口错开/只留最强档；首案例校准：默认同窗口可叠 |
| 3 | 量化 × Cache | 跨 seam（precision × step_decision） | 直接叠加 |
| 4 | **Cache + 量化 + 稀疏（三元）** | 跨 seam + step_decision 窗口可分 | **必须实测或显式豁免**（见下）；即 SKILL S4 目标档「8bit + 稀疏 + cache」 |

  当某维度在 framework×模型下不可用（support-matrix ❌/staying-dense/无单点）时，含该维度的
  必测行自动降级为「豁免」，但仍须登记原因——不豁免缺失，只豁免"某个维度不存在"。

### 裁决三态与收口

- 覆盖清单每行状态：`已测`（retain/reject，带 gate 证据） / `豁免`（框架不支持 / 能力互斥 /
  staying-dense / 同 serve 跨树不可行等，**带证据指针**） / `未裁决`（禁止 close）。
- **必测优先于自由候选**：先让必测集全部有裁决，再做探索性候选与 frontier 扩展；预算到点只可
  停自由候选，**不得把必测组合留作「未尝试」进 detail-report §C**——确未测者必须转「豁免」并写明阻断。
- **三元不达标不静默删行**：按层回退（降档 / 错窗 / 整维回退）并记录该组合的回退裁决与最终边界。

### 组合语义与命名（2026-09 定稿）

- **量化组合代表档 = `w8a8f8`**：量化与其他特性的必测组合一律以 w8a8f8 为代表档参与，在候选
  组合中选**最高性能**的组合；
- **Cache 与时间步优化二选一**：同一条组合/对比行只取其一（时间步为步级横切项，不与 Cache 并存
  于同一组合行）；
- **试验记录命名**：分类直接写 `算子(稀疏度xx%)`（如 `rf_v2(80%)` / `EagleQBSA(80%)`，稀疏×FA
  量化新算子标注 mask 算法与量化规格）；量化行带 `w8a8(mxfp8)` 式规格标注；
- 稀疏与量化(f8)叠加产生的新算子：按 mindiesd 算子性能分析，框架未接入先本地 benchmark 后再定
  接入（优选取性能者 + 对应 mask 算法）。

### 覆盖清单模板（放 run-state「候选与迭代表」必测子区，行首标 `[MUST]`）

> **机器生成**：把各维度的单点 frontier（每族最强档）喂给
> `model-auto-optimization/scripts/seam_check.py --required-combos {ids} --model {适配名}`
> 直接输出 [MUST] 清单 + 行内 seam 预判（两两 + 三元），再按迭代表逐行实测登记，防手写遗漏。

| round | 特性/实现 id（组合） | 必测 | 假说/预期 | 状态 | gate 证据 | 裁决/豁免理由 |
|-------|---------------------|------|-----------|------|-----------|----------------|
| r5 | 量化(w8a8)·mxfp8 × 稀疏(rf_v2 0.8) | [MUST] | 跨 seam 可叠加，预期组合收益≈单点收益相乘 | 已测 | evidence/S4/r5.log | retain/reject（…） |
| r6 | Cache(dit) × 稀疏(rf_v2 0.8) | [MUST] | step_decision 窗口可分；先查 dense_steps 冲突 | 已测 | evidence/S4/r6.log | … |
| r7 | **Cache + 量化(w8a8)·mxfp8 + 稀疏(rf_v2 0.8)** | [MUST] | S4 目标档三元；跨 seam + 窗口可分 | 已测/豁免 | evidence/S4/r7.log | retain / 回退（降档…）/ 豁免（…） |

## seam 声明表（草案 v1，首案例校准）

| seam | 含义（MindIE-SD 语境） | 涉及维度 | 组合规则 |
|---|---|---|---|
| precision | 同一 MatMul/Linear 的数值精度档唯一 | 量化档 × 融合 kernel（pattern） | 同 seam 至多一个 writer；量化档 × 融合 pattern 须先确认 kernel 接受量化输入/图形态匹配，否则运行期 fallback → 组合前查 features.md 支持矩阵 |
| attention_backend | 同一 attention 一次只一个后端 | FA / 稀疏 FA / 融合 FA（含 BSA） | 稀疏度参数与后端选择同 seam；同层只留最强档 |
| step_decision | 哪些 step 全稠密/稀疏/被缓存复用 | 缓存（dense_steps/阈值）× 稀疏（dense_steps/路由） | 同一 step 窗口内只允许一个 writer；不同窗口（phase）可叠加 |
| token_semantics（预留） | token 集合语义一致性 | 未来 token prune × 稀疏 attention | 共享 token 语义，先静态判再叠加 |
| vae_seam | VAE 编解码精度/编译 | VAE 编译、VAE 精度 | 独立于 DiT 侧维度，可自由叠加 |

> 读法：跨行（不同 seam）组合 → 直接进 Step 2；同行（同 seam）组合 → 只保留最强档，
> 不做双 writer 叠加。

### seam 表首案例校准（2026-09-05 · H3 × vLLM-Omni V1，见 vllm-omni-minimax-h3-case.md §4）

- precision × attention_backend 跨 seam 可叠加：INT8（MatMul precision）与 稀疏 mix（attention 内量化）同开有效（F5=28.2s@60）。
- attention_backend 同 seam：bf16 稀疏 vs EagleQBSA mix 取最强档（mix 更快、质量 -1%SSIM）；不同档算同 seam writer，勿双叠。
- step_decision：Cache × 稀疏同 step 窗口可叠加（默认 cache 即生效），未见 dense_steps 冲突；收紧
  max_continuous_cached_steps=1/threshold 0.12 为质量优先档（F5hq +24%墙钟换 +1.4%SSIM）。
- **稀疏档质量非线性**：0.6 稀疏的端到端 PSNR 反低于 0.8（轨迹敏感性）→ 组合前稀疏参数须逐档端到端 gate，
  不可按密度线性外推（呼应基准扫描纪律）。
- **单点行归因**：组合链必须含每特性的「单点行」（最强档单独、其余有损关）——首版缺「单独 Cache」单点被
  审阅发现；补测 Cache 单点（lossless 基底，默认档）38.4s@60（-67.5%、3.07×；质量 19.33/0.665），hq 档
  49.6s（21.69/0.763），与 768P cache 单点（-66.7%）交叉一致 → cache 是最轻损维度（默认档 SSIM 即 >0.66）。
- **质量绝对值口径**：组合/并行档质量一律 = vs 同构 lossless（USP4+INT8 85.1s → 19.43/0.692；
  +INT8+mix 52.05s → 15.97/0.523）；「有损 × 并行」交叉对比（USP2 vs USP4 同档）只作次级参考，
  不替代绝对值作对外数值。
- **覆盖范围声明**：mix（attention 内量化，EagleQBSA）的 int8 只覆盖稀疏块路径，dense FA 兜底 bf16
  （FA 兜底 4/步计数）→ 对外表述与报表量化修饰符（如 `f8`）必须写明覆盖范围；全注意力 8bit 未实现时
  标 ❓ 不虚列（报表命名/对照组规则见 model-auto-optimization `overview-report.md` §2.1–§2.3）。
- 跨窗口声明：frontier 判定以**同窗相邻对**为准（本机不同窗口绝对 e2e/首步耗时有 ~±10% 热/负载漂移）；性能主口径 = e2e 完整请求墙钟（锚点实测/中间行估算），首步耗时（warmup 后第 1 步）作辅助口径用于跨步数同口径对比（见 overview-report §1/§1.1）。

## 执行协议

**性能口径（同 model-auto-optimization overview-report §1/§1.1）**：性能主口径 = e2e 完整请求
墙钟——**base / 最终推荐 / 三元组合 = 实测**；中间行可估算标 `[估算]`（DiT 比例法 + VAE/其他，
公式见 overview §1.1）。性能选型/组合预筛可在少步下做（如 1 步识别收益，用首步耗时辅助口径同
口径对比，只筛方向与排序）；**进入 quality gate 的候选与最终采纳组合必须全量步数实测 e2e**
（少步结论不直接宣称）；每档/每组合登记实测步数与 e2e（实测或 `[估算]`，识别行注明「识别步数」）。

```text
Step 0 单特性基线：每个有损维度先开一档 → quality gate → 成为 frontier 单点
Step 0.5 建覆盖清单：对「量化/稀疏/Cache ≥2 个有单点 frontier」的任务，在迭代表建 [MUST]
        必测行（两两 + 三元 Cache+量化+稀疏）——先于自由候选规划
Step 1 静态筛查：按 seam 表排除同 seam 双 writer 与 kernel 不支持的「量化 × 融合」组合；
        必测行中同 seam（稀疏×Cache）按「窗口可分」判定，不直接排除
Step 2 逐组合叠加：先跑完覆盖清单 [MUST] 行，再做自由候选叠加（跨 seam 优先），登记候选
Step 3 验收：三层证据（图命中/kernel diff/首步耗时）+ quality gate + off-identity 复核；
       不达标 → 层回退（记名单）后复验（**回退后档位全量步数实测 e2e**，见「层回退策略」）；达标 → 保留进 frontier
Step 4 收敛与产物：覆盖清单收口（无未裁决 [MUST] 行）+ 候选登记表收尾 +
        final_report「优化方案」节 + 声明边界
```

稀疏度-性能曲线预扫描（benchmark-dev / mindie_bench）先行，为稀疏档参数提供实现级证据，
再进入端到端组合（与 model-auto-optimization 的 S4 纪律一致）。

## 候选登记表（产物模板）

| # | 组合（基底 + 新增） | seam 判定 | 必测 | 步数 | e2e(实测或[估算]) | quality gate | off-identity | 结论 |
|---|---|---|---|---|---|---|---|---|
| 1 | 量化 W8A8（单点） | — | — | 全量 | … | pass | n/a | frontier |
| 2 | 基底 1 + 缓存 | 跨 seam（precision × step_decision） | [MUST] | 全量 | … | pass | pass | frontier |
| 3 | 基底 2 + 稀疏 80% | 跨 seam；与缓存同 step 窗口 → 先查 dense_steps 冲突 | [MUST] | 全量 | … | fail（闪烁） | — | 回退：稀疏降 60% 或错开窗口 |
| 4 | Cache + 量化 + 稀疏（三元） | 跨 seam + 窗口可分 | [MUST] | 全量（实测） | …（e2e 实测，禁止估算） | … | … | 最强组合行 / 层回退后复验 |
| … | | | | | | | | |

## 层回退策略

- 触发：端到端 quality gate fail（伪影/定量超容差），或无损项被有损组合稀释。
- 顺序：先降该维度档位（稀疏度/缓存阈值）→ 再错开 step 窗口 → 最后整维回退；
  每次回退记名单（层/参数/原因）并复验；回退名单进 evidence.json 与 final_report。
- **回退必须全量实测 e2e（强制）**：对 cache/稀疏/量化任一维度发生回退（降档/错开窗口/换实现/
  整维回退后仍保留的档位）时，**回退后的最终档/边界档必须全量步数实测 e2e** 来识别性能加速
  效果——回退档是潜在采纳配置，是结论锚点，**禁止用少步识别或 `[估算]` 宣称回退后的性能**；
  回退前的高档失败数据可留少步/估算记录（进迭代表/§C），但回退后采纳的档必须实测（若该回退档
  即为最终推荐/三元组合，按 overview-report §1.1 锚点行实测）。

## 产物与接线

- 组合矩阵与回退名单 → final_report.md「优化方案（按实施顺序）」+ evidence.json
  （features / validation.technique_counters）；
- 质量判定 → `references/quality-gate.md` + 仓库 `evals/`；性能证据 → profiling 回路；
- 特性能力/支持矩阵 → `references/mindiesd-features.md`（唯一真相源，冲突时以它为准）；
- 首次真实组合试验后回填 seam 表（逐行校准确认），并更新本文件「维护与更新」。

## 与相邻边界

- 单特性有损判定 → quality-gate.md；
- 稀疏度-性能实现级扫描 → benchmark-dev；
- 编排层验收与声明纪律 → model-auto-optimization（阶段路由表 S4 / 声明纪律与真实性核验）。

## 维护与更新

- seam 表/协议变化 → 同步 model-auto-optimization 的 S4 引用与 `.agents/README.md` §4（S4-2）。
- 本文件被 performance-optimization（Step 4）与 model-auto-optimization（S4）引用。
