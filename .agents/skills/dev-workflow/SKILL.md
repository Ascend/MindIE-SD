---
name: dev-workflow
compatibility: 无额外工具依赖（负责编排其余 skill）；需 git 与仓库工作区
description: MindIE-SD 仓库开发总入口（侧轨）。当用户进行 MindIE-SD 的任何代码开发工作时使用此 skill——
             包括但不限于写 pattern、改测试、部署到昇腾、跑 benchmark、性能分析、多卡并行、复盘归档。
             模型/三方框架自动优化类任务（非本仓代码改动）由 model-auto-optimization 入口承接，
             本入口只在优化流程需要新增 pattern/算子/部署代码时承接其指向的开发子任务。
             即使用户未明确提到"开发流程"，只要涉及 MindIE-SD 代码改动都应触发。
---

# MindIE-SD 开发工作流

## 0. 入口分流

- 本入口是 **MindIE-SD 仓库开发入口（侧轨）**：改 pattern / 算子 / 图下发 / 测试 / 文档等本仓代码。
- 对一个**三方框架托管的模型**做接入、无损/有损优化、并行调优或性能收益确认（不指向本仓代码改动）
  → 先读 `model-auto-optimization/SKILL.md`，按 S0–S4 流水线路由到 env-install、framework-feature-enablement、
  profiling-collect、profiling-analyze、parallelism-strategy、performance-optimization
  等能力技能。
- 优化流程需要新增 pattern / 算子 / 部署代码时，由 model-auto-optimization 指向本入口承接开发子任务。
- 三方框架自身结构性缺口补齐（comm-stream / 缓存 / 稀疏/量化消费者等，经 model-auto-optimization §0
  用户确认）→ 先加载 `../framework-extension-dev/SKILL.md`（框架差异/注入点/合入姿势），
  实现仍按本入口开发子任务执行（Test-First → 部署 → 验证 → 复盘）。

### 0.1 开发子任务路由（按代码落点定侧）

框架 × mindiesd 的合作界面随框架而异，开发与特性开发常有重叠区；**分工按「改动文件在哪个仓库」
一刀切**，路由如下：

| 改动落点 | 路由 | 流程/说明 |
|---|---|---|
| mindiesd 本仓（pattern / kernel / 图下发 / 测试 / `mindiesd/parallel` 等） | 本入口 + compilation-dev / operator-dev / aclgraph-dev（按实现类型） | 本 SKILL 主流程：Test-First → 部署 → pytest → 复盘 |
| 三方框架仓 · 框架**已有**特性接线/开关/小修 | framework-feature-enablement | enablement 使能回路验证（计数契约 + 三层证据） |
| 三方框架仓 · 框架**未支持**特性的结构性开发 | framework-extension-dev（先加载，差异表/注入点/合入姿势） | 实现仍按本入口子任务执行 |
| 跨侧重叠（同一特性 mindiesd + 框架两侧都改） | **按文件拆两侧子任务**：mindiesd 侧走本入口开发技能；框架侧按上两行路由 | 对接点联调收口（输出一致 + 计数契约 + 三层证据）；接口差异记录 `framework-support-matrix.md` |
| 环境 / 权重 / 部署 | env-install + remote-access | 非代码开发 |

判断口径：先问「要改的文件在哪个仓库」；同一 MR 跨两仓时先确认合入渠道（上游 PR / fork 钉版本），
再拆子任务执行。

### 0.2 流程门禁（先验收后推进）

- 每个功能点闭环（写测试 → 实现 → 部署 → pytest）的验证输出（命令 / 退出码 / 关键结果）随
  实施记录留痕；缺失或与产物矛盾时不得宣称该功能点通过。
- 复盘检查（§6.3）在本轮全部功能点收尾后执行；发现流程偏离（如计划并行实际串行）先在复盘
  记录原因，再进入下一轮工作。
- 被 `model-auto-optimization` 指向的模型优化子任务，遵守其 run-state 推进表与
  `scripts/stage_gate.py` 门禁（见 `model-auto-optimization/references/run-state.md`），
  本入口按 dev-workflow 复盘与提交流程承接代码改动侧。

### 0.3 交付件契约（L1 编排协议：开发任务必交）

> 三层定位：本 SKILL 是 **L1 编排入口**——负责分流/路由、**顺序契约**（Test-First、并行、
> 复盘时机）与**交付件契约**；单任务的"最佳实现路径/回退"（pattern 生命周期、算子 DSL 选择、
> mismatch 回退）由 **L3 能力自身工作流纪律**承载（compilation-dev / operator-dev / …）；
> L2 为内联轻量（本节即 L1 契约，暂不拆独立 workflow 文件）。

**顺序契约（引用）**：功能点走 §1 Test-First（先测试后实现）；独立模块按 §3 并行、共享文件后
合并；复盘在 §6 收尾执行。

**交付件契约（开发任务必交，缺一不宣称完成/不进入提交）**：

- 功能点证据行：测试预期 FAIL → 实现 → 部署编译 → pytest PASS 的命令/退出码/关键结果留痕
  （与 §0.2 门禁一致）；
- pytest 通过输出（或明确注明未跑原因）；
- 复盘记录：§6.3 复盘清单执行结果；发现流程偏离先记原因；
- 回填与同步：可复用经验按 §6.4/`.agents/README.md` §7 回填（rework-lessons / case / skill
  刷新），涉及公共流程面同步 README；
- 提交规范：按 `mindie-sd-community-governance`（commit/PR 格式、模板四区块）；
- 被 model-auto-optimization 指向的子任务：遵守宿主 run-state 推进表/迭代表 + stage_gate 门禁，
  本入口只承接代码改动侧并按上述契约收口。

**为 dev 拆独立 L2（workflow 文件）的信号**：出现**跨多个能力、多阶段、有顺序门禁与逐阶段
确认点**的开发线（如端到端融合算子合入链）时，为该线新增 `workflows/{line}.md`，机制复用
L1 协议（迭代表/证据行/stage_gate），并把本节契约改为引用。

## 1. Test-First 流程

每个功能点必须遵循「先测试，后实现」的闭环：

```text
写测试（预期 FAIL） → 实现功能 → 远端部署编译 → 远端 pytest 验证 → 进入下一阶段
```

- 测试必须覆盖：输出正确性 + 耗时验证
- 新功能的测试应先写，确认 FAIL 后再写实现
- 编码过程中遵循 code-standards 规范（Ruff lint、pre-commit 钩子、代码风格约定）
- Markdown 文件的格式检查由 markdown-lint 规范覆盖，提交前需通过 `pre-commit run markdownlint` 检查

### 1.1 Pattern 开发专项

若任务是新增或调试 MindIE-SD compilation pattern（RMSNorm / RoPE / AdaLayerNorm / GELU 融合），
路由到 `compilation-dev` skill 获取全生命周期指导：模型代码分析 → pattern 创建 → 注册 → 单元测试 → mismatch 调试 → 集成验证 → Copy 消减。
算子本体（triton kernel 编写/调优）→ `operator-dev`；批量下发（aclgraph）→ `aclgraph-dev`。

## 2. 模型验证

写实现前，在 NPU 上用 dummy-run 的 Dummy Run 方法快速验证模型架构兼容性，
不必下载完整权重。如果已通过验证则跳过。
部署完成后，使用 framework-feature-enablement 验证已部署模型在框架侧的推理正确性（1 步推理/特性开关）。

## 3. 并行开发策略

无代码依赖的独立模块并行推进，共享文件最后合并：

- 每个模块走独立闭环：写测试 → 实现 → 部署 → 各自 pytest
- 多卡验证时通过 env-install + remote-access 部署到不同 NPU 卡隔离运行
- 共享文件（如 `patterns/__init__.py`、`passes/__init__.py`）的修改在最后统一合并
- 部署时一次性推送所有文件到远端，验证阶段使用不同卡 ID 并行运行

## 4. 远程部署

本地编码完成后用 env-install 的部署流程（remote-access 提供 SSH 工具）将代码推送到昇腾容器，编译验证。

→ 部署的 shell 脚本隔离、跨平台编码注意事项见 references/cross-platform.md。

## 5. 性能评估与优化

功能验证通过后，用 profiling-collect 采集真实 NPU 数据、profiling-analyze 分析并建立性能基线。
Benchmark 计时方法论（L2-flush 放计时区外、warm/cold 双档）见 compilation-dev/references/benchmark-guide.md。

采集完成后用 profiling-analyze 定位瓶颈，用 performance-optimization 选择优化方案。
多卡场景参考 parallelism-strategy 选择并行策略。

## 6. 复盘归档

每个 Phase 完成后按以下流程复盘：

1. 回顾本阶段问题点和改进点
2. 检查是否需要补充 references/rework-lessons.md
3. 交叉检查各模块 skill 需不需要更新
4. 同步刷新 `.agents/README.md` 技能总览与各 skill 状态

### 识别更新信号

出现以下情况时必须检查 skill 是否需要更新：

- 同一类问题重复出现 ≥ 2 次
- 开发流程偏离预期（如计划并行但实际串行）
- 发现新的可用算子或确认算子不可用
- 远端环境发生变化（torch/TorchNPU 版本更新）
- 出现新的有效工作方法

### 6.3 复盘检查清单

复盘时逐一确认：

| 检查项 | 说明 |
|---|---|
| 目标范围 | 是否在第一步就确认了文件路径与影响面？ |
| 文件编码 | 批量操作前是否验证了 UTF-8 含中文文件的编码安全性？ |
| 匹配覆盖 | 正则/替换模式是否覆盖了所有变体（缩进围栏、嵌套代码块）？ |
| 工具一致性 | 本地 markdownlint 版本与配置是否与 CI 一致？ |
| 总览/细分报表 | 模型优化类闭环是否输出 `overview_report.md` 与 `detail_report.md`（总览：基线=TP 多卡未优化、三层级行组 + 每特性组合搜索数据；细分：融合算子逐项收益与剩余机会、并行候选与带宽、稀疏/量化(FA+线性层)/cache 与采样步数、组合矩阵）？ |

### 6.4 实验 → case 回填

复盘确认新经验需沉淀为 case 时，按 `.agents/README.md` §7「case 回填规范」执行：
**先做「经验 vs 探针」判定**（临时方案/一次性补丁/诊断绕法 → 只作 `[探针]` 归档，不进
推荐姿势/经验槽位/宣称；发现的 durable 约束事实按经验沉淀并注明来源）→ 归属判定 →
case 模板 → SKILL.md 接线（Reference Files + 加载时机）→ evals 增补 →
隐私清理 → 校验（markdownlint / JSON / ruff）。

## Reference Files

- `../compilation-dev/SKILL.md` — 加载时机: 编写或修改 compilation pattern 时
- `../aclgraph-dev/SKILL.md` — 加载时机: 静态 shape 大 batch 需要批量下发时
- `../operator-dev/SKILL.md` — 加载时机: 算子本体开发/调优（复用 cannbot-skills）时
- `../framework-feature-enablement/SKILL.md` — 加载时机: 三方框架特性使能/验证（vLLM-Omni 等）时
- `../framework-extension-dev/SKILL.md` — 加载时机: 承接三方框架侧特性补齐开发子任务时（框架差异/合入姿势）
- `references/ascend-ops.md` — 加载时机: 涉及 NPU 算子调用或环境诊断时
- `references/cross-platform.md` — 加载时机: 跨平台部署或遇到 PowerShell/编码兼容问题时
- `references/rework-lessons.md` — 加载时机: 每次复盘归档时，或遇到相似问题需查历史教训时
- `../markdown-lint/SKILL.md` — 加载时机: Markdown 文件格式检查（跨 skill 引用）

## 维护与更新

当开发流程发生偏离、出现新的返工模式、或模块间关系变更时更新本 skill。
各子模块 skill 的更新触发条件参见各自的"维护与更新"章节。

## 新增 Skill 规范

当需要新建 skill 时，遵循 [Anthropic skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md) 指南，**以下为必检清单**：

- **目录结构**: `{skill-name}/SKILL.md` + 可选 `scripts/` `references/` `assets/` `evals/`
- **SKILL.md < 500 行**，超限用 references/ 拆分（progressive disclosure）
- **frontmatter 必须含 `compatibility`**：声明运行依赖（工具/环境/外部技能库）
- **description 要 pushy**：明确写清触发条件（what + when-to-trigger），覆盖 near-miss 边界
  （"看似相关但不应触发"的场景），避免 undertrigger 与误触发
- **必须建 `evals/evals.json`**：≥2 条真实用户 prompt（`prompt` / `expected_output` / `expectations`，
  格式遵循 skill-creator `references/schemas.md`）；新增或大改 skill 后必须同步增补
- **references 必须全部接线**：在 SKILL.md 列出 Reference Files + 加载时机；**>300 行的 reference 必须带目录**
- **references 必须含"维护与更新"章节**，写明更新触发条件
- **单 skill 单职责**：不同模块/领域的内容拆分为独立 skill
- **模型/框架专属知识放 `references/{variant}/` 子目录**（如 `references/models/minimax-h3.md`），避免平铺
- **命名**：小写连字符，业界通用名（如 `env-install`、`dummy-run`）
- **新 skill 必须包含"维护与更新"章节**
