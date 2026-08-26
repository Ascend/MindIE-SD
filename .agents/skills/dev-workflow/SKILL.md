---
name: dev-workflow
compatibility: 无额外工具依赖（负责编排其余 skill）；需 git 与仓库工作区
description: MindIE-SD 开发总入口。当用户进行 MindIE-SD 的任何开发工作时使用此 skill——
             包括但不限于写 pattern、改测试、部署到昇腾、跑 benchmark、性能分析、多卡并行、复盘归档。
             即使用户未明确提到"开发流程"，只要涉及 MindIE-SD 代码改动都应触发。
---

# MindIE-SD 开发工作流

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

写实现前，在 NPU 上用 dummy-run-dev 的 Dummy Run 方法快速验证模型架构兼容性，
不必下载完整权重。如果已通过验证则跳过。
部署完成后，使用 framework-integration §1 验证已部署模型的推理正确性。

## 3. 并行开发策略

无代码依赖的独立模块并行推进，共享文件最后合并：

- 每个模块走独立闭环：写测试 → 实现 → 部署 → 各自 pytest
- 多卡验证时通过 ascend-deploy 部署到不同 NPU 卡隔离运行
- 共享文件（如 `patterns/__init__.py`、`passes/__init__.py`）的修改在最后统一合并
- 部署时一次性推送所有文件到远端，验证阶段使用不同卡 ID 并行运行

## 4. 远程部署

本地编码完成后用 ascend-deploy 将代码推送到昇腾容器，编译验证。

→ 部署的 shell 脚本隔离、跨平台编码注意事项见 references/cross-platform.md。

## 5. 性能评估与优化

功能验证通过后，用 profiling-collection 采集真实 NPU 数据、performance-analysis 分析并建立性能基线。
Benchmark 计时方法论（L2-flush 放计时区外、warm/cold 双档）见 compilation-dev/references/benchmark-guide.md。

采集完成后用 performance-analysis 定位瓶颈，用 performance-optimization 选择优化方案。
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

## Reference Files

- 📋 `../compilation-dev/SKILL.md` — 加载时机: 编写或修改 compilation pattern 时
- 🔧 `../aclgraph-dev/SKILL.md` — 加载时机: 静态 shape 大 batch 需要批量下发时
- 🔧 `../operator-dev/SKILL.md` — 加载时机: 算子本体开发/调优（复用 cannbot-skills）时
- 🔧 `../framework-integration/SKILL.md` — 加载时机: 三方框架对接/验证（vLLM-Omni 等）时
- ⚡ `references/ascend-ops.md` — 加载时机: 涉及 NPU 算子调用或环境诊断时
- 🔧 `references/cross-platform.md` — 加载时机: 跨平台部署或遇到 PowerShell/编码兼容问题时
- 📝 `references/rework-lessons.md` — 加载时机: 每次复盘归档时，或遇到相似问题需查历史教训时
- 📄 `../markdown-lint/SKILL.md` — 加载时机: Markdown 文件格式检查（跨 skill 引用）

## 维护与更新

当开发流程发生偏离、出现新的返工模式、或模块间关系变更时更新本 skill。
各子模块 skill 的更新触发条件参见各自的"维护与更新"章节。

## 新增 Skill 规范

当需要新建 skill 时，遵循 [Anthropic skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md) 指南，**以下为必检清单**：

- **目录结构**: `<skill-name>/SKILL.md` + 可选 `scripts/` `references/` `assets/` `evals/`
- **SKILL.md < 500 行**，超限用 references/ 拆分（progressive disclosure）
- **frontmatter 必须含 `compatibility`**：声明运行依赖（工具/环境/外部技能库）
- **description 要 pushy**：明确写清触发条件（what + when-to-trigger），覆盖 near-miss 边界
  （"看似相关但不应触发"的场景），避免 undertrigger 与误触发
- **必须建 `evals/evals.json`**：≥2 条真实用户 prompt（`prompt` / `expected_output` / `expectations`，
  格式遵循 skill-creator `references/schemas.md`）；新增或大改 skill 后必须同步增补
- **references 必须全部接线**：在 SKILL.md 列出 Reference Files + 加载时机；**>300 行的 reference 必须带目录**
- **references 必须含"维护与更新"章节**，写明更新触发条件
- **单 skill 单职责**：不同模块/领域的内容拆分为独立 skill
- **模型/框架专属知识放 `references/<variant>/` 子目录**（如 `references/models/minimax-h3.md`），避免平铺
- **命名**：小写连字符，业界通用名（如 `ascend-deploy`、`dummy-run-dev`）
- **新 skill 必须包含"维护与更新"章节**
