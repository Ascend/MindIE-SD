# MindIE-SD Agent Rules

## 1. 定位

- 本文件只定义仓库级默认规则。
- 本文件负责工作流、资源边界、上下文压缩和 skill 路由。
- 本文件不承载领域细节。
- 领域规则由对应 `SKILL.md` 承接。

## 2. 工作流定义

### 2.1 任务进入

- 先识别任务类型：实现、修复、文档、治理、模板、流程、版本策略。
- 当请求包含 `commit`、`提交`、`rebase`、`squash`、`cherry-pick`、`PR`、`MR`、模板、发布、版本策略等关键词时，优先判定是否属于治理/流程类任务。
- 当请求包含 `docs`、`文档`、`README`、`中英文`、`Sphinx`、`Read the Docs`、`文档网站`、`编译`、`index.md`、`menu_user_manual.md` 等关键词时，优先判定是否属于文档类任务，并继续检查是否需要加载本仓库治理 skill。
- 先确认目标文件和目标行为，再决定读取范围。
- 不因为用户提到一个概念就默认读取整仓。

### 2.2 事实核实

- 先读最小必要文件，再下结论。
- 需要恢复历史行为时，先看当前文件，再看仓库历史或已知基线。
- 需要判断是否影响构建、版本、流程时，必须读对应事实源。

### 2.3 变更分类

- 实现类：以运行逻辑和测试为主。
- 文档类：以文档一致性和对外说明为主。
- 治理类：以规则一致性和联动文件为主。
- 模板/流程类：以提交入口、说明和联动规范为主。
- 版本策略类：以版本源、发布说明和相关流程为主。

### 2.4 最小读取集

- 只读取当前任务直接相关的规范源、事实源、实现源、验证源。
- 无法证明相关的文件，不进入默认读取集。
- 读取后只保留当前决策所需事实，不重复搬运全文。

### 2.5 编辑前检查

- 确认目标文件是否为当前任务的真实来源文件。
- 确认是否存在历史基线、旧实现或现成模板可恢复。
- 确认是否会扩大 public API、版本源或 contributor workflow。
- 涉及 commit message、提交历史整理、PR/MR 标题或正文、模板、版本策略时，必须先读取本仓库最贴合当前任务的 skill，再决定是否补充参考外部 skill 仓库。
- 涉及 `docs/` 文档内容、双语同步、首页目录、Sphinx 或 Read the Docs 构建时，必须先读取 `.agents/skills/mindie-sd-community-governance/SKILL.md`，再决定是否补充参考外部 skill 仓库。

### 2.6 交付输出

- 输出优先给结论。
- 结论后给依据文件。
- 最后给后续动作、验证命令或未验证项。
- 涉及提交、提交历史整理、推送、PR 或 MR 时，必须主动给出可直接使用的 commit 标题、PR 标题和 PR 正文草案；不能只说明格式规则。
- 不把长文复述当作交付。

## 3. 资源定义

### 3.1 规范源

- 用于确定仓库默认规则和流程。
- 最小示例：
  - `AGENTS.md`
  - `.agents/skills/*/SKILL.md`
  - `https://gitcode.com/Ascend/agent-skills`
  - `contributing.md`

### 3.2 事实源

- 用于确认当前仓库的真实状态和来源。
- 最小示例：
  - `README.md`
  - `setup.py`
  - `OWNERS`
  - `.gitcode/PULL_REQUEST_TEMPLATE.md`

### 3.3 实现源

- 用于确认行为、导出面和模块边界。
- 最小示例：
  - `mindiesd/__init__.py`
  - `mindiesd/*`
  - `build/*`
  - `csrc/*`

### 3.4 验证源

- 用于确认最小验证路径和已有测试约束。
- 最小示例：
  - `tests/*`
  - `tests/README.md`
  - `docs/*`

### 3.5 读取原则

- 先读规范源决定流程。
- 再读事实源确认真相。
- 只有当任务涉及行为改动时才扩展到实现源。
- 只有当任务需要验证时才扩展到验证源。

## 4. 上下文压缩规则

- 只保留当前任务所需事实。
- 先写结论，再写证据，不写长篇背景。
- 默认使用“结论 + 依据文件 + 后续动作”结构。
- 引用规则时优先写规则名和文件路径，不粘贴整段正文。
- 治理类任务优先提炼联动关系、影响面和缺口。
- 同一事实只保留一次，后续直接引用。
- 大文件只提取与当前决策直接相关的片段。

## 5. Skill 路由规则

- 优先查看本仓库 `.agents/skills/` 中是否已有最贴合当前任务的 skill。
- 本仓库没有合适 skill，或本地 skill 缺少必要规范时，再参考 `https://gitcode.com/Ascend/agent-skills`。
- 只选择最小、最贴合当前任务的 skill，不做全量加载。
- 对高风险流程类任务，`AGENTS.md` 允许显式指定必须优先加载的本地 skill。
- 以下请求必须先读取 `.agents/skills/mindie-sd-community-governance/SKILL.md`：
- commit message 格式调整
- 提交拆分、压缩、rebase、cherry-pick 或历史整理
- PR / MR 标题与正文格式
- PR / Issue 模板
- contributor workflow、治理规则、版本策略
- `docs/` 下文档内容修改
- 中文 / 英文配套文档检查
- `docs/index.md`、`menu_user_manual.md`、developer guide 入口调整
- `docs/conf.py`、`.readthedocs.yaml`、`docs/requirements-docs.txt` 修改
- 文档网站编译、Sphinx、Read the Docs 相关问题
- skill 与本文件冲突时，当前任务直接采用的 skill 优先于本文件；未命中 skill 的任务按本文件默认工作流执行。

## 6. 最小交付要求

- 说明做了什么。
- 说明依据了哪些文件。
- 说明跑了哪些验证，或哪些验证未跑。
- 涉及恢复类修改时，说明恢复到哪个基线。
- 涉及 public API、版本源、模板入口、流程定义时，明确说明是否发生变化。
