---
name: mindie-sd-community-governance
description: "Handle MindIE-SD documentation, governance, contributor workflow, commit or PR conventions, history cleanup, template changes, and version-policy updates with minimal context, fixed review steps, and deterministic output."
---

# MindIE-SD Community Governance Skill

## 1. 适用范围

仅在以下任务中使用本 skill：

- 修改 `docs/` 下的用户文档、开发者文档、首页、目录或导航
- 调整中英文文档内容、结构或入口
- 修改文档站点配置，例如 `docs/conf.py`、`.readthedocs.yaml`、`docs/requirements-docs.txt`
- 排查文档网站编译、Sphinx、Read the Docs、导航丢失、链接失效等问题
- 修改治理文档
- 修改贡献流程说明
- 调整 commit message 规范
- 整理提交历史，例如拆分、压缩、rebase、cherry-pick、squash
- 修改 PR / Issue 模板
- 修改 PR / MR 标题或正文规范
- 修改版本策略或发布规则
- 修改与角色、审批、仓库流程直接相关的规则文本

本 skill 不负责通用工程实现，不负责仓库默认工作流定义。仓库默认规则由 `AGENTS.md` 负责。

## 2. 输入与目标

输入是一个治理或流程类修改请求。
也可以是文档内容、双语同步或文档站点构建请求。

目标只有三个：

- 用最小上下文确认当前仓库事实
- 产出一致、可复核的治理或文档类修改结论
- 在文档场景下检查双语配套、入口导航和文档站点编译路径

不要默认全量阅读治理或文档文件。先按场景选最小读取集，再做一致性检查。

按需读取的资产：

- `assets/mr_ruleset_20260327101328.xlsx`

该资产是 MR 静态规则 / assert 依据文件。
只有当任务明确涉及代码、脚本、静态检查、MR 质量门禁或提交规范时才读取，不作为默认必读项。
读取时只提取与当前语言或当前变更类型直接相关的规则，不加载整表。
如果资产中没有 commit / PR 标题模板，不要从中反推标题格式。

## 2.1 强命中信号

出现以下任一信号时，本 skill 必须优先接管：

- `docs`、`文档`、`README`、`developer guide`
- `中英文`、`英文版`、`中文版`、`双语`
- `Sphinx`、`Read the Docs`、`RTD`、`docs/conf.py`
- `文档网站`、`编译`、`构建`、`sphinx-build`
- `index.md`、`menu_user_manual.md`、目录、导航、首页
- `commit`、`提交`、`commit message`
- `rebase`、`squash`、`cherry-pick`
- 提交历史整理、提交拆分、历史重写
- `PR`、`MR`、标题格式、正文格式
- 模板、治理、版本策略、发布规则

如果当前仓库已有提交使用 `docs:`、`fix:`、`chore:` 等 conventional commits 风格，而本 skill 要求使用 `[Type][Scope]Summary`，必须先指出冲突，再按本 skill 规范整理。

## 3. 按场景的最小读取集

### 3.1 PR / Issue 模板变更

- `AGENTS.md`
- `contributing.md`
- 目标模板文件
- 与模板直接相关的流程文件

### 3.2 提交与 PR 规范变更

- `AGENTS.md`
- `.gitcode/PULL_REQUEST_TEMPLATE.md`
- `README.md`
- `docs/zh/developer_guide/test.md`
- `docs/zh/developer_guide/tooling.md`
- `assets/mr_ruleset_20260327101328.xlsx`，按需读取

### 3.3 文档内容或站点构建变更

- `AGENTS.md`
- 当前被修改的文档文件
- 对应语言的配套文档
- `docs/index.md`，如果修改影响首页入口
- `docs/zh/menu_user_manual.md` 或 `docs/en/menu_user_manual.md`，如果修改影响导航
- `docs/conf.py`
- `.readthedocs.yaml`
- `docs/requirements-docs.txt`

### 3.4 Contributor workflow 变更

- `AGENTS.md`
- `contributing.md`
- 相关 workflow 文件
- 相关治理文档

### 3.5 Governance 文档变更

- `AGENTS.md`
- 当前被修改的治理文档
- 对应语言版本的配套文档
- `OWNERS`，如果变更涉及角色或审批

### 3.6 Version / release policy 变更

- `AGENTS.md`
- `CHANGELOG.md`
- `RELEASE.md`
- `mindiesd/_version.py`
- `pyproject.toml`
- 相关流程文件

## 4. 处理流程

1. 先确认任务属于哪个治理场景。
2. 只读取该场景的最小读取集。
3. 提炼当前规则、目标改动和缺口。
4. 检查是否需要联动：
   - 双语文档
   - `OWNERS`
   - 版本源
   - 发布说明
   - 模板入口
   - 提交格式
   - PR 标题与正文格式
   - 首页入口与目录导航
   - 文档站点编译配置
5. 再实施修改或给出修改建议。
6. 若任务涉及提交、提交历史整理、准备推送、准备提 PR/MR，必须主动产出可直接使用的 commit 标题建议、PR 标题建议和 PR 正文草案；不能只给格式规则。
7. 最后按固定输出模板总结，不追加自由发挥的长文。

## 5. 提交与 MR/PR 规范

### 5.1 Commit message 格式

- 推荐格式：`[Type][Scope]Summary`
- 无 scope 时允许：`[Type]Summary`
- `Type` 固定为：`Feature`、`Bugfix`、`Docs`、`CI`、`Refactor`、`Test`、`Chore`
- `Scope` 使用英文短词或仓库模块名，例如 `quant`、`ops`、`service`、`docs`、`build`
- `Summary` 使用单行，直接描述变更主目标
- 不使用空泛词，例如 `update`、`fix issue`、`misc`
- 若在整理历史或手动拆分提交，单个 commit diff 应控制在 1000 行以内；超过时按功能边界继续拆分
- 提交前至少自检一次标题是否满足 `^\[(Feature|Bugfix|Docs|CI|Refactor|Test|Chore)\](\[[A-Za-z0-9._-]+\])?.+`

禁用示例：

- `bugfix`
- `fix`
- `update log`
- 无类型前缀的随意中文短句

### 5.2 PR 标题格式

- PR 标题与 commit 标题同构，统一为 `[Type][Scope]Summary` 或 `[Type]Summary`
- PR 标题只描述本次变更主目标，不堆多个不相关事项
- 若任务已经形成一组待提交或待推送修改，必须至少给出 1 个可直接使用的 PR 标题，不允许只说明格式而不给标题草案

### 5.3 PR 正文格式

- 必须使用仓库现有 `.gitcode/PULL_REQUEST_TEMPLATE.md`
- 正文必须覆盖以下 4 块：
  - `Which issue(s) this PR fixes or accomplishes`
  - `Purpose`
  - `Test Plan`
  - `Test Report`
- Issue 关联写法遵循模板说明：
  - 完整解决使用 `Fixes #...`
  - 部分解决使用 `Fix part of #...`
- 若任务涉及提交、推送、合并请求、历史整理或“准备发 PR”，必须按模板 4 个区块主动产出一份可直接粘贴的 PR 正文草案
- 若缺少 issue 编号或测试结果，必须在草案中保留占位并明确标注待补充，而不是省略整个区块

### 5.4 提交前检查

- 本地提交前检查入口统一使用 `pre-commit`
- 事实依据来自 `README.md`、`docs/zh/developer_guide/test.md` 与 `docs/zh/developer_guide/tooling.md`
- 最小检查命令：
  - `python -m pip install -r requirements-lint.txt`
  - `pre-commit install`
  - `pre-commit run --all-files`
- `git commit --no-verify` 只在明确需要绕过时使用
- 对本次 commit 新增或修改的 `.md` 文件，额外做 markdownlint 专项检查：

  ```shell
  git diff --name-only HEAD~1..HEAD -- '*.md' | xargs markdownlint -c .markdownlint.json
  ```

- 批量修复 `.md` 文件时需确认编码不破坏：PowerShell `Get-Content`/`Set-Content` 读写 UTF-8 含中文文件会引入 BOM 与乱码，应使用 Python 或 Node.js 处理

### 5.5 Assert / MR 质量规则资产

- `assets/mr_ruleset_20260327101328.xlsx` 是 MR 静态规则 / assert 依据文件
- 它不是 commit 或 PR 标题模板来源
- 当前资产主要提供静态代码规则，不提供 `[Type][Scope]Summary` 的标题模板事实源
- 当任务涉及代码、脚本、静态检查或 MR 质量门禁时，按需引用该资产
- 只提取与当前语言或变更类型直接相关的规则

### 5.6 文档一致性与构建要求

- 修改 `docs/zh/` 下对外文档时，默认检查 `docs/en/` 是否存在对应配套页
- 修改 `docs/en/` 下对外文档时，默认检查 `docs/zh/` 是否存在对应配套页
- 修改首页、目录、用户手册入口时，检查 `docs/index.md`、`docs/zh/menu_user_manual.md`、`docs/en/menu_user_manual.md`
- 修改 developer guide 结构时，检查中英文 developer guide 入口是否同步
- 涉及文档文件、导航或文档配置修改时，默认给出最小站点验证路径：
- `python -m pip install -r docs/requirements-docs.txt`
- `python -m sphinx -b html docs docs/_build/html`
- 若未实际执行编译，必须明确写“未执行，仅给出建议验证命令”

## 6. 一致性检查项

- 修改是否与当前仓库事实一致
- 是否误改了非目标流程或非目标模板
- 是否需要中英文同步
- 是否需要 `OWNERS` 或角色信息联动
- 是否影响 `CHANGELOG.md`、`RELEASE.md`、`mindiesd/_version.py`、`pyproject.toml`
- 是否引入新的流程入口、模板字段或版本来源
- 是否引入新的 commit / PR 格式分歧
- 是否正确使用 `.gitcode/PULL_REQUEST_TEMPLATE.md`
- 是否按需使用 `assets/mr_ruleset_20260327101328.xlsx`
- 是否需要同步首页或目录入口
- 是否需要补充文档站点编译验证
- 是否只在必要范围内读取和修改文件

## 7. 固定输出模板

使用本 skill 时，结尾固定输出以下 6 项：

1. **变更类别**
2. **影响文件**
3. **读取依据**
4. **一致性检查结果**
5. **必需联动项**
6. **风险与缺口**

输出要求：

- 每项只保留当前任务所需事实
- 优先写结论，不复述长背景
- 依据中优先写文件路径和规则名
- 若涉及提交规范或 MR 质量要求，说明是否使用了 `assets/mr_ruleset_20260327101328.xlsx`，以及使用了哪类规则
- 若涉及文档修改，说明中英文是否同步，以及文档站点编译是否已执行
- 若涉及提交、推送或 PR/MR，追加给出：
- `Commit title proposal`
- `PR title proposal`
- `PR body draft`
- 如果没有联动项或风险，明确写“无”
