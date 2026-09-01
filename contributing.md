# Contributing to MindIE SD / MindIE SD 贡献指南

[简体中文](#简体中文) | [English](#english)

本文件是 MindIE SD 唯一的贡献流程与治理规则入口。

This file is the single source of truth for the MindIE SD contribution workflow
and governance rules.

## 简体中文

感谢你为 MindIE SD 做出贡献。提交 Issue、代码、测试、文档或评审即表示你同意
遵守[社区行为准则](CODE_OF_CONDUCT.md)。发现安全漏洞时，请勿提交
公开 Issue，应按照[安全政策](SECURITY.md)报告。

### 开始之前

- 搜索现有 Issue 和 PR，避免重复工作。
- Bug 修复和一般功能应关联清晰的 Issue。
- 重大架构、公开接口、行为或发布策略变更应先提交 RFC，并在方案获得认可后
  再开始大规模实现。
- 一个 PR 只处理一个清晰主题。

### 获取代码和创建分支

外部贡献者应先 Fork 仓库，再配置个人仓库和上游仓库：

```bash
git clone https://gitcode.com/<你的用户名>/MindIE-SD.git
cd MindIE-SD
git remote add upstream https://gitcode.com/Ascend/MindIE-SD.git
git fetch upstream
git switch -c feat/<功能名> upstream/dev
```

- 外部 PR 必须基于最新 `upstream/dev` 创建独立主题分支，并以 `dev` 为目标
  分支。
- `master` 仅用于经维护者授权的版本发布、紧急修复或 `dev` 到 `master` 的
  受控合入。
- 不得直接向 `dev` 或 `master` 推送未经评审的变更。
- 推荐使用 `feat/<功能名>`、`fix/<问题>`、`docs/<主题>` 或
  `refactor/<范围>` 作为分支名。

### 开发环境与本地检查

安装 lint 依赖和 pre-commit hook：

```bash
python -m pip install -r requirements-lint.txt
pre-commit install
```

提交前至少运行：

```bash
pre-commit run --all-files
bash tests/run_UT_test.sh
```

涉及 NPU 的变更还应按影响范围运行：

```bash
bash tests/run_test.sh --all
```

也可根据需要使用 `--cpu_only` 或 `--npu_only`。涉及模型输出或性能的变更必须
提供适用的准确性、性能或端到端测试结果。

更多构建、测试、环境、仓库结构、模式开发和性能分析指导，统一参阅
[`docs/zh/developer_guide/`](docs/zh/developer_guide/) 目录；本指南不重复列举
该目录下的单独页面。

### 代码与文档质量

- Python 代码遵循 `pyproject.toml` 中的 Ruff 配置，目标版本为 Python 3.10，
  行长度上限为 100，统一使用双引号。
- 导入按标准库、第三方库、本地模块分组；日志优先使用 `%` 风格参数，避免在
  日志模板中使用 f-string 或 `.format()`。
- C/C++ 代码使用仓库 `.clang-format`；Markdown 必须通过 markdownlint。
- 新功能和 Bug 修复应提供能够保护具体行为的测试。用户可见变更必须同步更新
  `docs/zh/` 和 `docs/en/` 的对应文档，重要变更还应更新版本说明。
- 避免重复代码、超大文件和不必要的 NPU-CPU 同步；不得使用 `pickle.loads()`
  或 `pickle.load()` 反序列化不可信数据。

### 提交与 PR 规模

- Commit 和 PR 标题使用 `[Type][Scope]Summary`；无 scope 时可使用
  `[Type]Summary`。
- `Type` 取值为 `Feature`、`Bugfix`、`Docs`、`CI`、`Refactor`、`Test`
  或 `Chore`；`Scope` 使用英文短词或模块名。
- 标题应单行、直接描述主目标，避免 `update`、`fix issue`、`misc` 等空泛表述。
- 单个 PR 的有效变更量原则上不得超过 1000 行。有效变更量按相对目标分支的
  新增行与删除行之和计算；自动生成文件、依赖锁文件、第三方镜像代码、
  二进制文件和大体量测试数据不计入。
- 无法合理拆分的变更必须关联已批准的 RFC，并在提交前获得 maintainer 对例外
  范围的确认。

### 提交 PR

使用仓库的 [PR 模板](.gitcode/PULL_REQUEST_TEMPLATE.md)，完整填写：

- `Which issue(s) this PR fixes or accomplishes`
- `Purpose`
- `Test Plan`
- `Test Report`

完整解决 Issue 时使用 `Fixes #<Issue ID>`；部分解决时使用
`Fix part of #<Issue ID>`。

当前 PR 流水线通过在 PR 评论中输入 `compile` 触发；需要触发权限时请联系
maintainer。

### 评审、合入与治理

角色名单以 Ascend community 中的
[MindIE-SD SIG 配置](https://gitcode.com/Ascend/community/blob/master/MindIE/sigs/MindIE-SD/sig-info.yaml)
为准：

- **Reviewer** 提供技术评审、协助问题复现，并审核代码、测试和文档。
- **Committer** 校验技术质量、测试证据和文档影响，执行最终批准并确认变更具备
  合入条件。
- **Branch keeper** 管理受保护分支策略、发布就绪检查和最终合入授权。

贡献者从 Contributor 成长为 Reviewer、Committer 或 Maintainer 的条件、提名和
投票流程，以 Ascend community 的
[角色定义及晋升机制](https://gitcode.com/Ascend/community/blob/master/docs/role-definition-and-promotion-mechanism.md)
为准。

普通 PR 按以下流程处理：

1. 贡献者提交代码、测试、文档和完整 PR 说明。
2. Reviewer 提供技术意见；所有意见都必须明确回复。
3. 需要修改的意见完成修改并请求复审；不采纳的意见说明理由并获得 reviewer
   确认。
4. 合入前未解决的评审会话数必须为 0；最后一次代码更新后仍须保有有效的
   Committer 批准。
5. 所有适用的必需 CI 在最新提交或预合并结果上通过。
6. Branch keeper 或授权 maintainer 将变更合入受保护分支。

发布与治理事项由 Branch keepers 协调分支和发布准备，Committers 确认代码、
文档与版本记录状态；发布仅在批准的 Ascend/NPU runner 上执行。

### Ascend 社区公共规范

除上述 MindIE SD 仓库规则外，贡献者还应遵守以下 Ascend 社区公共规范；发生
冲突时，以更严格且更贴合本仓库的规则为准。

- [Issue 提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-guide.md)
- [社区 Issue 处理流程指导](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-workflow-guidelines.md)
- [PR 提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)
- [Ascend 社区开发者测试贡献指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/developer-testing-guide.md)
- [Ascend 开源与第三方软件建仓及分支命名指导](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-repo-branch-guide.md)
- [Ascend 开源与第三方软件管理规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-software-management-guide.md)
- [社区安全设计规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/security-design-guideline.md)

## English

Thank you for contributing to MindIE SD. By submitting Issues, code, tests,
documentation, or reviews, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Do not open a
public Issue for a suspected vulnerability; follow the
[Security Policy](SECURITY.md) instead.

### Before You Start

- Search existing Issues and PRs to avoid duplicate work.
- Bug fixes and regular features should link a clear Issue.
- Major architecture, public API, behavioral, or release-policy changes should
  start with an RFC and receive design agreement before substantial
  implementation begins.
- Each PR must address one clear topic.

### Get the Code and Create a Branch

External contributors should fork the repository, then configure their fork and
the upstream repository:

```bash
git clone https://gitcode.com/<your-username>/MindIE-SD.git
cd MindIE-SD
git remote add upstream https://gitcode.com/Ascend/MindIE-SD.git
git fetch upstream
git switch -c feat/<feature> upstream/dev
```

- An external PR must use a dedicated topic branch created from the latest
  `upstream/dev` and must target `dev`.
- `master` is reserved for maintainer-authorized releases, emergency fixes, or
  controlled merges from `dev` to `master`.
- Do not push unreviewed changes directly to `dev` or `master`.
- Recommended branch names include `feat/<feature>`, `fix/<issue>`,
  `docs/<topic>`, and `refactor/<scope>`.

### Development Environment and Local Checks

Install the lint dependencies and pre-commit hooks:

```bash
python -m pip install -r requirements-lint.txt
pre-commit install
```

Run at least the following checks before submission:

```bash
pre-commit run --all-files
bash tests/run_UT_test.sh
```

Changes involving NPUs should also run the applicable tests through:

```bash
bash tests/run_test.sh --all
```

Use `--cpu_only` or `--npu_only` when appropriate. Changes affecting model
outputs or performance must provide applicable accuracy, performance, or
end-to-end test results.

For additional build, test, environment, repository-structure, pattern
development, and performance guidance, browse the
[`docs/en/developer_guide/`](docs/en/developer_guide/) directory. This guide
does not enumerate its individual pages.

### Code and Documentation Quality

- Python code follows the Ruff configuration in `pyproject.toml`, targets
  Python 3.10, uses a maximum line length of 100, and uses double quotes.
- Group imports into standard-library, third-party, and local modules. Prefer
  `%`-style arguments for logging instead of f-strings or `.format()` in log
  templates.
- Format C/C++ with the repository `.clang-format`; Markdown must pass
  markdownlint.
- New features and bug fixes should include tests that protect concrete
  behavior. User-visible changes must update the corresponding documents under
  both `docs/zh/` and `docs/en/`; significant changes must also update the
  release notes.
- Avoid duplicate code, oversized files, and unnecessary NPU-CPU
  synchronization. Never use `pickle.loads()` or `pickle.load()` to deserialize
  untrusted data.

### Commit and PR Scope

- Commit and PR titles use `[Type][Scope]Summary`, or `[Type]Summary` when no
  scope is needed.
- `Type` must be `Feature`, `Bugfix`, `Docs`, `CI`, `Refactor`, `Test`, or
  `Chore`. Use a short English term or module name for `Scope`.
- Keep the title on one line and describe the primary goal directly. Avoid vague
  wording such as `update`, `fix issue`, or `misc`.
- As a rule, a PR must not exceed 1,000 effective changed lines. Effective size
  is the sum of additions and deletions relative to the target branch,
  excluding generated files, dependency lock files, vendored code, binary
  files, and large test data.
- A change that cannot reasonably be split must link an approved RFC and obtain
  maintainer confirmation of the exception scope before submission.

### Submit the PR

Use the repository [PR template](.gitcode/PULL_REQUEST_TEMPLATE.md) and complete:

- `Which issue(s) this PR fixes or accomplishes`
- `Purpose`
- `Test Plan`
- `Test Report`

Use `Fixes #<Issue ID>` when the Issue is fully resolved and
`Fix part of #<Issue ID>` when it is only partially resolved.

The current PR pipeline is triggered by commenting `compile` on the PR. Contact
a maintainer when trigger permission is required.

### Review, Merge, and Governance

The role roster is maintained in the Ascend community
[MindIE-SD SIG configuration](https://gitcode.com/Ascend/community/blob/master/MindIE/sigs/MindIE-SD/sig-info.yaml):

- **Reviewers** provide technical review, assist with issue reproduction, and
  review code, tests, and documentation.
- **Committers** verify technical quality, test evidence, and documentation
  impact, issue final approval, and confirm that a change is ready to merge.
- **Branch keepers** manage protected-branch policy, release-readiness checks,
  and final merge authorization.

The eligibility, nomination, and voting process for progressing from Contributor
to Reviewer, Committer, or Maintainer follows the Ascend community
[Role Definitions and Promotion Mechanism](https://gitcode.com/Ascend/community/blob/master/docs/role-definition-and-promotion-mechanism.md).

A regular PR follows this process:

1. The contributor submits code, tests, documentation, and a complete PR
   description.
2. Reviewers provide technical feedback, and every comment receives an explicit
   response.
3. Requested changes are implemented and sent for re-review. A declined
   suggestion includes a rationale acknowledged by the reviewer.
4. There are zero unresolved review conversations before merge, and a valid
   Committer approval remains after the final code update.
5. Every applicable required CI check passes on the latest commit or pre-merge
   result.
6. A Branch keeper or authorized maintainer merges the change into the protected
   branch.

For release and governance matters, Branch keepers coordinate branch strategy
and release preparation, while Committers confirm the code, documentation, and
release-note status. Releases run only on approved Ascend/NPU runners.

### Shared Ascend Community Guidelines

In addition to the MindIE SD repository rules above, contributors must follow
the shared Ascend community guidelines below. If requirements differ, follow
the stricter rule that is more specific to this repository.

- [Issue Submission Guide](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-guide.md)
- [Community Issue Workflow Guidelines](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-workflow-guidelines.md)
- [PR Submission Guide](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)
- [Ascend Community Developer Testing Contribution Guide](https://gitcode.com/Ascend/community/blob/master/docs/contributor/developer-testing-guide.md)
- [Ascend Open-Source and Third-Party Repository and Branch Naming Guide](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-repo-branch-guide.md)
- [Ascend Open-Source and Third-Party Software Management Guide](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-software-management-guide.md)
- [Community Security Design Guidelines](https://gitcode.com/Ascend/community/blob/master/docs/contributor/security-design-guideline.md)
