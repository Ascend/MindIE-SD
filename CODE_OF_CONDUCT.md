# MindIE SD 社区行为准则与贡献规则 / Code of Conduct and Contribution Rules

[简体中文](#简体中文) | [English](#english)

## 简体中文

### 我们的承诺

为建设开放、友好、尊重的社区，MindIE SD 的贡献者和维护者承诺：无论年龄、
体型、残障、族群、经验、教育程度、社会经济状况、国籍、外貌、种族、宗教、
性别认同或性取向，所有人均可在不受骚扰的环境中参与项目和社区活动。

### 期望行为

- 使用友好、包容的语言。
- 尊重不同观点和经历。
- 以专业、建设性的方式提出和接受反馈。
- 优先考虑项目和社区的整体利益。
- 对其他贡献者和用户保持同理心。

### 不可接受的行为

- 使用带有性暗示的语言或图像，或作出不受欢迎的性关注。
- 恶意挑衅、侮辱、贬损，以及人身或政治攻击。
- 公开或私下骚扰他人。
- 未经明确许可发布他人的私人信息。
- 其他可被合理认定为不适合专业环境的行为。

### 项目贡献与合入规则

以下规则适用于外部贡献者和项目成员：

1. 外部贡献必须基于最新 `upstream/dev` 创建独立主题分支，并将 PR 的目标
   分支设置为 `dev`。`master` 仅接收经维护者授权的发布、紧急修复或
   `dev` 到 `master` 的受控合入。
2. 一个 PR 只处理一个主题，有效变更量原则上不得超过 1000 行。无法合理拆分
   的变更必须关联已批准的 RFC，并获得 maintainer 对例外范围的确认。
3. 所有评审意见必须形成闭环：已修改的意见完成复审，不采纳的意见说明理由并
   获 reviewer 确认；合入前未解决会话数必须为 0，最终代码更新后仍须保有
   有效的 Approver 批准。
4. 所有适用于该 PR 的必需 CI 必须在最新提交或预合并结果上通过。失败、取消、
   应运行但未运行的检查均不得视为满足合入条件。
5. 提交者应理解并能够解释全部变更，提供必要测试证据，并同步更新用户可见的
   中英文文档。

操作步骤、有效变更量口径和本地验证命令见[贡献指南](contributing.md)。

### 维护者责任与执行

维护者负责说明并执行社区行为和贡献标准。对于违反本准则的评论、提交、代码、
Wiki 编辑、Issue、PR 或其他贡献，维护者可以移除、编辑、拒绝或要求修改。

维护者可以对威胁、冒犯、骚扰或持续违反项目规则的参与者采取警告、临时限制或
永久禁止参与等措施。执行决定应保持公正，并尽可能记录在适当的公开治理渠道；
涉及隐私或安全的问题除外。

### 适用范围

本准则适用于项目空间，也适用于个人公开代表本项目或社区的场景，包括仓库、
Issue、PR、评审、文档、会议和社区沟通渠道。

### 举报与报告

如遇到或发现不可接受的行为，请通过项目安全与治理渠道向 Ascend 社区维护者
报告：

- [安全与升级处理指南](https://gitcode.com/Ascend/community/blob/master/docs/security.md)
- [Ascend 社区贡献指南](https://gitcode.com/Ascend/community/tree/master/docs/contributor)
- [MindIE SD 安全政策](SECURITY.md)

报告将尽快得到审查，并以公平、保密和符合安全要求的方式处理。

## English

### Our Commitment

To foster an open, welcoming, and respectful community, contributors and
maintainers of MindIE SD pledge to provide a harassment-free experience for
everyone, regardless of age, body size, disability, ethnicity, experience,
education, socioeconomic status, nationality, personal appearance, race,
religion, gender identity, or sexual orientation.

### Expected Behavior

- Use welcoming and inclusive language.
- Respect differing viewpoints and experiences.
- Give and accept feedback professionally and constructively.
- Prioritize the interests of the project and community.
- Show empathy toward other contributors and users.

### Unacceptable Behavior

- Sexualized language or imagery, or unwelcome sexual attention.
- Trolling, insulting or derogatory comments, and personal or political attacks.
- Public or private harassment.
- Publishing another person's private information without explicit permission.
- Other conduct that could reasonably be considered inappropriate in a
  professional setting.

### Project Contribution and Merge Rules

The following rules apply to external contributors and project members:

1. External contributions must use a dedicated topic branch created from the
   latest `upstream/dev`, and the PR must target `dev`. `master` only accepts
   maintainer-authorized releases, emergency fixes, or controlled merges from
   `dev` to `master`.
2. Each PR must address one topic and, as a rule, must not exceed 1,000 effective
   changed lines. A change that cannot reasonably be split must link an approved
   RFC and receive maintainer confirmation of the exception scope.
3. Every review comment must be closed out: implemented changes must be
   re-reviewed, and declined suggestions must include a rationale acknowledged
   by the reviewer. There must be zero unresolved conversations before merge,
   with a valid Approver approval after the final code update.
4. Every required CI check applicable to the PR must pass on the latest commit
   or pre-merge result. Failed, cancelled, or expected-but-not-run checks do not
   satisfy the merge requirements.
5. Submitters must understand and be able to explain every change, provide the
   necessary test evidence, and update user-visible documentation in both
   Chinese and English.

See the [Contributing Guide](contributing.md) for operational steps, the
effective-change-size definition, and local validation commands.

### Maintainer Responsibilities and Enforcement

Maintainers are responsible for clarifying and enforcing the community behavior
and contribution standards. They may remove, edit, reject, or request changes
to comments, commits, code, wiki edits, Issues, PRs, or other contributions that
violate this Code of Conduct.

Maintainers may warn, temporarily restrict, or permanently ban participants who
engage in threatening, offensive, harassing, or persistently non-compliant
behavior. Enforcement decisions should be fair and, where appropriate, recorded
in public governance channels, except for privacy- or security-sensitive cases.

### Scope

This Code of Conduct applies within project spaces and when an individual is
publicly representing the project or community, including repositories, Issues,
PRs, reviews, documentation, meetings, and community communication channels.

### Reporting

If you experience or witness unacceptable behavior, report it to the Ascend
community maintainers through the project security and governance channels:

- [Security and Escalation Guidance](https://gitcode.com/Ascend/community/blob/master/docs/security.md)
- [Ascend Community Contribution Guidance](https://gitcode.com/Ascend/community/tree/master/docs/contributor)
- [MindIE SD Security Policy](SECURITY.md)

Reports will be reviewed promptly and handled fairly, confidentially, and in
accordance with applicable security requirements.
