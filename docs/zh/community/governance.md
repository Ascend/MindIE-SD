# 社区治理

MindIE SD 采用 maintainer-led 治理模型，面向代码评审、受保护分支合入和版本发布进行统一管理。

## 角色定义

### Branch keepers

Branch keepers 负责：

- 版本发布就绪检查
- 受保护分支的合入策略
- 受保护分支上的最终合入授权

### Approvers

Approvers 负责：

- 审核已具备合入条件的变更
- 校验技术质量、测试证据和文档影响
- 确认用户可见变更已同步到文档与发布记录

### Reviewers

Reviewers 负责：

- 提供技术评审意见
- 协助问题复现
- 审核代码、测试与文档

## 决策流程

### 日常 Pull Request

普通 PR 按以下流程处理：

1. 先有清晰的问题描述、Issue 或已批准的 RFC。
2. 贡献者提交代码、测试和文档更新。
3. Reviewer 给出技术评审意见。
4. Approver 确认变更已具备合入条件。
5. Branch keeper 或具备授权的 maintainer 将变更合入受保护分支。

### 发布与治理决策

发布与治理类事项按以下流程处理：

1. Branch keepers 协调版本发布准备和分支策略。
2. Approvers 确认代码、文档和变更记录状态。
3. 发布仅在批准的 Ascend/NPU runner 上执行。

## 信息来源

角色名单与仓库工作流使用的角色来源以 [`OWNERS`](../../../OWNERS) 为准。

## 社区规范

- 遵守 [`CODE_OF_CONDUCT.md`](../../../CODE_OF_CONDUCT.md)。
- 用户可见变更必须同步更新测试、文档和变更记录。
- 重大接口变更、行为变更和发布策略变更应先提交 RFC。
