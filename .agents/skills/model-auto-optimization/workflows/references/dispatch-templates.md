# 派发模板与自验证回执（编排执行模式）

> 与 `workflows/optimization-flow.md` 配套。两种执行模式：自执行（默认，编排者按模板自检）与
> subagent 模式（运行环境支持 subagent 时，如 DSH / Claude Code；编排者按模板派发角色 subagent）。

## 通用派发规范

- 派发 prompt 只填模板字段与占位符，不转述上下文：执行者进场先读
  `{工作目录}/agentic/run-state.md` 与指定的支撑技能，共享上下文以文件为单一来源。
- 每条任务必须绑定支撑技能（「必须使用 skill / 脚本: X」），不允许执行者自由选路。
- 回执只回：结论摘要 + 证据文件路径 + 需编排者决策的问题；长日志 / 原始数据留 evidence/，
  不回传主上下文。
- 任何涉及代码改动的执行，先确认改动归属（mindiesd 仓内 → dev-workflow 子任务；三方框架 →
  framework-feature-enablement / framework-extension-dev），不静默越界。

## 角色与写权限

| 角色 | 职责 | 写权限 | 边界 |
|------|------|--------|------|
| 采集者 collector | profiling 采集（profiling-collect） | evidence/{stage}/ | 不改模型代码/配置 |
| 分析者 analyzer | profiling 分析 / 候选清单（profiling-analyze） | evidence/{stage}/ | 不改模型代码/配置 |
| 实施者 implementer | 按已确认方案实施（framework-feature-enablement 等） | evidence/ + 工作区；代码改动按归属子任务 | 唯一可改代码方；不自改已确认方案 |
| 复核者 reviewer | 验收复核（只读） | evidence/{stage}/review.md | 禁改模型代码/配置，不做自行修复 |

## 派发模板

### collector / analyzer（只读数据角色）

```text
工作目录: {work_dir}
角色: {collector | analyzer}
必须使用 skill: {profiling-collect | profiling-analyze}
任务: {采集 baseline/重采 profiling | 分析 {dir} 产出报告与候选清单}
口径: {从 run-state「任务与口径」读取，不另起口径}
产物: 写入 evidence/{stage}/，回执只回摘要 + 产物路径
```

### implementer（实施角色）

```text
工作目录: {work_dir}
角色: implementer
必须使用 skill: {framework-feature-enablement | performance-optimization | …（按 run-state 决策）}
任务: {阶段 {Sn} 实施：…}
方案要点: 见 run-state「决策与轮次」{记录 id}，不自改方案；发现问题停止并报告
自验证: 按下方「自验证回执格式」写入 evidence/{stage}/selfcheck.md 与 run-state 工作区
```

### reviewer（复核角色，只读）

```text
工作目录: {work_dir}
角色: reviewer（只读：禁改模型代码/配置，不做自行修复）
任务: 复核 {阶段 {Sn}} 实施结果
检查项: {验收 gate 清单（见 optimization-flow.md 对应阶段）}
产物: 结论（通过/FAIL + 证据与诊断）写入 evidence/{stage}/review.md
```

## 单点特性独立子 agent 与并行执行（无损 ∥ 有损）

- **一特性一 agent**：每个单点特性（`kernel融合` 的融合内容 / `并行` / `Cache` / `量化` /
  `稀疏` / `时间步优化`）可派发独立 implementer 子 agent 承担，各自：专属
  `evidence/{stage}/{feature}/` 目录 + run-state 迭代表一行（特性/实现 id 列）+ 自验证回执。
- **可并行关系**：在共享基线/口径/run-state 前提下，**无损组（S1 融合/S3 并行）与有损组（S4 各特性）
  本身即可并行**；组内不同有损特性（Cache×量化×稀疏×时间步）亦可在 seam 裁定后并行试验。
- **禁止并行**：同 seam 互斥组合（seam_check 判定，如 cache_dit×cache_attention、同
  attention_backend 双 writer）不得双开，裁定取最强档后顺序化；共享同一代码/环境的改动串行化。
- **并行护栏**：
  1. 推进表/迭代表仍由**编排者单写**；子 agent 只写自己的 evidence 与工作区，先读后追加；
  2. evidence 按 `{stage}/{feature}/` 隔离，回执路径与迭代表一致；
  3. 共享文件与卡组资源互斥调度（同卡并行验证需隔离，参照 dev-workflow §3 并行策略）；
  4. 每 agent 独立自验证 + 拒收语义；编排者按迭代表 retain/reject+签名裁决后，才允许结论
     进入报表；
  5. 收益一律在**同基线**上合并进报表（overview/detail），禁止并行中互相引用未裁决结果。

## 自验证回执格式（机械可查，实施者必填）

```text
### 自验证结果（阶段 {Sn}）
- 支撑技能/脚本: {skill 名}
- 运行证据: {命令 + 退出码 + 日志路径}
- 验收证据: {与 run-state 推进表一致的 evidence 路径}
- 异常记录: 无 / {描述}
```

编排者核验要点：四行齐全、验收证据路径与 run-state 推进表该阶段行一致、日志/产物真实存在。

## 拒收语义

- 回执缺自验证节、或记录与 evidence 矛盾 → 拒收并列出缺失项，要求补齐后重交；编排者不自行
  审查代码替代验证。
- 复核者报告 FAIL → 按 optimization-flow.md「阶段推进规则」进入修复轮（实施/复核合计上限
  5 轮，超限回退并报告阻塞点）。

## GOAL 骨架（复杂探索型任务 / subagent 可选增强）

借鉴 bounded search 的 goal 契约；用于 S3 并行选型、S4 组合搜索等探索型派发，把"做什么/
不许做什么/怎么算数/交什么"一次性钉死，随 prompt 下传或在 run-state「任务与口径」登记：

```text
## 任务边界（scope 固定项）
- 硬件/拓扑预算: {卡数与节点、同拓扑约束}
- 不可改动项: {steps/分辨率/seed/质量相关 dtype/调度器…；改动 = lossy = 越界}
## 指标口径（metric 定义）
- 计时边界: {起止 phase + 排除项 model_load/compile_prime/warmup}；rank0/p50 口径
- 正确性/质量口径: {无损=输出一致；有损=同 seed 冻结 baseline 帧对照，阈值引用 evals/profiles}
## 参照与前置（golden 锚）
- 先建/复用 baseline 产物（iter0）作为对照锚；未跑通 baseline 不进入探索
## 循环与预算（有界）
- 每轮一条假说入 run-state 迭代表（假说先行）→ 一次一候选 → gate → retain/reject+签名；
  max_rounds={N}，预算到点带 frontier 收尾，未尝试候选留清单
## guardrails
- OFF/默认路径保持可用（off-identity 可验）；单候选失败不结束，换假说继续；
  结构性缺口补齐先经用户确认（§0），不静默改三方框架
## deliverable spec（必交文件）
- evidence/{stage}/ 证据 + run-state 推进表/迭代表更新 + （闭环）overview/detail 报表节
```
