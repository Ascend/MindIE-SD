# 派发模板与自验证回执（编排执行模式）

> 与 `workflows/optimization-flow.md` 配套。两种执行模式：自执行（默认，编排者按模板自检）与
> subagent 模式（运行环境支持 subagent 时，如 DSH / Claude Code；编排者按模板派发角色 subagent）。

## 通用派发规范

> 交付件（派发单 / 回执 / 复核单 / 资源租约 / 交接单）的路径、最小字段与指针链单点见
> `../../references/agent-roles-and-handoff.md` §3；派发前先过**扇出判据**（同文件 §1）。本节只给字段写法。

- 派发 prompt 只填模板字段与占位符，不转述上下文：执行者进场先读
  `{工作目录}/agentic/run-state.md` 与指定的支撑技能，共享上下文以文件为单一来源。
- 每条任务必须绑定支撑技能（「必须使用 skill / 脚本: X」），不允许执行者自由选路。
- 派发单须写明**工作类型**（纯算子开发 / 框架接入 / 特性使能 / 优化验证）——**异类工作拆不同子 agent**
  （技能面与验证口径不同，混在一起会串用判据），**同类工作默认共用一个 agent**（判据见
  `../../references/agent-roles-and-handoff.md` §1.1）。
- 回执只回：结论摘要 + 证据文件路径 + 需编排者决策的问题；长日志 / 原始数据留 evidence/，
  不回传主上下文。
- 任何涉及代码改动的执行，先确认改动归属（mindiesd 仓内 → dev-workflow 子任务；三方框架 →
  framework-integration），不静默越界。

## 角色（四角色 · 单点在上层）

> **角色定义、写权限、交付件与指针链的单点 = `../../references/agent-roles-and-handoff.md` §2–§3**
> （**主控 orchestrator / 代码开发 developer / 结果分析 analyst / 部署与资源分配 deployer**）。
> 本文件只给**派发模板**；模板 ↔ 角色对应关系：

| 本文件模板 | 对应角色 | 说明 |
|------------|----------|------|
| `collector` / `analyzer` | 结果分析 analyst | 只读数据角色：采集与分析 |
| `implementer` | 代码开发 developer | 唯一可改代码方；不自改已确认方案 |
| `deployer` | 部署与资源分配 deployer | 环境就绪 + 卡组/窗口租约签发 |
| `reviewer` | 结果分析 analyst（**未参与实施的实例**） | 只读复核；禁改代码、不做自行修复 |

## 派发模板

### collector / analyzer（只读数据角色）

```text
工作目录: {work_dir}
角色: {collector | analyzer}
必须使用 skill: {profiling-collect | profiling-analyze}
任务: {采集 baseline/重采 profiling | 分析 {dir} 产出报告与候选清单}
口径: {从 run-state「任务与口径」读取，不另起口径}
产物: 写入 evidence/{task_id}/{stage}/，回执只回摘要 + 产物路径
```

### implementer（实施角色）

```text
工作目录: {work_dir}
角色: implementer
必须使用 skill: {framework-integration | performance-optimization | …（按 run-state 决策）}
任务: {阶段 {Sn} 实施：…}
方案要点: 见 run-state「决策与轮次」{记录 id}，不自改方案；发现问题停止并报告
自验证: 按下方「自验证回执格式」写入 evidence/{task_id}/{stage}/selfcheck.md 与 run-state 工作区
```

### reviewer（复核角色，只读）

```text
工作目录: {work_dir}
角色: reviewer（只读：禁改模型代码/配置，不做自行修复）
任务: 复核 {阶段 {Sn}} 实施结果
检查项: {验收 gate 清单（见 optimization-flow.md 对应阶段）}
产物: 结论（通过/FAIL + 证据与诊断）写入 evidence/{task_id}/{stage}/review.md
```

### deployer（部署与资源分配角色）

```text
工作目录: {work_dir}
角色: deployer
必须使用 skill: {env-install | remote-access}
任务: {环境/权重/容器就绪 | 为 {特性} 单元分配卡组与实验窗口}
资源: 卡组 id {…}（互斥，禁与他人对照臂同卡）；容器/远端 {…}；时间窗 {起-止}
产物: evidence/{task_id}/{stage}/lease-{group}.md（占用者 / 起止 / 释放确认）+ 环境指纹；
      回执只回摘要 + 租约路径 + 需主控决策的问题
禁区: 不裁决收益、不选档、不改模型代码
```

## 单点特性独立子 agent 与并行执行（无损 ∥ 有损）

- **一特性一 agent**：每个单点特性（`kernel融合` 的融合内容 / `并行` / `Cache` / `量化` /
  `稀疏` / `时间步优化`）可派发独立 implementer 子 agent 承担，各自：专属
  `evidence/{task_id}/{stage}/{feature}/` 目录 + run-state 迭代表一行（特性/实现 id 列）+ 自验证回执。
- **可并行关系**：在共享基线/口径/run-state 前提下，**无损组（S1 融合/S3 并行）与有损组（S4 各特性）
  本身即可并行**；组内不同有损特性（Cache×量化×稀疏×时间步）亦可在 seam 裁定后并行试验。
- **禁止并行**：同 seam 互斥组合（seam_check 判定，如 cache_dit×cache_attention、同
  attention_backend 双 writer）不得双开，裁定取最强档后顺序化；共享同一代码/环境的改动串行化。
- **并行护栏**：**单点在 `../../references/agent-roles-and-handoff.md` §5**（并行单元 = 特性 × 阶段 ×
  卡组、资源租约、并发上限、收口指针链）。派发侧只需记住两条：① 一单元一目录一实例，回执路径与
  迭代表一致；② 结论进报表前必须由主控按迭代表裁决（retain/reject + 签名），
  **禁止并行中互相引用未裁决结果**。

## 自验证回执格式（机械可查，实施者必填）

```text
### 自验证结果（阶段 {Sn}）
- 支撑技能/脚本: {skill 名}
- 运行证据: {命令 + 退出码 + 日志路径}
- 验收证据: {与 run-state 推进表一致的 evidence 路径}
- 异常记录: 无 / {描述}
```

编排者核验要点：四行齐全、验收证据路径与 run-state 推进表该阶段行一致、日志/产物真实存在。

### 脚本交付补充（交付 **gate / 校验器 / 对照脚本** 时必填）

**规则来源**：`../../../perf-gate/references/measurement-discipline.md` §11.1 —— 交付的判据脚本必须
自证"平台归属 + 能在目标平台跑通一次"，否则**不是证据**（典型失效形态：gate 脚本把仓库路径写死为
盘符绝对路径、其平台语义在目标平台根本不成立，且交付时没有任何一次运行记录）。

```text
### 脚本交付自证（回执必带；只交付脚本时这一节代替「自验证结果」）
- 脚本: {相对路径}  md5={…}
- 平台归属: {POSIX/Linux 容器/Windows；是否需要 /dev/shm、NPU、特定框架}
- 外部依赖: {目标主机可达/某目录已暂存/某包已安装}；依赖不可达时 → 判「无法判定」（不得降级为通过）
- 最小用例运行证据: {命令 + 退出码 + 日志路径 + 环境指纹（sys.platform / python 版本 / 关键目录是否存在）}
- 若本环境跑不了: {平台不可达的证据（端点超时/无 /dev/shm/无容器）} ⇒ 结论只能写「未证明」
- 机检: `python check_gate_script.py {脚本}` → error=N warn=N + 无法机检项逐条
  （脚本位置 `perf-gate/scripts/`，判据见 `../../../perf-gate/references/measurement-discipline.md` §11.1）
```

**拒收判据**：缺"平台归属"或"最小用例运行证据" ⇒ 拒收；把"跑不通"写成"通过"或"未通过" ⇒ 拒收
（正确写法是"未证明 + 不可达证据"）。

## 拒收语义

- **单点在 `../../references/agent-roles-and-handoff.md` §6**（拒收条件 + 六类多 agent 失败模式 +
  独立复核要求）：回执缺自验证节、或记录与 evidence 矛盾 → 拒收并列出缺失项要求补齐后重交；
  主控不自行审查代码替代验证。
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
- evidence/{task_id}/{stage}/ 证据 + run-state 推进表/迭代表更新 + （闭环）overview/detail 报表节
```
