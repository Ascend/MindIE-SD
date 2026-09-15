---
name: performance-optimization
compatibility: 环境可用（`import mindiesd` 成功或目标框架可 serve）, 模型已跑通一轮, 有 baseline 数字, NPU 复验
description: >
  优化域入口（L2）：拿到**已明确的瓶颈点 / 瓶颈标签**后，按标签把任务**分发**到四个优化模块
  （DiT 计算 `dit-perf-opt` / DiT 通信 `dit-parallel-opt` / VAE `vae-opt` /
  host `host-opt`），并给出域内**最小前置集**与**域级验收口径**（性能入库口径引
  `perf-gate`、精度判据引 `accuracy-gate`）。
  **本技能不再承载选档与实施内容**（Step 2–4 闭环、特性档位选择、组合试验已全部归
  `dit-perf-opt`）。
  入口信号（任一即可触发）：① 用户或编排层已给出**瓶颈点/瓶颈标签**（"瓶颈已经明确，按这个点优化"）；
  ② **框架侧特性没落地**（该开的开关/特性没开，需要先判走哪个模块）。
  near-miss（看似相关但不属本技能）：
  - **瓶颈未明**（"这个模型怎么加速""怎么跑通""采个 profile"，还要先定位）→ 先走
    `model-auto-optimization`（唯一有分析权的一方），本入口**不接受未定位的任务**。
  - 问"要不要开量化、开哪一档""量化/稀疏/Cache 怎么选怎么开" → `dit-perf-opt`（选档与实施在模块层）。
  - 多卡并行形态 / 通信掩盖 / TP·offload → `dit-parallel-opt`；VAE·TAE 解码段 →
    `vae-opt`；交付搬运 / 装载预热等固定开销 → `host-opt`。
  - 需要改本仓代码（pattern / 算子 / 测试 / 文档）→ `dev-workflow`。
  由 model-auto-optimization 的阶段路由与用户直接声明瓶颈两条路径触发。
---

# 优化域入口（分发与前置）

## 定位

本技能是**优化域入口（L2）**：唯一职责是**把已明确的瓶颈点分发给正确的模块**，并把域内的
**前置集**与**验收口径**固定下来。层级依据：L3 模块不放"业务顺序 / 门禁节奏 / 走哪个模块的姿势
决策"，这些正归 L2 入口；"选哪个特性档"是能力选择，归 L3 模块。

**本技能不做的事**：不定位瓶颈（归 `model-auto-optimization`）、不做占比与门限分析（归
`model-auto-optimization` 的阶段账）、不选特性档位 / 不开特性 / 不做组合试验（全部归各模块）、
不定义验收标准（归两个验收标准技能）。

## 0. 入口信号

进入本入口（任一）：

| # | 信号 | 说明 |
|---|---|---|
| ① | **瓶颈点已明确** | 用户带一句实测锚点声明，或编排层已交付瓶颈标签 |
| ② | **框架侧特性没落地** | 该开/该验的特性开关未使能，需先判"走哪个模块把它落地" |

**不进入本入口**（先做别的）：

- **瓶颈未明** → `model-auto-optimization`（编排层是**唯一有分析权**的一方）
- **要改本仓代码** → `dev-workflow`
- **只要单算子实现级实测对比** → `benchmark-dev`

## 1. 输入：瓶颈标签（只消费，不定义）

唯一输入是**瓶颈标签**。标签枚举（`DiT-计算受限` / `DiT-通信受限` / `非DiT-解码段` /
`非DiT-host段` / `一致性不达标`）与门限口径（含 10% 门限的分母 / 测点 / **步数档**）**单一真源**为
`model-auto-optimization/references/bottleneck-labels.md`。

本技能**只引用不复制**：不在本文件另列标签表、不另写门限数字——两处定义必然漂移。
标签之外的细分类（MatMul / Attention / Norm / 生效判据…）由对应模块在域内自行映射。

## 2. 分发路由（入口核心动作）

按标签 → 模块 → 产物 → 验收判据查 `references/dispatch-table.md`。

| 瓶颈标签 | 分发目标 |
|---|---|
| `DiT-计算受限` | `dit-perf-opt`（需要新能力时再走 `pattern-dev` / `operator-dev`） |
| `DiT-通信受限` | `dit-parallel-opt` |
| `非DiT-解码段` | `vae-opt` |
| `非DiT-host段` | `host-opt` |
| `一致性不达标` | `accuracy-gate`（判据）→ 排障 |

分发后由**模块**负责实施与自证生效；本入口不重复模块内的步骤，只在模块回流"需要新能力 /
需要改框架 / 瓶颈判定有误"时改道或退回编排层。

**标签缺失或与实测不符**（模块复工发现真正瓶颈在别处）→ 退回 `model-auto-optimization`
重新定位，**不得**在本入口内改判标签。

## 3. 最小前置集

三项全满足才可开工，缺任一 → 退回 `model-auto-optimization` 走 S0：

1. **环境可用**：`import mindiesd` 成功，或目标框架可 serve；
2. **模型已跑通一轮**：端到端可产出结果（不是"能 import"就算）；
3. **有 baseline 数字**：同口径的端到端 / 阶段账基线（无基线则后续任何收益都不可归因）。

**不含** S0 的安装与权重准备（那归 `env-install`）；本入口不代做环境安装。

## 4. 锚点要求（防"优化错对象"）

用户直接声明瓶颈时**必须带一句实测锚点**——阶段账某一行、或某 kernel 的占比读数。
理由：声明与实测常不符（真实案例：用户认为"解码慢"，实测 DiT 占 89%）。

- **有锚点** → 按锚点映射标签（映射口径见 `bottleneck-labels.md`），进入 §2 分发；
- **无锚点** → **退回 `model-auto-optimization`** 做定位，**不由本入口自己猜**，也不凭"感觉慢"选档。

编排层交付的标签同样附锚点，便于复核与回溯。

## 5. 域级验收口径

分发出去的任务，回流时按本节口径收口（口径本身归各标准技能，本入口只固定"必须过哪几关"）：

- **性能**：库内数字一律按 `perf-gate` 的**同窗 A/B** 口径取；跨窗口绝对值不可比；
  **只有验收结果才能写入总览表**（报表契约见 `model-auto-optimization/references/report-contract.md`）。
- **噪声门限**：与基线差距 **< 3%** 视为噪声，不下结论（阈值在本技能**单点维护**，其它技能只引用）。
- **精度**：**未使用有损特性时一致性验收强制调用** `accuracy-gate`（三级：逐位 →
  跨配置数值门 + md5 → 质量门）；有损项另过质量门
  （`../accuracy-gate/references/quality-gate.md` + 仓库 `evals/`）。
- **一致性不达标**（标签 `一致性不达标`）→ 由 `accuracy-gate` 给判据，再转对应对象的
  `troubleshooting-{对象}.md` 排障流程。

### 停止条件（域级口径，单点维护于本技能）

满足任一条件即停止优化循环：

1. **目标达成**: MindIE-SD compiled 在目标硬件上已满足性能预期
2. **噪声范围**: 与 baselines 差距 < 3%，继续优化无统计意义
3. **外部瓶颈**: 根因在 CANN / TorchNPU / HCCL 而非 MindIE-SD 代码
4. **硬件瓶颈**: 已改善但受限于 NPU 物理显存 / 带宽上限

## 6. 独立触发时的交付物

用户直接给瓶颈点、不经编排层时，本入口产出**域级优化报告**：瓶颈锚点 + 分发结论 + 各模块产物
指针 + 验收结论。若要进 `model-auto-optimization` 的总览表，按编排层契约（报表结构 / 白名单 /
`[profile].domain` 记法）回填，收口方写清。

## Reference Files

- `references/dispatch-table.md` — 加载时机: 拿到瓶颈标签、决定分发目标与验收判据时
  （标签 → 模块 → 产物 → 验收口径；路由含域外标准技能）
- `../model-auto-optimization/references/bottleneck-labels.md`（跨技能，**单一真源**）— 加载时机:
  需要标签枚举原文或 10% 门限口径时（**只引用，不复制**）
- `../accuracy-gate/references/quality-gate.md`（跨技能，质量门本体归属精度验收标准）—
  加载时机: 有损项 / 闭环端到端质量判定时（定量 + 视觉伪影 + off-identity；判定标准与工具在
  仓库 `evals/`）
- `../perf-gate/SKILL.md`（跨技能）— 加载时机: 报数、入库与验收口径核对时
- `dit-perf-opt/SKILL.md`（域内模块，非本 skill 文件）— 加载时机: 确认"DiT 计算侧选档/组合试验/
  5 步闭环"的具体内容时（入口不再复述）

## 维护与更新

- 触发：优化模块增减 / 改名、瓶颈标签枚举或门限口径调整、验收标准接线变化时更新本 skill 与
  `references/dispatch-table.md`；改名时本入口的模块名清单与 `dispatch-table.md` 必须同步。
- 校验：新增标签须同时出现在 `bottleneck-labels.md`、本入口分发表与 `dispatch-table.md`，否则视为孤儿标签。
- 与 `dit-perf-opt` 的 description 必须互斥：入口只讲**分发/前置/锚点/验收**，选档与实施描述只在模块侧出现。
