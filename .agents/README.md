# MindIE-SD 多模态解决方案 Skills

面向 MindIE-SD 多模态扩散模型（Wan2.2 / FLUX / Qwen-Image / MiniMax-H3）在昇腾 NPU 上的
**模型自动优化**技能集合。

组织方式（三层架构：编排 L1 · workflow L2 · 能力 L3，2026-09 重构）：

```text
L1 编排入口（2）：model-auto-optimization / dev-workflow——任务分流与路由、特性验证顺序（依赖派生）、
                交付件契约与编排机制所有权（run-state / stage_gate / 迭代表 / 覆盖清单 / 双报表）
L2 业务 workflow（按业务线）：特定业务线下的阶段化执行与单任务深挖——把 L3 能力组合成该业务的
                流程与"姿势/优化选择"（含单特性策略与组合回退裁决）；厚度因业务线而异（见「架构速览」）
L3 能力（16）：可复用的单一能力/知识/工具（含能力自身工作流纪律），可被任意 L2 编排或独立直达
预留槽位（7）：流程自动化所需但经验尚空的位置（§4），复盘回填
```

**入口互斥**：对一个模型/三方框架做「接入/加速/量化稀疏缓存/并行/性能确认」→
`model-auto-optimization`（L1 入口）；修改 MindIE-SD 代码（pattern/算子/测试/文档）→
`dev-workflow`（L1 入口）。单点问题（"帮我采个 profile"、"写个 pattern"）由描述最贴切的能力技能（L3）直接命中。

## 架构速览（三层：编排 L1 · workflow L2 · 能力 L3）

### 三层职责与判据

| 层 | 职责 | 放什么 | 不放什么 |
|----|------|--------|----------|
| **L1 编排入口** | 任务分流与路由、**特性验证顺序**（按依赖派生：无损先于有损、kernel 序列变化需重验）、**交付件契约**（双报表/§C/§E/必测组合覆盖/覆盖清单无未裁决/声明纪律）、编排机制所有权、**总览收口** | 入口 SKILL.md（<500 行）+ 机制资产（run-state/stage_gate/迭代表/证据行/报表规范，供所有 L2 复用） | 业务阶段细节、单特性策略、领域知识 |
| **L2 业务 workflow** | 特定业务线下的**阶段化执行** + **单任务/单特性深挖到最佳策略**（候选空间→扫描→组合试叠→回退→收口采纳档）；承载业务姿势/优化选择 | `workflows/*.md`（挂所属入口下）+ `workflows/references/`（dispatch/GOAL）；运行态：迭代表/覆盖清单执行 | 跨业务通用机制（上浮 L1）；领域事实/工具（下沉 L3） |
| **L3 能力** | 可复用单一能力：选项/接口/判据/工具；含**能力自身工作流纪律**（如 compilation-dev Phase、governance 固定步骤，显式区别于业务 L2） | 能力 SKILL + references/scripts/evals | 业务顺序、业务门禁节奏、本业务姿势决策 |

**内容归属三问判据**：① 换一条业务 workflow 仍遵守 → L1（机制/契约）；② 仅本业务线有效 → L2（阶段参数/姿势）；③ 领域事实/工具/判据 → L3。

### 横向顺序与契约（L1） vs 纵向单特性策略（L2）

- **L1 管横向**：特性验证顺序（依赖派生）、阶段门禁节奏、交付件契约（双报表、§C 候选治理、§E 质量证据、S4-2 必测组合 4 组覆盖、覆盖清单无未裁决）；运行态机制所有权——L2 **引用不重复**（单一真源）。
- **L2 管纵向**：对每个判定「做」的特性/任务，深挖其最佳策略（候选空间→经验档起扫→单变量扫描→替代算子/载体回退→与其它特性 seam 组合试叠→迭代表 retain/签名→收口采纳档进报表）；业务姿势链（稀疏候选选择链、融合开发链等）在此。
- **报表分工**：总览 `overview_report.md` 由 **L1 维护 = 覆盖收口表**（每特性 + 必测组合均有行与数据，数据由 L2 特性/组合反馈录入，L1 不自测）；**特性自身的报表**（detail 分节 + 迭代表过程 + §E 质量证据）= 该特性内部的方案选择与过程记录；两者不同、缺一不可。
- **执行波次**：单特性（判「做」者）可**多 agent 并行**（资源允许、同卡组互斥、迭代表单写）→ 组合（含 [MUST] 必测两两+三元）在**单特性收敛后启动**（防路径爆炸），多组合可并发、同 seam 不并行。
- **跨特性组合归属**：必测组合覆盖 = L1 契约；seam/组合协议 = L3（combination-search.md / seam_check 声明）；组合"试叠与组合回退裁决"执行点 = 各特性 L2 策略 + 迭代表（一次一候选、带签名）。
- **质量体系**：判定依据（profiles `[domain]/[decisions]`、rubric）= L3（产品 evals）；数值现算不入库、随报表 §E 逐行登记 = L1 契约；登记执行 = L2/闭环。

### 物理落点与加载链路

```text
L1 入口技能（model-auto-optimization / dev-workflow）
├── SKILL.md                 # L1：分流/路由/总纲/协议所有权
├── workflows/               # L2（挂所属入口下）
│   ├── optimization-flow.md           # 模型优化业务线 L2：S0/S1/S3/S4/S5(预留)+闭环 阶段骨架 + 特性策略
│   └── references/dispatch-templates.md  # L2 派发/GOAL
└── references/ + scripts/   # L1 机制资产（run-state / stage_gate / 报表与纪律；跨业务复用）
L3 能力技能（每能力 SKILL + references/scripts/evals）
```

加载链路：触发 → L1 SKILL 分流 → 按业务线 **Read 对应 L2 workflows/*.md**（不得绕过）→ 建
run-state（推进表/迭代表/覆盖清单）→ 逐阶段回写并跑 `stage_gate.py --stage {Sn}` → 具体执行
路由 L3 → 闭环按 L1 交付件契约输出 overview/detail 双报表（`--stage close` 校验）。

### 业务线 L2 厚度差异（关键：不是所有业务线一样厚）

- **模型自动优化**：L2 厚——单特性策略深挖为主（稀疏/量化/cache 候选选择链、组合回退、必测
  组合），L1 定顺序（S0/S1/S3/S4/S5 依赖序）与交付件契约。
- **仓库开发（dev-workflow）**：L2 薄——Test-First 闭环骨架内联（轻量 L2），单任务最佳路径
  （pattern 生命周期、算子 DSL 选择、mismatch 回退）由 **L3 能力自身工作流纪律**承载；为 dev
  建独立 L2 文件的信号 = 出现**跨多能力、多阶段、有顺序门禁与逐阶段确认点**的开发线（如端到端
  融合算子合入链），届时按 L1 机制复用（迭代表/证据行/门禁）。
- **纯 L3 域**（部署/环境、基准、治理单发等）：无 L1/L2，能力直达即可。

## 技能关系（可视化总览）

```text
                      编排层
   model-auto-optimization（模型优化流程）        dev-workflow（仓库开发流程）
      │ 路由 S0→S1→S3→S4→S5(可选)→闭环                          │ 路由/承接开发子任务
      ▼                                            ▼
 S0 准备         env-install（安装/权重确认/下载） ── 工具 ──> remote-access
                 dummy-run（简化基线/快验）
 S1 kernel融合优化 framework-feature-enablement(接入/compile) ◄──快验── dummy-run
                 profiling-collect ──> profiling-analyze（算子序→机会点→收益→开发/compile）
                 framework-feature-enablement（使能决策/异常回修）
                    └──需要新增 pattern/算子──> compilation-dev / operator-dev
 S3 并行通信     parallelism-strategy ◄──掩盖收益判定── profiling-analyze
 S4 有损优化     performance-optimization（features.md 选档）
                 framework-feature-enablement（框架侧开关）＋ benchmark-dev（稀疏度/量化档选型证据）
                 benchmark-dev ◄──实现级实测 vs 特性级选档──> performance-optimization
 闭环复验         标准回路：profiling-collect → profiling-analyze → performance-optimization
                 （产物规范 model-auto-optimization/references/artifact-layout.md）

 开发与规范支撑（被 dev-workflow 与上述场景调用）
   compilation-dev / operator-dev / aclgraph-dev / benchmark-dev（开发态）
   framework-extension-dev（三方框架侧特性补齐开发，被 §0 缺口路由调用）
   code-standards / markdown-lint / mindie-sd-community-governance
```

图例：`──路由/调用──>` 编排层→能力层调度；`──数据/产物──>` 上游产出喂下游；`◄──> / ──>` 边界互指（description 的 near-miss 句，如 collect↔analyze、benchmark-dev↔performance-optimization、parallelism-strategy↔performance-optimization、framework-feature-enablement↔env-install）；虚线场景按需加载（实现层/工具），非默认常载。各技能详情见 §1/§2，仓库级路由规则见 AGENTS.md §5。

> 编排层两技能内部结构：入口 SKILL.md（判定/路由）→ 执行定义（`model-auto-optimization` 已拆
> `workflows/optimization-flow.md` 阶段文件；`dev-workflow` 为轻量内联）→ 状态/门禁
> （run-state 推进表 + 候选迭代表 + 特性覆盖清单 + `scripts/stage_gate.py`）——详见「架构速览」。

---

## 1. 编排层（流程）

| 技能 | 一句话界面 | 触发 |
|------|-----------|------|
| **[model-auto-optimization](skills/model-auto-optimization/SKILL.md)** | 模型自动优化流程：S0 环境准备 → S1 kernel 融合优化（原 S1/S2 合并）→ S3 并行通信 → S4 有损优化 → S5 训练感知（预留）→ 闭环复验（内含 采集→分析→复验 标准回路；**闭环必列输出「优化总览报表 + 优化细分报表」**——总览基线=TP 多卡未优化，细分=融合算子/并行/稀疏/量化/cache 与步数/组合 构成口径，见 `references/overview-report.md` 与 `references/detail-report.md`；框架未提供的特性须明确标注「未提供」）。流程执行按 `workflows/optimization-flow.md` 阶段模板，阶段验收以 run-state 推进表 + `scripts/stage_gate.py` 门禁为准（见 `references/run-state.md`） | 模型名 + 优化/加速/跑通/采profile 等流程型目标 |
| **[dev-workflow](skills/dev-workflow/SKILL.md)** | 仓库开发流程：Test-First → 编码 → 部署 → pytest → 复盘（轻量内联，暂不拆 workflow 文件；流程门禁见其 §0.2） | MindIE-SD 代码改动 |

## 2. 能力层（16）

能力技能按域标签组织（一个技能可多标签），每条是干净的单职责接口。

| 能力 | 一句话界面 | 域标签 |
|------|-----------|--------|
| **[env-install](skills/env-install/SKILL.md)** | 环境安装与准备：mindiesd + 三方框架全栈安装、权重确认/下载（部署时先与用户确认是否已存在） | 安装准备 |
| **[remote-access](skills/remote-access/SKILL.md)** | 远程昇腾访问工具：SSH 连接复用 / 容器执行 / 空闲卡选择 / 传输 | 工具 |
| **[framework-feature-enablement](skills/framework-feature-enablement/SKILL.md)** | 三方框架语境下特性使能与验证：mindiesd 能力适配接入（runtime/compile）与框架特性开启、使能异常定位回修 | 使能验证 |
| **[framework-extension-dev](skills/framework-extension-dev/SKILL.md)** | 三方框架自身结构性缺口补齐开发（comm-stream / 缓存 / 稀疏量化消费者）：框架差异 + 注入点 + 合入姿势（平台注册优先，fork/monkey 备选） | 开发 |
| **[dummy-run](skills/dummy-run/SKILL.md)** | 简化模型代码快速验证载体（架构 / 算子先接入 / 融合快验），无需真实权重 | 验证 |
| **[profiling-collect](skills/profiling-collect/SKILL.md)** | 统一采集：mindiesd 自家脚本 + 三方框架补丁两入口，产出统一 ASCEND_PROFILER_OUTPUT | 采集 |
| **[profiling-analyze](skills/profiling-analyze/SKILL.md)** | 统一分析：5 层管道 + kernel diff，输出瓶颈与融合机会候选 | 分析 |
| **[performance-optimization](skills/performance-optimization/SKILL.md)** | 特性库与方案选择：mindiesd-features.md（唯一真相源）选量化/稀疏/缓存等方案并实施复验 | 特性 |
| **[parallelism-strategy](skills/parallelism-strategy/SKILL.md)** | 并行策略选型：USP→CP + few-step 多 rank 验证 | 并行 |
| **[benchmark-dev](skills/benchmark-dev/SKILL.md)** | 算子选型判断工具：单算子对比（选型证据）+ 算子接入测试（供 operator-dev） | 基准 |
| **[compilation-dev](skills/compilation-dev/SKILL.md)** | Pattern matcher / Inductor 后端开发（S1 融合需新增 pattern 时被指向）；Phase 1–7 出口证据行验收 + 候选判定 gate（验收与推进纪律） | 开发 |
| **[operator-dev](skills/operator-dev/SKILL.md)** | 算子开发（路由外部 cannbot-skills） | 开发 |
| **[aclgraph-dev](skills/aclgraph-dev/SKILL.md)** | NPU 图批量下发开发 | 开发 |
| **[code-standards](skills/code-standards/SKILL.md)** | Python 格式与 lint | 规范 |
| **[markdown-lint](skills/markdown-lint/SKILL.md)** | Markdown 格式检查 | 规范 |
| **[mindie-sd-community-governance](skills/mindie-sd-community-governance/SKILL.md)** | 文档/治理/提交/PR/版本规范 | 规范 |

## 3. 本仓技能变更对照（迁移记录）

| 旧 | 新 | 说明 |
|----|----|------|
| auto-optimization | 融入 model-auto-optimization | 采集→分析→复验回路并入入口（标准回路章节） |
| third-party-framework-profiling | 解散分发 | 采集 → profiling-collect；方法论 → profiling-analyze；使能姿势/案例 → framework-feature-enablement；权重准备 → env-install |
| ascend-deploy | env-install + remote-access | 安装部分 → env-install；远程登录/卡管理 → remote-access |
| framework-integration | framework-feature-enablement | 去安装职责；定位改为特性使能与验证 |
| profiling-collection | profiling-collect | 统一采集（含三方框架补丁入口） |
| performance-analysis | profiling-analyze | 统一分析（profiling 族命名一致） |
| dummy-run-dev | dummy-run | 去 -dev 误导，定位快速验证载体 |
| performance-optimization | 保留原名 | 仅校准 S4 定位与引用 |

## 4. 预留空缺槽位（待经验回填）

> 槽位编号沿用历史：S2-1 对应 S1 kernel 融合能力清单（阶段并入后编号保留）；新增 S5-1 预留
> 训练感知案例。

| 槽位 | 阶段 | 内容 | 现状 |
|------|------|------|------|
| S0-1 | S0 | 各模型/任务权重分区与下载经验（Qwen-Image / Wan2.2 / H3…） | 仅 MiniMax-H3 |
| S1-1 | S1 | 无抽象接口框架的算子注入方法（多框架泛化） | 空 |
| S2-1 | S2 | mindiesd 融合 kernel 能力清单（接口 → pattern → 验证过框架，唯一真相源） | 分散于各 case（H3×vllm-omni 已回填：`framework-feature-enablement/references/vllm-omni-minimax-h3-case.md` §6 —— 融合候选工作流「执行序→候选列表→mindiesd/CANN 能力对照→独立验证」+ eager 已融合热路径判定 + 单步 kernel 采集 hook；**2026-09-06 图像第二案例**：`vllm-omni-qwen-image-case.md` —— FA/AdaLN/RoPE/GELU eager 覆盖核验 + kprof 目标扩展 + compile 输出非无损否决） |
| S3-1 | S3 | NPU 拓扑/带宽矩阵与并行选型决策 | 910B 单点 + 2026-09 增 950PR×4：并行矩阵/ring 不可用/offload 解锁并行与通算掩盖 step_trace 评估（见同 case §7 与 parallelism-strategy「内存受限时的并行解锁」）；2026-09 实测增补：UB/HCCS 岛 vs SYS/PCIe 拓扑、bulk vs head-parallel 翻转、HCCL 带宽 bench 姿势与 set_device 陷阱、端口泄漏/卡组诊断 → 单源参考 `parallelism-strategy/references/ascend-topology-bandwidth-diag.md` + evals 4/5；**2026-09-06 图像案例（vllm-omni-qwen-image-case.md §3/§6）**：UB 岛 0-3/4-7（跨岛 SYS）同岛选卡 + 同岛亦受他户干扰（探活前置）、**并行候选矩阵勿漏 2 卡 USP（TP1×USP2 图像 20 步 > TP2 -12%）**、短任务 4-rank 病态回退 |
| S3-2 | S3 | few-step 多 rank 验证协议（脚本 + 判据） | 空 |
| S4-1 | S4 | 精度校验与单特性影响评估方法（端到端质量门禁：定量 + 视觉伪影 + off-identity） | 部分回填：方法见 performance-optimization `references/quality-gate.md`，工具与判定标准在仓库 `evals/`（契约/rubric/profiles/quality_compare.py）；单特性影响矩阵首个真实案例已回填（H3×vllm-omni case §4/§5 + `evals/profiles/minimax-h3.toml` 阈值校准）；**2026-09-06 图像第二案例**：`vllm-omni-qwen-image-case.md` §4/§6 + `evals/profiles/qwen-image-2512.toml`（21-seed 同 seed 像素对口径；图像质量域 >> 视频 → 阈值不跨域迁移） |
| S4-2 | S4 | 组合试验设计与层回退策略 | 已回填协议 + **seam 表首案例已校准**（2026-09-05 H3×vllm-omni V1：precision×attention 跨 seam 可叠、attention 同 seam 取最强档、cache×稀疏同 step 窗口可叠、稀疏档质量非线性、frontier 以同窗相邻对为准；见 combination-search.md 校准节与 vllm-omni-minimax-h3-case.md §4） |
| S5-1 | S5 | 训练感知案例（少步蒸馏 4/8 步 + SLA/QAT 叠加；蒸馏权重 modelscope 下载；质量-速度权衡口径） | 空（预留分支，案例待回填） |

## 5. 快速开始

```text
# 模型/框架自动优化任务（编排层）
加载 model-auto-optimization → 先 Read workflows/optimization-flow.md → 建 run-state
→ 按 S0/S1/S3/S4(/S5 可选) 路由到能力技能 → 每阶段 stage_gate 门禁 → 闭环双报表

# MindIE-SD 仓库开发任务（编排层）
加载 dev-workflow → Test-First → 编码 → env-install 部署 → pytest
```

Profiling 回路（S1 融合分析 / S4 / 闭环直接使用）：

```bash
# 采集（自家脚本入口；三方框架入口见 profiling-collect 补丁法）
python skills/profiling-collect/scripts/collect_profile.py \
    --script wan_infer.py --device-id 0

# 分析 + kernel diff
python skills/profiling-analyze/scripts/analyze_trace.py \
    --profile-dir ./profile_l1 --output-dir ./
python skills/profiling-analyze/scripts/compare_traces.py \
    --baseline {base}/kernel_details.csv --target {opt}/kernel_details.csv
```

## 6. 目录结构

```text
.agents/
├── README.md                                   # 本文件：架构（三层）与技能清单唯一权威
└── skills/
    ├── model-auto-optimization/                # 编排层 · 模型自动优化流程
    │   ├── workflows/optimization-flow.md       # 阶段执行模板（确认点/验收 gate/闭环强制交付）
    │   ├── workflows/references/dispatch-templates.md  # 角色派发/自验证回执格式
    │   ├── references/run-state.md              # 运行状态文件规范（推进表 + 候选迭代表单一真相源）
    │   └── scripts/stage_gate.py                # 阶段推进机械门禁（零 NPU）
    ├── dev-workflow/                           # 编排层 · 仓库开发流程
    ├── env-install/                            # 能力 · 环境安装与准备（含权重确认/下载）
    ├── remote-access/                          # 能力 · 远程昇腾访问工具
    ├── framework-feature-enablement/           # 能力 · 三方框架特性使能与验证
    ├── framework-extension-dev/                 # 能力 · 三方框架侧特性补齐开发（框架缺口）
    ├── profiling-collect/                      # 能力 · 统一采集
    ├── profiling-analyze/                      # 能力 · 统一分析
    ├── performance-optimization/               # 能力 · 特性库与方案选择（S4）
    ├── parallelism-strategy/                   # 能力 · 并行策略选型
    ├── benchmark-dev/                          # 能力 · 算子选型判断工具
    ├── dummy-run/                              # 能力 · 快速验证载体
    ├── compilation-dev/ operator-dev/ aclgraph-dev/   # 能力 · MindIE-SD 实现开发
    ├── code-standards/ markdown-lint/ mindie-sd-community-governance/  # 能力 · 规范
    └── (其他支持目录)
```

## 7. 贡献

- 新增/改动 skill 前读本 README（三层架构分层与变更门禁，见「架构速览」与 §1–§7）；
  参考 [Anthropic skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)。
- 每个 skill 均含 `evals/evals.json`（≥2 条 prompt + expectations）。
- **变更门禁**：① 决定归属——能力技能不带流程语境（不写"主轴/侧轨"），新流程 = 新增编排层 skill；
  ② 同步本 README（对应层/域标签/目录树/计数）；③ 路由声明成对（"由 X 调用"与路由方指向互为反向）；
  ④ evals 增补；⑤ 复盘（dev-workflow §6）新经验先判断归属槽位/能力域再回填。

### case 回填规范（实验 → skill 刷新）

实验/复盘产出可复用经验时，把案例按下列规范沉淀为能力技能的 `references/{主题}-case.md`
（或说明性 notes），保证每次刷新不破坏三层架构分层与 skill-creator 要求：

**经验 vs 探针 判定（先于归属与沉淀）**：先用三问分类——
① 是否一次性/未正式合入（本机补丁、fork 钉版、.bak 默认关、用后即弃载体）？
② 是否只用于诊断/验证（绕法试探、临时开关、本地 microbench 探测、未上线载体）？
③ 是否依赖特定临时环境/窗口（版本漂移前有效、仅本案例可复现）？
任一为「是」→ 该内容按**探针**处理：可在 case/notes 内以 `[探针]` 标注并存档
（用途 / 适用窗口 / 结论边界），**不进「推荐姿势 / 方法论 / 经验槽位 / 报表宣称」**；
三者皆「否」→ 才按经验/方法论沉淀。例外：由探针**发现的 durable 约束事实**
（CANN 行为、部署顺序要求、命名/默认值纪律、框架支持状态）属经验（可注明"来源探针"）。

1. **归属判定（先于书写）**：先判断经验属于哪个能力域（framework-feature-enablement / dummy-run /
   performance-optimization / parallelism-strategy …）或预留槽位（§4）；编排层跨阶段纪律才进
   model-auto-optimization。不新建技能，除非触发「新增 Skill 规范」（dev-workflow）。
2. **case 文件模板**（对齐现有案例，如
   `framework-feature-enablement/references/vllm-omni-minimax-h3-case.md`）：
   - 标题：`模型 × 框架 × 主题（阶段/特性）`；首行引用块写 日期/仓库基线/环境/硬件/对比口径；
   - 正文按「环境要点 → 使能面速查 → 结果表（含口径）→ 回修与坑 → 方法论 → 结论与证据」组织；
   - 一次只归因一个变量；数字带口径（同拓扑/同 seed/steady 次数）；单次结论不迁移；
   - 端到端案例附质量门禁结论与阈值回填（`evals/profiles/{model}.toml`）。
   - **文件命名规则（统一）**：kebab-case；案例统一 `{framework}[-{model}]-case.md`（框架前置，
     如 `vllm-omni-minimax-h3-case.md`）；`-case`=事实案例、`-notes`=方法论沉淀、`-pattern`=接入姿势、
     `-gate`=判定标准、`-guide/-checklist/-catalog`=流程/清单；故障排查类统一
     `troubleshooting-{对象}.md`（对象前置）；reference 一律连字符、不用下划线。
   - **报表规范 ↔ 产物镜像命名**：编排层报表规范用 `-report.md`（`overview-report.md` /
      `detail-report.md`），与运行时产物 `overview_report.md` / `detail_report.md` 词根镜像——
      规范文件（references 内）连字符，产物（runs/ 下，stage_gate 强制 basename）下划线；
      二者不同目录、同名不同分隔符是刻意设计，勿合并。
   - **workflow 局部模板例外**：`workflows/references/` 只放该 workflow 专属派发模板
      （`dispatch-templates.md`），不归技能顶层 `references/`；通用 reference 一律在顶层
      `references/`，避免与跨 workflow 机制混淆。
3. **接线**：在所属 SKILL.md 的 Reference Files 登记 + 加载时机；作某技能方法论配套案例时，
   在引用方一并注明（路由声明成对）。**涉及框架特性支持的更新，须同步刷新
   `framework-feature-enablement/references/framework-support-matrix.md`（按该文件 §刷新协议）。**
4. **evals 增补**：行为/流程变化的，同步所属技能 `evals/evals.json`（schema 见 skill-creator
   references/schemas.md），expectations 用可验证陈述。
5. **隐私清理**：真实 IP / 主机 / 容器名 / 用户名 / 内部路径 → 占位符（`{model_weight_dir}`、
   `{env_host}` 等）；提交前敏感模式 `git grep` 复扫 0 命中。
6. **校验**：markdownlint-cli v0.44.0（仓库 pin）0 违规；evals JSON 解析通过；含脚本跑 ruff + py_compile。
7. **变更门禁复核**：README §2/§4/§6（计数/槽位/目录树）与相关 description 定位句是否需要同步。

### 当前开发状态

- ✅ 18 个技能：编排层 2 + 能力层 16
- ⏳ 7 个预留槽位：S3-1 已回填拓扑/带宽单源参考与 evals 4/5、S4-1 部分回填（质量门禁）、S4-2 已回填协议草案（seam 表待校准），
  其余待回填（见 §4）

### 最近经验合并（2026-09 MiniMax-H3×LightX2V 会话蒸馏，已按隐私规则脱敏）

- 本次把 R19–R25 会话知识按 skill-creator 规范合并进现有 5 个技能（无新增技能）：
  - `parallelism-strategy`：SKILL.md 更新触发/新增 950PR 拓扑段；新参考 `references/ascend-topology-bandwidth-diag.md`（UB vs SYS、bulk vs head-parallel 翻转、HCCL bench 姿势、卡组诊断）；evals 增 4/5
  - `framework-feature-enablement`：case 追加 §10（R22–R25 S4 稀疏/量化结论 + 无损试验矩阵摘要）
  - `profiling-analyze`：方法论补 clean-window/融合阶梯/环境劣化判读
  - `env-install`：troubleshooting 补 §F 多卡运行期劣化诊断树
  - `dev-workflow`：rework-lessons 增 #35（远程多卡实验工具纪律）
- 隐私规则：上述内容不含主机 IP / 口令 / 用户名 / 容器名；实验脚本与原始日志保留在会话产物目录，不写入 skills

### 最近经验合并 2（2026-09 H3×LightX2V 优化收口，R26–R28）

- **R26 收敛复核**：卡组可用性须实测（npu-smi OK ≠ 可用；4-7 驱动损伤残留 43.6s/步均匀慢）；
  masking 已实测否决（~4% 理论）→ parallelism-strategy reference §2/§4 补注 + rework-lessons #36
- **跨框架差距/对比**：768p 同分辨率对照 vllm-omni（lossless 243.0s@50 步、INT8 -14.2%、
  mix -44.6%、Cache -66.7%、best -82.5%）；LightX2V 单步同级、步数裁剪拉平；量化缺口双 seam
  （线性层 + 稀疏）清单落 case §11 / session GAP·COMPARE 文档
- **线性量化打通（原生 scheme）**：LightX2V 权重树为自研 `MMWeight`（非 nn.Linear）→ 注册原生
  `dit_quant_scheme="npu-w8a8-mxfp8"`（内部直连 mindiesd `npu_quant_matmul`），model.py 门放行
  online scheme；实测 **clean-window -13%（2× 复现）、帧 SSIM 0.972（近无损）** → 最终部署配置
  `r28_final_24step_npu_mxfp8.json`；matrix W8A8_MXFP8 格 ❌→🟡
- **FA 量化 A5 可用（认知修正）**：mindiesd FP8/MXFP8 FA（`npu_fused_infer_attention_score_v2`
  fp8）微测通过（cos≈0.998）→ 「仅 A2」不成立；matrix FA 行 + case §11 P3 更新
- 载体：`framework-feature-enablement/references/lightx2v-mindiesd-case.md`（§10/§11 采纳项与
  落地状态）、`framework-support-matrix.md`（量化两格 + FA 行）；会话证据见
  `session_work/{R27_P1P2_PROGRESS,R28_FINAL_REPORT_LIGHTX2V_H3,ANALYSIS_LIGHTX2V_NATIVE_QUANT,
  CROSSFRAMEWORK_768P_GAP,R29_SPARSE_FA_LIGHTX2V}.md`
- **R29 稀疏 FA 质量结论更正（重要）**：LightX2V rf3（与 vllm-omni 同 mindiesd rf_v3 路径）eager
  质量梯度平滑（同 seed/21 帧 vs dense：sp0.3 SSIM 0.975 / sp0.5 0.960 / sp0.8 0.81）——
  历史「sp0.3-0.6 平台 0.82-0.84」**证伪（R19 口径混淆，证据作废）**；compile×rf3 Dynamo
  trace 错误待解；vllm RAINFUSION 几何契约（tail/prefix）非质量必需。经验教训：稀疏质量门禁
  必须同 seed/同配置/同窗（详见 case §10 与 matrix rf_v3 格）

### 最近经验合并 3（2026-09 MindIE-SD FFN 融合会话：operator-dev / compilation-dev 收敛，已脱敏）

- `operator-dev`：catlass 只读融合算子全链（vendored 头 → standalone 对拍 → 集成 → compile
  GraphPatternEntry 真图使能 → 开关治理）沉淀为 2 个新 references + SKILL 接线；**三约束落档**：
  算子开发必须加载 cannbot（与本仓经验并行使用、非互斥）、融合 DSL 分界（CV 含 matmul → catlass；
  VV 纯 vector → triton）；审查收敛（位级仅 h3 特例、layout 记法等价注、工作流与案例去重）；evals 同步
- `compilation-dev`：提炼（skill-creator 合规）：数值口径"位级 vs fp8 量化级"、GraphPatternEntry
  双案例（h3 / mm_gelu）、registration-checklist 补「路径 B：GraphPatternEntry」与默认值纪律
  （norm_rope 教训：不留默认 False 死代码）、pattern-templates 修 qwen rope 状态并去重 §4；evals 增补
- 产品代码（未提交，另组产品 MR）：`mm_swiglu_mxquant`/`mm_gelu_mxquant`（catlass 融合算子 +
  GraphPatternEntry 真图使能，FLUX -5.7% / Wan -0.8% / Qwen -4%）；`enable_minimax_h3_norm_rope`
  （triton 净负、默认 False 死代码）**已整体移除**（算子/pattern/开关/UT），教训入默认值纪律

### 编排层流程管控改造（借鉴 cannbot-skills 编排机制，仅流程不涉知识点）

针对「使用编排层 skill 时输出不及预期、不按要求执行」问题，向 cannbot 的流程管控机制
（workflow 与 SKILL 分离 / 运行状态单一真相源 / 机械门禁 / 角色派发与自验证回执 / 逐阶段确认点
与拒收语义）做等价落地，未改任何能力技能知识内容：

- `model-auto-optimization/SKILL.md`：瘦身为「入口 + 强制流程指针 + 路由表 + 纪律总纲」；
  阶段执行细节移至新 `workflows/optimization-flow.md`（每阶段模板 + 方案确认点 + 验收 gate +
  闭环强制交付 + FAIL 上限 5 轮回退协议）
- 新 `model-auto-optimization/references/run-state.md`：进行中状态单一真相源（推进表唯一写者 =
  编排者；证据目录 evidence/{stage}/；即时回写支撑上下文压缩重建），与 manifest/final_report
  分工（见 artifact-layout.md 增补节）
- 新 `model-auto-optimization/scripts/stage_gate.py`：阶段推进机械门禁（零 NPU：校验 run-state
  推进表 status=done + 验收证据存在；close 强制声明总览/细分双报表），进入下一阶段前必跑
- 新 `model-auto-optimization/workflows/references/dispatch-templates.md`：采集/分析/实施/复核
  角色写权限 + 派发模板 + 自验证回执四行格式 + 拒收语义
- `dev-workflow/SKILL.md`：新增 §0.2 流程门禁（功能点验证留痕、复盘先行、被指向子任务遵守
  run-state/stage_gate）
- `model-auto-optimization/evals/evals.json`：增补 2 条流程遵从断言（阶段推进先跑门禁；缺
  post-enable-review/质量门禁结论的收益回执拒收）
- 校验：markdownlint v0.44.0 0 违规；evals JSON 解析通过；stage_gate.py py_compile + dry-run 通过

### compilation-dev 流程约束强化（能力层，承接编排层改造，仅流程不涉知识点）

- `compilation-dev/SKILL.md` 新增「验收与推进纪律（先验收后推进，出口证据化）」：统一证据行
  格式（与编排层自验证回执同构：Phase 证据/命中/数值核验/收益判定+开关状态）、Phase 3/4/6/7
  出口拒收语义（Phase 6 三层顺序不可跳，三层全过 + eager vs compile 数值核验后才允许宣称
  融合完成/收益）、断点接力状态行；Phase 1 新增**候选判定行**（优化点识别 gate：复用决策/
  预期收益/路径预判 + 不做清单，不静默跳过）
- `compilation-dev/evals/evals.json`：增补 2 条流程遵从断言（未过 Phase 6 三层不得宣称完成；
  无数值核验/计数契约不得宣称收益，fail-closed）
- 能力层不加运行状态文件、不引入编排语境；既有机械脚本（check_fusion_hit.py /
  numeric_check_eager_compile.py）已在纪律中绑定为 Phase 6 出口动作

### 编排层过程交付件强化（借鉴 Sana sol-engine 编排机制，仅流程不涉知识点）

针对探索过程"无迹可循/只报采纳项"缺口，借鉴 bounded search（GOAL/JOURNAL/REPORT）补过程交付件：

- `model-auto-optimization/references/run-state.md`：新增「候选与迭代表」（JOURNAL 式）：每候选
  一行（round/假说/状态 retain·discard·reject·terminal_pending_review/gate 证据/拒绝签名），
  拒绝签名枚举（crash/implementation-wrong/degenerate/dominated/out-of-scope/no-gain）、
  structured negative 不终止、预算记录、未尝试候选保留清单
- `workflows/optimization-flow.md`：阶段推进规则补 7–9（假说先行一次一候选 / 预算到点带
  frontier 收尾交用户选档 / **契约变更即新版本**：改基线·拓扑·口径·组合·产物结构 = 关闭旧
  claims 重跑 smoke+formal；golden 锚 = 同 seed 冻结 baseline 帧）；S1/S4 门禁并入迭代表治理
- `references/detail-report.md`：新增 **§C 候选治理（强制）**——已否决假设表（round/假说/签名/
  证据/派生）+ 未尝试候选清单（预期价值/未试原因/建议）；`references/overview-report.md`
  §6 补候选治理指针（总览子表只引用结论）
- `workflows/references/dispatch-templates.md`：补 **GOAL 骨架**（scope 固定/metric 定义/golden
  锚/有界循环/guardrails/deliverable spec），供 S3/S4/subagent 探索型派发
- `model-auto-optimization/SKILL.md`：声明纪律补第 7 条（契约变更即新版本）
- `model-auto-optimization/evals/evals.json`：增补 2 条流程遵从断言（假说先行迭代表治理；闭环
  报表必须含已否决假设与未尝试候选）
- 校验：markdownlint（0.44 等价）0 违规；evals JSON 解析通过；description ≤500 字符不变

### 特性语义与命名对齐（kernel融合 单特性模型 + 收益小阈值 0.5%）

用户确认固定特性名语义：`kernel融合` 是**单一固定特性名**，其下全部融合内容与实现方法
（compile / API 接入）一律入说明列/细分报表、不派生特性名；识别到融合机会 → 需新算子 →
`operator-dev` 开发 → 融合前后算子执行序收益判定 → **整 block 耗时影响 <0.5% 视为收益小、
可不执行**（记录签名进迭代表/A.1.3/§C），决策必须进最终报表。据此落地：

- `workflows/optimization-flow.md`：S1 合并接入+融合为统一链（原 S1/S2 合并）
  （识别→operator-dev 开发→执行序对比→收益判定）+ **0.5% 收益小阈值单点定义**（口径=融合
  所在链路完整执行墙钟；噪声宣称 <3% 规则不变）
- `references/detail-report.md`：A.1 归属口径（固定特性名/子项/方法不派生名）；A.1.1 表增
  「实现方法」列 + 状态枚举 `采纳/既有/识别但不执行（整 block <0.5%）/回退`；A.1.3 改「识别到
  但未落地/未执行的融合机会」（含开发链与 0.5% 判定）；§C.1 增特性/实现 id 列与 0.5% 引用
- `references/run-state.md`：迭代表增「特性/实现 id」与「计数契约证据」列（S4/融合类必填，
  防 no-op 可对账）；拒绝签名 `no-gain` 注明 <0.5% 阈值语义
- `references/overview-report.md`：§2.1 固定名词表补 kernel融合 单特性/子项/方法不派生名句、
  f8 双实现互译（FP8RotateQuantFA docs 特性 vs EagleQBSA 案例实现）、时间步优化 vs 少步蒸馏
  按权重归组区分；§2.4 补「内容属 kernel融合、收益归因绑使能序列」兼容注；§2.5 补与矩阵图例
  互译行
- `framework-feature-enablement/references/framework-support-matrix.md`：首部报表映射补 f8 实现
  互译
- `scripts/feature_declarations.json`：models 增 `minimax-h3-lightx2v` 能力别名
- 校验：markdownlint（0.44 等价）0 违规；evals/declarations JSON 解析通过；SKILL description
  不变；**未改任何固定特性名（白名单不动）**

### 单点特性执行架构与语义补全（框架自带能力 / 量化能力组合 / 稀疏经验档 / 无损∥有损并行）

- `references/overview-report.md` §2.1：补 `Cache` 语义（固定特性名；实现优先框架自带能力
  如 vLLM-Omni cache_dit，mindiesd cache_agent 兜底；框架能力名/外部同名实现不得作特性名）
  与 `量化` 语义（单一固定特性名，修饰符 = 能力内容组合 w8a8/f8/w4a4/w8a8f8，非子特性）
- `workflows/optimization-flow.md` S4：补「实现来源（先框架后 mindiesd）」与「稀疏经验档起扫
  ——图像 sparsity 0.6 起步 / 视频 0.8 起步 → 稀疏度-性能曲线判断效果与算子性能；图像无 2D
  路径 staying-dense fail-closed」
- `workflows/references/dispatch-templates.md`：增「单点特性独立子 agent 与并行执行（无损∥有损）」：
  一特性一 agent + evidence 按 `{stage}/{feature}/` 隔离 + 同 seam 互斥禁止并行双开 + 推进表/
  迭代表编排者单写 + 同基线合并报表
- `references/run-state.md`：evidence 布局按特性隔离 + 并行写者规则（引用 dispatch 并行护栏）
- 校验：markdownlint（0.44 等价）0 违规；**未改任何固定特性名**

### 特性覆盖清单（任务级触发判定前置交付件）

借鉴 Sana method-baseline catalog/search_space 与 cannbot 探索 dashboard 的候选登记，补
"保证每个候选特性被显式触发判定、防漏想方向"的最后一环：

- `references/run-state.md`：新增「特性覆盖清单」规范 + 模板（固定特性全集 + 预扫缺口逐项：
  做-目标/经验档 / 分析后做 / 不做+理由（无瓶颈/预期收益小 <0.5%/框架不支持引矩阵）；round/
  阶段 + 证据指针；闭环前复核无未裁决项）
- `workflows/optimization-flow.md`：新增「特性覆盖清单（触发判定前置）」节 + 闭环复核点
- `SKILL.md`：恢复 §0 启动确认（此前瘦身误删、workflow/dispatch/evals 仍引用），并登记
  特性覆盖清单（含目标/验收、缺口补齐 5 选、实现边界、中途新缺口回补）
- `references/detail-report.md` §D：补覆盖清单闭环复核引用
- `evals/evals.json`：增补 1 条流程遵从断言（启动建清单逐特性触发判定、框架不支持不静默跳过、
  闭环无未裁决）
- 校验：markdownlint（0.44 等价）0 违规；evals JSON 解析通过；未改任何固定特性名

### framework × 特性/能力 支持档位（借鉴 method_baseline tier 语义）

在 `framework-support-matrix.md` 新增 §〇「framework × 特性/能力 支持档位」图例与记录规则
（区分"验证状态 ✅/🟡/❌/❓"与"实现就绪状态"），供特性覆盖清单排序与执行序：

- 档位：`已支持`（直接用）/ `待配置`（特性面存在但部分能力不全/差接线，如"量化有支持但缺
  mxfp8 型 w8a8"——能力实体已具备）/ `待开发`（无支持需完整实现：operator-dev 新算子 /
  framework-extension-dev 结构性实现，先估成本 + §0 确认）/ `上界`（**预留空间，定义待人工
  填写**——理想档/天花板探针，只诊断不宣称）
- 判定粒度（能力级）：量化按能力组合 w8a8/w4a4/f8（类型 mxFP8/mxFP4，与报表修饰符同源）；
  稀疏**与 mindiesd 稀疏算子对齐**逐算子判（rf_v2/ada_bsa/平台扩展）；cache/编译融合/并行按
  分项判；判据=能力实体是否存在 + 是否仅差接线
- `run-state.md`「特性覆盖清单」增「框架档位」列 + 执行排序（已支持/待配置先做，待开发先估
  成本确认）；`optimization-flow.md` 特性覆盖清单节补档位排序句
- 校验：markdownlint（0.44 等价）0 违规；未改任何固定特性名

### 编排层 workflow 架构刷新（README 架构对齐）

随编排层新增 workflow 分离（model-auto-optimization 的 workflows/optimization-flow.md 等），
刷新本 README 架构描述以反映"入口 SKILL + workflow 执行定义 + 状态/门禁"结构：

- 顶部「组织方式」注明编排层内部 = 入口 SKILL.md + workflows/ + run-state/stage_gate 状态门禁
- 新增「架构速览（编排层 = 入口 SKILL + workflow 分离）」：三件套目录结构与加载链路
  （触发 → SKILL 分流 → 必读 workflows → run-state → stage_gate 逐阶段 → 能力技能 → 闭环双报表）、
  能力层不带流程语境的边界、何时把阶段细节拆进 workflows/（model-auto-optimization 已落地；
  dev-workflow 轻量内联暂不拆）
- 技能关系图补编排层内部结构注释；§1 dev-workflow 行与 §5 快速开始同步（先读
  workflows/optimization-flow.md、dev-workflow 轻量内联说明）

### 最近经验合并 4（2026-09-08 cache-dit×MiniMax-H3 补测轮：framework-feature-enablement case + effort V4 校准，已脱敏）

- `framework-feature-enablement`：`references/cache-dit-minimax-h3-case.md` 重写固化补测轮经验——
  ① 稀疏 FA **两条路径区分**（eager rf_v3 外部逐层 mask 1.15–1.25× 近无损，kernel 级已确认；EagleQBSA
  融合 op 稀疏FA+FA量化 op 级快 ~6× 未接线）；② **mindiesd 自研算子部署坑**（`ASCEND_CUSTOM_OPP_PATH`
  由 `import mindiesd` 前置设置，先 import 再建 NPU 张量，否则 aclnn `inferShape does not exist`）；
  ③ mask 开销拆解（选择逻辑 ~3ms、rearrange/pool 搬运 ~220ms/步为主）；④ **叠加组合实测**（sparse×Cache
  3.4–3.7×、量化×稀疏×Cache 4.1–4.5×、start0 质量劣化）——性能表述仅加速比口径；SKILL.md 案例行同步
- `model-auto-optimization`：`effort-estimation.md` §1c 补组合批成本口径（8 serve≈4 卡×1.2h、质量对
  预算公式）
- **通用方法论再提炼（跨模型可复用，按域排布）**：`env-install/references/troubleshooting-env.md` 新增
  §G（mindiesd 自研 CANN 算子部署/`import mindiesd` 前置 `ASCEND_CUSTOM_OPP_PATH`、gloo
  `ss_family 10 vs 2` 偶发强制 IPv4）；`profiling-analyze/references/heuristics.md` 新增
  「Host/Kernel 与数据搬运归因」（step_trace Free≈0→device-bound、per-op 时长聚合区分计算 vs 搬运、
  单调用微基准 vs 模型步级矛盾核对项）；`framework-feature-enablement/SKILL.md` ③ 补「自研算子部署侧」
  根因 + 三层证据段补「稀疏/融合路径区分先于收益判定（eager 外部 mask vs 融合 op，op 级微基准先行）」
- 边界：仅改动 cache-dit 相关内容，未触碰其他框架（matrix/vllm-omni 0.28/qwen/lightx2v 等）内容

### 三层架构重构落地（README 架构分层）

三层模型（编排 L1 · workflow L2 · 能力 L3）文档化：组织方式与「架构速览」改为三层——
L1 管横向（分流/特性验证顺序/交付件契约/机制所有权/总览收口），L2 管纵向（业务 workflow：阶段化
执行 + 单特性策略深挖与组合回退裁决，厚度因业务线而异：模型优化厚、dev 薄内联、纯 L3 域无
L1/L2），L3 管能力与判据（含能力自身工作流纪律）；写入内容归属三问判据、跨特性组合归属、
执行波次（单特性∥→组合后置）、报表分工（总览 L1 收口 vs 特性报表）与加载链路、dev 拆 L2 信号。
机制单一真源在 L1（run-state/stage_gate/迭代表/双报表），L2 引用不重复。

### 命名体系审计落地（references 同级命名收敛，2026-09）

对 `.agents/skills` 全量文件做"名称 ↔ 功能 / 同级命名策略"审计后收敛（git mv + 全引用面同步，
本 README §6 树/§7 词表同步）：

- **N1 报表规范 ↔ 产物镜像**：`model-auto-optimization/references/optimization-report.md` →
  `overview-report.md`——spec 词根与产物 `overview_report.md` 对齐（与 `detail-report.md` ↔
  `detail_report.md` 同构）；§7 补「报表规范 ↔ 产物镜像命名」规则
- **N2 case 后缀归位**：`operator-dev/references/case-mmgelu-flux-wan-qwen.md` →
  `mmgelu-flux-wan-qwen-case.md`（唯一 `case-` 前缀反例 → 统一 `-case.md` 后缀）
- **N3 compilation-dev references 词族收敛**：`custom-graph-pass.md` → `custom-graph-pass-guide.md`（⚠️
  该文件后续已删除，见下条"废弃自定义 Graph Pass"）、
  `graph-pattern-rewrite.md` → `graph-pattern-rewrite-guide.md`、`benefit-rootcause.md` →
  `benefit-rootcause-guide.md`、`pattern-dev.md` → `pattern-dev-notes.md`（-dev 后缀与技能目录撞名，
  实为注册机制易错细节）
- **N4 operator-dev references 词族收敛**：`catlass-ffn-fusion-opdev.md` →
  `catlass-ffn-fusion-guide.md`、`mindiesd-fusion-experience.md` → `mindiesd-fusion-notes.md`；
  `operator-optimization-skill-map.md` / `catlass-kernel-integration.md` 保留（-map/-integration 自解释，
  与同目录 -guide/-notes/-case 不冲突）；`framework-support-matrix.md` 内「optimization-report §2.1」
  裸词引用同步 → overview-report
- **N5 workflow 术语避让**：`profiling-analyze/references/analysis-workflow.md` → `analysis-flow.md`
  （references 层不用 -workflow，避免与 L2 `workflows/` 概念同词异层）
- **N7 泛名精化**：`benchmark-dev/references/debugging.md` → `troubleshooting-benchmark.md`
  （归属 troubleshooting-{对象} 词表，对象=benchmark）；`dummy-run/references/model-common.md`
  保留（镜像代码目录 `model/common`，属「代码目录镜像」类功能性命名）；`cache-dit-minimax-h3-case.md`
  首行补命名说明（cache-dit=框架 repo 名，非特性 Cache）
- **N6 层级例外文档化**：README §7 补「workflow 局部模板例外」（workflows/references/ 只放该
  workflow 专属模板）；`scripts/README.md` 补「数据随脚本就近」说明（feature_declarations.json 不入
  references 词表）
- 校验：markdownlint（0.44 等价）0 违规；evals JSON 解析通过；旧 basename 全 token/裸词残留 = 0

### compilation-dev：废弃自定义 Graph Pass（2026-09，禁止手写 FX graph traversal）

对照代码现状审计后收敛（register_replacement 双参数 pattern 即可表达 `nn.Module` 权重，
freeze 前窗口命中；`GraphPatternEntry` 为手动改图唯一正解）：

- **删除** `compilation-dev/references/custom-graph-pass-guide.md`（git rm），**明确禁止
  自定义 FX Graph Pass**：不得手写遍历 `graph.nodes` 的 `_rewrite_*` 方法、不得在
  `graph_rewrite_after_freezing` 手写 node 替换（该路径仅历史留档于
  dev-workflow/rework-lessons.md §24/§25 并标注已废弃）
- **同步**：compilation-dev SKILL.md（生命周期 / Phase 2 路径表 ⛔ 禁行声明 / Phase 5 路径 B /
  Reference Files 表）、mismatch-catalog.md 类型 7（正确写法改为 weight 收进 pattern 输入 +
  register_replacement）、graph-pattern-rewrite-guide.md 路径表、evals.json eval 1
- **保留** `graph-pattern-rewrite-guide.md`（GraphPatternEntry 为 pattern matcher 原生 API，
  手动改图正解，不属自定义 pass）
- 校验：markdownlint（0.44 等价）0 违规；evals JSON 解析通过；`custom-graph-pass` 引用残留仅存于
  "已废弃/禁止"说明处

### 模型自动优化：少步性能识别 + 全量叠加 + 报表步数列（2026-09）

性能收益识别口径统一（profiling 手段，非 S4 少步产品特性）：

- **少步识别收益**：无损优化（S1/S3）与不做质量分析的有损性能分析可用很少步数（如 1 步预热 +
  1 步采集）识别收益——单步即代表算子形态与 kernel 序（profiling-collect「少步快速采集经验」）
- **同口径对比**：少步下加速比/百分比必须同为少步、同窗口同卡组对比；禁止少步结果 vs 全量基线
  混口径
- **最终叠加必须全量**：需要质量分析的有损行与所有叠加/组合/采纳行 = 全量步数实测；少步只筛方向
- **总览报表新增「步数」固定列**（优化类型/特性名/首步耗时/步数/加速比/质量/说明，7 列）：每行必填
  实测 denoise step，便于按步数核对耗时合理性（60 步 vs 24 步耗时是否成比例）
  （注：性能主口径后续定为 warmup 后第 1 步耗时，见下条）
- 同步：overview-report.md（§1.1 新增 + §2 列 4 + §2.3 + §5 子表 + §7 纪律）、detail-report.md
  （A.1.1/B 节步数）、run-state.md 迭代表（步数列）、optimization-flow.md（标准回路步数口径/
  S1/S4）、combination-search.md（执行协议 + 候选登记表步数列）、SKILL.md（总览收口/强制交付）、
  evals.json（eval 8 固定七列）

### 模型自动优化：e2e 主口径（锚点实测 / 中间估算）+ 首步辅助 + 字段枚举表（2026-09）

步数-耗时**非线性**（固定开销 + VAE tile 解码帧数 + 热效应随步数累积），完整请求墙钟不能按步数
等比例换算/直接比较 → 性能口径统一为：

- **性能主口径 = e2e 完整请求墙钟（s/请求）**：`base` / `最终推荐` / `三元组合` 三行（结论锚点）
  **e2e 必须实测**（禁止估算）；完整墙钟随步数非线性，不同步数行不按 e2e 等比例互比
- **中间行 e2e 可估算并标 `[估算]`**：`行 e2e[估算] = base DiT 耗时 × (行首步耗时/base 首步耗时)
  再加 (VAE + 其他固定耗时)`（§1.1，同步数前提）——少步/估算只铺中间链，锚点不估算
- **首步耗时（warmup 后第 1 步, s/step）= 辅助口径**：步数无关、跨 run 可比——用于少步性能识别
  （1 预热 + 1 采集）与中间行 e2e 估算输入；质量行/最终叠加仍全量步数实测
- **总览报表 8 列**：优化类型 / 特性名 / e2e(实测或[估算]) / 首步耗时 / 步数 / 加速比 / 质量 /
  说明；新增 §7.1 字段取值类型表（**枚举**=优化类型/特性名/无损质量词；**数值**=e2e/首步/步数/
  加速比/有损质量；**自由文本**=说明/子组合/证据——必含项固定，其余任意）
- 同步：overview-report.md（§1/§1.1/§2 列 3-8/§2.3/§5/§7/§7.1）、detail-report.md（头部基线 +
  A.2/B.1-B.3）、optimization-flow.md（标准回路/S1/S3/S4 验收）、combination-search.md（frontier/
  Step 3/候选登记表）、SKILL.md（证据/评估纪律）、evals.json（eval 8 八列 + 实测/估算 + 字段类型）
- **回退必须全量实测（2026-09 追加）**：cache/稀疏/量化发生回退（降档/错开窗口/换实现/整维回退后
  保留的档）时，**回退后最终档必须全量步数实测 e2e 识别性能加速**——回退档=潜在采纳锚点，
  禁止少步识别或 `[估算]` 宣称回退后性能；同步 combination-search「层回退策略」/Step 3、
  optimization-flow S4 试验记录、overview-report §1.1 锚点行与 §2.3、detail-report B.4、
  SKILL 有损纪律、evals（eval 6 回退全量）

### 遗留待办（人工项）

- 核对 `mindie-sd-community-governance/assets/mr_ruleset_20260327101328.xlsx` 与
  `mindie-sd-community-governance/SKILL.md` §5.5 的定位是否一致（该技能按约定不改写，需人工复核一次）
- `docs/zh/features` 增补 2026-09 落地特性（`mm_swiglu_mxquant`/FFN-MX compile 融合、FA A5
  可用性等）后跑 `performance-optimization/scripts/refresh_features.py` 同步
  `mindiesd-features.md`（产品 docs MR；skills 侧 framework-support-matrix 已登记，S2-1 槽位待回填）
- dummy-run 的 quant 细节（§A6）是否部分下沉 `references/model-common.md`（当前内联偏厚，观察项）
- S2-1（mindiesd 融合 kernel 能力清单，唯一真相源）回填时可参考 per-technique 目录化组织
  （每技术：描述 / 开关 / 验证点与运行期计数 / 模型适用矩阵）——结构模板借鉴
  跨栈方案的 per-technique 目录化组织（仅借结构，不搬其内容）

## 参考链接

- [Anthropic skill-creator 规范](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
- [cannbot-skills（算子开发外部技能库）](https://gitcode.com/cann/cannbot-skills)
- [MindIE-SD](https://gitcode.com/Ascend/MindIE-SD)
