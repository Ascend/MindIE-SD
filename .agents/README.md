# MindIE-SD 多模态解决方案 Skills

面向 MindIE-SD 多模态扩散模型（Wan2.2 / FLUX / Qwen-Image / MiniMax-H3）在昇腾 NPU 上的
**模型自动优化**技能集合。

组织方式（三层架构：编排 L1 · workflow L2 · 能力 L3，2026-09 重构）：

```text
L1 编排入口（2）：model-auto-optimization / dev-workflow——任务分流与路由、特性验证顺序（依赖派生）、
                交付件契约与编排机制所有权（run-state / stage_gate / 迭代表 / 覆盖清单 / 双报表）
L2 业务 workflow（按业务线）：特定业务线下的阶段化执行与单任务深挖——把 L3 能力组合成该业务的
                流程与"姿势/优化选择"（含单特性策略与组合回退裁决）；厚度因业务线而异（见「架构速览」）
L3 能力（18）：可复用的单一能力/知识/工具（含能力自身工作流纪律），可被任意 L2 编排或独立直达
预留槽位（8）：流程自动化所需但经验尚空的位置（§4），复盘回填
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
| **L3 能力** | 可复用单一能力：选项/接口/判据/工具；含**能力自身工作流纪律**（如 pattern-dev Phase、governance 固定步骤，显式区别于业务 L2） | 能力 SKILL + references/scripts/evals | 业务顺序、业务门禁节奏、本业务姿势决策 |

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
                       编排层（L1）
   model-auto-optimization（模型优化流程）        dev-workflow（仓库开发流程）
      │ S0→S1→S3→S4→S5(可选)→S6(条件进入)→闭环      │ 路由/承接开发子任务
      ▼                                            ▼
 S0 准备         env-install（安装/权重确认/下载） ── 工具 ──> remote-access
                 dummy-run（简化基线/快验）
      │  产出「阶段账 + 瓶颈点标签」 → 交给优化域入口
      ▼
 优化域入口      performance-optimization（L2，可独立触发：**只分发，不选档**）
                 └─ 按瓶颈标签分发 ▼（标签表：references/bottleneck-labels.md）
  S1 DiT·融合      dit-perf-opt（特性选档与实施）──需新能力──> pattern-dev / operator-dev
                   profiling-collect ──> profiling-analyze（算子序→机会点→收益）
                   framework-integration（分支 A 使能 / 分支 B 缺口补齐）
  S3 DiT·并行      dit-parallel-opt（选型 + 生效判定）◄──掩盖收益判定── profiling-analyze
  S4 DiT·有损      dit-perf-opt（组合/seam/回退）＋ framework-integration ＋ benchmark-dev
  S5 DiT·训练感知  MAO 自身 ＋ framework-integration（少步蒸馏 / SLA / QAT）
  S6 VAE + host    vae-opt（VAE 计算 + 通信）／ host-opt（交付搬运 + 装载预热）
                   —— 进入条件：非 DiT 段占比 ≥10%（bottleneck-labels.md，须带步数档）
  闭环复验         标准回路：profiling-collect → profiling-analyze → dit-perf-opt
                   验收：perf-gate（入库门）／ accuracy-gate（未用有损时的一致性）
                   （产物规范 model-auto-optimization/references/artifact-layout.md）

 开发与规范支撑（被 dev-workflow 与上述场景调用）
   pattern-dev / operator-dev / aclgraph-dev / quantization-dev / benchmark-dev / dummy-run
   framework-integration（三方框架侧：使能 + 缺口补齐，两分支）
   code-standards / markdown-lint / mindie-sd-community-governance
```

图例：`──路由/调用──>` 编排层→域入口→模块调度；`──数据/产物──>` 上游产出喂下游；`◄──> / ──>` 边界互指（description 的 near-miss 句，如 collect↔analyze、benchmark-dev↔dit-perf-opt、dit-parallel-opt↔performance-optimization、framework-integration↔env-install、vae-opt↔dit-parallel-opt）；虚线场景按需加载（实现层/工具），非默认常载。各技能详情见 §1/§2，仓库级路由规则见 AGENTS.md §5。

> 编排层两技能内部结构：入口 SKILL.md（判定/路由）→ 执行定义（`model-auto-optimization` 已拆
> `workflows/optimization-flow.md` 阶段文件；`dev-workflow` 为轻量内联）→ 状态/门禁
> （run-state 推进表 + 候选迭代表 + 特性覆盖清单 + `scripts/stage_gate.py`）——详见「架构速览」。

---

## 1. 编排层（流程）

| 技能 | 一句话界面 | 触发 |
|------|-----------|------|
| **[model-auto-optimization](skills/model-auto-optimization/SKILL.md)** | 模型自动优化流程：**S0 环境准备 → S1 DiT·融合 → S3 DiT·并行 → S4 DiT·有损 → S5 DiT·训练感知 → S6 VAE + host**（非 DiT 段占比 ≥10% 才启动；门限口径见 `references/bottleneck-labels.md`）**→ 闭环复验**（内含 采集→分析→复验 标准回路；**闭环必列输出「优化总览报表 + 优化细分报表」**——总览基线=TP 多卡未优化，细分=融合算子/并行/稀疏/量化/cache 与步数/组合 构成口径，见 `references/overview-report.md` 与 `references/detail-report.md`；框架未提供的特性须明确标注「未提供」）。流程执行按 `workflows/optimization-flow.md` 阶段模板，阶段验收以 run-state 推进表 + `scripts/stage_gate.py` 门禁为准（见 `references/run-state.md`）；**瓶颈点标签表的单一真源 = `references/bottleneck-labels.md`**（域入口据它分发） | 模型名 + 优化/加速/跑通/采profile 等流程型目标 |
| **[dev-workflow](skills/dev-workflow/SKILL.md)** | 仓库开发流程：Test-First → 编码 → 部署 → pytest → 复盘（轻量内联，暂不拆 workflow 文件；流程门禁见其 §0.2） | MindIE-SD 代码改动 |

## 2. 优化域入口与能力层（21）

> **分类**：优化域入口（L2，可独立触发）**1** · 优化域模块（L3）**4** · 能力供给（L3）**6** · 标准与方法（L3）**2** · 采集分析（L3）**3** · 环境与运行通道（L3）**2** · 规范（L3）**3**。
> **命名分层**：**入口用全称**；**模块 `{对象}[-{维度}]-opt`**；**验收标准 `-gate`**；其余角色词限定为 `-collect` / `-analyze` / `-dev` / `-lint` / `-standards` + 名词短语。**新增技能不得另造角色词**，例外须在本表标注理由。
> 另有**预留分类**：AR 优化（条件编码器 / 自回归模型）——本次只占位，不建技能。

### 2.1 优化域入口（L2，可独立触发）

| 技能 | 一句话界面 | 域标签 |
|------|-----------|--------|
| **[performance-optimization](skills/performance-optimization/SKILL.md)** | 优化域入口（L2）：拿到**已明确的瓶颈点 / 瓶颈标签**后按标签**分发**到四个模块（DiT 计算 / DiT 通信 / VAE / host），并持域内**最小前置集**与**域级验收口径**（性能入库引 `perf-gate`、精度判据引 `accuracy-gate`）；**本技能不再承载选档与实施**。瓶颈标签与 10% 门限的单一真源 = `model-auto-optimization/references/bottleneck-labels.md` | 入口 |

### 2.2 优化域 · 模块（4）

| 技能 | 一句话界面 | 域标签 |
|------|-----------|--------|
| **[dit-perf-opt](skills/dit-perf-opt/SKILL.md)** | DiT 计算模块：把**已定位的 DiT 计算瓶颈**落成特性级选档与实施——量化档（W8A16/W4A16/W8A8 系列/W4A4/MXFP8/FA 量化）、稀疏（rf_v2/ada_bsa）、缓存（DiTCache/AttentionCache/时间步优化）、编译启用的**开不开、开哪一档、怎么开、怎么复验**；依据 `docs/zh/features/*`（特性真源）+ `framework-integration/references/framework-support-matrix.md`（支持状态），组合/seam/回退见其 `references/combination-search.md` | 计算 |
| **[dit-parallel-opt](skills/dit-parallel-opt/SKILL.md)** | DiT 通信模块：并行策略选型（USP→CP + few-step 多 rank 验证）；**形态抉择**（纯 Ulysses vs 复合 AllGather-KV×Ulysses，按 GQA/跨岛带宽/形态 plumbing 条件化定胜负）；**并行 × 稀疏叠加**（seam 契约与「未生效」判定）；**通信掩盖上限**（`c/f` 决定 `1-1/n` 是否可达）+ **并行方案差异归因**；收编 offload/TP，并**承接「改了并行不报错但没生效」的生效判定**（见其 `references/scope-effectiveness-check.md`） | 并行 |
| **[vae-opt](skills/vae-opt/SKILL.md)** | VAE/TAE 解码段优化（**计算 + 通信同技能**）：解码器能否沿某轴切到多卡、怎么切才逐位精确、切了值不值；含跨轴耦合审计、潜帧边界切分＋末帧状态携带、与 rank 无关的判定、交换预算、已知静默陷阱、有损分片的精确性前置条件与自动修复；范围**含 VAE encode（待补，暂不纳入正文）** | 对象 |
| **[host-opt](skills/host-opt/SKILL.md)** | host / 辅助段优化：**交付与搬运**（mp4/图片编码、worker→API 通路、落盘、异步化）+ **装载与预热**（权重加载、编译与图下发预热、镜像/容器预热）；**明确不含并发与吞吐**（未来独立立项） | 对象 |

### 2.3 能力供给（6，阶段无关）

| 能力 | 一句话界面 | 域标签 |
|------|-----------|--------|
| **[pattern-dev](skills/pattern-dev/SKILL.md)** | PyTorch Inductor pattern matcher 机制开发：扩展本仓 compile 能力或三方框架自带的类似能力——Pattern 创建/注册/调试、GraphPatternEntry 手动改图、Copy（InplaceCopy/ViewCopy）消减全生命周期，含 Phase 1–7 出口证据纪律 | 开发 |
| **[operator-dev](skills/operator-dev/SKILL.md)** | 算子开发与调优（Triton / Ascend C / Catlass / PyPTO / TileLang；优先路由外部 cannbot-skills） | 开发 |
| **[aclgraph-dev](skills/aclgraph-dev/SKILL.md)** | NPU 图批量下发开发（NPUGraph 静态 capture / graph pool / lazy capture / 驱逐） | 开发 |
| **[quantization-dev](skills/quantization-dev/SKILL.md)** | 量化**格式契约**与位级对齐：从设备字节反推 MXFP8/int8 契约（编码公式/舍入/scale 粒度/退化块）并逐字节复现；含**除数须载入**、**8-bit 回绕非饱和**、组尺度耦合、位级对拍 SOP、**量化前移的字节精确条件** | 算子 |
| **[dummy-run](skills/dummy-run/SKILL.md)** | **能力供给的配套验收载体**：随机权重/精简代码快验（架构兼容性、算子先接入、融合可行性），提升开发效率；**非通用验证载体** | 验证 |
| **[framework-integration](skills/framework-integration/SKILL.md)** | 三方框架特性落地（**原 framework-feature-enablement ＋ framework-extension-dev 合并**）：一个入口信号「框架侧特性没落地」，内部分两分支——**分支 A 框架已有 → 使能与验证**（计数契约 + 三层证据 + 异常回修）；**分支 B 框架缺失 → 补齐开发**（注入点 + 注册机制 + 合入姿势：平台注册/上游 PR 优先，fork/monkey 备选） | 框架 |
| **[env-install](skills/env-install/SKILL.md)** | 环境安装与准备：mindiesd + 三方框架全栈安装、权重确认/下载（部署时先与用户确认是否已存在） | 安装准备 |
| **[remote-access](skills/remote-access/SKILL.md)** | 远程昇腾访问工具：SSH 连接复用 / 容器执行 / 空闲卡选择 / 传输 | 工具 |

### 2.4 标准与方法（2）

| 能力 | 一句话界面 | 域标签 |
|------|-----------|--------|
| **[accuracy-gate](skills/accuracy-gate/SKILL.md)** | **精度验收标准**（原 `lossless-op-replacement`）：凡是"改动不应改变结果"的场合（等价替换、融合、并行切分、编译与图下发、拷贝消减）都按本标准判合格/不合格——等价分层（逐位 / 数值门 / 有损门）+ **三级验收序**（① 同配置重跑逐位 → ② 跨配置数值门 + 产物 md5 不变 → ③ 质量门）+ 判据不达标时的排障入口；含**满足本标准的实现约定**（per-shape 对拍写进实现） | 验收 |
| **[perf-gate](skills/perf-gate/SKILL.md)** | **性能验收标准**（原 `npu-ab-measurement-discipline`）：判定"有没有效、有多少效"，并规定**只有验收态结果才能写入总览表**；分**探索态**（不展开、少步/单次、产物标 `[探索]`、不得入表）vs **验收态**（同窗 ≥N 复现）；含同窗 A/B 唯一权威、A/B/A 漂移校正、噪声地板、三证、md5 盲区与忠实度对照 | 验收 |

### 2.5 采集分析（3）

| 能力 | 一句话界面 | 域标签 |
|------|-----------|--------|
| **[profiling-collect](skills/profiling-collect/SKILL.md)** | 统一采集：mindiesd 自家脚本 + 三方框架补丁两入口，产出统一 ASCEND_PROFILER_OUTPUT | 采集 |
| **[profiling-analyze](skills/profiling-analyze/SKILL.md)** | 统一分析：5 层管道 + kernel diff，输出瓶颈与融合机会候选 | 分析 |
| **[benchmark-dev](skills/benchmark-dev/SKILL.md)** | 算子选型判断工具：单算子对比（选型证据）+ 算子接入测试（供 operator-dev）；计时方法论单点 = 其 `references/benchmark-guide.md` | 基准 |

### 2.6 规范（3）

| 能力 | 一句话界面 | 域标签 |
|------|-----------|--------|
| **[code-standards](skills/code-standards/SKILL.md)** | Python 格式与 lint | 规范 |
| **[markdown-lint](skills/markdown-lint/SKILL.md)** | Markdown 格式检查（工具名保留为业界通用名，属命名例外） | 规范 |
| **[mindie-sd-community-governance](skills/mindie-sd-community-governance/SKILL.md)** | 文档/治理/提交/PR/版本规范（全库唯一带仓名前缀者，属命名例外） | 规范 |

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
| **parallel-scope-diagnosis** | **撤销（内容三分）** | 依据「**各技能对自身使能成功负责**」：测量类 → `perf-gate`；生效/静默降级判据 → `dit-parallel-opt` 与 `vae-opt`；并行陷阱清单 → `dit-parallel-opt` |
| **npu-silent-wrong-output-hunt** | **撤销（经验落位）** | 它是「优化过程中的 bugfix 经验」：通用流程 → `accuracy-gate/references/silent-failure-localization.md`（一处）；案例 → `vae-opt/references/troubleshooting-cann-upsample.md`；实测数字出库 |
| **performance-optimization（拆分）** | 保留为**优化域入口（L2）** ＋ 新建 **dit-perf-opt** | 选档与实施内容全部归 `dit-perf-opt`；入口只负责分发 / 最小前置集 / 锚点要求 / 域级验收口径 |
| compilation-dev | **pattern-dev** | 语义收窄为「Inductor pattern matcher 机制开发」；`benchmark-guide.md` 迁 `benchmark-dev/references/` |
| parallelism-strategy | **dit-parallel-opt** | 模块命名统一为 `{对象}[-{维度}]-opt`；收编 offload/TP 与「并行改动没生效」的生效判据 |
| decoder-shard-equivalence | **vae-opt** | 范围扩为「VAE 计算 + 通信同技能」（含 encode，待补） |
| lossless-op-replacement | **accuracy-gate** | 定位改为**精度验收标准**；收编 `quality-gate`（三级验收的第三级） |
| npu-ab-measurement-discipline | **perf-gate** | 定位改为**性能验收标准**（双态 + 入库门）；`report-contract.md` 迁 `model-auto-optimization/references/` |
| framework-feature-enablement ＋ framework-extension-dev | **framework-integration** | 合并为一个技能、内部分两分支（A 已有→使能与验证；B 缺失→补齐开发）——**回退 2026-09 的拆分** |
| （新建） | **host-opt** | host / 辅助段模块：交付搬运 + 装载预热；**不含并发/吞吐** |
| mindiesd-features.md（docs 镜像） | **删除** | 方案 A：特性真源改为 `docs/zh/features/*` + `framework-support-matrix.md`；`refresh_features.py` 一并删除 |

## 4. 预留空缺槽位（待经验回填）

> 槽位编号沿用历史：S2-1 对应 S1 kernel 融合能力清单（阶段并入后编号保留）；S5-1 训练感知案例
> 已于 2026-09-12 回填（见下）。

| 槽位 | 阶段 | 内容 | 现状 |
|------|------|------|------|
| S0-1 | S0 | 各模型/任务权重分区与下载经验（Qwen-Image / Wan2.2 / H3…） | **modelscope 优先 + 分区约定已立；H3 已填，其余待回填**：下载源优先级 = **默认 modelscope**（`modelscope download` / `snapshot_download` + `--local_dir` 直落、国内可达、HF gated 仓库在 modelscope 镜像通常免鉴权）→ **次选 HuggingFace / gated**（需 token，走 `hf`/`huggingface-cli login` 或 `--token`，镜像回退 hf-mirror）；分区约定 = `{model_weight_dir}/{模型名}/{任务变体}`（模型根目录直接 serve）；落位表（模型 / 目录落位 / 任务变体 / 仓库 id / 依据）见 `env-install/references/weights-prep.md` §2.2（源优先级见同文件 §2.1）——**H3 全列已填**（`MiniMax/MiniMax-H3`：根 diffusers + `FL2VA/`、`Ref2VA/` 子分区），Qwen-Image / Wan2.2 / FLUX 的**仓库 id 与任务变体列标 `待回填`**；权重确认纪律（先确认远端已存在、无 `.incomplete`、分片齐全/文件数一致、有校验和则逐文件核对）保留于同文件 §5 |
| S1-1 | S1 | 无抽象接口框架的算子注入方法（多框架泛化） | **已回填（2026-09-13，两阶段法）**：**① API 优先**（runtime 注入——配置注册表 / 框架侧 dispatch / **直接改模型代码 `import mindiesd` 并替换调用点**）→ 验证接口可行 + 同 seed 数值对拍 → **② 再走 compile 机制**做图级适配（pattern 图形态变体 / `_compiled_call_impl` 写入 / 平台注册 backend）。三组判据齐备：「为何 API 先行」（改动面小、失败早暴露、先拿到可归因的单点收益与对拍基线）、「为何 API 不替代 compile」（层内单算子 vs 整链融合 + 图级拷贝消除）、「**何时停在 API 阶段即可**」（eager 已覆盖热路径/compile 无正收益/输出非逐字节/适配落探针 ⇒ 按证据回退，不留半开）；含**收益量级对照**（API vs compile 的相对贡献，各标「本组合观测」，不记绝对耗时）与**各框架注入点差异表**（注册表替换 / 模型层直接改写 / 框架侧 dispatch / `_compiled_call_impl` 原地写入 / 平台注册表 / env 门控 fork）。落点：`framework-integration/SKILL.md` §②「两阶段顺序纪律」+ §运行时算子接入（阶段 1）/ §compile 融入（阶段 2）；**compile 阶段适配动作**落 `pattern-dev/references/fusion-enablement-notes.md` §4（1→7 顺序，与该 SKILL §② 路由成对） |
| S2-1 | **S1**（原 S2，已并入 S1） | mindiesd 融合 kernel 能力清单（接口 → pattern → 验证过框架，唯一真相源） | 分散于各 case，已按新分层回填到**方法单点 + 框架开启方式**（H3×vllm-omni）：`model-auto-optimization/references/lossless-methodology-notes.md` §A/§D —— 融合候选工作流「执行序→候选列表→mindiesd/CANN 能力对照→独立验证」+ eager 已融合热路径判定 + 残余候选池对抵判定 + 量化后融合重审；`framework-integration/references/vllm-omni-enablement.md` §3.1/§5 —— 自动路由面核验（FA/AdaLN/RoPE/GELU eager 覆盖）+ 单步 kernel 采集 hook（`[探针]`）；**2026-09-06 图像第二案例**：同文件 §5.3 kprof 目标扩展 + §3.5 compile 输出非无损否决；实测数字归档于会话产物目录 `{run_results_dir}/archive/`） |
| S3-1 | S3 | NPU 拓扑/带宽矩阵与并行选型决策 | 910B 单点 + 2026-09 增 950PR×4：并行矩阵/ring 不可用/offload 解锁并行与通算掩盖 step_trace 评估（见 `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §7 与 `dit-parallel-opt`「内存受限时的并行解锁」）；2026-09 实测增补：UB/HCCS 岛 vs SYS/PCIe 拓扑、bulk vs head-parallel 翻转、HCCL 带宽 bench 姿势与 set_device 陷阱、端口泄漏/卡组诊断 → 单源参考 `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` + evals 4/5；**2026-09-06 图像案例（见 `ascend-topology-bandwidth-diag.md` §7 + `vllm-omni-enablement.md` §2.2/§3.3）**：UB 岛 0-3/4-7（跨岛 SYS）同岛选卡 + 同岛亦受他户干扰（探活前置）、**并行候选矩阵勿漏 2 卡 USP（图像 20 步短任务 TP1×USP2 优于 TP2，本组合观测）**、短任务 4-rank 病态回退 |
| S3-2 | S3 | few-step 多 rank 验证协议（脚本 + 判据） | **已回填（2026-09-13）**，核心是确立 **DiT-only 口径**——并行对比**只取 DiT 去噪阶段**（`<Pipeline>.diffuse` 阶段墙钟，微秒级、每 rank 一行取 min），**VAE 解码 / 文本编码 / 权重装载 / warmup / 响应编码一律排除**；理由：少步档固定开销占比畸高、且**固定开销自身的波动远大于并行差异**（本组合观测：同配置相邻两次同参请求 DiT 阶段波动个位数百分比，VAE 解码波动数十个百分点）⇒ 计入 decode 量到的其实是 VAE 噪声。协议与判据落 `dit-parallel-opt/references/few-step-multirank-protocol.md`（口径边界与取数 → 矩阵设计「必须含同卡数不同切分的一对」+ 形态是启动级参数/步数是请求级参数 ⇒ 少步与锚点可同 serve 交错 → 噪声纪律 ≥3 rep 取中位、**离散度 >3% 该格不可用** → **外推判据** `r = DiT单步(锚点) ÷ DiT单步(少步)`：`\|r−1\|≤5%` 且两档排序一致才可外推，**排序翻转一律以锚点档为准** → 必回较高步复验的 7 个触发条件 → 通信占比随步数漂移与 profiler 膨胀口径 → 9 条踩坑清单）；SKILL.md 新增「少步 × 多 rank 验证协议」节为摘要并接线；采集脚本 `dit-parallel-opt/scripts/fewstep_multirank_probe.py`（跑矩阵 + 出 DiT-only 证据包，`--parse-only` 零 NPU 复盘）。实测数字归档于会话产物目录 |
| S4-1 | S4 | 精度校验与单特性影响评估方法（端到端质量门禁：定量 + 视觉伪影 + off-identity） | 部分回填：方法见 `accuracy-gate/references/quality-gate.md`，工具与判定标准在仓库 `evals/`（契约/rubric/_template/quality_compare.py + gen_profile/check_profile）；单特性影响矩阵首个真实案例已回填（H3×vllm-omni V1：阈值校准见 quality-gate.md，实测数字归档于会话产物目录 `{run_results_dir}/archive/`；运行时 profile 由 `evals/scripts/gen_profile.py` 生成到 `runs/{task_id}/profiles/`，不入库）；**2026-09-06 图像第二案例**：`accuracy-gate/references/quality-gate.md`「图像第二案例」校准段 + `framework-integration/references/vllm-omni-enablement.md` §3.2–§3.4（21-seed 同 seed 像素对口径；图像质量域 >> 视频 → 阈值不跨域迁移）。**待办（缺什么，未闭环）**：① **视觉门=待判（按需触发）**——两案例均为 `visual_artifact inconclusive`，且**成因须分开标注**：①-1 无图像输入能力 / ①-2 用户未要求判别（本仓默认不主动判、不催促判）/ ①-3 已判但结论不明确（判据见 `accuracy-gate/references/quality-gate.md`「判定要点」三成因）；**判卷材料与待判卷提示已就绪**（同 seed/同帧号并排对比图 + 抽帧蒙太奇 + ASCII 亮度机读存证，落会话产物目录 `{run_results_dir}/s41_visual_gate/`，**不入库**），**待具备图像输入的人或工具**按 `evals/rubrics/visual-artifact-gate.md` **回填 `pass` / `fail` / `inconclusive` 后**才可宣称质量通过；未判时报表质量列照常给定量结论与变化度但**不得写「质量通过」**、须与说明列 `视觉门=待判（用户未要求判别）` 标注同时出现（口径见 `model-auto-optimization/references/overview-report.md` §2/§4）；② **跨模型覆盖缺**——单特性影响矩阵只有 H3×vllm-omni（V1）+ Qwen-Image×vllm-omni（V3）两条 vLLM-Omni 链，LightX2V / DiffSynth-Engine / cache-dit 列无单特性质量矩阵；③ **图像域阈值未固化进 profile 模板**——图像档阈值（同 seed 像素对口径）仍写在 quality-gate.md 文字里，尚未落到 `gen_profile.py` 的 profile 模板字段（跨域阈值不可迁移，缺模板字段就会默认套视频阈值） |
| S4-2 | S4 | 组合试验设计与层回退策略 | 已回填协议 + **seam 表首案例已校准**（2026-09-05 H3×vllm-omni V1：precision×attention 跨 seam 可叠、attention 同 seam 取最强档、cache×稀疏同 step 窗口可叠、稀疏档质量非线性、frontier 以同窗相邻对为准；见 combination-search.md 校准节；该案例实测数字归档于会话产物目录 `{run_results_dir}/archive/`） |
| S5-1 | S5 | 训练感知案例（少步蒸馏 4/8 步 + VAE解码替换 + SLA/QAT 叠加；蒸馏权重 modelscope 下载；质量-速度权衡口径） | **已回填（2026-09-12）**，且按「**方法与产物隔离**」拆两件：① **通用方法** `framework-integration/references/train-aware-lossy-method.md`（分类与归组判据 / 三条前置契约 / 协同定位比值法 / 质量分层「接口正确性→画质档位」/ 归因链 / **数字纪律：不写绝对耗时与绝对质量分值，大致加速比与质量变化度照写**）；② **框架差异** `references/vllm-omni-train-aware-enablement.md`（vLLM-Omni 开启方式 + 该模型侧契约 + 并行负载前提 + 产物坐标指针） |

## 5. 快速开始

```text
# 模型/框架自动优化任务（编排层）
加载 model-auto-optimization → 先 Read workflows/optimization-flow.md → 建 run-state
→ 按 S0/S1/S3/S4（/S5 可选 / S6 条件进入）路由到能力技能 → 每阶段 stage_gate 门禁 → 闭环双报表

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
    ├── model-auto-optimization/                # 编排层 · 模型自动优化流程（L1）
    │   ├── workflows/optimization-flow.md       # 阶段执行模板（确认点/验收 gate/闭环强制交付）
    │   ├── workflows/references/dispatch-templates.md  # 角色派发/自验证回执格式
    │   ├── references/run-state.md              # 运行状态文件规范（推进表 + 候选迭代表单一真相源）
    │   ├── references/bottleneck-labels.md      # 瓶颈点标签表 + 非 DiT 段 10% 门限（域入口的唯一输入）
    │   ├── references/report-contract.md        # 报表列契约（L1 单点；自 perf-gate 迁入）
    │   ├── references/overview-report.md detail-report.md  # 双报表规范
    │   └── scripts/stage_gate.py report_lint.py # 阶段门禁（含 S6）/ 报表结构 lint
    ├── dev-workflow/                           # 编排层 · 仓库开发流程（L1）
    ├── performance-optimization/               # 优化域入口（L2，可独立触发；只分发不选档）
    │   └── references/dispatch-table.md         # 瓶颈标签 → 四模块 的路由表
    ├── dit-perf-opt/                           # 模块 · DiT 计算（特性选档与实施）
    │   ├── references/optimization-dimensions.md  # 标签 → 方案映射（阈值只引用 analyze）
    │   └── references/combination-search.md       # 组合 / seam / 回退
    ├── dit-parallel-opt/                       # 模块 · DiT 通信（并行选型 + 生效判定）
    │   ├── references/comm-masking-method.md    # 掩盖上限三量模型 / 分块 recipe / 生效判据
    │   ├── references/scope-effectiveness-check.md  # 并行改动是否真的生效（自 PSA 迁入）
    │   ├── references/ascend-parallel-traps.md  # 并行侧静默陷阱清单（自 PSA 迁入）
    │   ├── references/parallel-plan-attribution-method.md  # 阶段 Δ 分解 / 集合通信归属 / 通信量下限
    │   ├── references/parallel-form-selection-method.md    # 形态抉择七关 / 条件化结论 / 重判触发
    │   ├── references/cp-sparse-combination-method.md      # 并行 × 稀疏 seam 四条契约 / 验收 / 失效模式
    │   └── scripts/mask_bound_calc.py collective_attribution.py bubble_attribution.py
    ├── vae-opt/                                # 模块 · VAE/TAE 解码段（计算 + 通信同技能）
    │   ├── references/troubleshooting-cann-upsample.md  # 该对象的缺陷判定实验与规避
    │   └── references/{independence-proof,sliced-state-decode,verification-and-budget,parallel-scope-effectiveness}.md
    ├── host-opt/                               # 模块 · host/辅助段（交付搬运 + 装载预热）
    │   └── references/host-overhead-account.md  # 固定开销账口径与常见错账
    ├── pattern-dev/                            # 能力供给 · Inductor pattern matcher 机制
    ├── operator-dev/                           # 能力供给 · 算子开发（路由外部 cannbot-skills）
    ├── aclgraph-dev/                           # 能力供给 · 图批量下发
    ├── quantization-dev/                       # 能力供给 · 量化契约与位级对齐
    │   └── references/contract-reverse-engineering.md # 契约逆向六步法
    ├── dummy-run/                              # 能力供给 · 配套验收载体（快验，非通用验证）
    ├── framework-integration/                  # 能力供给 · 三方框架特性落地（两分支：使能 / 补齐）
    │   ├── references/framework-support-matrix.md  # 框架 × 特性支持状态（支持状态单点）
    │   ├── references/{vllm-omni,lightx2v,diffsynth-engine}-enablement.md  # 框架差异真源
    │   └── scripts/ascii_luma_preview.py        # 无图像输入时的解码输出亮度预览存证
    ├── accuracy-gate/                          # 标准与方法 · 精度验收标准（含 quality-gate 第三级）
    │   ├── references/quality-gate.md           # 三级验收的第三级（有损档）
    │   ├── references/silent-failure-localization.md  # 判据不达标时的排障流程（一处）
    │   └── references/{bit-exact-harness,equivalence-criteria}.md
    ├── perf-gate/                              # 标准与方法 · 性能验收标准（入库门 + 双态）
    │   ├── references/window-ab-protocol.md      # 同窗 A/B 与 A/B/A 漂移校正
    │   ├── references/measurement-discipline.md  # ABBA / 敏感度地板 / 假数字三证 / 上报模板
    │   └── references/evidence-toolbox.md        # 三次证明 / md5 盲区 / 忠实度对照
    ├── profiling-collect/ profiling-analyze/   # 采集分析 · 统一采集 / 统一分析
    ├── benchmark-dev/                          # 采集分析 · 算子实现级基准（含 timing 方法论单点）
    │   └── references/benchmark-guide.md        # 计时方法论（自 pattern-dev 迁入）
    ├── env-install/ remote-access/             # 环境与运行通道
    └── code-standards/ markdown-lint/ mindie-sd-community-governance/  # 规范
```

## 7. 贡献

### 技能准入与命名判据（新增技能前必读）

**准入判据**（四条全满足才允许新建技能）：

1. **独立耦合结构 / 几何契约**：换模型即失效的结构（如潜帧边界、状态前缀依赖）——通用手段不算。
2. **独立失败模式**：有只属于它的静默失败（状态携带、设备缺陷、交付链路语义等）。
3. **跨 ≥2 种手段**：单手段的应并入该手段技能。
4. **方法论可由既有技能引用而不复制**：否则只是把重复搬家。

**角色判据**（手段 / 对象 / 判定）：

- 技能按 **手段（How）/ 判定（Judge）** 划分；**对象**（DiT / VAE / host / AR）可作技能内部的章节维度，也可独立成对象技能（须过上面四条；AR 目前仅占分类位）。
- **针对某类动作的第三方判定 → 不独立成技能**（并入该动作技能，例：并行作用域生效判定并入 `dit-parallel-opt` / `vae-opt`）。
- **跨动作共享的方法与标准 → 保留为 L3**（例：`perf-gate` / `accuracy-gate`），并由编排层验收步骤**强制引用**（否则会退化成"引用靠自愿"的孤儿技能）。

**命名（角色词集合，新增技能不得另造）**：`-opt`（优化模块）· `-gate`（验收标准）· `-collect` / `-analyze`（观测动作）· `-dev`（开发）· `-lint` / `-standards`（规范）；**入口用全称**，其余用名词短语。
命名例外（须在此标注理由）：`markdown-lint`（markdownlint 是业界工具名）· `mindie-sd-community-governance`（跨仓治理规范需自识别）。

- 新增/改动 skill 前读本 README（三层架构分层与变更门禁，见「架构速览」与 §1–§7）；
  参考 [Anthropic skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)。
- 每个 skill 均含 `evals/evals.json`（≥2 条 prompt + expectations）。
- **变更门禁**：① 决定归属——能力技能不带流程语境（不写"主轴/侧轨"），新流程 = 新增编排层 skill；
  ② 同步本 README（对应层/域标签/目录树/计数）；③ 路由声明成对（"由 X 调用"与路由方指向互为反向）；
  ④ evals 增补；⑤ 复盘（dev-workflow §6）新经验先判断归属槽位/能力域再回填。

### case 回填规范（实验 → skill 刷新）

> 标题沿用旧称。**`-case.md` 政策（2026-09-14 重新裁定）**：案例类文件**可保留但受限**——必须在所属技能
> 的 Reference Files 登记并标注「**不作为推荐加载入口**」，且**实测数字按数字纪律一律出库**
> （`{run_results_dir}/archive/`，skill 内只留判据）。**回填仍优先按下述四分类**（新建 `-case.md` 非首选）。

实验/复盘产出可复用经验时，把经验按下列规范沉淀为能力技能的**四分类落点**
（`{主题}-method.md` / `{framework}-{主题}-enablement.md` / `-notes.md` / 实测记录→会话产物目录），
保证每次刷新不破坏三层架构分层与 skill-creator 要求：

**经验 vs 探针 判定（先于归属与沉淀）**：先用三问分类——
① 是否一次性/未正式合入（本机补丁、fork 钉版、.bak 默认关、用后即弃载体）？
② 是否只用于诊断/验证（绕法试探、临时开关、本地 microbench 探测、未上线载体）？
③ 是否依赖特定临时环境/窗口（版本漂移前有效、仅本案例可复现）？
任一为「是」→ 该内容按**探针**处理：可在 case/notes 内以 `[探针]` 标注并存档
（用途 / 适用窗口 / 结论边界），**不进「推荐姿势 / 方法论 / 经验槽位 / 报表宣称」**；
三者皆「否」→ 才按经验/方法论沉淀。例外：由探针**发现的 durable 约束事实**
（CANN 行为、部署顺序要求、命名/默认值纪律、框架支持状态）属经验（可注明"来源探针"）。

1. **归属判定（先于书写）**：先判断经验属于哪个能力域（dit-perf-opt / dit-parallel-opt / vae-opt / host-opt / framework-integration …）或预留槽位（§4）；编排层跨阶段纪律才进
   model-auto-optimization。不新建技能，除非触发「新增 Skill 规范」（dev-workflow）。
2. **落位四分类（按有效性范围选一类或多类）**：先判断经验「换框架是否成立、
   换模型是否成立」，再决定落到哪几件（同一批经验常需**同时**写方法与开启方式两件）：
   - **域级方法** `{主题}-method.md`（换框架、换模型仍成立）：标题 `方法：{主题}`；正文按
     「分类/术语 → 判据与流程 → 契约与前置 → 判定法 → 归因链 → 声明纪律 → 索引」组织；
     只写**可迁移的判据与流程**，**不含产品名**；数字按「数字纪律」办——**不写绝对耗时与绝对
     质量分值**，**要写**大致加速比（量级/约数）与质量变化度（降幅/差值）。
   - **框架开启方式** `{framework}-{主题}-enablement.md`（仅该框架成立）：标题
     `{框架}：{主题}的开启方式（框架差异记录）`；正文按「开启姿势（命令/开关）→ 前置与版本边界 →
     日志与计数契约 → 该框架的坑（`[探针]` 标注）→ 并行与负载前提 → 产物坐标指针」组织。
   - **平台/模型底座** `-notes.md`（换框架仍成立、换模型不成立）：模型结构、几何契约、平台拓扑。
   - **实测记录** → 会话产物目录（`runs/`、报表、`evidence.json`），**不入 skills**；迁移既有内容时
     **先导出归档再删**。
   通用纪律：一次只归因一个变量；数字带口径（同拓扑/同 seed/steady 次数）；单次结论不迁移；
   端到端验证附质量门禁结论与阈值回填（运行 profile 由 `evals/scripts/gen_profile.py` 生成到
   `runs/{task_id}/profiles/{model}.toml`，不入库；close 前 `check_profile.py` 强校验）。
   - **文件命名规则（统一）**：kebab-case；`-method`=**通用方法（与具体产物隔离）**、
     `-enablement`=**某框架/对象的开启方式（框架差异）**、`-notes`=平台/模型底座与方法论沉淀、
     `-pattern`=接入姿势、`-gate`=判定标准、`-guide/-checklist/-catalog`=流程/清单；
     故障排查类统一 `troubleshooting-{对象}.md`（对象前置），且**同一对象的排查流程只留一处**——
     其它技能只写指针、禁止复制判据与阈值；reference 一律连字符、不用下划线。
     **边界（勿读成穷尽白名单）**：上述四分类后缀是**案例/经验回填产物**（`{主题}-method.md` /
     `{framework}-{主题}-enablement.md` / `-notes.md` 及 `-case.md` 迁移件）的命名规则；
     **其余 references 只要 kebab-case + 自解释即可，不强行套后缀**——机制/运行态类
     （`run-state.md`、`artifact-layout.md`、`manifest-schema.md`）、方法/启发式类
     （`heuristics.md`、`analysis-flow.md`、`performance-analysis-methodology.md`）、
     矩阵/清单/协议类（`framework-support-matrix.md`、`capability-matrix.md`、
     `few-step-multirank-protocol.md`）、故障诊断类（`ascend-topology-bandwidth-diag.md`）、
     框架环境类（`vllm-omni-build.md`、`lightx2v-env.md`、`weights-prep.md`）等一律按「名即其义」
     命名，**不为套后缀改名**（`gate-check-rules.md` 亦保留）；与同目录不撞义即可，硬约束只有
     **kebab-case + 连字符（不用下划线）**。
     **`-case.md`（受限保留，2026-09-14 重新裁定）**：优先按上面四类落位；确需保留案例文件时，
     必须在技能 Reference Files 登记并标注「不作为推荐加载入口」，且**实测数字一律出库**
     （`{run_results_dir}/archive/`），skill 内只留判据。
   - **方法与产物隔离（强制）**：把「**跨框架、跨模型可复用的方法**」与「**具体产物**（某模型 /
     某框架 / 某权重）」分开写，按**有效性范围**落四类——**域级方法** `{主题}-method.md`（换框架、
     换模型仍成立）/ **框架开启方式** `{framework}-{主题}-enablement.md`（仅该框架成立）/
     **平台·模型底座** `-notes.md`（换框架仍成立）/ **实测记录** → 会话产物目录（只本次环境成立）。
     理由：**特性分类跨框架相似，但开启方式按框架不同**。
   - **数字纪律（强制）**：**看趋势与量级，不看绝对值**——
     - **不写入**：**绝对耗时**（单次总耗时 / 每请求耗时 / 每步耗时的**实测绝对值**）与**绝对质量分值**
       （**PSNR / SSIM 的实测绝对值**）——只在本次环境成立，写进 skills 会被误读为该组合的预期值；
     - **要写入**：**大致加速比**（量级/约数，如「约 3 倍」「两位数倍」「一个数量级」）与
       **质量变化度**（相对基线的**差值/降幅**，如「SSIM 降约 0.2」「PSNR 降约 4 dB」「质量基本无感」）
       —— 这两类才是跨组合可读的信息；
     - **比例关系**（案例说明的核心）：单点收益**排序**、独立性 / 协同强度（如「≈1 独立可连乘 /
       某两维呈正协同」）、组合能否由「两两 × 剩余单点」**完全解释**、某阶段是否为端到端节省的
       **全部来源**、收益随规模**线性还是反转**；
     - **方向选择**：采纳哪一档、为何、回退了什么、资源下一步投向哪里——**并标注「本组合观测」**；
     - **判定以最新数据为准**：策略选择与方向结论**每次按最近一次实测重判**（同「方向动态优选」），
       案例里的历史结论只作参照，**不得替代本任务复测**。
     - **豁免**：`model-auto-optimization/references/effort-estimation.md` 是**耗时/成本估计模型**，其预算与耗时样本属**规划输入**（非性能宣称），可保留；但其中引用的**模型实测加速比**仍须去掉。
     - **同属豁免**：**报告模板与校验夹具**（如 `overview-report.md` 的示例行、`report_lint_cases.md`、`compile-ab-report-template.md`）中为演示/断言所必需的数字。这两类**不清数字**，但引用它们时须标明其性质（模板示例 / 断言夹具），**不得当作本仓实测收益引用**。
   - **数字必带环境作用域（强制 · 防跨环境照抄）**：**写进 skill 的每个数字、阈值与「最佳参数」都必须带上它的
     作用域**——它们是**一次「硬件 + 拓扑 + 软件栈 + 模型」组合的快照**，不是普适事实：
     - **来源环境必写**：skill 须写明其来源环境（芯片/代际、卡数与拓扑、关键版本、模型），并**明确要求读者
       本地重测**；无作用域的数字会被直接抄到别的硬件上，**这是已知失败模式**，须按此拦截；
     - **默认择优而非照抄**：套用任何已记录参数时，默认**枚举 ≥2 个候选 → 同口径实测 → 先比正确性、再比
       性能 → 取最优**；**数值错误的候选无论多快一律淘汰**（实测实例中「最快」的 tile 配置正确性不达标，
       已被否决）；
     - **结论同样带作用域**：「X 更快 / X 已被证伪」也**依赖环境**，换环境**至少复测一次**再沿用。
   - **记录的结论有寿命（强制 · 防固化已修复问题）**：**某个版本需要绕行的问题，下个版本可能已经是正常功能** ——
     只记录问题而不写「如何判定它仍存在」，会**主动误导**后续 agent：既让它做无谓的绕行工作，
     更糟的是**阻止它使用已经修好的原生路径**（「必须绕行 X」会被当成永久禁令）：
     - **必写复核方法（强制）**：每条问题 / 陷阱都要写清「**如何判定它仍存在**」（一条探针 / 一条 `grep` /
       一个最小复现），并明确要求读者**先复核、再套用绕行**；无复核方法的条目不予回填；
     - **优先记录判定方法，其次才是绕行方案** —— 判定方法换版本后仍然有用，**绕行方案会作废**；
     - **升级软件栈 / 硬件后逐条复核，不再复现的必须删除**，不要留「以防万一」；留着已修复的条目
       只会制造假结论与无谓绕行（与「方向动态优选」「数字必带环境作用域」同源）；
     - **禁止把一次性缺陷固化成结构**：绕行一律走**可一键关闭、可整体删除**的开关，
       不长期分裂代码路径、不写兼容 shim，以便确认修复后**整块删掉**。
   - **skill 只承载知识与流程（强制 · 防混入交接内容）**：skill 面向**未来在新模型 / 新框架上复用**，
     不是项目进展记录。**状态、进度、证据清单、校验和、备份数量、私有路径、以及「某次会话里我……」
     式的叙述，一律不进 skill** —— 这些归 `HANDOFF_*.md`（交接文档）：
     - **该进 skill**：机制与根因、契约与格式、公式、判据、流程与检查单、错误码表、**复核触发**；
     - **不该进 skill**：`已集成 / 未集成 / 已停用 / 待收尾 / 未闭环` 等状态；md5 等校验和；
       文件清单与证据树；备份个数；容器名、主机名与私有绝对路径；会话叙述与顺序性回顾；
     - **数字可以进，但只能作为「示例 / 量级参照」**，且须带环境作用域（见上条），
       **不得写成「我们项目的结果」**；
     - **判断标准一句话**：「**换一个新模型 / 新框架，这条还成立吗？**」—— 成立 ⇒ 留；
       只在描述「这个项目现在到哪一步」⇒ 拿走；
     - **界限（防过度清理）**：本仓特有的文件名 / 门控名**可保留**（属定位信息），但**不带行号、
       不带「当前已改到第几版」**；`[探针]` / 默认关 / `.bak` 作为**流程约定**保留，
       但不记录其当前数量。
   - **方向动态优选（强制 · 防路径固化）**：**同一方法在不同框架下由不同能力实体承载（融合 op /
     eager 路径 / 框架自带开关），收益因此不同，优选方向也随之不同** ⇒ **禁止把某个模型的优化路径
     固化成 skills 里的固定路线**（如「长视频优先投稀疏、其次缓存」只能作**该组合观测**记录，
     不得作为其他框架 / 模型 / 规模的执行序）。技能只固定**判据与流程**：本任务口径下测单点 →
     比值法定协同 → 结合能力面（`framework-support-matrix.md`）与本任务实测收益**在任务内定方向**；
     换框架 / 换模型 / 换负载规模 = **重判**。
   - **报表规范 ↔ 产物镜像命名**：编排层报表规范用 `-report.md`（`overview-report.md` /
      `detail-report.md`），与运行时产物 `overview_report.md` / `detail_report.md` 词根镜像——
      规范文件（references 内）连字符，产物（runs/ 下，stage_gate 强制 basename）下划线；
      二者不同目录、同名不同分隔符是刻意设计，勿合并。
   - **workflow 局部模板例外**：`workflows/references/` 只放该 workflow 专属派发模板
      （`dispatch-templates.md`），不归技能顶层 `references/`；通用 reference 一律在顶层
      `references/`，避免与跨 workflow 机制混淆。
3. **接线**：在所属 SKILL.md 的 Reference Files 登记 + 加载时机；作某技能方法论配套案例时，
   在引用方一并注明（路由声明成对）。**涉及框架特性支持的更新，须同步刷新
   `framework-integration/references/framework-support-matrix.md`（按该文件 §刷新协议）。**
4. **evals 增补**：行为/流程变化的，同步所属技能 `evals/evals.json`（schema 见**外部** skill-creator 仓的
   `references/schemas.md`），expectations 用可验证陈述。
5. **隐私清理**：真实 IP / 主机 / 容器名 / 用户名 / 内部路径 → 占位符（`{model_weight_dir}`、
   `{env_host}`、`{run_results_dir}` 等）；提交前敏感模式 `git grep` 复扫 0 命中。
6. **校验**：markdownlint-cli v0.44.0（仓库 pin）0 违规；evals JSON 解析通过；含脚本跑 ruff + py_compile。
7. **变更门禁复核**：README §2/§4/§6（计数/槽位/目录树）与相关 description 定位句是否需要同步。

### 当前开发状态

- ✅ **23 个技能**：编排层 2（L1）＋ 优化域入口 1（`performance-optimization`，L2，可独立触发）＋ 能力层 20（L3）
  —— 分类见 §2（域入口 1 + 模块 4 + 能力供给 6 + 标准与方法 2 + 采集分析 3 + 环境通道 2 + 规范 3）
- ⏳ 8 个预留槽位：S0-1 已立 modelscope 优先 + 分区约定（H3 已填，其余待回填）、S1-1 已回填两阶段法、
  S3-1 已回填拓扑/带宽单源参考与 evals 4/5、S4-1 部分回填（质量门禁，缺项见 §4 该行）、S4-2 已回填协议 + **seam 表首案例已校准**（见 §4 该行，与本节旧措辞取后者为准）、
  **S5-1 已回填（2026-09-12 训练感知案例）**，其余待回填（见 §4）

> **命名约定**：以下历史条目保留**当时的技能名**（如 `compilation-dev` / `parallelism-strategy` /
> `framework-feature-enablement` / `lossless-op-replacement` / `npu-ab-measurement-discipline`、
> 已撤销的 `parallel-scope-diagnosis` / `npu-silent-wrong-output-hunt`）——改名与撤销对照见 §3，
> 不再逐条回改（改历史条目会掩盖当时的真实状态）。

### 最近经验合并（2026-09 MiniMax-H3×LightX2V 会话蒸馏，已按隐私规则脱敏）

- 本次把 2026-09 批会话知识按 skill-creator 规范合并进现有 5 个技能（无新增技能）：
  - `parallelism-strategy`：SKILL.md 更新触发/新增 950PR 拓扑段；新参考 `references/ascend-topology-bandwidth-diag.md`（UB vs SYS、bulk vs head-parallel 翻转、HCCL bench 姿势、卡组诊断）；evals 增 4/5
  - `framework-feature-enablement`：case 追加 §10（2026-09 增补：S4 稀疏/量化结论 + 无损试验矩阵摘要）
  - `profiling-analyze`：方法论补 clean-window/融合阶梯/环境劣化判读
  - `env-install`：troubleshooting 补 §F 多卡运行期劣化诊断树
  - `dev-workflow`：rework-lessons 增 #35（远程多卡实验工具纪律）
- 隐私规则：上述内容不含主机 IP / 口令 / 用户名 / 容器名；实验脚本与原始日志保留在会话产物目录，不写入 skills

### 最近经验合并 2（2026-09 H3×LightX2V 优化收口）

- **收敛复核（2026-09）**：卡组可用性须实测（npu-smi OK ≠ 可用；驱动损伤残留表现为整组**每步时长均匀
  抬高一个量级**）；masking 已实测否决（每步理论上限仅个位数占比）→ parallelism-strategy
  reference §2/§4 补注 + rework-lessons #36
- **跨框架差距/对比**：768p 同分辨率对照 vllm-omni——**单点降幅排序 Cache > 稀疏 mix > 量化**，
  最优组合更强；LightX2V 单步同级、步数裁剪拉平；量化缺口双 seam
  （线性层 + 稀疏）清单落 `lightx2v-enablement.md`（+ matrix「跨框架待补充能力清单」）与框架仓内
  会话过程文档（非本仓、不入库）；**绝对耗时与加速比见归档** `{run_results_dir}/archive/lightx2v-mindiesd-case.md`
- **线性量化打通（原生 scheme）**：LightX2V 权重树为自研 `MMWeight`（非 nn.Linear）→ 注册原生
  `dit_quant_scheme="npu-w8a8-mxfp8"`（内部直连 mindiesd `npu_quant_matmul`），model.py 门放行
  online scheme；实测**步时明显下降（2× 复现）、帧 SSIM 属近无损档** → 最终部署配置（原生
  w8a8-mxfp8 档 + 24 步；配置文件名以归档为准）；matrix W8A8_MXFP8 格 ❌→🟡（读数见归档
  `{run_results_dir}/archive/lightx2v-mindiesd-case.md`）
- **FA 量化 A5 可用（认知修正）**：mindiesd FP8/MXFP8 FA（`npu_fused_infer_attention_score_v2`
  fp8）kernel 级微测通过（量化级精度）→ 「仅 A2」不成立；matrix FA 行 + `lightx2v-enablement.md`
  §3.5 更新
- 载体：`framework-integration/references/lightx2v-enablement.md`（§3.3–§3.5 采纳项与
  落地状态）、`framework-support-matrix.md`（量化两格 + FA 行）；实测数字归档于
  `{run_results_dir}/archive/lightx2v-mindiesd-case.md`；会话证据见框架仓内会话过程文档
  （非本仓、不入库）——本 README 只保留载体指针与结论，不再登记其路径与编号；
  绝对数字与过程细节一律以归档与该文档为准
- **稀疏 FA 质量结论更正（重要）**：LightX2V rf3（与 vllm-omni 同 mindiesd rf_v3 路径）eager
  质量梯度**平滑单调**（同 seed / 同帧门禁 vs dense）——
  历史「某稀疏区间质量平台化」**证伪（早期批口径混淆，证据作废）**；compile×rf3 Dynamo
  trace 错误待解；vllm RAINFUSION 几何契约（tail/prefix）非质量必需。经验教训：稀疏质量门禁
  必须同 seed/同配置/同窗（详见归档 §10 与 matrix rf_v3 格）

### 最近经验合并 3（2026-09 MindIE-SD FFN 融合会话：operator-dev / compilation-dev 收敛，已脱敏）

- `operator-dev`：catlass 只读融合算子全链（vendored 头 → standalone 对拍 → 集成 → compile
  GraphPatternEntry 真图使能 → 开关治理）沉淀为 2 个新 references + SKILL 接线；**三约束落档**：
  算子开发必须加载 cannbot（与本仓经验并行使用、非互斥）、融合 DSL 分界（CV 含 matmul → catlass；
  VV 纯 vector → triton）；审查收敛（位级仅 h3 特例、layout 记法等价注、工作流与案例去重）；evals 同步
- `compilation-dev`：提炼（skill-creator 合规）：数值口径"位级 vs fp8 量化级"、GraphPatternEntry
  双案例（h3 / mm_gelu）、registration-checklist 补「路径 B：GraphPatternEntry」与默认值纪律
  （norm_rope 教训：不留默认 False 死代码）、pattern-templates 修 qwen rope 状态并去重 §4；evals 增补
- 产品代码（未提交，另组产品 MR）：`mm_swiglu_mxquant`/`mm_gelu_mxquant`（catlass 融合算子 +
  GraphPatternEntry 真图使能；单点收益排序 FLUX > Qwen > Wan，约 6% / 约 4% / 约 1%（Wan 接近零）——本组合观测）；`enable_minimax_h3_norm_rope`
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
- `framework-integration/references/framework-support-matrix.md`：首部报表映射补 f8 实现
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
  framework-integration 结构性实现，先估成本 + §0 确认）/ `上界`（**预留空间，定义待人工
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

- `framework-feature-enablement`：`references/cache-dit-enablement.md`（原 `-case.md` 已按新规迁移，
  实测数字归档于 `{run_results_dir}/archive/cache-dit-minimax-h3-case.md`）固化补测轮经验——
  ① 稀疏 FA **两条路径区分**（eager rf_v3 外部逐层 mask：近无损但收益有限；EagleQBSA
  融合 op 稀疏FA+FA量化：op 级收益显著、本链未接线）；② **mindiesd 自研算子部署坑**（`ASCEND_CUSTOM_OPP_PATH`
  由 `import mindiesd` 前置设置，先 import 再建 NPU 张量，否则 aclnn `inferShape does not exist`）；
  ③ mask 开销拆解（选择逻辑可忽略、rearrange/pool 搬运为主且随层数线性放大）；④ **叠加组合实测**
  （稀疏在 Cache 之上仍有效、start0 质量劣化）——性能表述仅留方向与比例口径；SKILL.md 案例行同步
- `model-auto-optimization`：`effort-estimation.md` §1c 补组合批成本口径（8 serve≈4 卡×1.2h、质量对
  预算公式）
- **通用方法论再提炼（跨模型可复用，按域排布）**：`env-install/references/troubleshooting-env.md` 新增
  §G（mindiesd 自研 CANN 算子部署/`import mindiesd` 前置 `ASCEND_CUSTOM_OPP_PATH`、gloo
  `ss_family 10 vs 2` 偶发强制 IPv4）；`profiling-analyze/references/heuristics.md` 新增
  「Host/Kernel 与数据搬运归因」（step_trace Free≈0→device-bound、per-op 时长聚合区分计算 vs 搬运、
  单调用微基准 vs 模型步级矛盾核对项）；`framework-integration/SKILL.md` ③ 补「自研算子部署侧」
  根因 + 三层证据段补「稀疏/融合路径区分先于收益判定（eager 外部 mask vs 融合 op，op 级微基准先行）」
- 边界：仅改动 cache-dit 相关内容，未触碰其他框架（matrix/vllm-omni 0.28/qwen/lightx2v 等）内容

### 最近经验合并 5（2026-09-12 MiniMax-H3×vLLM-Omni 0.28 训练感知有损优化会话蒸馏，已脱敏）

- **新 case（S5-1 槽位回填），按「方法与产物隔离」拆两件**：
  ① **通用方法** `framework-integration/references/train-aware-lossy-method.md` ——
  换入外部训练权重换速度并承担质量代价（少步蒸馏 / VAE解码替换 / 可训练稀疏 / QAT）：
  **跨框架、跨模型复用，不含模型/框架/权重产品名**（数字口径见 §7「数字纪律」，该条随后更新）；
  ② **框架差异** `references/vllm-omni-train-aware-enablement.md` —— 该框架的开启方式（命令/开关/
  日志契约）、该模型侧契约、并行负载前提、产物坐标指针。**特性分类跨框架相似、开启方式按框架分列**，
  故差异内容独立成文，不混进通用方法
- **新增通用声明纪律：绝对数字不写入 skills**（`train-aware-lossy-method.md` §6）——比例是
  「模型 × 框架能力 × 负载规模」的函数，跨模型/框架不可迁移，写进 skills 会被误读为预期值；
  对外只给相对结论与机制（谁主导、谁与谁协同、随什么变化），实测数字留在会话产物目录
  （⚠️ 2026-09-13 数字纪律更新：该条现为「**不写绝对耗时与绝对质量分值；大致加速比（量级/约数）
  与质量变化度（降幅/差值）照写**」，见 §7「数字纪律（强制）」）
- `model-auto-optimization`：S5 路由行由「预留」改为已回填（支撑技能 + 验收点含**两条并列部署建议**）；
  评估纪律「训练感知（S5）」段补**三条前置契约**（步数语义 = 评估次数 ≠ 区间数 /
  装载计数防 no-op 假加速 / 换组件前先读参考实现的判定契约）、**S4 seam 语义已界定**（训练感知档可叠
  免训练有损档；少步档 Cache 可能整体失效，判据 = 开/关产物 md5 字节级等价）、预览级组件质量三项
  （灰度相关 / 钳位占比 / 输出 std）
- `model-auto-optimization/references/overview-report.md`：§2.1 固定名词表**训练感知组增 `VAE解码替换`**
  （归组判据 = 是否换入外部训练权重；具体 checkpoint/结构布局属实现入说明列）——**白名单变更已获用户
  确认**；§2.3c 补「训练感知组必须给保画质档 + 预览档两条并列建议」；§4 补 VAE解码替换行质量列三项口径
- `framework-integration/SKILL.md`：案例表登记上述两件（通用方法 + 框架差异）；description 定位句
  补 S5 调用方；`framework-support-matrix.md` 增证据码 V4 与「训练感知」表（分类跨框架一致 / 开启方式按列）
- **跨侧实现（探针）**：VAE解码替换为框架未提供能力 → 在框架源码树内新增解码器模块 + pipeline 分派，
  未合入上游 / `.bak` 保留 / 默认关，按「经验 vs 探针」标 `[探针]`；由它发现的 durable 契约事实
  （帧数契约、latent 约定、参考实现判定契约）按经验沉淀
- 新增工具：`framework-feature-enablement/scripts/ascii_luma_preview.py`（零依赖；执行侧无图像输入时
  把解码输出落成 ASCII 亮度图 + 聚合指标，补视觉判卷 inconclusive 的存证缺口）
- 校验：markdownlint 0 违规；evals/declarations JSON 解析通过；新脚本 ruff + py_compile 通过；
  敏感模式 `git grep` 0 命中

### 最近经验合并 6（2026-09-12 批 3 收尾：case 全量迁移 + 数字纪律巡检 + 悬空引用修复，已脱敏）

- **DiffSynth-Engine 两件合一**：原 `diffsynth-engine-case.md` + `diffsynth-engine-notes.md`（两件均已
  `git rm`）→ **`references/diffsynth-engine-enablement.md`**（画像与版本边界 / 部署与启动前置
  （`--no-deps` + `SETUPTOOLS_SCM_PRETEND_VERSION`）/ 特性开关面板（`compile_backend="mindie"` +
  `_compiled_call_impl` **原地写入** + backend 实例复用；RoPE 实数域改写与 text encoder key
  归一化两处命中前置；**三层证据**使能判断）/ 与其它框架差异照搬清单 / `[探针]` 回修与三个陷阱
  （日志 2048 截断 · **图命中≠运行期全部生效** · 不做耗时比较）+ `residual_gate_add` 4D fallback +
  **attention 进图中性回退** / 产物坐标指针）；两个源文件 `git rm`，绝对数字原文归档
  `{run_results_dir}/archive/diffsynth-engine-case.md` 与 `…-notes.md`（后者**无绝对性能数字**，
  仅结构条目留档）
- **算子融合样例并入方法单点**：原 `mmgelu-flux-wan-qwen-case.md`（已 `git rm`）→
  `operator-dev/references/mindiesd-fusion-notes.md` **§7**（语义与真实图链 / kernel 要点（gelu 恒等式、mx 量化尾
  复用、bias 实测全 0 与装载 API 坑、判别法）/ 使能载体形态 / 工程坑速查）；**单点收益排序
  （FLUX（约 6%）> Qwen（约 4%）> Wan（约 1%，接近零），本组合观测）**入 `pattern-dev/references/fusion-enablement-notes.md`
  §3.6；绝对数字归档 `{run_results_dir}/archive/mmgelu-flux-wan-qwen-case.md`
- **`-notes.md` 保留但去数字**：`dummy-run/references/minimax-h3-notes.md` 按「平台·模型底座」**保留**
  （换框架仍成立、换模型不成立），绝对耗时 / 加速比 / 峰值显存 / 质量匹配率移入
  `{run_results_dir}/archive/minimax-h3-notes-numbers.md`；正文只留**比例关系**（单点排序、协同、
  占比、随规模线性/超线性）、**判定阈值**、**结构契约**（帧数 `17n+5`、参数量、kernel 计数契约）
  与**方向**（各标「本组合观测」）
- **数字纪律巡检**：`rework-lessons.md`（故障指纹改「**量级 + 指纹**」）、`benefit-rootcause-guide.md`、
  `cache-enablement-pattern.md` §4、`dummy-run/SKILL.md`、`dit-parallel-opt/SKILL.md` §9
  通信掩盖表、`pattern-dev-notes.md`、`graph-pattern-rewrite-guide.md`、`profiling-analyze/SKILL.md`、
  `optimization-dimensions.md`、`troubleshooting-vllm-omni.md`、`troubleshooting-env.md`、
  `copy-elimination-guide.md`、`benchmark-guide.md` → 绝对耗时/加速比改**量级与方向**，
  **占比 / 阈值口径 / 结构契约保留**
- **豁免写入 §7**：报告模板与校验夹具（`overview-report.md` 示例行、`report_lint_cases.md`、
  `compile-ab-report-template.md`）**不清数字**，
  但须标明性质（模板示例 / 断言夹具 / docs 宣称），**不得当作本仓实测收益引用**；
  四个文件头部同步加性质标注
- **悬空引用修复 8 处**：已删除的自定义 Graph Pass 指南死链（compilation-dev SKILL 改为指引现有
  GraphPatternEntry 指南并标「已退役」）；cannbot skill 的目录索引与算子经验文件（裸路径 → 标明
  「**外部** cannbot skill 自身路径，非本仓文件」）；模型知识路径（原按 `references/models/` 子目录写，
  本仓无该目录）→ `dummy-run/references/minimax-h3-notes.md`；脚本目录说明 →
  `model-auto-optimization/scripts/README.md`；schema 出处 → **外部** skill-creator 的
  `references/schemas.md`；算子在位基准入口 → 主仓 `benchmarks/scripts/mindie_bench.py`；
  部署脚本 → `env-install/scripts/deploy_to_remote.py`；另把「case 回填规范」的 `-case.md`
  配方改指四分类
- **交叉引用同步**：DSE 合并件的 8 处指称（SKILL 案例表 / framework-extension-dev /
  profiling-collect / matrix V2 证据码 / evals.json / cache-enablement-pattern /
  feature_declarations.json）与 mmgelu 案例的 7 处指称（operator-dev SKILL /
  catlass-ffn-fusion-guide / catlass-kernel-integration / pattern-templates /
  graph-pattern-rewrite-guide / operator-optimization-skill-map）全部改指新落点
- 校验：markdownlint-cli 0.44.0 **0 违规**；`.agents/**/*.json` 全解析；敏感模式 `git grep`
  **0 命中**；悬空引用复扫 **0**

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
  `mmgelu-flux-wan-qwen-case.md`（唯一 `case-` 前缀反例 → 统一 `-case.md` 后缀；⚠️ 该文件后续已
  按「`-case.md` 类别取消」并入 `mindiesd-fusion-notes.md` §7 并 git rm，见下条"批 3 收尾"）
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
  保留（镜像代码目录 `model/common`，属「代码目录镜像」类功能性命名）；`cache-dit-enablement.md`
  首行补命名说明（cache-dit=框架 repo 名，非特性 Cache）
- **N6 层级例外文档化**：README §7 补「workflow 局部模板例外」（workflows/references/ 只放该
  workflow 专属模板）；`model-auto-optimization/scripts/README.md` 补「数据随脚本就近」说明（feature_declarations.json 不入
  references 词表）
- 校验：markdownlint（0.44 等价）0 违规；evals JSON 解析通过；旧 basename 全 token/裸词残留 = 0

### compilation-dev：废弃自定义 Graph Pass（2026-09，禁止手写 FX graph traversal）

对照代码现状审计后收敛（register_replacement 双参数 pattern 即可表达 `nn.Module` 权重，
freeze 前窗口命中；`GraphPatternEntry` 为手动改图唯一正解）：

- **删除** `pattern-dev/references/custom-graph-pass-guide.md`（git rm），**明确禁止
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

### USP2 通信分布补测回填（2026-09-09，仅涉 cache-dit）

- `framework-integration/references/cache-dit-enablement.md`（原 `-case.md` §8 的**采集方法**已迁移：
  开启/产物指针见该文件 §6，方法落 `profiling-collect`）：USP2（2 卡，DiT 减步）全链路通信分布实测——
  DiT Ulysses a2a **为每步通信的绝对主导项（~182 次/步；载荷为数十 GB/步量级、接近该步通信全部）**、text encoder TP
  all_reduce ~101 次/encode、VAE video all_gather 载荷 GB 级/请求、audio VAE 无并行通信；含采集 shim 姿势
  （**encode_ids 锚**、torch 根导入 hook 时机、懒装锚、同 prompt 缓存陷阱）与 moved 口径
- `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` 增 **§6**：运行时通信分布采集
  shim 法（通用可复用，跨框架）；报表 `comm_analysis_usp2.md`（会话产物，不入库）

### 会话蒸馏：掩盖上限与并行归因（2026-09-13，H3 × vLLM-Omni 8 卡，无新增技能）

- **归属判定**：掩盖机制与其上限、并行方案差异归因属并行能力域 → 回填 `parallelism-strategy`；
  量化前移的数值契约属量化契约域 → 回填 `quantization-dev`。**未新增技能**（§7.1 归属判定）。
- `parallelism-strategy`：
  - 新参考 `references/comm-masking-method.md`（域级方法）——掩盖上限三量模型（`C`/`F`/`n` ⇒
    `c/f` 决定 `1-1/n` 是否可达；`f<c` 时上限退化为 `1-[c+(n-1)(c-f)+drain]/C` 且**与 n 无关**）、
    分块实现 recipe（一次预置换 / 侧流只放集合通信 / 块级 event / 不交错 communicator）、
    生效判据（该族次数上升 ≈n 倍、侧流 AI-core 核数 = 0、md5 逐字节一致）与静默失效清单
    （门控读取顺序、启动脚本 `unset`）；
  - 新参考 `references/parallel-plan-attribution-method.md`（域级方法）——阶段 Δ 分解与闭合校验、
    集合通信按 communicator 分组 + 次数对账、`union==sum` 判重叠真实性、隐含带宽 vs 隔离微基准
    （**先对齐并发度再归因**；未解释项必须标注）、跨岛/同岛通信量下限与 **GQA 决定 CP 下限**、
    投影纪律、4→8 卡线性度口径；
  - 新脚本 `scripts/mask_bound_calc.py`（上限与达成率，含 >100% 自检告警）与
    `scripts/collective_attribution.py`（族/communicator 分组 + union/sum + 跨族重叠 + per-stream 表）；
  - SKILL.md 增两节摘要 + Reference Files 接线 + description 触发面补齐；
    **evals 增 8/9/10**（掩盖率达不到 1-1/n / 两形态差异拆解 / 隐含带宽只有隔离一半）。
- `quantization-dev`：SKILL.md 增 **§四·B 量化前移的字节精确条件**——按 scale 的**归约域**分三类
  （分片局部 ⇒ 对齐式 `S % (world × b) == 0`；跨分片全局 ⇒ 量化前须补 `all_reduce(MAX)` 并计入
  收益账；全量派生的 mask ⇒ 走池化侧信道或接受漂移）、收益/成本两算与判定工作流；§九 分工补指针；
  **evals 增 8**。
- 数字纪律：本批回填只写**比例、比值、排序与达成率**（如 `c/f`、上限达成率、跨岛/同岛约 2 倍、
  线性度 78%→90%），绝对耗时与绝对带宽不进 skills；一次性探针（本地补丁、默认关门控）
  按 `[探针]` 口径处理，不写成推荐姿势。
- **探针 vs 经验**（按 §7 四分类判定）：分块掩盖的**实现注入点**是 `[探针]`（已在
  `comm-masking-method.md` §3 显式标注适用窗口与复核方法）；由它发现的**机制级约束**
  （侧流只放集合通信 / 块级 event / 不交错 communicator / 上限公式 / 归因判据）属**经验**，
  按域级方法沉淀。

### 会话蒸馏 2：形态抉择与「并行 × 稀疏」seam（2026-09-13，同日续）

- `parallelism-strategy`：
  - 新参考 `references/parallel-form-selection-method.md`（域级方法）——**序列并行形态抉择**
    （纯 Ulysses vs 复合 AllGather-KV×Ulysses）：七关流程（G1 形态可用性 ⇒ G2 显存/驻留 ⇒
    G3 每层通信量下限（跨岛/同岛两列 + **GQA 是决定性变量**）⇒ G4 形态自带计算 plumbing ⇒
    G5 掩盖上限 c/f ⇒ G6 同窗口 A/B + 线性度 ⇒ G7 投影与反超阈值）、判据表、**条件化结论模板**、
    重判触发清单；核心纠正两条——「拓扑更优 ≠ 更快」（a2a 进同岛换来 K/V 汇聚跨岛）与
    「无 GQA 时两形态跨岛字节几乎相同」；
  - 新参考 `references/cp-sparse-combination-method.md`（域级方法）——**CP × 稀疏 seam 契约**：
    可行性分形态判（环状不可用 / AG-KV 可用）、四条契约（先汇聚后稀疏 / 窗口偏移 =
    「完整 KV 长度 − 本 rank 逻辑 Q 长度」/ 分片块对齐 `S % (并行度×块) == 0` / per-head 掩码不可
    跨片复用）、掩盖叠加规则（复合形态自有分块 + 「先全部 a2a 再全部 AG」）、验收判据（含
    「与 lossless 逐字节相同 ⇒ 判未生效」）与失效模式表；
  - SKILL.md 增两节摘要 + Reference Files 接线 + description 触发面（含 CP×稀疏与 CP/USP 选型
    的用户原话兜底句）；**evals 增 11/12**。
- `performance-optimization/references/combination-search.md`：新增 **seam 表第三案例校准**
  （序列并行 × 稀疏 × 掩盖）——组合层只记规则增量：形态可行性与「稀疏生效」是两件事、
  「未生效」以字节级等价判定、**正协同 ≠ 整体更快**（协同强度不替代选型结论）、
  **载荷档变化会反转形态排名 ⇒ 组合顺序应先定量化/载荷档再比形态**。
- `framework-feature-enablement`：`vllm-omni-enablement.md` §2.2 补「CP 形态可用面」（复合可跑通、
  环状对稀疏不可用、4 卡 2×2 未跑通）、§3.3 补「与序列并行叠加的前置」与生效判定；
  `framework-support-matrix.md` CP 行由 `❓` 升为 **✅ V4**（带条件与反例）+ 表头 last-checked 追加。
- **机器声明联动的检查结论（支持矩阵 §三.5 要求）**：`feature_declarations.json` 只声明
  特性级 seam/互斥（`exclusive_seams` / `window_conflict_warning_seams`），**并行形态不是已声明特性**，
  而 CP×稀疏是**依赖型要求**（需要全量 KV）而非互斥 ⇒ **本次不改声明文件**；若后续要机器校验
  该依赖，需先给 schema 增「依赖框架能力」字段（记此判断留痕）。

### 会话蒸馏 3：暴露归属与空泡归因（2026-09-13，同日续）

- 触发：实测追问「输出的 O 未掩盖」与「FA 前的空泡是否该靠异步/多线程下发掩盖」。
- `parallelism-strategy`：
  - 新脚本 `scripts/bubble_attribution.py`（零依赖）——三个判据一次给出：**暴露归哪一族**（某族耗时
    大部分落在空隙里才是关键路径）、**释放判据**（空隙"恰好被某次集合通信的完成释放"才算真依赖；
    "空隙里有通信在跑"只是必要条件）、**生产-消费判据**（消费者核启动 − 它**自己那片**通信的完成，
    用来判通信是否已跑在消费前面）；
  - `comm-masking-method.md`：新增 **§4.5 归属与空泡归因**（两个判据 + 报数纪律）与 recipe
    第 8 条 **「下发顺序就是调度」**——同一 communicator 的集合通信按下发顺序串行（`union/sum = 1.00`），
    「先全部前向、再全部反向」会把反向整段暴露；修法是**软件流水式下发**而非逐片交替；
    recipe 第 7 条补 **「回退分支必须留命中/回退计数器」**（本组合实测出现过"开了门控但每次静默
    回退"的臂：md5 与耗时都与基线相同，看上去像优化无效）；
  - SKILL.md 摘要同步 + 接线新脚本；`evals` 增 13。
- 本组合观测（进 skills 的只有量级、方向与排序）：反向输出 a2a 占该族字节 **1/4** 却贡献**过半**的暴露通信、
  与注意力核重叠**几乎为零**；前向 q/k/v 暴露比例**约三成**（重叠约六成）；计算流空隙中
  **约九成有集合通信在跑**但只有**约七成是被集合通信完成所释放**（其余是"传输还在途就恢复执行"），
  真空气泡**约一成**（占整步为个位数百分点，与 `Free` 列同量级）；
  **生产-消费判据**：注意力核与其**自己那片**前向 a2a 完成的间隔**中位数远高于判据门槛（毫秒量级）**、
  350/350 全部越过门槛 ⇒ **前向通信跑在消费前面，FA 从不等自己的数据**（FA 前的空隙合计每步毫秒量级）——
  这条推翻了"空隙=FA 在等自己的 qkv"的初判，故 §4.5 由两判据改写为三判据，并列为强制一起报。

### 遗留待办（人工项）

- 核对 `mindie-sd-community-governance/assets/mr_ruleset_20260327101328.xlsx` 与
  `mindie-sd-community-governance/SKILL.md` §5.5 的定位是否一致（该技能按约定不改写，需人工复核一次）
  ——**「不改写」的范围限定**：仅指 **`assets/mr_ruleset_*.xlsx` ↔ SKILL §5.5 定位核对**这一条
  （人工复核期间该技能正文规则不动），**不覆盖** frontmatter `description` 的触发质量——description
  与其余 17 个技能同受 skill-creator 要求约束（**「什么时候用」必须进 description**、用户原话兜底
  句、长度 ≤1024），2026-09 已按其正文 §2.1「强命中信号」补齐触发覆盖；**勿把本待办读成「整个技能
  不许动」**
- `docs/zh/features` 增补 2026-09 落地特性（`mm_swiglu_mxquant`/FFN-MX compile 融合、FA A5
  可用性等）后，核对 skills 侧引用（`framework-integration/references/framework-support-matrix.md`
  已登记，S2-1 槽位待回填）；**原 `refresh_features.py` 与 `mindiesd-features.md` 镜像已随方案 A 删除**，
  特性事实一律以 `docs/zh/features/*` 为真源。
- dummy-run 的 quant 细节（§A6）是否部分下沉 `references/model-common.md`（当前内联偏厚，观察项）
- S2-1（mindiesd 融合 kernel 能力清单，唯一真相源）回填时可参考 per-technique 目录化组织
  （每技术：描述 / 开关 / 验证点与运行期计数 / 模型适用矩阵）——结构模板借鉴
  跨栈方案的 per-technique 目录化组织（仅借结构，不搬其内容）

## 参考链接

- [Anthropic skill-creator 规范](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
- [cannbot-skills（算子开发外部技能库）](https://gitcode.com/cann/cannbot-skills)
- [MindIE-SD](https://gitcode.com/Ascend/MindIE-SD)

### 经验积累 7（2026-09-14 H3 交付档收口：三类判断方法）

> **本节记方法与判据，不记"谁更快/谁更好"的名单。** 所有绝对值都是"实测实例在某口径下的观测"，
> 换硬件/版本/形状/拓扑/窗口就必须重测；每条经验都给出了"失效信号"，出现它就说明旧结论过期。

新增三个能力技能（§2 由 18 → 21）：

| 技能 | 回答什么问题 | 什么时候该重测 |
|---|---|---|
| `accuracy-gate`（`references/silent-failure-localization.md`） | 结果**为什么是错的**（无报错、无 NaN、md5 稳定但内容错） | 任何"结果看着不对"的场合；换设备/版本后同类现象重新走一遍逐层打点 |
| `perf-gate` | 优化**有没有效、有多少效** | 每次要下性能结论时；换窗口/换机器/换形状先重测噪声地板 |
| `vae-opt` | 解码**能不能切、怎么切才逐位精确** | 网络结构或并行拓扑一变就重做独立性审计 |
| `accuracy-gate` | 热点算子**等价替换**怎么做才算无损、怎么证明 | 换实现后先重做 per-shape 逐位对拍；换硬件/形状/版本要重验 |

三条经验（都指向"怎么判"，而不是"结论是什么"）：

1. **"交付产物看着不对"要先当静默错误查，而不是当模型质量问题。** 实测实例最终定位到设备算子
   （见技能内案例），而排除过程靠的是**造参照 + 切维度 + 做对照**：CPU 真值 → 逐层打点找首个分叉 →
   算子隔离 → 规模阈值扫描（区分"固定索引"与"规模阈值"）→ 跨运行相关性（区分"陈旧内存"与"数值漂移"）。
   **判据是流程，不是数字**：阈值与首坏索引随 kernel/形状/版本变化，必须现场重扫（本案只二分到
   166–200 区间，报告里也如实标注了这一点）。**失效信号**：换版本后首坏索引变了、或同形状复现不出。
2. **跨窗口比数字是无效结论。** 窗口内自带基线、必要时 A/B/A 校正；另外两条硬经验也要按"判据"理解：
   ① arm 日志的 `t=` 会被 profiler 阶段/排队污染（**判据**：与服务端阶段账/节拍交叉验证，冲突时以
   服务端为准）；② **产物 md5 门禁有盲区**——基线与各臂共用同一（可能有缺陷的）解码器时，"逐字节一致"
   什么都证明不了，必须另设与**独立真值**的忠实度对照（**判据**：相关性/MAE 要达到与真值同量级；
   实测实例中两套权重一个相关性接近 1、一个接近 0，而两者都能通过 md5 门禁）。**失效信号**：同窗对照差异落进
   地板 ⇒ 标"不可分辨"，不要硬报收益。
3. **"每卡各解一遍"这类重复计算要切，但切法由网络的耦合结构决定，不由我们想切哪根轴决定。**
   逐帧解码网可自由按帧轴切；含状态块（mem/past）的递归解码网**不可沿时间轴直接切**（第 t 帧依赖
   整段前缀），正确做法是按状态边界切段并**逐段携带末状态**——**判据**是闸①"CPU 上整段 vs N 段逐位
   相等"，段数以现场阈值扫描为准；这条切分还能顺带绕开设备算子缺陷。**失效信号**：切分"看起来生效但
   结果不对" ⇒ 先跑闸① 二分"切分不精确"还是"并行没生效"。

4. **"等价替换"是性价比最高也最容易自欺的一类优化，所以要把等价性写进代码而不是写进结论。**
   实测实例：某类抗混叠反卷积**几乎占满所在阶段耗时**，多相分解重写后单次耗时降了一个数量级、
   整段解码同量级下降（绝对值见归档），而**每个形状 `torch.equal` 成立**（ndiff = 0）、产物 md5 逐字节不变。
   **判据**：L1 逐位（`torch.equal`，不是"平均误差很小"）→ 端到端产物 md5 → 同窗 A/B 归因；
   **手段**：实现里带 per-shape 自验证断言（第一版 SNR 远低于门限、被它当场抓住，根因是相位接线
   而非精度）；**失效信号**：换硬件/版本/形状后 kernel 或前置条件变化 ⇒ 逐位与收益都要重验
   （实测实例中就有"只在片长 ≥ 某阈值时逐位精确"的条件性等价）。

三条通用纪律（已写进各技能的收尾自检）：**一次只动一个维度**；**噪声地板内不下结论**；
**无效的改动要撤回**（实测实例中有两个补丁被证明与旧路径逐位等价，已显式回退——树里不能沉淀
"看起来在起作用"的开关）。
