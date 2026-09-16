# MindIE-SD 多模态解决方案 Skills

面向 MindIE-SD 多模态扩散模型（Wan2.2 / FLUX / Qwen-Image / MiniMax-H3）在昇腾 NPU 上的
**模型自动优化**技能集合。

组织方式（三层架构：编排 L1 · workflow L2 · 能力 L3，2026-09 重构）：

```text
L1 编排入口（2）：model-auto-optimization / dev-workflow——任务分流与路由、特性验证顺序（依赖派生）、
                交付件契约与编排机制所有权（run-state / stage_gate / 迭代表 / 覆盖清单 / 双报表）
L2 业务 workflow（按业务线）：特定业务线下的阶段化执行与单任务深挖——把 L3 能力组合成该业务的
                流程与"姿势/优化选择"（含单特性策略与组合回退裁决）；厚度因业务线而异（见「架构速览」）
L3 能力（21）：可复用的单一能力/知识/工具（含能力自身工作流纪律），可被任意 L2 编排或独立直达
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

## 2. 优化域入口与能力层（22）

> **分类**：优化域入口（L2，可独立触发）**1** · 优化域模块（L3）**4** · 能力供给（L3）**7** · 标准与方法（L3）**2** · 采集分析（L3）**3** · 环境与运行通道（L3）**2** · 规范（L3）**3**。
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

### 2.3 能力供给（7，阶段无关）

| 能力 | 一句话界面 | 域标签 |
|------|-----------|--------|
| **[pattern-dev](skills/pattern-dev/SKILL.md)** | PyTorch Inductor pattern matcher 机制开发：扩展本仓 compile 能力或三方框架自带的类似能力——Pattern 创建/注册/调试、GraphPatternEntry 手动改图、Copy（InplaceCopy/ViewCopy）消减全生命周期，含 Phase 1–7 出口证据纪律 | 开发 |
| **[operator-dev](skills/operator-dev/SKILL.md)** | 算子开发与调优（Triton / Ascend C / Catlass / PyPTO / TileLang；优先路由外部 cannbot-skills） | 开发 |
| **[aclgraph-dev](skills/aclgraph-dev/SKILL.md)** | NPU 图批量下发开发（NPUGraph 静态 capture / graph pool / lazy capture / 驱逐） | 开发 |
| **[quantization-dev](skills/quantization-dev/SKILL.md)** | 量化**格式契约**与位级对齐：从设备字节反推 MXFP8/int8 契约（编码公式/舍入/scale 粒度/退化块）并逐字节复现；含**除数须载入**、**8-bit 回绕非饱和**、组尺度耦合、位级对拍 SOP、**量化前移的字节精确条件** | 算子 |
| **[dummy-run](skills/dummy-run/SKILL.md)** | **能力供给的配套验收载体**：随机权重/精简代码快验（架构兼容性、算子先接入、融合可行性），提升开发效率；**非通用验证载体** | 验证 |
| **[framework-integration](skills/framework-integration/SKILL.md)** | 三方框架特性落地（**原 framework-feature-enablement ＋ framework-extension-dev 合并**）：一个入口信号「框架侧特性没落地」，内部分两分支——**分支 A 框架已有 → 使能与验证**（计数契约 + 三层证据 + 异常回修）；**分支 B 框架缺失 → 补齐开发**（注入点 + 注册机制 + 合入姿势：平台注册/上游 PR 优先，fork/monkey 备选） | 框架 |
| **[fusion-scope-analyze](skills/fusion-scope-analyze/SKILL.md)** | **融合范围与收益分析**：判定"哪些计算该融进同一个融合单元、边界画在哪、融了值不值"——结构侧按**融合单元构造规则**切边界（函数边界即候选组；FA 及 BSA 等变体为**硬锚点**、不吸收也不跨越；以 norm / rope / FA 分界分区，**全 Vec 优先试融**；区内有 Cube 时以 **Cube 为首算子**向后包裹 Vec，**Σvec < Cube** 或遇下一 Cube 封口；纯 Vector 单元之间迭代再融；MLP / MoE 以 MatMul / BatchMatMul / GroupedMatmul 为界）；收益侧按计算单元利用率**判型**（`memory_bound`、`*_vec / *_mac / *_mte2 / *_mte3` 的 ratio 与 time、`cube_utilization`）+ 带宽下限 / 区域占比 / 天花板法 / 地板先行 / Amdahl 传导校验给出 go-no-go；**只出判定与建议**（选点与派活归 `dit-perf-opt`，采集与瓶颈定位归 profiling 管道） | 分析 |
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

> 历史**改名 / 合并 / 撤销**对照不入本文件（属过程记录；旧名在各处出现时已 inline 自标，
> 其余查 `git log`）。技能层与产品侧的依赖方向见 §7。

## 4. 预留空缺槽位（待经验回填）

> 槽位编号沿用历史：S2-1 对应 S1 kernel 融合能力清单（阶段并入后编号保留）；S5-1 训练感知案例
> 已于 2026-09-12 回填（见下）。

| 槽位 | 阶段 | 内容 | 现状 |
|------|------|------|------|
| S0-1 | S0 | 各模型/任务权重分区与下载经验（Qwen-Image / Wan2.2 / H3…） | **modelscope 优先 + 分区约定已立；H3 已填，其余待回填**：下载源优先级 = **默认 modelscope**（`modelscope download` / `snapshot_download` + `--local_dir` 直落、国内可达、HF gated 仓库在 modelscope 镜像通常免鉴权）→ **次选 HuggingFace / gated**（需 token，走 `hf`/`huggingface-cli login` 或 `--token`，镜像回退 hf-mirror）；分区约定 = `{model_weight_dir}/{模型名}/{任务变体}`（模型根目录直接 serve）；落位表（模型 / 目录落位 / 任务变体 / 仓库 id / 依据）见 `env-install/references/weights-prep.md` §2.2（源优先级见同文件 §2.1）——**H3 全列已填**（`MiniMax/MiniMax-H3`：根 diffusers + `FL2VA/`、`Ref2VA/` 子分区），Qwen-Image / Wan2.2 / FLUX 的**仓库 id 与任务变体列标 `待回填`**；权重确认纪律（先确认远端已存在、无 `.incomplete`、分片齐全/文件数一致、有校验和则逐文件核对）保留于同文件 §5 |
| S1-1 | S1 | 无抽象接口框架的算子注入方法（多框架泛化） | **已回填（2026-09-13，两阶段法）**：**① API 优先**（runtime 注入——配置注册表 / 框架侧 dispatch / **直接改模型代码 `import mindiesd` 并替换调用点**）→ 验证接口可行 + 同 seed 数值对拍 → **② 再走 compile 机制**做图级适配（pattern 图形态变体 / `_compiled_call_impl` 写入 / 平台注册 backend）。三组判据齐备：「为何 API 先行」（改动面小、失败早暴露、先拿到可归因的单点收益与对拍基线）、「为何 API 不替代 compile」（层内单算子 vs 整链融合 + 图级拷贝消除）、「**何时停在 API 阶段即可**」（eager 已覆盖热路径/compile 无正收益/输出非逐字节/适配落探针 ⇒ 按证据回退，不留半开）；含**收益量级对照**（API vs compile 的相对贡献，各标「本组合观测」，不记绝对耗时）与**各框架注入点差异表**（注册表替换 / 模型层直接改写 / 框架侧 dispatch / `_compiled_call_impl` 原地写入 / 平台注册表 / env 门控 fork）。落点：`framework-integration/SKILL.md` §②「两阶段顺序纪律」+ §运行时算子接入（阶段 1）/ §compile 融入（阶段 2）；**compile 阶段适配动作**落 `pattern-dev/references/fusion-enablement-notes.md` §4（1→7 顺序，与该 SKILL §② 路由成对） |
| S2-1 | **S1**（原 S2，已并入 S1） | mindiesd 融合 kernel 能力清单（接口 → pattern → 验证过框架，唯一真相源） | 分散于各 case，已按新分层回填到**方法单点 + 框架开启方式**（H3×vllm-omni）：`model-auto-optimization/references/lossless-methodology-notes.md` §A/§D —— 融合候选工作流「执行序→候选列表→mindiesd/CANN 能力对照→独立验证」+ eager 已融合热路径判定 + 残余候选池对抵判定 + 量化后融合重审；`framework-integration/references/vllm-omni-enablement.md` §3.1/§5 —— 自动路由面核验（FA/AdaLN/RoPE/GELU eager 覆盖）+ 单步 kernel 采集 hook（`[探针]`）；**2026-09-06 图像第二案例**：同文件 §5.3 kprof 目标扩展 + §3.5 compile 输出非无损否决；实测数字归档于会话产物目录 `{run_results_dir}/archive/`） |
| S3-1 | S3 | NPU 拓扑/带宽矩阵与并行选型决策 | 跨代设备的拓扑/带宽矩阵与并行可行性（ring 可用性、offload 解锁并行、通算掩盖 step_trace 评估）→ 单源参考 `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §7 与 `dit-parallel-opt`「内存受限时的并行解锁」；拓扑分域（域内/跨域、SYS/PCIe）、bulk vs head-parallel 翻转条件、HCCL 带宽 bench 姿势与 set_device 陷阱、端口泄漏/卡组诊断 → 同上文件 + evals 4/5；**图像案例（见 `ascend-topology-bandwidth-diag.md` §7 + `vllm-omni-enablement.md` §2.2/§3.3）**：同域选卡 + 同域亦受他户干扰（探活前置）、**并行候选矩阵勿漏 2 卡 USP（短任务下 TP×USP 组合需实测取舍）**、短任务低 rank 数病态回退。**设备型号 / 卡数 / 实测数字归档于会话产物，不进本索引** |
| S3-2 | S3 | few-step 多 rank 验证协议（脚本 + 判据） | **已回填（2026-09-13）**，核心是确立 **DiT-only 口径**——并行对比**只取 DiT 去噪阶段**（`<Pipeline>.diffuse` 阶段墙钟，微秒级、每 rank 一行取 min），**VAE 解码 / 文本编码 / 权重装载 / warmup / 响应编码一律排除**；理由：少步档固定开销占比畸高、且**固定开销自身的波动远大于并行差异**（DiT 阶段与 VAE 解码的波动量级差异见会话产物归档）⇒ 计入 decode 量到的其实是 VAE 噪声。协议与判据落 `dit-parallel-opt/references/few-step-multirank-protocol.md`（口径边界与取数 → 矩阵设计「必须含同卡数不同切分的一对」+ 形态是启动级参数/步数是请求级参数 ⇒ 少步与锚点可同 serve 交错 → 噪声纪律 ≥3 rep 取中位、**离散度 >3% 该格不可用** → **外推判据** `r = DiT单步(锚点) ÷ DiT单步(少步)`：`\|r−1\|≤5%` 且两档排序一致才可外推，**排序翻转一律以锚点档为准** → 必回较高步复验的 7 个触发条件 → 通信占比随步数漂移与 profiler 膨胀口径 → 9 条踩坑清单）；SKILL.md 新增「少步 × 多 rank 验证协议」节为摘要并接线；采集脚本 `dit-parallel-opt/scripts/fewstep_multirank_probe.py`（跑矩阵 + 出 DiT-only 证据包，`--parse-only` 零 NPU 复盘）。实测数字归档于会话产物目录 |
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
├── README.md                     # 本文件：架构（三层）与技能清单唯一权威
├── scripts/                      # 门禁脚本（说明在同目录 README.md：规则/豁免/历史/盲区/积压）
│   ├── kb_lint.py                # 知识层结构门禁（13 条规则，CI 强制）
│   └── run_evals.py              # 技能 evals 裁定覆盖度门禁（会话内跑）
└── skills/                       # 24 个技能；每个 = SKILL.md + references/ scripts/ evals/
    ├── model-auto-optimization/  # L1 编排 · 模型自动优化流程
    ├── dev-workflow/             # L1 编排 · 仓库开发流程
    ├── performance-optimization/ # L2 优化域入口（只分发不选档）
    ├── dit-perf-opt/             # L3 模块 · DiT 计算（特性选档与实施）
    ├── dit-parallel-opt/         # L3 模块 · DiT 通信（并行选型 + 生效判定）
    ├── vae-opt/                  # L3 模块 · VAE/TAE 解码段（计算 + 通信）
    ├── host-opt/                 # L3 模块 · host/辅助段（交付搬运 + 装载预热）
    ├── pattern-dev/              # L3 能力 · Inductor pattern matcher 机制
    ├── operator-dev/             # L3 能力 · 算子开发（路由外部 cannbot-skills）
    ├── aclgraph-dev/             # L3 能力 · 图批量下发
    ├── quantization-dev/         # L3 能力 · 量化契约与位级对齐
    ├── dummy-run/                # L3 能力 · 快验载体（非通用验证）
    ├── framework-integration/    # L3 能力 · 三方框架特性落地（使能 / 补齐两分支）
    ├── fusion-scope-analyze/     # L3 能力 · 融合范围与收益分析（边界判定 + 收益前置评估）
    ├── accuracy-gate/            # L3 标准 · 精度验收标准
    ├── perf-gate/                # L3 标准 · 性能验收标准（入库门 + 双态）
    ├── profiling-collect/        # L3 采集分析 · 统一采集
    ├── profiling-analyze/        # L3 采集分析 · 统一分析
    ├── benchmark-dev/            # L3 采集分析 · 算子实现级基准
    ├── env-install/              # L3 环境 · 安装 / 权重准备
    ├── remote-access/            # L3 环境 · 远端执行通道
    ├── code-standards/           # L3 规范 · Python 格式与 lint
    ├── markdown-lint/            # L3 规范 · Markdown 格式
    └── mindie-sd-community-governance/  # L3 规范 · 文档 / 治理 / 提交规范
```

> 本树**只到技能一级**：技能内部的 `references/` `scripts/` `evals/` 明细不在此列（避免与 §2 的
> 清单重复、也避免维护一份易腐烂的文件清单）。查具体文件用 `ls` 或各 `SKILL.md` 的 Reference Files 表。

## 7. 贡献

### 技能准入与命名判据（新增技能前必读）

**准入判据**（五条全满足才允许新建技能）：

1. **独立耦合结构 / 几何契约**：换模型即失效的结构（如潜帧边界、状态前缀依赖）——通用手段不算。
   **例外（规范类）**：`-lint` / `-standards` / 治理类**不以结构耦合为判据**（通用规范天然无此结构），
   但须**过第 2/3/4 条**——现存 `code-standards` / `markdown-lint` / `mindie-sd-community-governance`
   即属此类（若无此例外，任何人新增规范类技能都会撞上本条而被拒）。
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
  ④ evals 增补；⑤ 复盘（dev-workflow §6）新经验先判断归属槽位/能力域再回填；
  ⑥ 跑 `kb_lint` error=0（pre-commit 已强制；改门禁脚本本身时另跑 `--selftest`）。

### case 回填规范（实验 → skill 刷新）

> **⚠️ 先读 · 可迁移性（强制）**：沉淀到 skill 正文（`SKILL.md` 与 `references/*.md`）的
> **只能是可以换设备、换模型、换框架仍然成立的判据 / 方法 / 契约**。**四类不进正文**：
> **设备身份**（型号 / 代际 / SKU / `soc_version` / 芯片代号）、**代际常量**（核数 / UB 容量 /
> grid 上限 / 单卡显存 / 拓扑成员编号）、**单案例绝对读数**（绝对耗时 / 带宽 / 占比 / 计数 / 形状
> 含模型几何）、**把版本号当规则前提的版本钉**（CANN / torch / torch_npu / 框架版本）——
> 它们落会话产物或 `-case.md`。正文只写三件：**判据 + 现场取数方式 + 归档指针**。
> 豁免：契约类事实（CSV 列名 / 文件布局 / 错误码 / API / 格式名 / dtype）、与设备无关的比值门限、
> 模板与断言夹具数字。细则见下文「落位四分类 → **可迁移性**」与「数字纪律」；**提炼完成后跑 `kb_lint`**。
>
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
     **`-case.md`**：受限保留，**政策见本节开头**（本处只留指针，不重复要求）；优先按上面四类落位。
   - **方法与产物隔离（强制）**：上条四分类的判据来源——**特性分类跨框架相似，但开启方式按框架不同**
     ⇒「可复用的方法」与「具体产物（某模型 / 某框架 / 某权重）」必须分开落位（同批经验常需同时写两件）。
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
   - **可迁移性（强制 · skill 正文只放可迁移信息）**：skill 正文（`SKILL.md` 与 `references/*.md`）**只承载可迁移的
     判据、方法与契约**。下面四类**不进正文**，落会话产物或 `-case.md`：
     - **设备身份**：型号 / 代际 / SKU / `soc_version` / 芯片代号；
     - **代际常量**：核数、UB 容量、grid 上限、单卡显存、拓扑分域的具体成员编号；
     - **单案例绝对读数**：绝对耗时 / 带宽 / 占比 / 计数 / 形状（含模型几何）；
     - **版本钉**：CANN / torch / torch_npu / 框架版本被当作**规则前提**时。
     正文取而代之写三件：**判据** + **现场取数方式**（`npu-smi -t topo` / `-t memory` / 设备属性 / 算子 UT）+
     **归档指针**（case 记录坐标）。**不受此限**：契约类事实（CSV 列名 / 文件布局 / 错误码 / API / 格式名 /
     dtype 名）、与设备无关的比值门限（噪声地板、占比门限、`1-1/n` 上界）、模板与断言夹具数字。
     `-method.md` 本就要求「**不含产品名**」，同此办理；`-notes.md` 只放平台/模型**几何契约**，不重复放代际常量。
   - **数字必带环境作用域（强制 · 防跨环境照抄）**：**凡按上条保留在正文里的数字、阈值与「最佳参数」都必须带上它的
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
6. **校验**：`kb_lint`（`.agents/scripts/kb_lint.py`，**提交/CI 强制**，见下「知识层门禁」）
   error=0；markdownlint-cli **v0.44.0**（仓库 pin，`.agents/**` 已由 pre-commit 强制）0 违规；
   evals JSON 解析通过；含脚本跑 ruff + py_compile。
   ⚠️ **工具版本纪律**：markdownlint 必须用 pin 版 0.44.0 判读——用更高版本（如 0.49.x）
   会对**同一批文件**产生上千条 `MD060` 版本性假违规（2026-09 实测：0.44.0 = 0 违规，
   0.49.1 = 1943 条，全部为 MD060）。比对版本不一致时，先对齐版本再下结论。
7. **变更门禁复核**：README §2/§4/§6（计数/槽位/目录树）与相关 description 定位句是否需要同步。

### 知识层门禁（`kb_lint` / `run_evals`）

两个**零依赖静态门禁**（纯 stdlib、零网络、零模型、只读幂等）：
`.agents/scripts/kb_lint.py` —— 结构门禁（链接图 / 接线 / 计数 / 仓内坐标 / 反向边，**13 条规则**）；
`.agents/scripts/run_evals.py` —— 技能 evals 的**裁定覆盖度**门禁。
**规则细节、豁免机制、历史理由、已知盲区与积压登记一律记在 `.agents/scripts/README.md`
（与脚本同源同改），本节只给当前口径与用法。**

- **当前口径**：`kb_lint` **error=0 / warn=0**（13 条规则或为 error、或积压已清零）；evals 裁定
  **120 用例 / 473 条 expectation 全覆盖**。任一项非 0 即视为知识层有未收口项。
- **违规怎么办**：先判「是内容真错了，还是规则误报」——**误报就改规则**（先例：23 条「缺加载时机」
  逐条复核后**全是误报**；markdownlint 高版本对同一批文件报 1943 条版本假违规），
  **不要为过门禁去写无意义的内容**；新增规则前先量积压，一次报出几十条几乎都是规则过宽。
- **依赖方向**：`.agents` 引用产品侧资产（`evals/`、`benchmarks/` …）是**正向**，受 KB009 覆盖；
  **产品侧不反向引用 skills**（`evals/`、`benchmarks/` 已清空跨层引用，且刻意不加门禁兜底——
  正确解法是消除引用，而非让 skills 层去管产品侧目录）。
- **变更门禁 ⑥**：提交前跑 `kb_lint` error=0；改门禁脚本本身时另跑 `--selftest`（pre-commit 已挂钩）。

```bash
python .agents/scripts/kb_lint.py --selftest     # 负样本自测：每条规则必须能被触发
python .agents/scripts/kb_lint.py                # 全量（error=0 方可提交）
python .agents/scripts/kb_lint.py --list-rules   # 规则与级别
python .agents/scripts/run_evals.py --pack  --out {run_results_dir}/skill_evals.json
python .agents/scripts/run_evals.py --check --results {run_results_dir}/skill_evals.json
```

> 接线：`.pre-commit-config.yaml` 的 `kb-lint` / `kb-lint-selftest` / `run-evals-selftest`；
> CI 的 `CodeCheck_pre_commit` job 会执行 pre-commit，故**无需额外 CI 配置**即生效。

## 参考链接

- [Anthropic skill-creator 规范](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
- [cannbot-skills（算子开发外部技能库）](https://gitcode.com/cann/cannbot-skills)
- [MindIE-SD](https://gitcode.com/Ascend/MindIE-SD)
