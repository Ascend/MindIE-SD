---
name: framework-integration
compatibility: vLLM-Omni / DiffSynth-Engine / LightX2V / diffusers 等第三方推理框架；远端昇腾容器 + CANN；mindiesd 已安装（安装见 env-install）
description: >
  三方框架特性落地（原 framework-feature-enablement ＋ framework-extension-dev 合并）：
  一个入口信号「框架侧特性没落地」，内部分两分支——**分支 A 框架已有 → 使能与验证**
  （计数契约 + 三层证据 + 使能异常回修）；**分支 B 框架缺失 → 补齐开发**
  （注入点 + 代码组织/注册机制 + 合入姿势：平台注册优先 / 上游 PR / fork 钉版 / monkey 备选）。
  分支 A 触发词：量化/稀疏/缓存开关没生效、融合算子没命中、使能异常或精度不符、
  "起 vllm 服务""模型在框架里跑不通""把 mindiesd 接到其他框架"。
  分支 B 触发词：框架没 comm-stream 掩盖、框架没缓存消费者、框架不支持某 collective/mask、
  feature_* NotImplemented、要给框架补一条结构性能力。
  near-miss：纯安装/权重 → env-install；纯采集/分析 → profiling-collect / profiling-analyze；
  纯选档（该不该开量化、开哪一档）→ dit-perf-opt；
  mindiesd 仓内 pattern / 算子 / 图下发开发 → dev-workflow + pattern-dev / operator-dev / aclgraph-dev。
  由 model-auto-optimization 的 S1（融合接入）/S4（有损使能）/S5（训练感知）与 §0 缺口补齐策略路由，
  亦由 dev-workflow 的框架接入/验证场景指引加载。
---

# 三方框架特性落地：使能与验证（分支 A） / 缺口补齐开发（分支 B）

> **合并说明**：原 `framework-feature-enablement`（= 本文件分支 A）与原 `framework-extension-dev`
> （= 分支 B）已合并为本技能——**入口前不再判「已有 / 未支持」**，判定收进 §0 的两分支表内。
> **框架差异知识单点**：各框架的注入点 / 注册机制 / 合入形态一律以
> `references/{framework}-*-enablement.md` 为真源（§3），本文件只留两分支共通的方法与判据。

## 0. 入口信号与分支判定

### 0.1 入口信号

**只要求一个信号：框架侧特性没落地。** 三种表现都从本技能进：

| 表现 | 例子 |
|---|---|
| 开关 / 接口存在，但没开、或开了看不出效果 | 量化 / 稀疏 / Cache 开关、`compile_backend` 配了但 0 命中 |
| 机制存在但当前形态不命中 / 报错 / 精度不符 | 融合算子没命中、4D attention 使算子运行期 fallback、`inferShape does not exist` |
| **框架自身没有这条路径** | 无 comm-stream 掩盖、无缓存机制、无稀疏/量化消费者、不支持某 collective / mask |

提问者不需要先判定"已有还是未支持"——由上表在 §0.2 分流；但**分支持续变更要留一条记录**（不静默切换）。

### 0.2 两分支判定

| 观察 | 分支 | 出口证据 |
|---|---|---|
| 框架该能力**已存在**（开关 / 接口 / 机制具备），只是没使能或没生效——含**为使能所需的最小 glue 修复**（如 `supports_packed_mask_free` classmethod） | **A：使能与验证**（§1） | 计数契约 + 三层证据（图命中 → kernel diff → 墙钟） |
| 框架该能力**未支持**（无机制 / 无注册点 / 无消费者 / 无注入点） | **B：补齐开发**（§2） | 输出一致 + 计数契约 + 三层证据；fork/探针须标 `[探针]` |
| mindiesd 仓内要新增 / 适配 pattern、kernel、图下发、后端 | **越界**：非本技能 | dev-workflow + `pattern-dev` / `operator-dev` / `aclgraph-dev` |
| 缺口归属不清 | 先判归属（§2.1，**以代码落点定侧**） | 判定行写入任务记录后再动手 |

- **是否补、补到哪一档**由 `model-auto-optimization` §0 的「缺口补齐策略」经用户一次性确认决定；
  本技能只提供「怎么使能 / 怎么补」的知识与流程，不代替该确认。
- 跨侧特性（mindiesd 提供能力接口 + 框架提供接入点）按 §2.1 拆两侧子任务，对接联调在本技能收口。

### 0.3 定位与分工（相邻能力）

| 相邻能力 | 分工 |
|---|---|
| env-install | 环境安装与权重准备（本技能前置）；**装库 / 权重 / 容器类异常直接归它** |
| profiling-collect / profiling-analyze | 证据侧：采集与 kernel diff / 三层证据 |
| perf-gate | 性能验收口径：同窗 A/B、噪声阈值、探索态 vs 验收态 |
| accuracy-gate | 精度验收口径：等价分层与三级验收（有损档的质量门） |
| pattern-dev / operator-dev | 实现侧：新增 / 适配 pattern、算子本体 |
| dummy-run | 快速验证载体：融合可行性、使能集合预判 |
| model-auto-optimization | 编排层：阶段账 / 门禁 / 报表 / 产物目录 / 缺口策略 |
| **本技能** | **框架侧落地**：分支 A 使能 + 验证 + 回修；分支 B 补齐开发 + 注入点 + 合入姿势 |

前置：环境与权重就绪（见 env-install：`import mindiesd` 成功、三方框架装好、模型权重可用且经其确认）。

---

## 1. 分支 A：框架已有 → 使能与验证

### 1.1 使能与验证回路

```text
① 使能评估 → ② 路径决策 → ③ 使能异常定位与回修
```

**① 使能评估**：目标特性需要 mindiesd 的什么 kernel / pattern？框架侧接入点是什么（配置注册表 /
特性开关 / compile 入口 / 图形态）？特性来源是 mindiesd 能力还是框架原生（决定查 `docs/zh/features/*`
还是该框架 enablement 文件）？

**② 路径决策**：

```text
a) 有现成融合 / 单算子 → 开启并验证生效
   姿势：特性开关 / 注册表替换 / _compiled_call_impl 赋值（见 §1.6）
   → 三层证据确认（图命中 → kernel diff → 墙钟，证据走 profiling-analyze）
b) 融合不支持（无 pattern / 图形态不匹配 / 无抽象接口）
   → 接入 mindiesd 算子（runtime 替换，见 §1.6）
     或走 compile（新增 / 适配 pattern，图形态差异加变体）
   → 需要实现改动时指向 pattern-dev / operator-dev
   → 融合可行性先用 dummy-run 快速验证（pattern 命中不依赖层数 / 权重），
     再在真实权重下 kernel diff + 墙钟复验
```

a) / b) 是**判据**；**执行顺序另有纪律**（见 §1.2）：无抽象接口框架的算子注入一律两阶段——
先 API（runtime 注入）→ 再 compile（图级适配）。下表方向与档位一律「本组合观测」，
**判定以最近一次实测为准**（历史结论只作参照，不得替代本任务复测）。

**③ 使能异常定位与回修**：见 §1.5。

### 1.2 两阶段顺序纪律（API 优先 → compile 适配）

1. **阶段 1 · API 接入（runtime 注入）**：在**最小改动面**上把 mindiesd 单算子接到调用点。
   落点三选一——框架配置注册表（`rms_type` / `rope_type` / `*_REGISTER`）、框架侧算子 dispatch 点、
   或**直接改模型代码**（模型 / 层源码内 `import mindiesd` 并替换调用点）。**不预设"无注册表就只能
   走 compile"**。出口条件：接口签名与几何约定对得上、同 seed 数值对拍通过（等价或落在该算子
   量化级容差内）、单点收益方向明确。
2. **阶段 2 · compile 适配接入**：接口确认可行后**再**走 mindiesd 的 compile 机制做**图级**适配
   （新增 / 适配 pattern、图形态变体、`_compiled_call_impl` 写入、平台注册 backend）。
   适配动作清单与开关治理见 `pattern-dev/references/fusion-enablement-notes.md`
   「compile 阶段的适配动作」（1→7 顺序不可跳）。出口条件：真图命中（pattern 计数）→ kernel diff
   （融合 kernel 出现、被替代 kernel 消失）→ 数值核验（eager vs compile）→ 墙钟收益过噪声阈值。

**为何 API 先行**：① 改动面小、成本低（pattern + 注册链 + 后端是两侧成倍工作量）；② 失败早暴露
（接口签名 / 几何约定 / 自研算子部署顺序错配在最小改动下立即暴露，不被 compile 的图形态问题掩盖）；
③ 先拿到可归因的单点收益（eager 侧收益是后一阶段的分母与数值对拍基线）。

**为何 API 不替代 compile**：API 只作用于被替换的调用点（层内单算子），框架侧的拷贝 / 调度开销与
**未展开的整链融合**仍在；compile 才能在图级命中原生 API 无法表达的多算子链，并消减
functionalization 引入的拷贝（见 `pattern-dev` Phase 7）。

**何时停在 API 阶段即可（按证据判定，不留半开状态）**：

- **收益已达标**：eager 单算子已覆盖热路径，再叠 compile 的 kernel 级结果为负 ⇒ 停在 API 并记录回退
  （本组合观测：vLLM-Omni 侧 FA / norm / rope / adaln 由单算子承载，叠 compile 无正收益）。
- **compile 无正收益或改数值语义**：受控交错 A/B 落在噪声阈值内（<3%）⇒ 回退；或图级编译改变数值
  语义（输出**非逐字节**，如 compiled FA 的数值 / seed 语义）⇒ 即便有方向性收益也按「非无损」回退。
- **适配代价落在探针侧**：接口可行性已验证，但 compile 适配需改框架仓且未合入上游 ⇒ 按 `[探针]` 标注
  （经验 vs 探针判定见 `.agents/README.md` §7），不进推荐姿势与报表宣称。

**收益量级参考**（两阶段的**相对贡献**，一律「本组合观测」；换框架 / 模型 / 负载规模 = 重判；
本表不记绝对耗时，机制与读数以下表「落点文件」为准）：

| 链 | 阶段 1（API / runtime 注入） | 阶段 2（compile 图级适配） | 落点文件（真源） |
|---|---|---|---|
| LightX2V × 视频 DiT（有注册表接口） | 注册表替换，**累计约一成** | 在其之上再拿**个位数百分点**（长视频档更小）；另有一项收益来自 a2a 留 eager 的通信项 | `lightx2v-enablement.md` §3.1/§3.2 |
| vLLM-Omni × 视频（无注册表接口） | eager 单算子（自动路由面）已覆盖热路径 | **无正收益 → 回退、默认关** | `vllm-omni-enablement.md` §3.1/§3.5 |
| vLLM-Omni × 图像 | 同上（单算子 + FA 路由） | 编译成功但方向性收益小、且**输出非逐字节** ⇒ 回退 | `vllm-omni-enablement.md` §3.5 |
| cache-dit × vLLM-Omni 0.26 | 框架原生 Cache / 量化开关 | 逐 block compile **噪声内无稳健收益 → 回退** | `cache-dit-enablement.md` §3.4 |
| DiffSynth-Engine × 图像（无注册表接口） | 模型层直接注入（RoPE 实数域改写、text encoder key 归一化）**是阶段 2 命中的前置** | 图级命中集合与端到端收益（个位数百分点量级） | `diffsynth-engine-enablement.md` §3.2–§3.4 |

- 各框架的**注入点 / 注册机制 / 合入形态**差异见 §3 的真源文件（同一「注入」动作落在不同层，
  姿势不可照搬）；本文件不复述逐框架机制。
- 使能判断用 dummy run 即可（pattern 匹配基于图结构）；使能集合与真实权重一致。
- **单次结果不迁移**：换框架 / 并行配置必须按回路重新验证。

### 1.3 框架侧验证（已部署模型，真实权重）

框架无关的验证骨架：**启动服务 → 健康探针 → 一次真实请求 → 校验产物（字节数 / 形状 / md5）**。

- **具体启动命令与端点选型**（vLLM-Omni 全栈）：`references/vllm-omni-enablement.md`
  §2.1（启动姿势 + 启动自检 curl + `/v1/images/generations` vs `/v1/images/edits` 端点选型）；
  Edit / I2I 模型（`QwenImageEditPlusPipeline` 等）**必须走 multipart edits 端点**，
  否则 500 `Missing preprocess images`。
- **启动期环境类异常**（缺 X11 库 / 权重分片缺失 / HCCL ranktable / 设备版本变量）：归 env-install 与
  本技能 troubleshooting → `env-install/references/troubleshooting-env.md` §H、
  `references/troubleshooting-vllm-omni.md` §E1/E2（**本文件只留指针，不复制命令**）。
- **权重落位**：按 `env-install/references/weights-prep.md` §2.2「落位约定」书写
  （模型根目录直接 serve，模型名之下不再加厂商 / 组织层；框架侧历史写法已弃用）。

**验证通过标准**（框架无关四项）：推理无异常（无 `RuntimeError` / `OOM` / `CANN error`）；输出合法
（shape > 0、非全零、产物字节数与预期一致）；显存峰值 < 物理显存 90%；特性叠加的开关生效**且过
§1.4 计数契约**（量化看权重精度与 `--quantization` 档位、稀疏看 sparsity 参数、Cache 看命中日志，
框架逐步计数器未填充时用 kernel 级计数兜底）。

### 1.4 计数契约（真实性核验，防 no-op 假加速）

宣称收益前，除"开关生效"外还须有技术真实参与的**运行期计数**并写入产物——缓存复用次数、
稀疏 / 量化 kernel 调用数、融合 kernel 实际执行次数等；可设 fail-closed 断言
（如「预期 dense N / sparse M / actual kernel M」，不符即运行契约失败）。计数口径记入
`model-auto-optimization/references/artifact-layout.md` 的产物（technique_counters）。

- **框架的逐步统计属性未填充时，用 kernel 级计数契约兜底**：若框架自身暴露的逐 step 计数器
  （缓存命中步 / 累计属性）在该框架路由下未填充（改三方仓补 glue 需另行确认），不要因此判
  「特性无效」——改用 **kernel 级计数**作真实参与证据：被跳过的阶段应表现为相关 kernel 分类计数
  **骤降一个量级**（而非归零或持平），实现「步级跳过」类特性（Cache 等）的等价契约。
- **三层证据（可靠性递增）**：`MINDIE_LOG_LEVEL=DEBUG` 日志 → `graph_log_url` DOT 图 →
  `kernel_details.csv`。
  - ⚠️ 日志 2048 字符截断会误判 0 命中 → 以 DOT 图 / kernel_details.csv 为准
  - ⚠️ 图命中 ≠ 运行期全部生效：`residual_gate_add` 对 4D attention 张量运行期 fallback，
    需看融合 kernel 实际执行次数
  - 不做耗时比较：dummy 与真实权重的耗时差异由层数主导，耗时评估必须真实权重 + kernel diff
    （证据走 profiling-collect → profiling-analyze）
- **收益判定用受控交错 A/B + 噪声阈值**：同卡组同日做 C1→E1→C2→E2 交错、n≥4 取中位消除热漂移；
  差异落在噪声阈值内（本仓惯例 <3%）**不宣称收益**，判「回退（默认关）」，不留半开状态
  （口径与归档见 `perf-gate`、`cache-dit-enablement.md` §3.4）。
- **稀疏 / 融合的「路径区分」先于收益判定**：eager 公共 API（外部逐层 mask，如 rf_v3）与融合算子
  （op 内 mask+BSA 一体，如 EagleQBSA）是两条不同实现，墙钟 / 收益口径不可混称；先 op 级微基准
  （同几何对拍 dense vs 目标路径，`import mindiesd` 前置，warm+稳态取 ms）确认 kernel 价值，再接线
  e2e 复验（方法见 profiling-analyze heuristics「Host/Kernel 与数据搬运归因」）。
  - **eager 外部 mask 路径的收益边界**：该路径「能生效但收益有限」——mask 选择逻辑开销可忽略，
    大头是每层每步的全尺寸 rearrange/pool 数据搬运（随层数线性放大）⇒ 只省下部分步时，且前缀
    dense 步进一步摊薄端到端收益，**op 级单调用甚至可能慢于 dense**。故：融合 op 存在时优先融合 op；
    未接线时其收益**只能先按 op 级证据固化**，e2e 组合为**待决项**。
  - **宣称纪律（fail-closed）**：必须有**本步 kernel 计数证据**（dense FA 计数下降 + 稀疏 kernel 出现）
    **或**同几何同 sparsity 的 op 级对拍；**无证据不宣称「稀疏生效」**，兜底日志（如 `staying dense`）
    与 lossless **逐字节相同**时按「未生效」处置（≠「收益小」）。
- **有损档**（缓存 / 稀疏 / 量化）过 `accuracy-gate` 的质量门；探针类修复标 `[探针]`。

### 1.5 使能异常定位与回修

1. 定位**算子执行差异**：图命中 vs 运行期实际 kernel（`graph_log_url` DOT 图 →
   `kernel_details.csv`）、数值 / 精度差异、shape / 布局差异
   （如 4D attention 张量使 `residual_gate_add` 运行期 fallback）。
2. 判定根因（三类，指向不同修复侧）：
   - **mindiesd 侧**：pattern 实现错误 / 注册缺失 / kernel 语义不符 → 指向 `pattern-dev` /
     `operator-dev` 修复；
   - **mindiesd 自研算子部署侧**：自定义 CANN 算子产物只在 repo `mindiesd/ops/vendors/*`，运行期靠
     `import mindiesd`（env.py 设 `ASCEND_CUSTOM_OPP_PATH`）——**必须先 import mindiesd 再初始化
     NPU / 建张量**，否则 `aclnnXxx … inferShape function does not exist`（部署 / 顺序问题，
     非参数问题；**部署校验与 golden 通过判据**见
     `operator-dev/references/custom-op-runtime-deploy-verify.md`，框架侧同款姿势见
     `cache-dit-enablement.md` §2.2）；
   - **框架适配侧**：调用姿势（`_compiled_call_impl` 未赋值）、图形态与 pattern 期望不符、开关未生效、
     Dynamo 重编译（backend 实例未复用）→ 本技能内修复框架侧适配（§1.6）。
3. 修复后复验：重新采集 + kernel diff + 墙钟（profiling-collect → profiling-analyze），确认根因消除。

### 1.6 接入姿势（运行时算子接入 / compile 融入）

**运行时算子接入（阶段 1）**：把算子替换为 mindiesd 单算子（如 `npu_rms_norm`、`npu_rotary_mul`）；
有注册表接口的框架改配置即可，无接口的改模型 / 层源码直接 `import mindiesd` 替换调用点。
单点收益读数与口径以该框架 enablement 文件为准（绝对数字归档 `{run_results_dir}/archive/`）。

**compile 融入（阶段 2）跨框架通用姿势**：

- ⚠️ `torch.compile(submodule, backend=...)` **不赋值不生效**，必须写入 `submodule._compiled_call_impl`
  （等价 `nn.Module.compile()`）。未赋值时 warmup 与 eager 一致、pattern 0 命中。
- **backend 实例复用**：每次新建 `MindieSDBackend()` → Dynamo `BACKEND_MATCH failure` 反复重编译
  直至 `recompile_limit` 后静默退回 eager；须缓存单实例。
- **compile backend 经平台注册表注入（推荐形态）**：不要在框架核心硬编码第三方包名；平台侧注册懒
  工厂、core 只查注册表、未知名 **raise** 不静默回退，实例仍须缓存单份（先例：LightX2V #1471）。
- **collective 留 eager（防 a2a 退化）**：多卡序列并行（USP / Ring）下若把 collective 编进图，
  `split_sizes` 可能被推断错导致**变长 alltoall 路径退化**（`hcom_alltoallv`，单次耗时比固定等分
  alltoall **放大数十倍**，比例关系）→ 用 `torch._dynamo.disable` / `torch.compiler.disable` 把
  collective 留在 eager（graph-break），计算段继续融合。⚠️ **优先框架原生可选机制，不要全局改共享
  collective 类**（合入版把"全局钉 disable"收敛为平台注册 + 配置可选）。
- **图形态差异**：同一算子在不同框架 Dynamo 展开形态不同（如 `chunk(2)` 在 diffusers 是单 split 双
  getitem、在 LightX2V 是两个独立 split 节点）→ pattern 需加变体（`pattern-dev`）；诊断用
  `graph.print_readable()` 对比 pattern 期望与真实图。
- 逐框架的具体改动点不在此复述：DiffSynth-Engine 的 3 处适配见
  `references/diffsynth-engine-enablement.md` §3.1/§3.2，LightX2V 见 `references/lightx2v-enablement.md`，
  vLLM-Omni 见 `references/vllm-omni-enablement.md` §3.5。

---

## 2. 分支 B：框架缺失 → 补齐开发

### 2.1 缺口判定与归属（框架 × mindiesd 合作界面）

模型优化的「预期关键路径」有时在目标框架自身缺失（无 comm-stream 掩盖、无缓存机制、无量化 kernel
消费者、Ring 不支持 attn_mask…）。**判据 = 该能力框架是否已存在**（已存在 → 回分支 A）。
归属**以代码落点（仓库）定侧**：

| 落点 | 归属 |
|---|---|
| mindiesd 仓文件（含 `mindiesd/parallel` 等本仓实现） | dev-workflow + `pattern-dev` / `operator-dev` / `aclgraph-dev` 开发子任务 |
| 三方框架仓文件：结构性能力（comm-stream masking、缓存框架、稀疏 / 量化消费者、平台注册后端…） | **本技能分支 B** |
| 三方框架仓文件：接线 / 小修（使能所需的最小 glue） | 分支 A（§1） |
| 重叠区（同一特性两侧都有改动） | **按文件拆两侧子任务**，互不越界，对接点统一收口 |

合作界面（注册表 / 平台分发 / 注入点 / 合入形态）随框架而异 → 两侧分别开发后在**对接点联调验证**
（输出一致 + 计数契约 + 三层证据，姿势见 §1）；该框架的合作界面差异记录进
`framework-support-matrix.md` 的姿势列，避免同一框架反复摸索。

### 2.2 补齐实现流程

1. **最小复现缺口**：用最小用例 / 框架入口复现「关键路径缺失」证据（报错 / 回退 / 无机制）。
2. **形态选择**（按 §0 用户确认执行，不静默切换）：见 §2.4 四档合入姿势。
3. **实现**：按 §2.3 的注入点分类落到目标框架的对应层（editable 安装改后即生效；无 `.git` 环境按
   env-install 的版本处理姿势）。
4. **验证**：输出一致性 + 计数契约（预期计数 fail-closed）+ 三层证据（图命中 / kernel diff / 墙钟，
   见 §1.4）；有损类补齐过 `accuracy-gate` 质量门。
5. **收口**：合入上游时遵循评审收敛姿势——common 零改动、平台注册 + 配置可选、未知名 **raise**
   不静默回退、backend 实例单例、collective 留 eager（先例：LightX2V #1471 `hccl_eager`）。
6. **回填**：更新 `framework-support-matrix.md`（状态 / 新特性行 + 证据码 + 版本 + 日期）、
   在对应框架的 `{framework}-*-enablement.md` / `-notes.md` 补开启方式与坑
   （`-case.md` 类别已取消，见 `.agents/README.md` §7 四分类），需要时增补本技能 evals。

### 2.3 注入点与代码组织（分类，逐框架实例见 §3 真源）

| 注入点类型 | 载体 | 适用 |
|---|---|---|
| 配置注册表替换 | 框架配置字段（`rms_type` / `rope_type` / `*_REGISTER`） | 框架有抽象接口 ⇒ 阶段 1 零代码 |
| 框架侧算子 dispatch | 平台实现内的 CustomOp dispatch | mindiesd 进 venv 后自动路由，不改模型代码 |
| 模型层直接改写 | 模型 / 层源码内 `import mindiesd` 并替换调用点 | 无配置接口时的阶段 1 落点；也是阶段 2 命中的前置 |
| `_compiled_call_impl` 原地写入 | 逐 submodule 编译并写回 | compile 接入（不赋值不生效） |
| 平台注册表注入 backend / 后端 | 平台侧注册懒工厂 + 配置字段 | 推荐形态：core 不硬编码第三方包名、未知名 raise |
| env 门控 fork 适配 | fork 补丁 + env 开关 | 未合入上游 ⇒ `[探针]`、默认关 |

> 表格给的是**注入点类型学**（跨框架复用）；`LightX2V` / `vLLM-Omni` / `DiffSynth-Engine` 各自
> 落在哪种类型、具体代码地图与注册机制，**单点在 §3 的 per-framework enablement 文件**。

### 2.4 合入姿势（四档，按影响面从小到大收敛）

| 档 | 形态 | 适用 / 代价 |
|---|---|---|
| ① **平台注册 + 合入上游**（推荐） | 能力经注册表 / 平台实现接入，common 零改动；随上游 PR 合入 | 影响面最小、可持续；需按上游评审收敛（未知名 raise、配置可选、单例） |
| ② **上游 PR**（不经平台注册层） | 直接改框架公共路径并追求合入 | 公共路径改动影响所有平台，评审收敛周期长 |
| ③ **fork 本地补丁（钉 `base_commit`）** | fork 仓补丁 + 版本钉死 | 快速可用；**不得当最终能力宣称**，回填须标 `[探针]` |
| ④ **monkey-patch（运行期注入）** | 运行期替换对象 / 方法 | 漂移风险最高，仅作验证与过渡；默认关 + 保留 `.bak` |

- 选择依据：**改动归属（改哪个仓）+ 是否接受改三方框架仓 / 合入上游**，由 §0 用户确认后落定；
  同一任务内不静默换档（换档要留记录）。
- 先例（可照抄的收敛姿势）：LightX2V #1471 把「全局 `@torch._dynamo.disable` 改共享 collective 类
  与 core 硬编码 mindiesd 包名」收敛为**平台注册表 + 配置可选键**（`hccl_eager` /
  `COMPILE_BACKEND_REGISTER`），common 零改动。

### 2.5 关键纪律

- **框架能力版本漂移**：实现前先钉框架 `base_commit`；合入上游前不得把 fork 结果当最终能力宣称。
- **结构性实现不并入 enablement 的 case**：enablement 只做接线 / 小回修，结构性能力各自单点维护。
- **最小复现先行**：无缺口证据（报错 / 回退 / 无机制）不动手，避免把"配置没开"误判为"框架没有"。
- 本技能只提供「怎么补」的知识与流程；**是否补、补到哪一档**由 §0 的确认决定。

---

## 3. 框架差异真源与 references 登记（加载时机）

> **同一框架的开发范式 / 注入点 / 注册机制 / 合入形态 / 坑，只写在下面这一份里**——本 SKILL 与
> `framework-support-matrix.md` 只做指针与状态，不复制机制细节；分支 A（使能）与分支 B（补齐）
> 共用同一份（"注入点速查表"不另存第二份）。

| 真源文件 | 覆盖（框架差异 / 机制） | 加载时机 |
|---|---|---|
| `references/vllm-omni-enablement.md` | vLLM-Omni 0.28：画像与版本边界、启动与并行（`--omni` 姿势 / `--text-encoder-tp-size` 契约 / worker 残留清理 / DLO offload）、特性开关面板（自动路由面 / 量化 w8a8·mxfp8 / 稀疏 `RAINFUSION_ATTN` 几何前置 / Cache `cache_dit` / `torch.compile` 接入点 / 计数契约）、回修与 `[探针]` 清单 | 使能或排查 vLLM-Omni 侧特性时 |
| `references/diffsynth-engine-enablement.md` | DiffSynth-Engine：无算子注册表接口 ⇒ 走 compile；`compile_backend="mindie"` + `_compiled_call_impl` 原地写入、backend 实例复用、RoPE 实数域改写 / text encoder key 归一化（命中前置）、三层证据、与 LightX2V 的差异照搬清单 | 接入 DSE / 排查其命中与使能时（原 `-case.md`+`-notes.md` 已合并进本件） |
| `references/lightx2v-enablement.md` | LightX2V（#1471 合入版）：注册表接入、平台注册懒工厂 + 配置可选键、代码地图、三条必修坑（a2a 不进图 / backend 单例 / swiglu 双 split 变体）、`[探针]` 清单（含框架侧结构性实现的跨侧指针 → §2） | 接入 LightX2V 或照抄其合入姿势时 |
| `references/cache-dit-enablement.md` | cache-dit × vLLM-Omni 0.26：trunk 托管链画像与版本边界、自研算子部署前置（"先 import mindiesd" 顺序）、Cache 主导旋钮、稀疏两条路径、受控交错 A/B、kernel 级计数契约兜底 | cache-dit / 缓存类特性使能时 |
| `references/train-aware-lossy-method.md` | **跨框架通用方法（与具体产物隔离）**：训练感知有损的分类与归组判据 → 四步使能回路 → 三条前置契约（步数语义 / 装载计数防 no-op / 组件契约先读参考实现）→ 判定三件（比值法 + 逐步墙钟 + 质量分层，**PSNR 不可判接错**）→ 归因链 → 数字纪律 | S5 训练感知类任务（少步蒸馏 / 换解码器） |
| `references/vllm-omni-train-aware-enablement.md` | vLLM-Omni 训练感知开启方式：少步蒸馏 LoRA 链（仅 0.28 有；`model_index.json` 不得 pin `base_schedule`；装载计数日志契约 `num_modules=259` / rank `0 -> 64`）、`VAE解码替换` 框架未提供 ⇒ fork 探针、视频 VAE 帧数契约、少步档 Cache 结构性失效 | 该框架训练感知档 |
| `references/cache-enablement-pattern.md` | 三方框架用库内 CacheAgent / DiTBlockCache / AttentionCache 做有损缓存的通用姿势（接入三要素 + 验证三件套 + 坑速查） | 框架无原生缓存、改用库内缓存时 |
| `references/run-entry-and-request-tiers.md` | 部署完成后的**运行入口与档位**：vLLM-Omni serve 目标二选一 + **请求级 `extra_params.task` 档位切换**；LightX2V `torchrun` 入口 + **`--config_json` 档位语义**（并行档位语义指回 `dit-parallel-opt`）；使能异常时的**回退姿势**（选哪一档退归 `dit-perf-opt`） | 需要"起服务 / 换运行档 / 请求级切 task / 档位报错回退"时 |
| `references/troubleshooting-vllm-omni.md` | vLLM-Omni 构建 / 启动 / 运行期异常（§E1 构建、§E2 启动、§E3 运行期；编排层口径见其开头指针） | 该框架异常排查时 |
| `references/framework-support-matrix.md` | 跨框架支持状态矩阵 + 证据码 + 版本（特性命名以 `docs/zh/features/*` 为准） | 使能前查支持面 / 分支 B 缺口判定 / S4 组合前裁剪候选 / 框架或 docs 版本升级后刷新 |

**接线（非本技能内容，按需跳转）**：证据（采集 / 分析 / kernel diff）→ `profiling-collect` /
`profiling-analyze`；性能与精度验收口径 → `perf-gate` / `accuracy-gate`；
实现层改动（pattern / 算子）→ `pattern-dev` / `operator-dev`；快速验证 → `dummy-run`；
环境与权重 → `env-install`（`references/{troubleshooting-env,weights-prep}.md`）；
产物目录 / 运行态 / 报表契约 / 编排阶段 → `model-auto-optimization`（`references/{artifact-layout,run-state}.md`）。

> 单次结论不迁移：案例中"某修复必要 / 有效 / 无效"仅在该框架 + 并行配置下成立；
> 换框架 / 换配置必须按 §1 回路重新验证。

## 4. 存量待处置 references（7 件；政策待父 agent 统一处置）

`.agents/README.md` §7 已取消 `-case.md` 类别（案例按「方法 / 开启方式 / 底座 notes / 实测出库」四分类
落位，实测表先导出到会话产物目录再删 skills 内副本）。以下 7 件为**存量未迁移**，**当前不作为推荐
加载入口**（`README.md` §7 需与现状对齐后统一处置，本批次不擅自删除）：

| 存量文件 | 建议处置（报父 agent） |
|---|---|
| `references/diffsynth-engine-case.md` | 与 `diffsynth-engine-enablement.md` / `-notes.md` 合并（README “批 3 收尾”记为已合一，需核对）→ 实测表出库后删 |
| `references/cache-dit-minimax-h3-case.md` | 实测表出库（`{run_results_dir}/archive/`），结论已在 `cache-dit-enablement.md` → 删 |
| `references/lightx2v-mindiesd-case.md` | 同上（README 已记归档路径）→ 删 |
| `references/vllm-omni-case.md` | 并入 `vllm-omni-enablement.md` 后删 |
| `references/vllm-omni-minimax-h3-case.md` | 并入 `vllm-omni-enablement.md` §3/§5 后删 |
| `references/vllm-omni-qwen-image-case.md` | 并入 `vllm-omni-enablement.md` §3.5 后删 |
| `references/diffsynth-engine-notes.md`（非 case） | README 记两件已并入 `diffsynth-engine-enablement.md`，但本文件仍在库 → 核对其独有内容后并入删；若保留则补登 §3 |

> 处置纪律：先按 README §7「经验 vs 探针」三问分类，再「实测表出库 → 删 skills 内副本」；
> 删除动作留给统一执行该政策的批次，本技能只登记现状与建议。

## 5. 故障排查

- 通用部署问题（环境依赖、缺 X11 库、权重分片缺失、装库 / 权重缺口）→
  `env-install/references/troubleshooting-env.md`（§H 与正文）
- SSH / CRLF / 传输域 / 本地开发机 TLS（schannel）→ `remote-access/SKILL.md`「故障排查」表与
  `remote-access/references/transport-troubleshooting.md`
- 自研算子部署顺序与部署校验（golden）→ 本文件 §1.5 与
  `operator-dev/references/custom-op-runtime-deploy-verify.md`
- 多卡运行期劣化（SIGKILL 残留态 / 端口 bind / 热降频与 clean-window 口径 / 拓扑选卡）→
  `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §1–§5
- vLLM-Omni 构建 / 启动 / 运行期异常 → `references/troubleshooting-vllm-omni.md`（§E1–E3）
- 编排层口径（基线冻结 / md5 口径 / 产物同目录 / 报表）→ `model-auto-optimization/**`
  （`workflows/optimization-flow.md`、`references/{run-state,artifact-layout,overview-report}.md`）；
  测量口径 → `perf-gate`

## 6. Reference Files

- 本技能的 references **全部单点登记于 §3**（真源 / 覆盖 / 加载时机），此处不重复登记；
  `references/framework-support-matrix.md` 为状态查询入口，其余为框架差异与方法件。
- 跨技能接线（环境 / 证据 / 验收口径 / 实现层 / 快验 / 编排）→ §3 末「接线」段。
- 开发流程与复盘规范 → `dev-workflow/SKILL.md`。

## 7. Bundled Scripts

- `scripts/ascii_luma_preview.py` — 解码输出亮度预览与对拍（零依赖、零 NPU、零网络）：读帧目录出
  逐帧聚合指标（均值 / 标准差 / 钳位占比）+ ASCII 亮度图，或两目录同尺寸帧对拍（逐帧灰度相关系数）。
  加载时机: 质量门禁只能记 `inconclusive`（尤其「无图像输入能力」成因）而需留下「可机读 + 可人眼读」
  的视觉存证时——把「无法看图」与「质量通过」明确分开。材料落 `{run_results_dir}/`。

## 8. 维护与更新

当 vllm / vllm-ascend / vllm-omni 版本或框架接入姿势变化、新增框架使能 / 回修 / 补齐经验、
或融合 pattern 使能集合变化时，按 dev-workflow 的复盘流程更新本 skill；框架特有开启方式与坑写入
对应 `references/{framework}-*-enablement.md`（实测数字入会话产物归档），正文只保留两分支共通回路与判据。

> 体积提示：本 skill 是能力层中负载最重者。若继续膨胀，优先把框架特定章节下沉为
> `references/{framework}-{variant}.md`（渐进披露），正文只保留 §0 分支判定与 §1/§2 共通判据。
> 边界复查（改名后）：**分支 A** = 开启 / 使用框架**已存在**的特性（含为使能的最小 glue 修复）；
> **分支 B** = 框架**未支持**特性的从无到有开发（合入姿势见 §2.4）；**mindiesd 仓内开发** →
> dev-workflow + `pattern-dev` / `operator-dev` / `aclgraph-dev`。
