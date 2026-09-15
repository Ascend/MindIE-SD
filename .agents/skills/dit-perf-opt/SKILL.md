---
name: dit-perf-opt
compatibility: 已安装 mindiesd, `docs/zh/features/*`（特性真源，仓内直读）, NPU 复验
description: >
  DiT 计算模块（L3）：把**已定位的 DiT 计算瓶颈**落成特性级选档与实施——
  量化档（W8A16 / W4A16 / W8A8 系列 / W4A4 / MXFP8 / FA 量化）、稀疏（rf_v2 / ada_bsa）、
  缓存（DiTCache / AttentionCache / 时间步优化）、编译启用（MindieSDBackend / Pattern 融合 / ACLGraph）
  的**开不开、开哪一档、怎么开、怎么复验**；依据是 `docs/zh/features/*`（特性真源）+
  framework-integration/references/framework-support-matrix.md（支持状态）。
  即使用户只说"这个模型怎么加速""量化/稀疏/Cache 怎么选怎么开""要不要开量化、开哪一档""这个档位开了有没有效果"
  而未提 profiling，也应触发。
  **入口条件**：瓶颈点已明确（用户带一句实测锚点，或编排层交付标签）时由域入口
  `performance-optimization` 按标签分发到本技能；**瓶颈未明（"怎么加速 / 跑通 / 采 profile"）先走
  `model-auto-optimization` 定位**，不在本技能内做占比分析。
  near-miss：多卡并行形态 / 通信掩盖 / TP·offload 选型 → `dit-parallel-opt`；VAE 解码段与
  host 固定开销 → 各自模块（VAE / host）；单算子实现级实测选型（mindie_bench）→ `benchmark-dev`；
  需要新增 pattern / 算子才能落地本档 → `pattern-dev` / `operator-dev`；框架侧开关与使能验证 →
  `framework-integration`；量化器位级契约与精度对齐（编码公式 / 舍入 / scale 粒度）→
  `quantization-dev`；精度验收判据 → `accuracy-gate`；数字入库口径 →
  `perf-gate`。
---

# DiT 计算侧选档与实施

## 定位

本技能是**优化域的 DiT 计算模块**：输入是**已定位的 DiT 计算瓶颈**（标签 `DiT-计算受限`），
输出是**可复验的特性档位组合**。范围只含 DiT 主体（Transformer block 的 MatMul / Attention /
Norm / 激活级）的计算侧手段；**不含**并行与通信（→ `dit-parallel-opt`）、**不含** VAE 解码段与
host 固定开销（→ 各自模块）。

不做的事：**不重新定位瓶颈**（占比分析、阶段账、标签判定归编排层 `model-auto-optimization`）、
**不实现新算子/新 pattern**（归 `pattern-dev` / `operator-dev`）、**不定义验收标准**
（性能口径归 `perf-gate`，精度判据归 `accuracy-gate`）。

## 优化闭环

```text
建立基线 → 瓶颈分析 → 根因定位 → 保守修补 → 复验
   ↑                                              │
   └──────────────────────────────────────────────┘
```

### Step 1: 建立基线

使用 profiling-collect 采集 + profiling-analyze 分析建立基线，记录模型 / 分辨率 / 帧数 / 精度 /
NPU 数等配置。基线必须与待验档位**同窗可比**（跨窗口绝对值不可比，口径见
`perf-gate`）。

### Step 2: 获取分析诊断

从 profiling-analyze 的 5 层分析报告中获取：

- Layer 1: 瓶颈阶段（DiT vs VAE）
- Layer 2: 算子分类占比（FA/MatMul/Vector/Comm）
- Layer 3: Host Bound / 通信暴露 / 融合机会
- Layer 5: 优化方向（P0-P2 优先级 + 引用 `docs/zh/features/*` 章节）

分析报告给出的是**优化方向**（如"量化方向"、"缓存方向"），**具体档位在本 Step 选取**。
若报告或标签指向的瓶颈不在 DiT 计算侧（通信 / VAE / host / 框架侧未使能），按下方
「与相邻模块的边界」转交，不在本技能内硬做。

### Step 3: 选取具体方案（选档）

基于分析报告的优化方向，查 `docs/zh/features/{quantization,sparse,compilation,cache,cpu_offload,parallelism}.md`
对应节确定具体 API 和参数，支持状态查
`framework-integration/references/framework-support-matrix.md`：

```text
正例: "分析报告显示 MatMul 占 DiT 58%，优化方向→量化。
       查 docs/zh/features/quantization.md §Linear量化，选取 W8A8_MXFP8"
反例: "感觉矩阵乘法比较慢，试试量化"
```

选择时需考虑：

- 硬件约束（docs 对应节的硬件列）
- 模型兼容性（`framework-support-matrix.md` 的支持状态）
- 精度 vs 速度权衡
- 有损 / 无损归属：无损档优先；有损档的判据与阈值引 `accuracy-gate`
- 多方案时按优先级：MindIE-SD Pattern > 量化 > 稀疏 > 缓存 > 通用

**单一真源纪律**：特性 API / 算法名 / 硬件约束只从 `docs/zh/features/*` 读；支持状态只从
`framework-support-matrix.md` 读；**不得引用任何仓内 docs 镜像副本**（镜像会过期，且已删除）。

### Step 4: 实施 + 验证

优化方案从 `docs/zh/features/*` 中选取，按档位落地的决策树见
`references/optimization-dimensions.md`。

| ✅ 允许 | ❌ 禁止 |
|---------|---------|
| 启用已有的、经验证的 kernel | 削弱输出正确性（cosine similarity 下降） |
| 修复遗漏的 fast path | 改变测试负载后宣称优化有效 |
| 减少不必要的同步/warmup | 仅为单框架/单硬件优化而破坏兼容性 |
| 添加有证据支撑的启发式配置 | 从单一 trace 数据得出普适结论 |

**使能成功由本技能负责**（"开了 ≠ 生效"）：档位声明开启后须给出该特性**确实参与**的证据
（图命中 / kernel diff / 特性 active 计数 / 抽样步 kernel），**不得只看墙钟**；
与无损基线的产物**逐字节相同** ⇒ 判**未生效**（登记为"能力未生效"，**不得**记作"收益近似为零"）。

多特性组合试验（叠加顺序 / seam 冲突 / 层回退 / 必测覆盖集）见 `references/combination-search.md`。

### Step 5: 复验

- 重新运行 profiling-collect + profiling-analyze 复验相同配置
- 重新运行 profiling-analyze 确认 5 层分析指标变化
- 差距 **< 3%** 视为噪声（噪声阈值与停止条件为**域级口径**，单点维护于域入口
  `performance-optimization` §5，本技能只引用）
- 有损档另过**三级精度验收**（逐位 → 跨配置数值门 + md5 → 质量门），判据归
  `accuracy-gate`：`../accuracy-gate/references/quality-gate.md` + 仓库 `evals/`；
  **只过墙钟不过门禁不得宣称有损加速**
- 数字入库前按 `perf-gate` 的同窗 A/B 口径复核（只有验收结果可写入总览表）

## 优化维度

→ `references/optimization-dimensions.md`（决策树：编译路径 / Attention / MatMul / 显存 / 缓存）
→ `docs/zh/features/*`（特性 API/算法真源，按需直读）
→ `framework-integration/references/framework-support-matrix.md`（框架侧支持状态）

## 停止条件

停止条件（目标达成 / 噪声范围 / 外部瓶颈 / 硬件瓶颈）与 **3% 噪声阈值**是**域级口径**，
单点维护在域入口 `performance-optimization` §5，本技能不另立一套。

## 与相邻模块的边界

| 情形 | 去向 |
|---|---|
| 瓶颈是**多卡并行形态 / 通信掩盖 / TP·offload** | `dit-parallel-opt` |
| 瓶颈在 **VAE / TAE 解码段**（计算或通信） | VAE 模块（`vae-opt`，计算 + 通信同技能） |
| 瓶颈是**交付/搬运/装载/预热等固定开销** | host 模块（`host-opt`） |
| 需要**新增 pattern / 融合 / 算子**才能落地该档 | `pattern-dev` / `operator-dev` |
| 框架侧**开关未使能 / 生效验证** | `framework-integration` |
| 量化器**位级契约/精度对不上**（编码公式、舍入、scale 粒度） | `quantization-dev` |
| 需要**同一算子多实现实测对比**（同 peak 口径） | `benchmark-dev`（结论回填本技能 Step 4 作实施证据） |
| 瓶颈**尚未定位** | `model-auto-optimization`（唯一有分析权） |

## Reference Files

- `references/optimization-dimensions.md` — 加载时机: 确定优化方向、按档位落地的决策逻辑时
  （编译路径 / Attention / MatMul / 显存 / 缓存；阈值只引用 profiling-analyze 与验收标准）
- `references/combination-search.md` — 加载时机: 需同时开启 ≥2 个有损维度时
  （seam 静态判定 + 必测覆盖集 + 单变量叠加 + frontier 保留 + 层回退）
- `references/quant-tier-device-mapping.md` — 加载时机: 需要核对**档位名 ≠ 实际算法**（同一档名在不同设备代际映射到不同算法/精度档）、或用户问"这个档位在 A2 / 950PR 上到底是什么实现"时（自 `dummy-run` 下沉的选档语义）
- `references/resource-fallback-tiers.md` — 加载时机: **显存不足（OOM）要选一组降档顺序时**，
  或**某档使能后算子 crash / 劣化 / 静默无效、要判"退到哪一档"时**
  （显存档位表 + 回退顺序 + 回退判据；使能验证与回退姿势归 `framework-integration`，
  部署可见性 / golden 校验归 `operator-dev`）
- `docs/zh/features/*`（仓内真源，非本 skill 文件）— 加载时机: 确定具体 API/算法/硬件约束时
  （量化→`quantization.md`、稀疏→`sparse.md`、编译→`compilation.md`、缓存→`cache.md`、
  显存→`cpu_offload.md`；支持状态查
  `framework-integration/references/framework-support-matrix.md`）
- `../accuracy-gate/references/quality-gate.md`（跨技能，质量门本体归属精度验收标准）—
  加载时机: 有损档的端到端质量判定时（定量 + 视觉伪影 + off-identity；判定标准与工具在仓库 `evals/`）
- `../perf-gate/SKILL.md`（跨技能）— 加载时机: 报数 / 入库 / 验收时

## 维护与更新

当新的 DiT 计算优化维度经验证有效、硬件平台升级或发现新的优化模式时，
按 dev-workflow 的复盘流程更新本 skill；档位与阈值变化时同步 `docs/zh/features/*` 与本 skill
Step 3/Step 5，不在此处另立一套阈值。
