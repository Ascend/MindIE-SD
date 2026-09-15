# 分发路由表（瓶颈标签 → 优化模块）

> **归属**：`performance-optimization`（优化域入口，L2）。入口的**核心动作**就是按本表分发；
> 分发后由模块负责实施与自证生效，入口不复述模块内的步骤。
> **标签单一真源**：`model-auto-optimization/references/bottleneck-labels.md`（本表只**引用**标签名，
> 不重定义枚举与门限口径；该文件变更时本表同步）。
> **模块名说明**：本表用**当前生效的技能名**；技能改名时本表与入口 SKILL 的分发表随之同步
> （见文末「维护与更新」）。

## 1. 路由表

| 瓶颈标签（真源见 `bottleneck-labels.md`） | 分发目标（域内模块） | 需要新能力时的域外供给 | 产物 | 验收判据 |
|---|---|---|---|---|
| `DiT-计算受限` | `dit-perf-opt` | `pattern-dev`（需要新 pattern / 融合）、`operator-dev`（需要新算子 / kernel）、`quantization-dev`（量化器位级契约对不上）、`benchmark-dev`（单算子实现级选型） | 单特性档位记录 + 组合矩阵（含 `[MUST]` 覆盖清单）+ 回退名单 + 每档使能（生效）证据 | 性能：`perf-gate` 同窗 A/B；有损档另过 `accuracy-gate` 三级验收（含质量门） |
| `DiT-通信受限` | `dit-parallel-opt` | `framework-integration`（框架侧掩盖/集合通信路径未接线时） | 并行方案（形态 + 掩盖配置）+ 多 rank 证据 + 差异归因 | 同上；**掩盖达成率**须与 profiling 口径同窗，不得跨窗口比 |
| `非DiT-解码段` | `vae-opt`（VAE 计算 + 通信同技能） | `pattern-dev` / `operator-dev`（VAE 侧算子与编译能力） | VAE 方案（切分 / 状态携带 / 交换预算 / 计算手段）+ 一致性证据 | 同上；**未使用有损手段时一致性验收强制调用** `accuracy-gate` |
| `非DiT-host段` | `host-opt` | `env-install`（权重/镜像准备侧）、`pattern-dev` / `aclgraph-dev`（编译预热的图 pool 与缓存接线） | 固定开销账（交付搬运 + 装载预热逐项）+ 优化项 | 同上；占比口径与门限继承 `bottleneck-labels.md`，本表不另定 |
| `一致性不达标`（结构性改动后） | `accuracy-gate`（判据）→ 转对应对象的 `troubleshooting-{对象}.md` | — | 一致性结论（逐位 / 数值门 + md5 / 质量门三层）+ 首个分叉点 | `accuracy-gate` 的分级判据；排障产物按对象落位 |

## 2. 分发时的三条纪律

1. **先核前置集**：环境可用 + 模型已跑通 + 有 baseline，缺任一退回 `model-auto-optimization` 走 S0
   （入口 SKILL §3）。
2. **锚点随行**：分发时把锚点（阶段账某行 / kernel 占比读数）一并交给模块；无锚点不分发
   （入口 SKILL §4）。
3. **不越界**：标签与实测不符（模块复工发现真瓶颈在别处）→ 退回 `model-auto-optimization` 重新定位，
   **不在入口内改判标签、不跨模块硬做**。

## 3. 入库口径（所有行共用）

- 数字一律按 `perf-gate` 的同窗 A/B 口径取；**只有验收结果才能写入总览表**
  （报表契约见 `model-auto-optimization/references/report-contract.md`）。
- 与基线差距 **< 3%** 视为噪声（阈值单点维护于入口 SKILL §5）。
- 域外标准技能是**强制项**不是可选项：性能入库引 `perf-gate`，精度判据引
  `accuracy-gate`。

## 4. 新增标签 / 新增模块时怎么改

- 新增标签：先在 `bottleneck-labels.md` 定枚举与门限，再回填本表与入口 SKILL 的分发表——三处齐了才算接线。
- 新增模块：在本表加一行（含产物与验收判据），并把模块名登记到入口 SKILL §2 与「Reference Files」。
- **失效判据**：本表出现 `bottleneck-labels.md` 未定义的标签、或模块名与 `.agents/skills/` 实际目录不一致 → 视为接线失效。

## 维护与更新

- 触发：优化模块增减 / 改名、瓶颈标签枚举变化、验收标准接线变化时更新本表，并与入口 SKILL 同步。
- 校验：`git grep` 本表的模块名应能在 `.agents/skills/` 找到同名目录（找不到即接线失效）。
