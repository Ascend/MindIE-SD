# 资源不足与使能异常时的档位回退（选档）

> **加载时机**：显存不足（OOM）需要选一组"先试哪个"的降档顺序时；或某档使能后**算子 crash / 明显劣化 /
> 静默无效**、需要判"退到哪一档"时。
> **边界**：本文件只做**选档**。使能本身与生效验证（计数契约 / 三层证据 / 回退姿势怎么做）归
> `framework-integration`；瓶颈定位与占比分析归 `model-auto-optimization` / `profiling-*`；
> 读数口径与入库归 `perf-gate`；精度判据归 `accuracy-gate`。
> **数字纪律**：本文件不写绝对耗时 / 显存 / 带宽读数与百分比——本组合观测读数的归档坐标见
> `framework-integration/references/*-enablement.md` 的产物节（`{run_results_dir}/archive/`），
> 拓扑与带宽口径单点在 `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md`。

## 1. 显存不足（OOM）时的候选档（改动面从小到大）

**先分段，再选档**：OOM 发生在**构造阶段**（`from_config` 等，host 侧）还是**推理阶段**（device 侧）？

- **构造阶段 OOM**：不是"降档"问题，走构造方式（如 meta → `to_empty` 构造）——降分辨率 / 关 CFG
  对构造期内存无效。
- **推理阶段 OOM**：按下表取候选档；顺序 = 改动面从小到大，**先试不改变输出语义的档**。

| 候选档 | 性质 | 归属 / 真源 |
|---|---|---|
| 降 batch / 并发 | 无损（只改负载） | `optimization-dimensions.md` §4；负载定义须与对照一致，否则不可比 |
| 降分辨率 / 帧数 / 时长 | 改负载（结果非逐字节可比） | 同 §4 决策树的显存分支；对照实验必须**同负载**才有意义 |
| 关 CFG（`guidance_scale=1.0`） | 有损（丢掉无分类器引导双分支） | 同 §4/§5；仅在 CFG 双分支确实占显存时用，并注明质量影响 |
| CPU offload / 分层 offload | 无损（时间换显存） | `optimization-dimensions.md` §4；接口 `docs/zh/features/cpu_offload.md`；框架侧 DLO 姿势见 `framework-integration/references/vllm-omni-enablement.md` §2.3 |
| TP / 并行分摊 | 无损（形态变化，跨配置非逐字节） | `dit-parallel-opt`（形态选择与拓扑判据） |
| Activation checkpoint | 无损（重计算换显存） | `optimization-dimensions.md` §4 |
| 层数裁剪 | **仅 dummy run** | `optimization-dimensions.md` §4；真实权重场景不得用它当"优化" |

**判据**：offload 类候档的收益上限由**未重叠通信/搬运占比**决定，形态变化会改变该占比 ⇒ 选档后按
`dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §7 重估；采纳与否按 `perf-gate`
同窗 A/B 结论，不以单次运行判定。

## 2. 使能档 crash / 劣化时的回退顺序

1. **先排除"不是档位问题"**：确认是**算子路径**还是参数 / 几何 / 部署顺序问题——
   同几何 op 级对拍（dense vs 目标路径）、`import mindiesd` 顺序与算子可见性检查
   （`operator-dev/references/custom-op-runtime-deploy-verify.md`）。
2. **按档位逐级回退**（保留开关与实现，不删代码）：关 compile / 关该融合档 → 关稀疏档 →
   关缓存档 →（多卡）回退并行形态；每步只动一个变量。
3. **判"留在哪一档"**：差异落在噪声阈值内、或图级编译改变数值语义（输出非逐字节）⇒
   **回退（默认关）**，不留半开状态；判据与措辞见 `framework-integration/SKILL.md` §1.2/
   §1.4 与 `accuracy-gate`。
4. **回退不是终点**：根因属 mindiesd 实现 → `pattern-dev` / `operator-dev`；
   属框架接线 / 结构性缺口 → `framework-integration`（分支 A / B）。
5. **记录**：回退结论按「经验 vs 探针」分类留痕（`.agents/README.md` §7），
   并注明作用域（框架 × 模型 × 规模 × 卡组）。

**不做的事**：不为了"让档位跑起来"而削弱输出正确性；不用跨窗口 / 跨负载读数证明档位优劣
（`perf-gate`）；不在本技能内做占比分析（归 `model-auto-optimization`）。

## 维护与更新

当新增降显存特性（docs `cpu_offload.md` / `parallelism.md` 变化）、出现新的"档位 crash"模式，
或回退判据（噪声阈值 / 数值语义）变化时，按 dev-workflow 的复盘流程更新本文件；
阈值与精度判据仍以 `performance-optimization` §5 / `accuracy-gate` 为单点，不在此另立一套。
