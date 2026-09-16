---
name: dit-parallel-opt
compatibility: 无额外工具（参考数据来自 dummy-run/references/minimax-h3-notes.md §9）；多卡需 HCCL 环境
description: 分布式并行策略选型与实测（USP / CP 通信掩盖 / CFG / TP/RSP/PP 概览；含拓扑相关选型
             （按实测拓扑分域条件化：域内 bulk vs 跨域 head-parallel 翻转）与 AlltoAllV 缺陷绕过）。在
             model-auto-optimization 中承担 S3：优先 USP、结合拓扑带宽差异选 CP，少量 step +
             多 rank 验证特性开启与掩盖（产物：并行方案 + 多 rank 证据）。当用户需要多卡并行
             策略选择、**序列并行形态抉择**（纯 Ulysses vs 复合 AllGather-KV×Ulysses：按 GQA /
             跨域带宽 / 形态 plumbing 条件化定胜负）、**并行 × 稀疏叠加**（seam 契约：先汇聚后稀疏、
             窗口偏移、块对齐、per-head 掩码；含「稀疏看似生效实则未生效」判定）、通信掩盖调优
             （含**掩盖率上限**：1-1/n 何时成立、c/f 决定的真实上限、没生效的排查）、**并行方案
             差异归因**（阶段 Δ 分解 / 集合通信按 communicator 归属 / 4→8 卡线性度），或排查多卡
             跑不动 / 通信暴露大 / 换卡组 / 端口 bind / HCCL 带宽验证问题时使用；**并收编原并行作用域诊断**：
             改了 SP/CP/Ulysses/AllGather-KV 后**不报错但结果没变/性能没变**、或小规模能跑大规模崩
             （如 Ascend EE1003 coreDim 超限）时，用本技能证明"改动到底有没有生效"（判别量逐层收窄 +
             两侧对照 + 日志≠生效，见 `references/scope-effectiveness-check.md`）；
             即使用户只说
             "多卡跑不动""通信暴露大""为什么没达到 6/7 的掩盖率""CP 和 USP 该选哪个""CP 叠稀疏
             怎么不生效"而未说并行，也应触发。特性档位/接口事实见 `docs/zh/features/parallelism.md`
             /`usp.md`（仓内真源），框架侧开启见 framework-integration；
             本技能承载选型决策、monkey-patch 掩盖与多卡诊断实测。
             由 dev-workflow 多卡场景触发，亦由 model-auto-optimization 的 S3 阶段触发。
---

# 并行策略选择

## 卡组拓扑规则（强制 · 选卡前提）

> 任何多卡运行（含 S3 并行验证、S4 组合、质量补测、复验）的卡组选择必须先满足本规则；
> 违反拓扑的卡组不采纳。判据来自 `npu-smi info -t topo` 的**互连分域**（同域 = UB/HCCS 全互联，
> 跨域 = SYS/PCIe；文中「同岛 / 跨岛」是同一概念的历史用词，**等同「同域 / 跨域」**）——
> **具体成员编号以现场 topo 输出为准，本文件不写死卡号**。

- **合法卡组（按卡数，成员按现场 topo 判定）**：
  - **2 卡** = 一对**同域**相邻卡（topo 显示同 UB/HCCS 域）；
  - **4 卡** = 一个**完整同域**组（topo 显示整组同域）；
  - **8 卡** = 全部卡；
  - 其余（跨域拼组、跨 pair）**不合法**——不采纳、不得用于正式运行；
    若因卡健康/占用需跨域，必须显式注明「非同一互连域、带宽次优」并在对比纪律中同域对比。
- **执行要点**：
  1. 选卡前 `npu-smi info -t topo` 核对目标组内互连为 UB（同岛）；
  2. `npu-smi info` 核 Health：Alarm 卡弃用；Warning 需探活确认；被其他租户占用（proc-mem
     见 rtp_llm 等）的卡不可用；
  3. 多卡 run 必须**4 步 smoke 验证 per-step 时长**（per-step 时长整组均匀抬高约一个量级即 SIGKILL 残留态，须驱动复位
     或换组，npu-smi Health=OK 不能证明组可用——见 ascend-topology-bandwidth-diag §4）；
  4. 卡组变更 = 契约变更即新版本：换卡组后锚点行（基线/采纳档）须在**新卡组同窗重测**，
     不得混用旧卡组数值作分母。

## 策略一览

| 策略 | 适用场景 | 通信模式 | 关键参数 |
|------|---------|---------|---------|
| Ulysses 并行 (USP) | 多模态扩散模型（序列维度切分） | all-to-all | ulysses_size |
| CFG 并行 | 开启 classifier-free guidance 的模型 | 双分支独立推理 | cfg_parallel |
| 张量并行 (TP) | 单层参数量超单卡显存 | all-reduce / all-gather | tp_size |
| 环状序列并行 (RSP) | 长序列场景，通信可被计算掩盖 | P2P 环形传递 | world_size |
| 流水线并行 (PP) | 层数多、单层显存可承受 | send/recv | pp_size |

> 边界：本技能主轴 = **并行策略选型决策 + 调优验证 + 现成实测/缺陷绕过**（S3 承担，选型为主、
> 开发为辅）。官方特性档位与接口事实查 `docs/zh/features/parallelism.md` / `usp.md`
> （仓内真源，直读；mindiesd 未来官方并行接口落地后同样先查该处），
> 框架侧开启与生效验证走 framework-integration；框架自身无机制、需**结构性开发**
> （非现成 patch 可注入）→ `framework-integration`（经 model-auto-optimization §0 确认）。

## 机制与触发条件（CP / 通信掩盖 / AlltoAllV 绕过 / 显存解锁）

> 实测读数与案例细节归档于 `dummy-run/references/minimax-h3-notes.md` §9 与会话产物；本节只写机制与判据。

### Ulysses USP 触发条件

Ulysses 的 all_to_all FA 切头路径**不会自动生效**：仅给 attention processor 设 seq 分片时
走的是非 CP 路径（profile 只有 allGather 无 allToAll）。必须给每个
`attn.processor._parallel_config` 设 `ParallelConfig(context_parallel_config=cp_cfg)`
（注意是 `ParallelConfig` 包装，不是 `ContextParallelConfig`），才会触发切头路径。

### 通信掩盖（comm-stream masking）

把 `torch.distributed._functional_collectives.all_to_all_single` /
`all_gather_tensor` monkey-patch 为 `mindiesd.parallel` 的专用流版本
（实现注入点 `examples/dummy_run/masking.py` —— **该坐标属产品侧/另一 MR，本仓不含**，
已按 `[探针]` 口径处理，见 `references/comm-masking-method.md` 的适用窗口与复核方法）：
HCCL 集合跑在独立 comm stream 上，
与 caller stream 的计算重叠（compute 记 ready 事件 → comm stream 等 → HCCL → 记 done → compute 等）。

实测收益（mask on/off）：

| 指标 | unmasked CP | masked CP | 改善 |
|---|---|---|---|
| kernel 总耗时 | 基线 | 降至约三分之一量级 | 明显下降（本组合观测） |
| Communication（未掩盖） | 基线 | 降至约二十分之一量级 | 未掩盖通信基本消除（本组合观测） |
| Stage（设备时间线） | 基线 | 降至约二分之一量级 | 明显下降（本组合观测） |

> 边界：本节只覆盖「现成 `mindiesd.parallel` patch 的注入与实测」。若目标框架**自身无 comm-stream
> 机制**且 monkey-patch 不可行、需要为框架结构性开发该机制 → 属框架侧补齐，
> 经 model-auto-optimization §0 用户确认后走 `framework-integration`（本 skill 不做框架结构性开发）。

### HcclAlltoAllV 缺陷与绕过（重要）

部分 CANN 版本上 `HcclAlltoAllV`（split 路径）**SIGSEGV**（等分 `HcclAlltoAll` 正常）——
**用前先按复核方法确认该缺陷是否仍存在**（症状→复核触发纪律见 `references/ascend-parallel-traps.md`）。
绕过方案：**pad + 等分**——把 input 各块 pad 到 `S_PAD`（128 倍数，由全局 max(out_sizes)
推导，全 rank 一致），用等分 `HcclAlltoAll(count=S_PAD×row_elems)` 交换，再 slice 各块前
`out_sizes[j]` 行。生效判据：kernel 名从 `hcom_alltoallv` 变为 `hcom_alltoall`（等分），
且交换结果与参考实现逐位对齐。

### 内存受限时的并行解锁

更优并行策略常因单卡显存不可行（例：BF16 单 rank 全量驻留超单卡容量，更高度序列并行不可行）：

- **先在无损阶段尝试 offload 类特性降显存解锁**，而非直接放弃或跳到有损。特性按框架命名不同、语义都是
  "权重/激活出显存"：vllm-omni `--enable-cpu-offload` / `--enable-layerwise-offload` /
  `--enable-distributed-layerwise-offload`（DLO，host 存 1/DP + H2D/AllGather 重叠，官方支持叠加
  online INT8/FP8/MXFP8）；mindiesd `enable_offload`；PyTorch FSDP/CPU-offload 语义开关。
  注意互斥与副作用（**逐框架、逐代际复核**）：部分框架/模型档拒绝任何 offload；**部分代际上普通 layerwise offload 会触发
  OOM killer（用 DLO 而非普通 layerwise）；DLO 的 no-AllGather 路径（rank-local H2D）
  **明显更慢（数倍量级）** → 选 AllGather 路径。
- **有损（量化等）完成后复查被显存卡住的通信组合**：降显存会解锁新并行（量化后原本不可行的形态
  变为可行——收益读数见归档 `{run_results_dir}/archive/`）；每个量化档落地后回跑「并行×显存余量」
  候选快测。
- 并行选型后做通算掩盖评估：用单步捕获的 step_trace（Computing/Communication(未重叠)/Free）量化暴露通信
  ——有未重叠通信（`Overlapped=0`）时，**掩盖空间上限 = 未重叠通信占比**（读数见归档），但需框架侧
  comm-stream 支持；实现参照本仓 mindiesd/parallel + LightX2V `hccl_eager` 合入姿势；
  **占比随并行策略与负载规模变化**，compute-bound 时先长序列复测再投入）。

### 少步 × 多 rank 验证协议（**DiT-only 口径**）

> 问题：**用少步跑多 rank 验证并行配置，结论能否外推到全步长？** 答：能，但有闸门。
> 完整协议 / 阈值 / 踩坑见 `references/few-step-multirank-protocol.md`；
> 采集脚本 `scripts/fewstep_multirank_probe.py`（跑矩阵 + 采 DiT-only 指标 + 出证据包，
> `--parse-only` 可零 NPU 离线复盘）。

- **口径（强制前置，不是可选优化）**：并行对比指标**只取 DiT 去噪阶段**
  （`<Pipeline>.diffuse` 阶段墙钟，微秒级、每 rank 一行取 min）；VAE 解码 / 文本编码 /
  权重装载 / warmup / 响应编码一律排除。理由：少步档固定开销占比畸高，且**固定开销自身的
  波动远大于并行差异**——同配置相邻两次同参请求里，DiT 阶段波动比 VAE 解码阶段小一个量级以上
  （读数见归档）⇒ 把 decode 计入端到端，量到的其实是 VAE 噪声。
  （这正是「聚焦 DiT 优化即可不受 VAE 占比影响」的可执行化。）
- **判据（只判一个量：DiT 单步耗时）**：`r = DiT单步(锚点档) ÷ DiT单步(少步档)`。
  `|r−1| ≤ 5%` 且各格离散度 `≤ 3%` 且两档**排序一致** ⇒ 少步排序可外推选型；
  **排序翻转 ⇒ 一律以锚点档（或更高）为准**；任一格离散度 > 3% ⇒ **判据不成立**，先修噪声。
- **噪声纪律**：每格 ≥3 次重复取中位并报离散度；共租宿主上同配置离散度可达 7% 量级 ⇒
  **单次 / 单窗对比不可用**；冷 / 热双窗自查。
- **必须回较高步复验**：排序翻转 / 单步漂移超阈 / **未重叠通信占比漂移** / 短任务通信病态 /
  **特性叠加改变单步结构**（量化缩短 compute ⇒ 通信占比抬升；稀疏换算子；缓存改步级是否
  forward）/ 显存解锁类改动改变并行可行域。
- **形态 vs 步数的变量分离**：并行形态是**启动级**参数（换形态必须重启 serve），步数是
  **请求级**参数 ⇒ 少步/锚点可在**同一 serve 内交错**发（漂移在档间均摊），跨形态只能靠
  **相邻窗口 + 第二窗自查**。
- **通信占比口径**：遮盖空间上限 = 未重叠通信占比（`step_trace_time.csv` 精确列名取值）；
  该占比随并行策略变化（既有结论），**且会被采集方式扭曲**——profiler 单 forward 捕获膨胀该次
  墙钟而未重叠通信秒数基本不变 ⇒ 可比口径是「未重叠通信秒数 ÷ **未开 profiler** 的单步耗时」，
  捕获须放在被排除的 warmup 请求里。⚠️ **捕获落点还必须在稳态之后**：serve 起后的前几个请求
  单步明显更快（同环境实测首请求比稳态快约一至一成半），若捕获落在 ramp 段，占比的**分母**就被
  系统性压小（详见 reference §6/§7）。
- **矩阵可行性先核**：纳入矩阵前先确认形态能跑通——本组合观测：**纯 TP 形态（无序列并行）在本框架
  × 本模型 × 该负载下整体不可用**（两种编码器 TP 模式各 fail 在不同位置，根因是形态而非超参，
  且有对照臂佐证）⇒ 该组合的「同卡数、不同切分」对取不到 2 卡版本，改报 4 卡对
  （`TP1×USP4` vs `TP2×USP2`）；**不得用跨卡数对比冒充形态对比**。

### 拓扑相关选型

同一并行形态在不同互连拓扑下排名可能反转，**别无条件复用历史结论**（详见
`references/ascend-topology-bandwidth-diag.md`）：

- 读取 `npu-smi info -t topo`：UB=HCCS 同域、SYS=跨 PCIe/NUMA；4 卡 a2a 优先单个同域组
- 同域单组 → USP4 **bulk** 最优（comm busy 最小、同步事件最少）；跨域组 → **head-parallel**
  更优（本组合观测：rank0 clean-window 更快，2×2 复现；绝对值见归档
  `{run_results_dir}/archive/`）：逐头小 a2a 全异步重叠，bulk 大 alltoall
  跨岛串行暴露。判据墙钟/clean-window 为准——head-parallel kernel-sum 更高（跨流多计数）
  但墙钟更低
- compile × head-parallel 不兼容（Dynamo recompile_limit 静默回退 eager）
- 带宽/环境诊断与恢复姿势（HCCL 微基准须先 `set_device` 否则报端口 bind、hccl_test 宿主可能
  不兼容、端口泄漏/卡组受损→换组验证）见同 reference

## 通信掩盖：分块设计与上限判定（2026-09-13 回填）

> 详细方法：`references/comm-masking-method.md`（含上限公式、实现 recipe、生效验证判据与静默失效清单）；
> 上限计算脚本 `scripts/mask_bound_calc.py`。以下为摘要与判据。

- **先算上限，再谈实现**。三个量（都按每层口径）：`C` = **被掩盖那一族**的通信总量、
  `F` = 可与它并行的计算（本组合 = 稀疏 FA 核总耗时）、`n` = 分块数；派生 `c=C/n`、`f=F/n`。
  - `f >= c`（等价 `F >= C`）⇒ 上限就是直觉式 **`1 - 1/n`**（只暴露 1 组头的通信）；
  - `f < c` ⇒ **`1 - 1/n` 不可达**，上限退化为 `1 - [c + (n-1)(c-f) + drain] / C`
    （填充 + 稳态吃不完的部分 + 排空），且**与 `n` 无关**：此时把块数从 7 加到 14 也不涨收益。
  - 判据顺序：先 `c/f` → 再看实测掩盖率占上限的多少。`c/f > 1` 时唯一的出路是**压通信分子**
    （量化载荷、减跨岛流量、合并集合），或把更多**独立**计算搬进同一段。
- **本组合观测（量级参照，勿当预期值）**：纯 8 卡序列并行 `c/f≈1.3` ⇒ 上限约 3/4，
  实测掩盖率约 0.54（占上限约七成）；复合 AG-KV×Ulysses `c/f≈2.0` ⇒ 上限约 1/2，
  实测约 0.50（**几乎贴上限 98%**）；4 卡序列并行 `c/f≈0.46` ⇒ 理想界可达（约 6/7），
  实测仅约 0.43（占上限约一半，实现侧仍有填充/排空损失）。
  ⇒ 「没达到 6/7」的根因是 `C > F`，不是实现没写好；但 4 卡那一格说明实现仍可再榨。
- **实现要点**（照做可省一轮返工）：块索引放最外层**一次**预置换（块即连续切片，避免
  `as_strided` 物化）；**侧流只放集合通信**、全部 AI-core 算子留计算流（本组合实测：把变换
  放进侧流会多出上千个 AI-core 核与百毫秒级物化算子，直接抢 vector core）；用**块级 event**
  而非 `wait_stream`；同一时刻只让一个 communicator 活跃（交错下发会让 HCCL 进入不一致态）；
  块数取「整除本地头数的最大值」。
- **生效判据**（缺一不可）：该族算子**次数上升 ≈ n 倍**（次数没变 = 没生效）、侧流 AI-core 核数 = 0、
  `Communication(Not Overlapped)` 下降（列名必须精确匹配）、产物 md5 与不开时**逐字节一致**
  （Ulysses a2a 无规约 ⇒ 掩盖是纯置换）。日志说「已启用」不算；**回退分支要留命中/回退计数器**
  （本组合出现过"开了门控但每次都静默回退"的臂：md5 与耗时都与基线相同，看着像"优化没用"）。
- **下发顺序就是调度**：同一 communicator 的集合通信按**下发顺序串行执行**（实测 `union/sum = 1.00`）
  ⇒「先发完所有前向、再发所有反向」会让反向通信整段落在本层计算之后。本组合观测：反向输出
  a2a 只占该族字节 1/4，却贡献**过半的暴露通信**、与注意力核重叠率 **0%**。
  修法是**软件流水式下发**（把第 g+1 片前向排在 g 片反向之前），不是「逐片交替」（会把流水压成串行）。
- **下一步往哪投：三个判据一起看**（`scripts/bubble_attribution.py` 一条命令）：
  ① **暴露归哪一族**——某族耗时大部分落在计算流空隙里才是关键路径（本组合：反向 a2a 几乎全部暴露
  vs 前向 q/k/v 只有约三成暴露；精确读数见归档）；② **释放判据**——空隙里"有集合通信在跑"**只是必要条件**，必须是"空隙**恰好
  在某次集合通信完成时结束**"才算真依赖（本组合：空隙中"有通信在跑"的那部分，大部分才是被通信
  完成释放的；其余是**传输还在途就恢复执行**，等的是别的东西）；③ **生产-消费判据**——比较消费者
  核启动与它依赖的那次通信完成：本组合注意力核与其**自己那片**前向 a2a 完成的间隔中位数为毫秒量级、
  逐例都远大于判据容差（无一例"紧接着就启动"）⇒ **前向通信跑在消费前面，FA 从不等自己的数据**（FA 前的空隙占单步可忽略）。
  ⚠️ 这三条一起看才不会误判：只看①会以为"FA 在等自己的 qkv"，只看②会把归因引向错误的族。
  真空气泡（什么都不跑）本组合约占空隙的一成、占整步的百分之几（与 `Free` 列同量级）⇒ 不是 host 下发受限。
- **别踩的两个静默坑**：① 门控读取顺序 —— 分派在进入并行形态分支**之前**读父门控，只开子门控
  时永远走不到子分支（复合形态需**同时**开父与子）；② 共用启动脚本里的一句 `unset` 会让新开关
  永远为 0（启动后 grep 实际生效的环境变量）。

## 并行方案归因对比（2026-09-13 回填）

> 详细方法：`references/parallel-plan-attribution-method.md`；脚本
> `scripts/collective_attribution.py`（族分组 + union/sum + 跨族重叠 + per-stream 表）。

- **阶段 Δ 分解**：取 `step_trace_time.csv` 逐列相减（`Computing` / `Communication(Not Overlapped)` /
  `Overlapped` / `Free` / `Stage`），并校验闭合 `ΔStage ≈ ΣΔ`。判读顺序：先看未重叠通信与 `Free`，
  再看 `Computing` —— 后者是**形态自带的 plumbing 成本**，不会被掩盖掉，不能当通信问题处理。
- **集合通信归属**：kernel 名带 communicator id（`hcom_allGather_AicpuKernel_<gid>_...`），
  按 `(族, gid)` 分组，再用「`次数 ≈ 层数 × 每层调用数 × 步数`」对账；**除不尽**说明有同族额外通信
  （逐层 offload 的权重 gather、TP、编码器、VAE）。offload 表现为**同一个组里**的额外通信，
  不是「多了一套卡」；判定必须靠同窗口 A/B（offload 开/关）。
- **重叠真实性（union == sum）**：`union/sum ≈ 1` ⇒ 该族是**传输时间**而非依赖等待，优化只能压载荷；
  `< 1` ⇒ 存在并发，报耗时要说清是「占用」不是「暴露」。
- **带宽归因要先对齐并发度**：模型内隐含带宽（载荷 ÷ 单次耗时）与隔离微基准（torchrun + torch_npu，
  必须 `set_device` 后再 init）**不可直接比**——微基准只有一个通信子，是上界；要归因先在微基准里
  复现同样的并发通信子数量。本组合观测：同一对 rank 的 2 卡 all-gather 跨岛耗时约为同岛的 2 倍，
  0.25×–2× 尺寸上耗时严格随载荷线性（纯带宽）；而**模型内同形状的隐含带宽只有隔离值一半左右，机制当时未定位**
  —— 必须作为**未解释项**保留，不能当成已解释。
- **通信量下限模型**：把每层每 rank 字节**分跨岛 / 同岛两列**算，下限取 `max(计算, 通信)`；
  **GQA 是决定性变量**：有 GQA 时 CP 的 K/V 跨岛量按 KV 头比例下降，无 GQA（Q=K=V 头数相同）时
  两形态跨岛量相同、CP 的下限并不更低 ⇒ 「CP 更省通信」必须以 GQA 为前提。
- **投影纪律**：给「修好 X 则 Y 反超」的结论时，只改被修复项、保留对方已实测的额外开销、
  标明是投影并给出反超阈值；只算通信项会得出相反的答案（本组合观测过一次完整反转）。
- **4→8 卡线性度**：同口径（DiT 阶段墙钟）测，线性度 = 加速比 ÷ 卡数比；**线性度差的第一嫌疑是
  暴露通信而不是并行度**（本组合观测：加掩盖后 4→8 卡线性度由约 78% 升到约 90%）。
  注意卡数变化同时改变拓扑（4 卡 = 单 UB 岛，8 卡 = 跨岛），要看拓扑效应必须另设对照。

## 形态抉择：纯 Ulysses vs 复合 AllGather-KV × Ulysses（2026-09-13 回填）

> 详细流程：`references/parallel-form-selection-method.md`（七关抉择 + 判据表 + 条件化结论模板 +
> 重判触发清单）。以下为摘要。

- **先纠正直觉**：**「拓扑更优的形态一定更快」是错的**。复合形态常把 a2a 全放进**同岛**
  （a2a 耗时降到约 1/4 量级），代价是把 K/V 汇聚搬到**跨岛**，而那一步的绝对耗时同量级甚至更大
  ⇒ 本组合净亏。**要比的是「每 rank 每层总字节 × 各自链路带宽」，不是「哪个通信更快」。**
- **无 GQA 时两形态跨岛字节几乎相同**（本组合差 < 1%）⇒「复合形态更省通信」**只在有 GQA 时成立**
  （按 KV 头比例下降）。这是抉择的**第一 gating 变量**。
- **七关流程**（任一关否决即出局）：G1 形态可用性（**未跑通的形态不进矩阵**，更不能用理论为它留位；
  本组合观测：Ring 被框架硬拒且其路径**绕过注意力后端** ⇒ 稀疏静默失效；4 卡的 2×2 变体起不来）
  → G2 显存/驻留（先用无损 offload 解锁，**顺序不能反**）→ G3 通信量下限（跨岛/同岛两列 + GQA）
  → G4 **计算 plumbing**（复合形态在 `Computing` 里的固有增量，**不会被掩盖**；本组合占两形态
  每步差的约 1/3）→ G5 掩盖上限 `c/f`（不同形态被掩盖的族不同）→ G6 同窗口 A/B + 线性度
  → G7 投影与反超阈值。
- **「谁更能被掩盖」≠「谁更快」**：本组合观测复合形态掩盖兑现率 ~100%（每片计算≈每片通信）、
  纯形态只有 ~60%，但复合形态端到端**仍慢约 10%**。
- **条件化结论模板（强制）**：写「当 <GQA/带宽效率/量化档> 满足 <不等式> 时 X 胜，否则 Y 胜；
  当前实测落在 <哪一侧>，差距 <比例>；抑制结论的未知量 = <未解释项 + 判定方法>；
  反超阈值 = <水平>」。本组合实例：无 GQA + 汇聚带宽约为隔离值一半 ⇒ 纯形态胜约 10%；
  修好带宽或改为带 GQA 则复合形态预计反超约 7%；**两边同时做通信前量化则再次反转**
  （计算项占比升高）⇒ 投影必须把计算项一起算。

## 序列并行 × 稀疏注意力叠加（seam，2026-09-13 回填）

> 详细契约：`references/cp-sparse-combination-method.md`（可行性判定 + 四条契约 + 掩盖叠加规则 +
> 验收判据 + 失效模式表）。seam 的**登记与裁决协议**在
> `dit-perf-opt/references/combination-search.md`。

- **这类 seam 最会「看起来生效」**：某些并行路径会**绕过注意力后端**（稀疏被接受、无异常、
  输出与无损**逐字节相同**）⇒ 必须把「与 lossless 逐字节相同」判成**未生效**，不是无损。
- **可行性分形态判**：环状 CP × 稀疏**不可用**（框架硬拒 + 路径绕过后端）；AllGather-KV CP × 稀疏
  **可用**（本组合实测跑通且带掩盖后与无掩盖臂逐字节一致）。
- **四条契约（缺一即崩或静默错）**：① **先汇聚后稀疏**（选块需全序列，否则是「局部稀疏」，
  数值与收益都不可比）；② 稀疏窗口全局偏移 = `完整 KV 长度 − 本 rank 逻辑 Q 长度`
  （**不是** `k_len − q_len`；无 joint 张量时为 0）——这是稀疏实现与 CP 记账的唯一接口，
  写错直接报 `Q window starts at X, outside the key sequence length Y`；③ 分片长度须是
  稀疏/量化块的整数倍（判据式 `S % (并行度 × 块) == 0`，与「量化前移」同一根因）；
  ④ 掩码是 per-head 的，**不能跨 head 分片复用**（复用要汇齐所有 head 的 K ⇒ 丢掉分块流水，净负）。
- **掩盖叠加两条承重规则**：复合形态要**自己的一套分块实现**（a2a + AllGather 两种集合）；
  且**先发完所有 a2a 再发所有 AllGather**（交错下发会让 HCCL 失步，本组合实测约两分钟后死于
  集合通信内部错误）。
- **报 seam 强度**：本组合观测 **CP × 掩盖为正协同**（掩盖在复合形态兑现率 ~100% vs 纯形态 ~60%，
  即「CP 让掩盖更值钱」），但**正协同不等于整体更快**（复合形态端到端仍慢约 10%）。

## WIP 待定内容

- [ ] 各策略在**目标代际**上的完整实测性能对比表
- [x] HCCL 拓扑感知的策略选择决策树（同域/跨域判据已回填：见上「拓扑相关选型」+ `references/ascend-topology-bandwidth-diag.md`）
- [ ] 混合并行策略的配置模板（如 USP + CFG 组合）
- [x] 策略切换的性能对比方法论（**少步 × 多 rank 验证协议已回填**：见上「少步 × 多 rank 验证协议」，落点 `references/few-step-multirank-protocol.md` 与 `scripts/fewstep_multirank_probe.py`）

## Reference Files

- `../dummy-run/references/minimax-h3-notes.md` §9 — 加载时机: 需要 CP/USP 实测细节、mask 注入代码或 AlltoAllV 绕过实现时
- `references/scope-effectiveness-check.md` — 加载时机: **改了 SP/CP/Ulysses/AllGather-KV 后「不报错但没生效」（结果没变、性能没变、小规模能跑而大规模崩）时**（先证明分片是否发生：状态量探针 + 判读表、两侧对照、判别量逐层收窄；含 `sp_plan_hooks_applied` 误判与身份比较陷阱）
- `references/ascend-parallel-traps.md` — 加载时机: **遇到具体报错码（`EE1003 coreDim` 超限、gloo 地址族、Q≠KV 守卫、`auto_pad` × 后端互斥）、或怀疑「某个配置被静默忽略」时**（逐条陷阱的 症状→原因→处置 + **「如何判定它仍存在」的复核触发** + 陷阱寿命纪律）
- `references/ascend-topology-bandwidth-diag.md` — 加载时机: **多卡拓扑选型（同域/跨域）**、HCCL 带宽验证（hccl_test 或 torchrun 等价工具）、端口 bind/卡组受损等环境诊断时
- `references/comm-masking-method.md` — 加载时机: **要做/要做完通信掩盖（分块流水掩 a2a）、判断「掩盖率为什么达不到 1-1/n」、估算掩盖上限、或掩盖开了却没生效时**（含上限公式、实现 recipe、生效判据、静默失效清单）
- `references/parallel-plan-attribution-method.md` — 加载时机: **比较两个并行形态/特性档的耗时差、需要把差异拆成可归因分项、判断「通信慢是传输还是等待」、建模跨岛/同岛通信量下限、或做 4→8 卡线性度分析时**
- `references/parallel-form-selection-method.md` — 加载时机: **要在纯序列并行与复合（AllGather-KV × Ulysses）等形态之间做抉择、判断「哪个形态更快」、或要写「若修好 X 则反超」的投影结论时**（七关流程 + 判据表 + 条件化结论模板 + 重判触发清单）
- `references/cp-sparse-combination-method.md` — 加载时机: **要把序列并行（CP/Ulysses）与稀疏注意力叠加、判断该叠加在框架侧是否可行、排查「稀疏看起来生效其实没生效」、或叠加掩盖时定分块与集合通信顺序时**（四条契约 + 验收判据 + 失效模式表）
- `references/few-step-multirank-protocol.md` — 加载时机: **少步/短任务下验证并行配置、判断「少步结论能否外推全步」、设计并行对比矩阵、需要 DiT-only 计时口径或通信占比采集口径时**（含阈值、复验触发条件与踩坑清单）
- `scripts/fewstep_multirank_probe.py` — 加载时机: 需要**实际执行**「多 rank × 多步数档」矩阵并把 DiT-only 指标与排序一致性自动出成证据包时（`--parse-only` 可零 NPU 复盘既有日志）
- `scripts/mask_bound_calc.py` — 加载时机: 拿到掩盖 ON/OFF 的 profile 后，**算每层 C/F、c/f、1-1/n 是否可达、真实上限与达成率**时（零依赖，只读 profile）
- `scripts/collective_attribution.py` — 加载时机: 需要**把集合通信归到具体并行组、判重叠真实性（union/sum）、或检查侧流是否混入 AI-core 核**时（零依赖，只读 profile）
- `scripts/bubble_attribution.py` — 加载时机: 需要**判定「暴露的是哪一族通信」「计算流空泡是等网还是等下发」、或决定下一步该投掩盖/压载荷还是投下发与图捕获**时（零依赖，只读 profile）

## 维护与更新

当新的并行策略经验证有效、多卡互联拓扑发生变化或发现新的分布式训练模式时，
按 dev-workflow 的复盘流程更新本 skill。
