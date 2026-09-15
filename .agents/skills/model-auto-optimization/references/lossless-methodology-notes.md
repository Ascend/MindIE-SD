# 无损阶段方法论补充笔记（S1 计算融合（原 S2）/ S3 并行解锁 / 量化后融合重审）

> 2026-09-05 由 MiniMax-H3 × vLLM-Omni 0.28 实测回填（该框架的**开启方式**见
> framework-integration/references/vllm-omni-enablement.md；实测记录与绝对数字已归档到会话
> 产物目录 `{run_results_dir}/archive/`，**不可跨模型 / 框架 / 规模 / 窗口引用**）。
> 作用：给 model-auto-optimization 的 S1（融合）/S3/S4 纪律补充可操作动作序列，供编排时直接套用
> （A：S1 融合（原 S2）；B：S3 并行解锁与量化后复查；D：量化后融合重审；E：量化后通信重审；F：无损评估提速）。
> （A：S1 融合（原 S2）；B：S3 并行解锁与量化后复查；D：量化后融合重审；E：量化后通信重审）。

## A. 计算（S1 融合）：候选列表 → mindiesd/CANN 融合能力对照 → 独立验证

用户经验原文（本笔记是其方法化）：

> 通过代码或者 profiling 的算子执行序给出可以融合的 kernel 的列表，然后基于 mindiesd 以及 CANN 的
> 融合能力识别融合位置，以及需要额外准备的融合算子。然后进行独立验证辨识是否生效。

标准动作序列（并入 S1 融合「候选集+尝试+评估」之前）：

1. **列候选 kernel**：从代码（模型 forward 结构）或 profiling 的算子执行序（kernel_details 执行序，
   相邻/链式的小 kernel）导出「可融合 kernel 列表」（如 RMSNorm 分解链、RoPE slice/cat 链、
   SwiGLU split+silu+mul、AdaLN index_select+scale/shift、gate 链、residual add）。
2. **对照融合能力定位**：查 mindiesd（runtime 单算子 + torch.compile pattern 两套）与 CANN/torch_npu 已有
   融合算子（npu_rms_norm / npu_rotary_mul / npu_swiglu / FA(varlen) / fused qk-norm-rope / aclnn* 系），
   判定每个候选：已覆盖（无需做）/ 需新增 pattern（mindiesd compile 侧）/ 需新融合算子（实现侧）。
   同时记录「**需额外准备的融合算子**」清单（如与 CANN 语义顺序相反的变体、tensor-scale 调制等）。
3. **独立验证**：对每个候选做单变量验证——编译/替换后先看 图命中（MINDIE_LOG→DOT）、再 kernel diff
   （融合 kernel 是否真执行、分解链是否消失、kernel 计数/拷贝计数）、最后墙钟；三层不一致时以墙钟+kernel 为准。

关键经验（负面结果同样要记录，避免重复投入）：

- **eager 已融合的热路径不要再叠 compile**：H3 DiT 的 RMSNorm/RoPE/FA/编码器 SwiGLU 在 eager 已由 mindiesd/
  CANN 单算子覆盖；对其再叠 MindIE compile 后 kernel 行数与 Copy/Move 计数上升、computing 与墙钟转负
  （输出一致但无收益）→ 判定不采纳（读数见归档 `{run_results_dir}/archive/`）。判断口径：先看目标链是否已被 eager 单算子消费。
- 融合收益要放真实规模（层数/分辨率/步数）看；短跑（少量步）墙钟中性不代表无收益，但也不代表有——以
  kernel diff（单步捕获）为准最稳：**通用做法 = 平台内加一个 env 门控的「单 forward kernel 采集 hook」**
  （torch_npu Level1 + tensorboard handler，事后 `torch_npu.profiler.profiler.analyse(dir)` 聚合出
  kernel_details.csv / step_trace_time.csv），任意 eager/compile/并行配置都能拿同口径单步 kernel 数据。
- VAE/编码/封装等非 DiT 耗时单列：DiT 通常主导（~83%+），其余按阶段（文本编码/VAE decode/多 rank 交接）简析即可。
- **热路径已被 eager 单算子覆盖后，剩余融合候选池落在「布局/小算子链」**（split / silu / cat / index
  等）：这类候选的收益必须先与 compile 的图内拷贝 / 调度开销**对抵**再判——整图编译后 kernel 行数与
  Copy/Move 计数上升、墙钟中性偏负，正是「剩余候选收益 < 图开销」的表现。判定动作 = kernel diff
  分类计数（新增拷贝 vs 消失的分解链）+ 同窗相邻对墙钟；**收益不足即回退（默认关），不留半开状态**。

## B. 通信（S3 并行）：内存受限 → offload 解锁；有损完成后复查组合

用户经验原文（本笔记是其方法化）：

> 当发现更好的并行策略，但是受限内存时，可以尝试使用 CPU Offload 的特性（包括 FSDP），这个特性在不同
> 框架上有不同的名字，但是作用就是降低显存，这个本身在无损阶段就需要尝试。同时，在完成量化等行为后，
> 也可以进一步地识别是否还有受到内存影响的通信组合未尝试。

标准动作：

1. 并行选型时若某更优策略（如更高 USP 度/TP1 全量驻留等）因单卡显存不可行，**先在无损阶段尝试 offload
   类特性解锁**，而不是直接放弃或跳到有损：
   - vllm-omni：`--enable-cpu-offload` / `--enable-layerwise-offload` /
     `--enable-distributed-layerwise-offload`（DLO：host 存 1/DP、H2D+AllGather 重叠；官方支持与
     online INT8/FP8/MXFP8 组合；`--dlo-resident-layers`/`--dlo-use-allgather` 调档）
   - mindiesd：`enable_offload(model, blocks, ...)`
   - PyTorch 侧：FSDP/CPU-offload 语义的对应开关（按框架名查）
   - ⚠️ 与部分特性互斥需查（如 vllm-omni FastH3 拒绝任何 offload）；offload 也可能被 OOM killer 触发
     （950PR recipe：--enable-layerwise-offload 主动有害 → 用 DLO 而非普通 layerwise）。
2. **有损完成后复查**：量化（INT8/FP8）等降显存动作会解锁新的通信组合（例：H3 BF16 单 rank 全量
   驻留超单卡容量 → 无法用更高度序列并行；INT8 online 后单 rank 显存降到可行区间 → 该形态可行，
   相对对照形态的收益见归档 `{run_results_dir}/archive/`）。
   因此 S4 每个量化档落地后，回头跑一遍「并行×新显存余量」候选（至少 10 步快测），把新增可行组合纳入矩阵。
   **offload 解锁实证（2026-09-05，H3×vllm-omni 0.28/950PR）**：无损阶段用 vllm-omni DLO
   （`--enable-distributed-layerwise-offload`，默认 AllGather：host 存 1/DP + H2D/AllGather 重叠）
   解锁了原本因单卡容量不可行的 BF16 高序列并行形态（同窗对照为**略慢**，属形态置换的代价，
   绝对耗时见归档）；DLO 的 no-AllGather 路径（rank-local H2D）**明显更慢（数倍量级）**
   → **选 AllGather 路径**；mindiesd `enable_offload`、PyTorch FSDP 同族。
3. 通信分析（选型后）用 step_trace_time 给 compute/comm/free 拆分（单步捕获）：本案例 USP2 单步
   Computing 占大头 / Comm(未重叠) 占其余（具体占比见归档 `{run_results_dir}/archive/`）/ Overlapped 0 → 通算掩盖空间明确但框架无机制，compute-bound 时
   先长视频/高分复测再投入实现（参照 mindiesd/parallel + LightX2V `hccl_eager` 合入姿势）。

## D. 量化后融合重审（kernel 序列变化事件 → 回到 S1 融合回路）

> 流程级规则（入 model-auto-optimization SKILL S4 纪律 6）：**任何改变 kernel 序列的特性使能后**
> （量化档最典型；稀疏后端/档、缓存、并行拓扑亦同），S1 融合的既有融合判定只对「使能前序列」有效，
> **不得跨序列沿用**；须以新序列重走 S1 融合回路「候选 → 尝试 → 评估（图命中 → kernel diff → 墙钟）」。

1. **为什么**：量化把原 MatMul/Addmm 序列替换为新的量化 kernel（H3 w8a8 单步实测：266 MatMul →
   6 遗留 + 260 对 `DynamicQuantV2 → QuantBatchMatmulV3`，总行 2625→2885，**零新增布局搬运**），
   producer/consumer 邻接整体重排 → 旧融合候选可能失效（bf16 MatMul 已被单量化 GEMM 替代）、
   新候选出现（量化 GEMM 与上游 norm/SwiGlu 的 epilogue、编译 pattern、FA 路径布局等）。
2. **标准动作**：
   - 采集新序列单步 kernel（profiling-collect 的 env 门控单 forward hook，同口径）→ 按类别计数 +
     **邻接分析**（量化 kernel 的前驱/后继直方、量化对是否紧邻、有无独立 dequant/新布局搬运）；
   - 对照融合能力（同 A.2）列新候选并登记进 S1 融合候选清单；
   - 逐候选单变量 AB（同窗同卡组）；**compile 必须按新序列重测**——旧序列负收益（bf16 compile 的
     kernel 行数与 Copy/Move 计数上升）不代表量化序列负收益（量化序列多出可融合的小 kernel 对，
     图级收益可能翻转）；
   - 采纳/回退带三层证据；量化档 × 融合 pattern 组合前查 combination-search 的 precision seam
     （kernel 是否接受量化输入/图形态匹配，否则运行期 fallback）。
3. **首个案例（H3 w8a8，2026-09-06）**：w8a8 场景融合机会清单 O1–O7（DQ 上游 epilogue 融合 /
   GEMM 吸收 A 侧动态量化 / FA 路径布局对齐 / 量化域贯通 / 通信重叠 / 新序列 compile 重测 /
   宽层回退收口）与量化单步证据，见归档 `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md`
   与归档分析 `H3_w8a8_fusion_analysis.md`。头号教训：量化后 **GEMM 级融合已到位**
   （bias/anti-quant/per-token scale 均在 QuantBatchMatmulV3 内、无独立 dequant kernel），
   新机会集中在 **DQ 两侧**（上游 norm/SwiGlu epilogue、与 GEMM 合并）+ 注意力路径布局 + 通信重叠；
   每 DiT block 恰 5 个量化 GEMM（qkv/out/fc1/down/adaln，fc1 为 merged）→ 模型层无共享输入重复量化。

## E. 量化后通信重审（comm 占比、量化/压缩通信与掩盖）

> 流程级规则（与 §D 同因，入 SKILL S4 纪律 6）：量化等特性改变 kernel 序列的同时也改变**数据口径与
> 单步耗时结构**——GEMM 加速使 comm 占比上升、通信张量的数值域/产生位置变化 → 通信侧的旧结论
> （掩盖空间、拓扑收益）须按新占比重估，并**审视通信数据本身是否可量化/压缩**。

1. **为什么**：量化档落地后 compute 缩短而 comm 基本不变 → comm 占比抬升（掩盖空间变大）；同时
   通信内容从「bf16 网络层输出」变为「量化 GEMM 的部分和/经 anti-quant 的张量」，出现低精度
   传输的候选点（TP allreduce 部分和、USP/SP 注意力跨 rank 的 K/V 交换、跨卡激活）。
2. **标准动作**：
   - 每量化档落地后重采一次 step_trace_time（同口径），记录 compute/comm(未重叠)/free 占比；
   - 占比升 → 重估掩盖空间（Overlapped 是否可用）与并行/拓扑矩阵（交叉引用 §B 的量化后复查）；
   - 扫描通信数据可量化点并登记候选：量化 allreduce（INT8/FP8 + scale 传输后反量化）、K/V
     低精度交换、压缩类集合通信——**每个候选是有损维度**：误差影响走质量门禁 + off-identity +
     计数契约（真实低精度传输计数），并进 S4-2 combination-search 当新维度叠加；
   - 实现归属：集合通信/comm-stream/量化 allreduce 多属框架结构性缺口 → model-auto-optimization
     §0 补齐策略经 `framework-integration` 确认（钉版本/合入上游），编排层只登记候选证据，不静默改框架。
3. **首个案例（H3 × vllm-omni 0.28，768P 单步 step_trace）**：lossless 单步 **Comm(未重叠) 占比
   逐档显著抬升**（lossless → +INT8 → +mix；compute 单步耗时同步下降、comm 基本不变；
   绝对耗时与占比见归档 `{run_results_dir}/archive/`），Overlapped=0 → **掩盖空间上限 = 该占比**、
   随量化抬升（comm-overlap 实现评估列 P1）；通信内容 = TP2 列并行
   allreduce 的量化 GEMM 部分和 → 低精度 allreduce / GEMM-comm 重叠为候选（量化 GEMM 后继 63 处
   comm，见 §D 邻接证据）；FP8 KV 属缓存侧量化，另列（support-matrix W8A8_MXFP8 ❓）。

## F. 无损评估提速（少量 step 快测 + mindiesd dummy-run）

> 用户经验（方法化）：无损链路（融合/并行/compile 候选）的评估不总是需要「完整 serve + 全步长」——
> 用少量 step 快测与 mindiesd dummy-run 提升并行度、压缩评估时间，让候选迭代更快（呼应 SKILL 无损
> 纪律 4 与 effort-estimation §1）。

1. **两条提速轨**：
   - **少量 step 快测**（serve，10 步 smoke）：单请求耗时远低于全步长档（量级随负载 / 卡数换算）
     ——承载 FA/并行/compile 等候选的**方向性 AB 与单步 kernel 采集**（同窗相邻对）；采纳项再回
     工作负载**同款步数**稳态复验（采集侧「60 步 → 2 步」式提速属**采集手段**，见 profiling-collect
     「少步快速采集经验」；耗时量级与读数见归档 `{run_results_dir}/archive/`）；
   - **mindiesd dummy-run**（examples/dummy_run，`*_infer.py` 统一 `--quant/--compile` 接口）：绕开
     text encoder / VAE / HTTP / 多 rank 启动，直接 python 驱动模型 forward——融合/量化/compile 候选
     可高频迭代，且天然便于**多候选并行**（提升并行度 = 同一时间窗评估更多候选，压缩排程等待）。
2. **覆盖边界（何时必须回 serve）**：
   - dummy-run 不覆盖：并行拓扑与通信（TP/USP/ring 形态、hccl）、Cache-DiT 机制、HTTP e2e/吞吐、
     编码/VAE/全程步数口径 → 这些候选直接 serve 少量 step 验证；
   - 快测/dummy-run 的**方向性结论不得直接作对外收益**：正式宣称走同窗同卡组全步长（声明纪律）。
3. **参考**：dummy-run 技能（A6 量化/精度模式、references/model-common.md、minimax-h3-notes）、
   effort-estimation.md §1。

## C. 相关落点

- 归档实测（绝对数字不入 skills）：会话产物目录 `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md`；
  框架开启方式：framework-integration/references/vllm-omni-enablement.md
- 质量门禁首个案例：accuracy-gate/references/quality-gate.md + MindIE-SD/evals/（契约/工具/profile minimax-h3.toml）
- 有损档位表与组合：H3_TUNING_REPORT 归档（见 `vllm-omni-enablement.md` §6 产物坐标指针）

## G. 维护与更新

- **触发条件**：① §B 的框架 offload 开启姿势变化（vllm-omni `--enable-distributed-layerwise-offload`
  / `--dlo-resident-layers` 等开关、mindiesd `enable_offload` 语义）；② 有损档使能改变 kernel 序列，
  使 §D 的 `DynamicQuantV2 → QuantBatchMatmulV3` 邻接证据与 O1–O7 机会清单失效、或 §E 的
  Comm(未重叠) 占比结论漂移；③ §F 两条提速轨（少量 step 快测 / mindiesd dummy-run）的覆盖边界变化；
  ④ §C 的归档坐标（`vllm-omni-minimax-h3-case.md`、`H3_w8a8_fusion_analysis.md`）改名或迁移。
- **复核方法**：用 profiling-collect 的 env 门控单 forward hook 对新档重采一次同口径单步
  kernel_details / step_trace_time，按 §D 重做邻接计数、按 §E 看 compute / comm(未重叠) / free
  占比是否仍复现原结论；不复现即改该节，并同步 post-enable-review 对应面。
