---
name: parallelism-strategy
compatibility: 无额外工具（参考数据来自 dummy-run/references/minimax-h3-notes.md §9）；多卡需 HCCL 环境
description: 分布式并行策略选型与实测（Ulysses USP / CP 通信掩盖 / CFG / TP/RSP/PP 概览），
             含 910B 实测、HcclAlltoAllV 缺陷绕过（pad+等分）与 950PR 拓扑相关选型
             （UB 岛 bulk vs SYS 跨岛 head-parallel 翻转）。在 model-auto-optimization 中承担
             S3：优先 USP、结合拓扑带宽差异选 CP，少量 step + 多 rank 验证特性开启与掩盖
             （产物：并行方案 + 多 rank 证据）。当用户需要多卡并行策略选择、通信掩盖调优，
             或排查多卡跑不动/通信暴露大/换卡组/端口 bind/HCCL 带宽验证问题时使用；即使用户只说
             "多卡跑不动""通信暴露大"而未说并行，也应触发。特性档位/接口事实见 mindiesd-features.md（由 performance-optimization 刷新维护），
             框架侧开启使能见 framework-feature-enablement；mindiesd 未来官方并行接口（docs 落地后）
             同样按该真相源选档、开启走 enablement；本技能承载选型决策、monkey-patch masking、
             缺陷绕过与多卡环境/带宽诊断类实测。由 dev-workflow 多卡场景触发，亦由
             model-auto-optimization 的 S3 阶段触发。
---

# 并行策略选择

## 策略一览

| 策略 | 适用场景 | 通信模式 | 关键参数 |
|------|---------|---------|---------|
| Ulysses 并行 (USP) | 多模态扩散模型（序列维度切分） | all-to-all | ulysses_size |
| CFG 并行 | 开启 classifier-free guidance 的模型 | 双分支独立推理 | cfg_parallel |
| 张量并行 (TP) | 单层参数量超单卡显存 | all-reduce / all-gather | tp_size |
| 环状序列并行 (RSP) | 长序列场景，通信可被计算掩盖 | P2P 环形传递 | world_size |
| 流水线并行 (PP) | 层数多、单层显存可承受 | send/recv | pp_size |

> 边界：本技能主轴 = **并行策略选型决策 + 调优验证 + 现成实测/缺陷绕过**（S3 承担，选型为主、
> 开发为辅）。官方特性档位与接口事实查 `performance-optimization/references/mindiesd-features.md`
> （并行/通信掩盖章，docs 同步刷新；mindiesd 未来官方并行接口落地后同样先查该真相源），
> 框架侧开启与生效验证走 framework-feature-enablement；框架自身无机制、需**结构性开发**
> （非现成 patch 可注入）→ `framework-extension-dev`（经 model-auto-optimization §0 确认）。

## 已实测（910B NPU，2026-08，4 卡 CP + 通信掩盖）

> 数据来源：`dummy-run/references/minimax-h3-notes.md` §9（MiniMax-H3 256×384×124, 2 layers, bf16）。

### Ulysses USP 触发条件

Ulysses 的 all_to_all FA 切头路径**不会自动生效**：仅给 attention processor 设 seq 分片时
走的是非 CP 路径（profile 只有 allGather 无 allToAll）。必须给每个
`attn.processor._parallel_config` 设 `ParallelConfig(context_parallel_config=cp_cfg)`
（注意是 `ParallelConfig` 包装，不是 `ContextParallelConfig`），才会触发切头路径。

### 通信掩盖（comm-stream masking）

把 `torch.distributed._functional_collectives.all_to_all_single` /
`all_gather_tensor` monkey-patch 为 `mindiesd.parallel` 的专用流版本
（见 `examples/dummy_run/masking.py`）：HCCL 集合跑在独立 comm stream 上，
与 caller stream 的计算重叠（compute 记 ready 事件 → comm stream 等 → HCCL → 记 done → compute 等）。

实测收益（mask on/off）：

| 指标 | unmasked CP | masked CP | 改善 |
|---|---|---|---|
| kernel 总耗时 | 37.5ms | 12.1ms | -67.7% |
| Communication（未掩盖） | 31.4ms | 1.5ms | -95.2% |
| Stage（设备时间线） | 71.5ms | 35.5ms | -50.3% |

> 边界：本节只覆盖「现成 `mindiesd.parallel` patch 的注入与实测」。若目标框架**自身无 comm-stream
> 机制**（如 vLLM-Omni 0.28）且 monkey-patch 不可行、需要为框架结构性开发该机制 → 属框架侧补齐，
> 经 model-auto-optimization §0 用户确认后走 `framework-extension-dev`（本 skill 不做框架结构性开发）。

### HcclAlltoAllV 缺陷与绕过（重要）

CANN 9.1.0 环境 `HcclAlltoAllV`（split 路径）**SIGSEGV**（等分 `HcclAlltoAll` 正常）。
绕过方案：**pad + 等分**——把 input 各块 pad 到 `S_PAD`（128 倍数，由全局 max(out_sizes)
推导，全 rank 一致），用等分 `HcclAlltoAll(count=S_PAD×row_elems)` 交换，再 slice 各块前
`out_sizes[j]` 行。实测 err=0.0，kernel 名从 `hcom_alltoallv` 变为 `hcom_alltoall`（等分）。

### 内存受限时的并行解锁（2026-09，H3 × vllm-omni 0.28 实测回填）

更优并行策略常因单卡显存不可行（例：H3 BF16 单 rank 全量 135G > 128GB，无法 TP1×USP4）：

- **先在无损阶段尝试 offload 类特性降显存解锁**，而非直接放弃或跳到有损。特性按框架命名不同、语义都是
  "权重/激活出显存"：vllm-omni `--enable-cpu-offload` / `--enable-layerwise-offload` /
  `--enable-distributed-layerwise-offload`（DLO，host 存 1/DP + H2D/AllGather 重叠，官方支持叠加
  online INT8/FP8/MXFP8）；mindiesd `enable_offload`；PyTorch FSDP/CPU-offload 语义开关。
  注意互斥与副作用：如 vllm-omni FastH3 拒绝任何 offload；950PR 上普通 layerwise offload 会触发
  OOM killer（用 DLO 而非普通 layerwise）。
- **有损（量化等）完成后复查被显存卡住的通信组合**：降显存会解锁新并行（INT8 online 后 H3 可 TP1×USP4，
  60 步 85.1s，-11.7% vs USP2-int8）；每个量化档落地后回跑「并行×显存余量」候选快测。
- 并行选型后做通算掩盖评估：用单步捕获的 step_trace（Computing/Communication(未重叠)/Free）量化暴露通信
  （H3 USP2 单步实测未重叠通信占 27.5%、Overlapped=0 → 掩盖空间明确但需框架侧 comm-stream 支持，
  实现参照本仓 mindiesd/parallel + LightX2V `hccl_eager` 合入姿势；compute-bound 时先长序列复测再投入）。

### 拓扑相关选型（2026-09，950PR × LightX2V 实测回填）

同一并行形态在不同互连拓扑下排名可能反转，**别无条件复用历史结论**（详见
`references/ascend-topology-bandwidth-diag.md`）：

- 读取 `npu-smi info -t topo`：UB=HCCS 同岛、SYS=跨 PCIe/NUMA；4 卡 a2a 优先单 UB 岛
- 单 UB 岛 → USP4 **bulk** 最优（comm busy 最小、同步事件最少）；SYS 跨岛组 → **head-parallel**
  更优（实测 rank0 clean-window -11.5%，2×2 复现）：逐头小 a2a 全异步重叠，bulk 大 alltoall
  跨岛串行暴露。判据墙钟/clean-window 为准——head-parallel kernel-sum 更高（跨流多计数）
  但墙钟更低
- compile × head-parallel 不兼容（Dynamo recompile_limit 静默回退 eager）
- 带宽/环境诊断与恢复姿势（HCCL 微基准须先 `set_device` 否则报端口 bind、hccl_test 宿主可能
  不兼容、端口泄漏/卡组受损→换组验证）见同 reference

## WIP 待定内容

- [ ] 各策略在昇腾 910B 上的完整实测性能对比表
- [x] HCCL 拓扑感知的策略选择决策树（950PR UB/SYS 判据已回填：见上「拓扑相关选型」+ `references/ascend-topology-bandwidth-diag.md`）
- [ ] 混合并行策略的配置模板（如 USP + CFG 组合）
- [ ] 策略切换的性能对比方法论

## Reference Files

- `../dummy-run/references/minimax-h3-notes.md` §9 — 加载时机: 需要 CP/USP 实测细节、mask 注入代码或 AlltoAllV 绕过实现时
- `references/ascend-topology-bandwidth-diag.md` — 加载时机: 950PR/多卡拓扑选型、HCCL 带宽验证（hccl_test 或 torchrun 等价工具）、端口 bind/卡组受损等环境诊断时

## 维护与更新

当新的并行策略经验证有效、多卡互联拓扑发生变化或发现新的分布式训练模式时，
按 dev-workflow 的复盘流程更新本 skill。
