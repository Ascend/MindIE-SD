---
name: parallelism-strategy
compatibility: 无额外工具（参考数据来自 dummy-run-dev/references/minimax-h3-notes.md §9）；多卡需 HCCL 环境
description: 大模型分布式并行策略选型与实测：Ulysses USP（all_to_all FA 切头）、
             Context Parallel 通信掩盖（comm-stream masking）、CFG 并行、TP/RSP 概览。
             含 910B 实测数据与 HcclAlltoAllV 缺陷绕过方案（pad+等分）。
             当用户需要多卡并行策略选择、上下文并行/通信掩盖调优、或排查多卡通信问题时使用此 skill。
             即使用户只提到"多卡跑不动"或"通信暴露大"而未说并行，也应触发。
             由 dev-workflow 在多卡场景中触发。
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

## 已实测（910B NPU，2026-08，4 卡 CP + 通信掩盖）

> 数据来源：`dummy-run-dev/references/minimax-h3-notes.md` §9（MiniMax-H3 256×384×124, 2 layers, bf16）。

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

### HcclAlltoAllV 缺陷与绕过（重要）

CANN 9.1.0 环境 `HcclAlltoAllV`（split 路径）**SIGSEGV**（等分 `HcclAlltoAll` 正常）。
绕过方案：**pad + 等分**——把 input 各块 pad 到 `S_PAD`（128 倍数，由全局 max(out_sizes)
推导，全 rank 一致），用等分 `HcclAlltoAll(count=S_PAD×row_elems)` 交换，再 slice 各块前
`out_sizes[j]` 行。实测 err=0.0，kernel 名从 `hcom_alltoallv` 变为 `hcom_alltoall`（等分）。

## WIP 待定内容

- [ ] 各策略在昇腾 910B 上的完整实测性能对比表
- [ ] HCCL 拓扑感知的策略选择决策树
- [ ] 混合并行策略的配置模板（如 USP + CFG 组合）
- [ ] 策略切换的性能对比方法论

## Reference Files

- 📝 `../dummy-run-dev/references/minimax-h3-notes.md` §9 — 加载时机: 需要 CP/USP 实测细节、mask 注入代码或 AlltoAllV 绕过实现时

## 维护与更新

当新的并行策略经验证有效、多卡互联拓扑发生变化或发现新的分布式训练模式时，
按 dev-workflow 的复盘流程更新本 skill。
