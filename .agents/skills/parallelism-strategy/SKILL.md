---
name: parallelism-strategy
description: [WIP] 大模型分布式并行策略选型参考。当前提供策略分类概览（TP/USP/RSP/CFG），
             具体决策树和实测性能数据待后续补充。
             当用户需要在多个 GPU/昇腾设备上训练大模型、了解可选并行策略时使用此 skill。
             由 dev-workflow 在多卡场景中触发。
---

# 并行策略选择

> **⚠️ WIP — 此 skill 尚未完成。** 当前仅提供策略分类大纲，具体实施指南待后续补充。

## 策略一览

| 策略 | 适用场景 | 通信模式 | 关键参数 |
|------|---------|---------|---------|
| Ulysses 并行 | 多模态扩散模型（序列维度切分） | all-to-all | ulysses_size |
| CFG 并行 | 开启 classifier-free guidance 的模型 | 双分支独立推理 | cfg_parallel |
| 张量并行 (TP) | 单层参数量超单卡显存 | all-reduce / all-gather | tp_size |
| 环状序列并行 (RSP) | 长序列场景，通信可被计算掩盖 | P2P 环形传递 | world_size |
| 流水线并行 (PP) | 层数多、单层显存可承受 | send/recv | pp_size |

## WIP 待定内容

- [ ] 各策略在昇腾 910B 上的实测性能数据
- [ ] HCCL 拓扑感知的策略选择决策树
- [ ] 混合并行策略的配置模板
- [ ] 策略切换的性能对比方法论

## 维护与更新

当新的并行策略经验证有效、多卡互联拓扑发生变化或发现新的分布式训练模式时，
按 dev-workflow 的复盘流程更新本 skill。
