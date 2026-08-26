---
name: operator-dev
compatibility: 外部技能库 cannbot-skills（线上 https://gitcode.com/cann/cannbot-skills，本地可选 clone）；triton / torch_npu / CANN 环境
description: 算子级开发与性能优化：Triton / Ascend C / Catlass / PyPTO / TileLang 算子
             编写、精度对齐与性能调优。优先路由到外部 cannbot-skills 技能库（不重复其内容），
             本 skill 只保留场景 → skill 映射与 MindIE-SD 特有补充。
             当用户需要新增/优化算子、定位算子性能或精度问题时使用此 skill。
             即使用户只提到"写个 triton kernel"或"这个算子怎么加速"而未说算子，也应触发。
             由 dev-workflow 或 compilation-dev 的 replacement kernel 场景触发。
---

# 算子开发（复用 cannbot-skills）

## 定位

算子级开发/优化任务按场景路由到**外部技能库 `cannbot-skills`**（完整链路：
triton-task-extractor → triton-op-designer → triton-op-coding → triton-latency-optimizer →
triton-simulator-optimizer / triton-precision-debug / triton-op-verifier，以及 Ascend C /
Catlass / PyPTO / TileLang 各 DSL 链）。

**本 skill 不复制外部库内容**，只提供：

1. 场景 → skill 映射（见 `references/operator-optimization-skill-map.md`）
2. MindIE-SD 特有、cannbot 未覆盖的补充经验（下表）

## 场景路由

→ `references/operator-optimization-skill-map.md`（完整路由总表）

## 本仓库补充经验（cannbot 无对应内容）

| 场景 | 入口 |
| --- | --- |
| **register_replacement pattern 命中≠收益**：kernel diff → 逐 pass AB → R1-R5 根因目录（含"负收益先查 kernel 形态"教训） | `compilation-dev/references/benefit-rootcause.md` |
| **kernel diff 方法论**：kernel_details.csv 聚合对比、L2-flush bench 必须放计时区外、warm/cold 双档测量 | `compilation-dev/references/benefit-rootcause.md` §2 + `dev-workflow/references/rework-lessons.md` |
| **模型级验证闭环**（不只看单测）：compute-precision 图验证、叠加 AB、远端 NPU 部署流程 | `dev-workflow/references/rework-lessons.md`、`dummy-run-dev/` |
| MiniMax-H3 算子上下文（npu_swiglu 语义、表 [3,D] L2 驻留、真实图形态） | `dummy-run-dev/references/minimax-h3-notes.md` §10 |

## 使用约束

- **外部库不可用时**（未 clone / 无网络）：跳过外部引用，仅使用上表本仓库补充经验，
  并把场景记入 dev-workflow §6 复盘（触发 cannbot 可用性确认）。
- 不把 cannbot-skills 的内容抄入本 skill；引用时给出 skill 名与场景即可。
- 本仓 fusion pattern 的 replacement kernel（triton 自研）走 `compilation-dev` 的
  pattern 生命周期，本 skill 只负责 kernel 本体开发与调优。

## Reference Files

- 🗺️ `references/operator-optimization-skill-map.md` — 加载时机: 任何算子开发/优化任务开始前（场景路由）
- 🔗 `../compilation-dev/references/benefit-rootcause.md` — 加载时机: replacement kernel 命中但收益存疑时
- 📝 `../dummy-run-dev/references/minimax-h3-notes.md` — 加载时机: 涉及 MiniMax-H3 算子语义/图形态时

## 维护与更新

当外部 cannbot-skills 结构变化、新增已验证的算子 DSL/优化方法、或本仓沉淀新的算子
教训时，按 dev-workflow 的复盘流程更新本 skill 与 skill-map。
