# 组合搜索编排与多 agent 扩展（可选自动化）

> 定位：把「候选集 + 尝试 + 评估」的**人工推进节奏**升级为可自动推进的编排——默认仍保持人工确认
> 节奏（§0/预算点），仅在需要并发维度探索时启用多 agent。不引入外部运行时框架，是本入口的
> 轻量流程契约 + 护栏。VLM 判卷为可选（合规 API 可用才接入，否则 inconclusive + 并排存证）。

## 1. 单会话自动推进协议（组合搜索编排）

输入：候选队列（来自 `combination-search.md` 候选登记表 + `framework-support-matrix` 裁剪 +
seam_check 静态通过）。规则：

1. **队列与顺序**：按「低成本高概率优先」，先跨 seam 组合，单变量叠加；每候选先 seam_check。
2. **预算**：设单候选评估次数与总预算（时间/NPU 时数）；到点**带证据收尾**（采纳/回退 + 记录），
   不无限调优（同 model-auto-optimization 限时收敛纪律）。
3. **NPU 单飞**：同卡组一次只跑一个实验（资源锁/队列），防止并发污染对比口径。
4. **推进判定**：每候选 = 三层证据 + 质量门禁（quality-gate/evals）+ frontier 保留
   （质量不降且墙钟提升）→ 采纳入队尾或回退记录。
5. **暂停点**：每完成一个特性族（如全部 cache 档）停下来向用户汇报并确认下一族，保留人工闸门。
6. **产出**：候选登记表 + 每特性内部组合搜索数据子表 → 喂 `overview-report.md` 分特性部分。

## 2. 多 agent 扩展（并发维度探索时启用）

- **形态**：一个编排主代理 + 每维度 executor（后台子代理，如 cache / sparse / quant / parallel 各一）；
  executor 只在自己的维度内按本协议产出候选与证据，回填共享候选登记表（共享工作区文件）。
- **护栏（必须）**：
  - NPU 单飞：executor 抢占前先登记卡组，一次一实验（可用远端空闲卡选择 + 锁文件）；
  - 卡组隔离：不同 executor 分配互斥卡组，禁止同卡混跑；
  - 预算与 watchdog：每 executor 设 max_iters 与超时；结构性失败记为 proposal 不静默终止；
  - 结果汇总：master 负责 dedupe / 冲突消解 / frontier 合并（禁止两个 executor 改同一配置/文件）。
- **触发条件**：仅在「同一模型多维度可并行探索」且你有意并行时启用；单会话默认人工节奏。

## 3. 与既有产物的衔接

- seam/能力判定 → `scripts/seam_check.py`（候选入队前静态过滤）；
- 可复现运行 → `references/manifest-schema.md` + `scripts/manifest_dryrun.py`（每候选一个 manifest）；
- 组合细节/层回退 → `combination-search.md`（候选登记表是共享数据面）；
- 总览/子搜索数据 → `overview-report.md`（编排结束直接生成报表）。

## 4. 可选：VLM 判卷

判定标准已在 `evals/rubrics/visual-artifact-gate.md`；工具层不强制封装。当合规的外部 VLM API 可用时，
可加 wrapper 复用 rubric 提示；不可用则维持人工并排 + 定量（inconclusive 路径属合法结论）。

## 维护与更新

编排规则/护栏变化 → 同步本文件与 model-auto-optimization SKILL（阶段纪律/自动化执行要点）；
每轮编排复盘后回填「编排经验」到候选登记表/报表，不重复维护。
