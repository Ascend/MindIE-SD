# model-auto-optimization scripts

NPU 前结构性门禁脚本（零数据、零 NPU；产物数据不入 git）。

## seam_check.py —— 特性组合静态判定

特性声明（`feature_declarations.json`）**跨模型**：声明每个 mindiesd/框架特性作用的 seam /
显式互斥 / 依赖模型能力 / 激活窗口；模型只提供能力子集（decl 的 `models` 适配示例，完整适配记 case）。

```bash
# 无冲突组合（跨 seam）：通过
python seam_check.py --features quant_w8a8_dynamic,sparse_rf_v2,cache_dit \
    --model minimax-h3-vllm-omni
# 同 seam 双 writer（量化×量化）→ error；双缓存（DiTCache×AttentionCache）→ error
python seam_check.py --features cache_dit,cache_attention --model minimax-h3-vllm-omni
# 稀疏×缓存同 step_decision → warning（需错开窗口/只留最强档）
python seam_check.py --features sparse_rf_v2,cache_dit --model minimax-h3-vllm-omni
```

## manifest_dryrun.py —— manifest.toml 可复现门禁

```bash
python manifest_dryrun.py --manifest {plan}.toml
```

做：schema/枚举 → 特性 seam（调 seam_check）→ 路径只读检查 → 按框架渲染启动计划
（vllm-omni / lightx2v；其余框架报 unsupported 不猜测）。依赖 python ≥3.11 tomllib，
不依赖 NPU/权重/框架运行。

## stage_gate.py —— 编排层阶段推进门禁

```bash
# 证据按任务隔离（task_id = runs 目录名）；校验 evidence/{task_id}/ 前缀（防旧残留冒充）
python stage_gate.py --stage S3 --task-id 20260908_minimax-h3_optimization --run-dir {工作目录}/agentic
# close 自动联动：report_lint.py（总览表结构）+ evals/scripts/check_profile.py（profile 强校验）
python stage_gate.py --stage close --task-id 20260908_minimax-h3_optimization --run-dir {工作目录}/agentic
```

做：解析 run-state.md（`references/run-state.md`）「阶段推进表」→ 校验目标阶段 status=done 且
声明的验收证据路径存在；提供 `--task-id` 时校验证据落 `evidence/{task_id}/`（任务隔离，2026-09-08
起强制——防旧任务残留文件充当本轮证据）；close 额外强制声明 overview_report.md / detail_report.md
（缺任一视为未闭环）并自动跑报表 lint 与 profile 校验（`_run_close_tools`）。
退出码 0=通过（可推进/可宣称闭环），1=存在 error（不得推进）。零 NPU、零数据。

## report_lint.py —— 总览表结构校验（close 前置 · 机器校验）

```bash
python report_lint.py {overview_report.md}
python report_lint.py ../runs/20260908_minimax-h3_optimization/overview_report.md
```

校验 overview_report.md 主表：8 列表头（优化类型|特性名|e2e 耗时|首步耗时|步数|加速比|质量数据|
说明）精确匹配；每行优化类型枚举/特性名非空/e2e·首步·步数·加速比单值；锚点行（基线/三元/最强/
最终推荐）e2e 禁 [估算]；质量列无损=输出一致、有损=数值非空。自测样例 `report_lint_cases.md`。
背景：2026-09-08 报表曾以旧 6 列经验交付（列不符缺口）——本 lint 强制报表结构与
`references/overview-report.md` §2/§7 契约一致，禁止凭人肉对照交付。

## 维护

- 特性命名/声明以 `docs/zh/features` 与 `framework-support-matrix.md` 为准；docs 或框架升级时
  同步刷新 `feature_declarations.json`（同 support-matrix 刷新协议）。
- 新框架接入 dry-run → 在 `manifest_dryrun.py` 的 `RENDER_HINTS` 补渲染提示，并在
  `references/manifest-schema.md` 登记。
- **数据随脚本就近**：`feature_declarations.json`（数据）与 `README.md`（文档）放在 `scripts/`
  是因为 `seam_check.py` / `manifest_dryrun.py` 以同目录相对路径读取、脚本与数据同源同改；
  它不属于 `references/` 的命名词表（该词表只管案例/方法论/流程类 md）。
