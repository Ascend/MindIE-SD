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
python stage_gate.py --stage S0 --run-dir {工作目录}/agentic
python stage_gate.py --stage close --run-dir {工作目录}/agentic
```

做：解析 run-state.md（`references/run-state.md`）「阶段推进表」→ 校验目标阶段 status=done 且
声明的验收证据路径存在；close 额外强制声明 overview_report.md / detail_report.md（缺任一视为
未闭环）。退出码 0=通过（可推进/可宣称闭环），1=存在 error（不得推进）。零 NPU、零数据。

## 维护

- 特性命名/声明以 `docs/zh/features` 与 `framework-support-matrix.md` 为准；docs 或框架升级时
  同步刷新 `feature_declarations.json`（同 support-matrix 刷新协议）。
- 新框架接入 dry-run → 在 `manifest_dryrun.py` 的 `RENDER_HINTS` 补渲染提示，并在
  `references/manifest-schema.md` 登记。
- **数据随脚本就近**：`feature_declarations.json`（数据）与 `README.md`（文档）放在 `scripts/`
  是因为 `seam_check.py` / `manifest_dryrun.py` 以同目录相对路径读取、脚本与数据同源同改；
  它不属于 `references/` 的命名词表（该词表只管案例/方法论/流程类 md）。
