---
name: benchmark-dev
description: MindIE-SD 核心算子（FA/BSA/GMM/MM）性能基准工具链：开发为主（benchmarks/ 架构、CLI/口径/schema 约定、扩展新算子/指标/config 键、数据异常排查），使用为辅（mindie_bench CLI 在模型调优中对单算子做性能分析、按 dtype/量化档/参数形态对比选出性能最优算子配置）。当用户需要新增或修改 benchmark 代码（mindie_bench CLI、MFU/MBU 口径、HTML 报告、op_defs/vendor、schema、example/）、排查 benchmark 数据异常、给基准加算子或指标、或在模型调优中用基准对比 FA/BSA 等单算子配置选型时应触发；即使用户只说"benchmark 数据不对""给基准加个算子""报表格式改一下""对比下这几个 FA 实现哪个快"也应触发。
---

# Benchmark 工具链开发与算子选型（MindIE-SD 核心算子）

`benchmarks/` 下的算子性能基准工具链（FA / BSA / GMM / MM 微基准、报告与 drift 门禁）。
本 skill 覆盖**两个方向**：

1. **开发 benchmark**（本上下文主体）：修改/扩展工具链代码本身——CLI、口径、schema、报告、数据异常排查
2. **使用 benchmark**：mindie_bench CLI 的应用——**模型调优中对单算子做性能分析，选出性能最优的 FA/BSA 实现**（命令行的约束与使用也是应用该工具的过程）

## 1. 定位：开发 + 使用

### 1.1 开发（面向工具链代码）

架构三层（common 口径单点 / xpu-perf-plugin 运行时 / scripts 报告）、test-first、schema 数据驱动、数据守卫。见 §2-§7。

### 1.2 使用：模型调优中的算子选型（面向推理模型）

在模型调优/性能优化流程中，用 benchmark 对**单个算子**做实测对比，为模型选择最优算子配置：

- **对比不同 dtype/量化档（真实内核分发）**：内核按 dtype/quant_algo 自动选择（fa bf16→torch_npu.npu_fusion_attention、mxfp4→torch.ops.mindiesd.quant_flash_attn；mm NO_QUANT→matmul、W8A8→npu_dynamic_quant...），用 `--config {dtype: [...]}` 一次对比不同实现；`func` 键只是报告系列标签（fn=），**不切换内核**。如

  ```bash
  python benchmarks/scripts/mindie_bench.py run --op "{fa: {}}" \
      --config "{seqlen: [4096, 8192], dtype: [bf16, fp8, mxfp4], timeout: 300, peak_flops: 377.78}"
  # report --report-dir <父目录> 合并不同 run，对比曲线
  ```

- **对比不同参数形态**：heads / head_dim / 量化档（dtype）/ 稀疏度（BSA sparsity）对性能的影响

  ```bash
  # 同 dtype 下扫 heads，选吞吐最优的 head 配置
  python benchmarks/scripts/mindie_bench.py run --op "{fa: {num_heads: 16}}" --config "{seqlen: [8192], dtype: [bf16], peak_flops: 377.78}"
  python benchmarks/scripts/mindie_bench.py run --op "{fa: {num_heads: 32}}" --config "{seqlen: [8192], dtype: [bf16], peak_flops: 377.78}"
  # BSA 扫稀疏度，看 MFU/latency 随 sparsity 的收益曲线
  python benchmarks/scripts/mindie_bench.py run --op "{bsa: {}}" \
      --config "{seqlen: [8192, 16384], dtype: [bf16], sparse: [0.6, 0.8, 0.95, 0.99], timeout: 300, peak_flops: 377.78}"
  ```

- **对比多档位**：一次 `--config` 内列表值笛卡尔积（heads × dtype），一次跑完对比
- **多 run 合并**：不同实现的 run 放同一 report 父目录下，`report` 合并为单 HTML 对比

**选型依据**：MFU（计算利用率）、latency；FA/BSA 对比时注意口径一致性（**同一 `peak_flops`/`peak_bw` 输入**、同 seqlen、同 dtype）。选出的最优配置再进入模型（结合 performance-optimization 的方案实施与复验）。

## 2. 架构（三层，口径单点）

```text
benchmarks/
├── common/                    # 口径单点维护（运行时 + 离线共享，禁止两处漂移）
│   ├── schema.py              # OP_SLOT_ARGS / OP_SEQ_AXIS / OP_SERIES_KEY /
│   │                          #   OP_DISPLAY_METRICS / BASELINE_METRICS / COMPARE_METRICS
│   ├── metrics.py             # util_metrics（MFU/MBU 公式 + 钳位 ≤1）——唯一公式
│   └── env_util.py            # env 工具（load_peaks 保留兼容、无调用方；峰值来源已改为 --config 输入）
├── scripts/
│   ├── mindie_bench.py        # 推荐 CLI 入口（run/report/compare），纯逻辑可单测
│   └── benchmark_report.py    # baseline 导出 / 快照 / HTML / compare（离线纯 Python）
├── xpu-perf-plugin/           # 运行时（基于 xpu-perf micro_perf，spawn 多进程）
│   ├── npu_launch.py          # 旧入口（保留兼容）
│   ├── backend_npu.py         # BackendNPU：墙钟计时、per-case 超时、异常透传、数据守卫
│   ├── op_defs/               # 基础实现：FLOPs/字节记账 + schema（vendor 未实现时抛 NotImplementedError）
│   │   └── _common.py         # tensor_bytes / quant_flops / attention_valid_parts（纯函数可测）
│   └── vendor_ops/NPU/        # NPU 实现（峰值不内置，由使用者 --config 输入）
├── example/                 # 样例脚本 + 使用说明（example/README.md）
└── tests/UT/benchmark/       # 离线单测（无 NPU 依赖）
```

数据流：`run`（vendor 执行 → jsonl 原始值，case arguments 携带 peak_flops/peak_bw）→ `report`（离线用 entry 携带的 peak 重算 MFU/MBU → baseline + 快照 + HTML + CSV）→ `compare`（drift 门禁）。
**关键契约**：运行时与离线通过 `common/` 共享口径（schema/metrics）；jsonl 行格式 `{"op_name","arguments","targets"}`。

## 3. 开发流程（test-first）

1. **先写测试**（`tests/UT/benchmark/`，确认 FAIL）——离线纯 Python 部分全部可本地测：
   - env 解析 / MFU/MBU 公式与钳位 / slot 键规范化 / 记账纯函数 / CLI 解析 / 报告聚合渲染
   - fixture 用合成 jsonl（格式与真实一致）
2. 实现功能（code-standards：ruff、<100 行/行、导入排序）
3. 远端部署验证（ascend-deploy：SSH 同步 + docker exec + 实测跑通）
4. 涉及运行时（vendor/backend）的改动需远端 NPU 实测；纯离线改动本地单测即闭环

```bash
python -m pytest tests/UT/benchmark -q        # 本地
python -m ruff check benchmarks tests/UT/benchmark
```

## 4. 核心设计约定（改代码前必读）

### 3.1 CLI 参数约定（严格）

- **不随意新增命令行参数**：需要新配置时优先进 `--config` 键（`seed`、`timeout`、`peak_flops`、`peak_bw` 都是 config 键而非独立参数）
- 职责分离：`--op` = 算子选择 + 结构参数（num_heads/head_dim/K/N/... + 保留键 `func`=内核来源标签，**不切换内核**）；`--config` = 扫描键 + 峰值（seqlen→各 op 扫描轴 / dtype / sparse→sparsity / quant_algo / seed / timeout / peak_flops / peak_bw），列表=笛卡尔积、标量=固定
- `--op` 不允许扫描键；`--config` 白名单在 `CONFIG_ALLOWED_KEYS`
- 裸形式不带引号（shell 拆 token 由 `nargs="+"` + `_join_nargs` 合并；宽松解析用正则补引号，注意相邻裸值逗号用 lookahead 不消费）

### 3.2 口径单点

- MFU/MBU 公式**只在** `common/metrics.py util_metrics`；运行时（op_defs `MfuMbuSummaryMixin`）与离线（benchmark_report `recompute_util`）都调它，改一处即可
- 钳位 ≤1 在公式层（量化档真实吞吐可能超 bf16 峰值口径）；**数据层保持小数，百分比只是展示层**（`_pct`）
- 峰值：**不内置**——`peak_flops` / `peak_bw` 必须由使用者通过 `--config` 输入（代码中不允许出现硬编码峰值如 377.78）；随 case arguments 走 jsonl，离线 `recompute_util` / 报告 env 段从 entry 读；无 env.json / --env

### 3.3 schema 数据驱动

- slot 键 / 序列轴 / 系列键 / 展示指标都是 schema 表——新增 op 或维度改表，报告聚合（`_aggregate_cases`）与渲染（`build_op_html`）按表驱动，避免硬编码特判蔓延
- `OP_DISPLAY_METRICS`：每 op 展示哪些指标（FA/BSA/MM 仅 MFU，GMM MFU+MBU）——图与表格列都按它动态生成
- fa/bsa slot 键含 num_heads/head_dim/func（func 缺省省略），报告系列标签 `dtype h{heads} d{dim} [fn=函数名]`

### 3.4 数据有效性守卫

- 异常 case 不得产生数据行：`collect_baseline` 过滤无有效测量的 entry（崩溃/异常 case 的 targets 为空）
- vendor 层可做输出有效性校验（如 BSA 全零输出 → 抛 RuntimeError → case 标记无效），校验只做一次（warmup 内，不进计时区）

### 3.5 报告

- 每 op 一个章节：一张/多张综合图（按 OP_DISPLAY_METRICS）+ 图下 per-series 数据表
- 「Command」段（run 写 `run_command.txt`）+ 「Peak config (CUBE flops / bandwidth)」段
- 多 run 合并：`load_report_entries` rglob 递归 + 同 slot 取最新 mtime
- **CSV 是数据源**：`reports/<op>.csv` 每行含 peak_flops/peak_bw；在 CSV 中更新 peak 后重跑 `report`，`read_peaks_from_csv` 读回新值 → `apply_csv_peak_updates` 覆盖 entry peak → 重算 MFU/MBU 与 Peak config 段（未改 CSV 时以 run 的 --config 峰值为准）

## 5. 扩展点

### 4.1 新增算子（五步）

1. `op_defs/<op>.py`：base（`MfuMbuSummaryMixin + BasicOp`，FLOPs/字节记账 + `vendor_impl_run` 抛 NotImplementedError）
2. `vendor_ops/NPU/<op>.py`：`register_vendor_impl("<op>", "NPU")` 真实 kernel（**调用参数必须对齐算子自身 UT**，见调试章节）
3. `common/schema.py`：`OP_SLOT_ARGS` / `OP_SEQ_AXIS` / `OP_SERIES_KEY` / `OP_DISPLAY_METRICS`
4. `scripts/mindie_bench.py`：`VALID_OPS` + `OP_DEFAULTS`
5. 记账纯函数加进 `op_defs/_common.py` 并单测

### 4.2 新增 config 键

1. `CONFIG_ALLOWED_KEYS` 加键（`peak_flops`/`peak_bw` 已在其中：消费方在 op_defs `MfuMbuSummaryMixin.summary` 与离线 `recompute_util` 从 args_dict/entry arguments 读取）
2. 消费方实现（case 参数 → 运行时行为，如 seed 在 op prepare_args 设 RNG、timeout 在 backend `_op_timeout` 读 args_dict）
3. 不进 `OP_SLOT_ARGS`（不影响 slot/报告）

### 4.3 改展示 / 口径

- 展示指标：`OP_DISPLAY_METRICS`
- 公式/钳位：`common/metrics.py`（**先确认是否影响 baseline/compare 兼容性**）

## 6. 调试（benchmark 数据异常 = 代码问题）

排查方法论（完整案例见 `references/debugging.md`）：

1. **先跑算子自身 UT**（`tests/plugin/test_<op>.py`）分流：UT 通过 → benchmark 调用/计时代码问题；UT 失败 → 算子/环境问题
2. **恒定 latency** = 计时区被固定开销污染（如 mask 构造每 iter 分配）或 kernel 空转——检查计时区内是否有可移到区外的构造/分配
3. **全零输出** = kernel 未执行：对照算子 UT 调用参数（BSA `inner_precise` 在 950 设备必须为 4，用 0 会全零；mask 需 per-row uniform 避免全零行崩溃）
4. **偶发污染**：异常大 latency → 增量重跑该 case 确认；加数据守卫
5. **长序列超时**：`--config {timeout: 300}`（默认 5s），backend 按 case 读
6. **增量合并**：每次补跑独立 report_dir；mtime 决定覆盖顺序，避免坏值覆盖好值

## 7. 测试

- `tests/UT/benchmark/`：env_util / metrics / schema / op_defs 记账 / mindie_bench 解析 / benchmark_report 聚合渲染 / 异常 entry 过滤
- conftest：sys.path 注入（benchmarks/ + scripts/ + xpu-perf-plugin/）、tmp_path 重定向（沙箱环境）、禁 cacheprovider
- 渲染测试用合成 report dict 验证 HTML 结构（图数量、百分比、Command 段、heads/dim/func 标签）

## 8. 维护与更新

出现以下情况时更新本 skill：

- 工具链接口变化（CLI 参数、schema、报告格式、config 键）
- 新算子/新指标加入
- 新的数据异常模式被定位（追加 references/debugging.md）
- 口径变更（MFU/MBU 公式、峰值来源、展示方式）
- 按 dev-workflow §6 复盘流程同步刷新 `.agents/README.md` 技能总览

## 参考文件

- 📋 `references/debugging.md` — 加载时机: benchmark 数据异常（全零/恒定 latency/偶发污染/增量合并陷阱）排查时
- 📄 `../ascend-deploy/SKILL.md` — 加载时机: 远端同步/运行/容器操作时
- 📄 `../code-standards/SKILL.md` — 加载时机: 编写 Python 代码时
- 📄 `../dev-workflow/SKILL.md` — 加载时机: 开发流程/复盘归档时
