---
name: profiling-collect
compatibility: torch_npu.profiler, CANN set_env.sh, paramiko（collect_profile.py）, tar
description: profiling 数据统一采集：远端昇腾采集 NPU 性能数据回传本地，供 profiling-analyze 分析。
             两条入口——mindiesd 自家脚本入口（collect_profile.py）与三方框架入口
             （collect_patch_template.py 补丁 + torchrun，含 vLLM-Omni HTTP 服务等分场景）；
             warmup 在 profiler 外（默认 5 步，compile ≥10）。
             只要用户想"开 profiler/采 profile/拿算子级数据"，自家脚本或 vllm、LightX2V 等框架都先
             用本入口；分析/瓶颈定位/优化选型/并行/基准请求分别属 profiling-analyze、
             dit-perf-opt、dit-parallel-opt、benchmark-dev，本入口不承接。
             由 model-auto-optimization 的 S1（融合分析）阶段调用，亦由 profiling-analyze 与 dev-workflow 的
             采集场景指引加载。
---

# Profiling 数据采集

在远端昇腾设备上采集性能 profiling 数据，为 profiling-analyze 提供标准化输入。

## 核心流程

```text
部署代码（env-install） → 开启 Profiler → 运行推理 → 压缩 → 回传本地
```

> 部署由 env-install/scripts/deploy_to_remote.py 完成。本 skill 仅负责 profiling 采集。

## 三方框架入口（补丁采集法）

适用：框架自带 CLI/runner 且无 `--profile` 入口（LightX2V / DiffSynth-Engine / vLLM-Omni
等），需要任意 python 入口 + torchrun 多卡采集时。方法要点（模板见
`scripts/collect_patch_template.py`）：

- 在框架**顶层推理方法**上打最小侵入包装补丁，不改框架主循环；只对 warmup 后的第 1 步开 profiler
- warmup 必须在 profiler 外（默认 5 步；MindieSDBackend 编译场景建议 ≥10 覆盖 JIT）
- 只 rank0 采集，避免多卡重复输出
- `tensorboard_trace_handler` 产出标准 `ASCEND_PROFILER_OUTPUT/`
  （kernel_details.csv + trace_view.json + step_trace_time.csv，与自家入口同格式 → 同一分析管道）

torchrun 启动示例：

```bash
export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_NPU_SOCKET_PORT_RANGE=20000-21000   # 防与 vLLM 等既有 HCCL 进程的 16666 端口冲突
source /usr/local/Ascend/ascend-toolkit/set_env.sh

nohup torchrun --standalone --nproc_per_node=4 /home/{user}/collect_patch.py \
  --model_cls {框架模型类} --task {任务} --model_path {权重目录} --seed 42 \
  > {model_weight_dir}/prof.log 2>&1 < /dev/null &
```

回传：远端打包 `ASCEND_PROFILER_OUTPUT` 目录为 tar.gz → 本地解压后直接喂 profiling-analyze。

三方框架采集常见坑：

- 卡 Health 会波动（`UB LINK ERROR` 等瞬时错误）→ 重跑即可；对比须同卡组
- 容器内 `ss -tlnp` 可能查不到 HCCL 端口（走 NPU RoCE 网卡）→ 端口"看似空闲"仍冲突时直接加端口范围
- Windows 编辑的脚本上传报 `$'\r'` → 先转 LF
- 远端 `pgrep` 等待不可靠 → 每次只启动一个 run，用日志行数外部轮询，完成后再启下一个
- 日志含 ANSI 转义 / emoji → grep 或脚本解析前先 `strings` / 专用清洗脚本，否则命中判断失真
- **容器内 `/tmp` 与宿主 `/tmp` 不是同一目录** → 取产物一律 `docker cp`，勿在宿主同名路径找
- 单 forward kernel 采集 hook 里直接 `analyse()` 会报「daemon 不可解析」→ **离线用独立进程
  analyse**（`torch_npu.profiler.profiler.analyse`），勿在采集进程内跑

### 三方框架分场景采集要点

同一补丁思路落到具体框架时，采集挂点与 warmup 姿势不同，按下述分场景执行。

vLLM-Omni（HTTP 服务）：`vllm serve` 以 OpenAI 兼容 HTTP 服务形态运行，无 CLI
`--profile` 入口，profiling 采集点应挂在 transformer forward（包装 `pipe.transformer`
的 forward）而非服务入口：

- 用服务预热请求代替 CLI warmup：先经 HTTP 接口（如 `curl /v1/images/generations`）
  发 1-2 次真实推理请求，触发编译与显存分配稳定后，再让补丁对后续前向开 profiler
- 补丁只对预热后的下一次前向开启 profiler，其余前向原样透传；warmup 全部在 profiler 外
- 多卡走 `--usp`/多进程部署时同样只 rank0 采集，避免多卡重复输出
- spawn 多进程注意：worker 由 spawn 拉起（`VLLM_WORKER_MULTIPROC_METHOD=spawn`），
  子进程会重新 import 入口模块，补丁 import 必须落在子进程加载路径上（入口脚本顶层
  或其 import 链上），否则只有父进程被打补丁、worker 前向漏采
- 补丁后仍产出统一 `ASCEND_PROFILER_OUTPUT/`，与自家入口同格式

LightX2V / DiffSynth-Engine（torchrun / 独立 CLI）：直接用
`scripts/collect_patch_template.py` 模板包装其顶层推理方法（LightX2V 如
`MiniMaxH3TransformerInfer.infer`，DiffSynth-Engine 为 pipeline 顶层推理调用）：

- torchrun 启动前设 `HCCL_NPU_SOCKET_PORT_RANGE`（如 20000-21000），防与既有 HCCL
  进程的 16666 端口冲突
- 只 rank0 采集；warmup 在 profiler 外（默认 5 步，compile 场景 ≥10）

diffusers 单进程：无需 torchrun，本地脚本直接包装 pipeline 调用即可
（`pipe(...)`，与「前置验证」节等价），warmup/采集规则不变。

补丁加载方式二选一：经 `PYTHONPATH`/`sys.path` 把包装模块挂到框架入口脚本可见路径，
或把包装模块直接 import 进入口脚本顶层（spawn 场景必须能被子进程加载，见上）。

产出契约与框架无关：补丁统一产出 `ASCEND_PROFILER_OUTPUT/`（kernel_details.csv +
trace_view.json + step_trace_time.csv + **单元利用率档**）→ 打包回传后喂同一 profiling-analyze 管道
（analyze_trace.py / compare_traces.py），无需按框架分化。

**单元利用率档（强制 · 融合判型的唯一数据源）**：融合范围与收益判定（`fusion-scope-analyze`）依赖各计算
单元的占用比，必须采集 `--task-time=l1 --aic-mode=task-based --aic-metrics=PipeUtilization`——
否则 `memory_bound` / `*_vec_ratio` / `*_mac_ratio` / `*_mte2_ratio` / `*_mte3_ratio` / `cube_utilization(%)`
等字段一律为 `N/A`（官方口径：`task_time=l0|off` 时不产出 AI Core / AI Vector PMU 数据）。
不要只采 duration：**没有利用率档的采集对融合判定不可用**。

**执行完成检查门禁（fail-closed）**：回传/交付前逐项断言，任一不满足即判本次采集**不合格**并给出重采命令，
**不得**交给下游：

1. 执行序可用：存在含 `Name`（或 `Op Name`）/ `Duration(us)` / `Task Type` 的文件；
2. 利用率可用：存在含 `*_vec_ratio`、`*_mac_ratio`、`*_mte2_ratio`、`*_mte3_ratio` **四族**的文件
   （`op_summary_*.csv` 为权威来源；若所用版本的 `kernel_details.csv` 含同名列则等价），
   且**待判型的目标算子**这些列非 `N/A` 且非空。`memory_bound` 是**可算字段**
   （公式 `mte2_ratio / max(mac_ratio, vec_ratio)`），实测导出常不含它 ⇒ **不作必需列**，判型时现算；
3. 口径可用：warmup 已在 profiler 外剔除（见「预热」节），步数与 profile 配置随产物留痕。

> 反例（必须拦住）：把 `N/A` 当成 0 或忽略缺列 → 下游会把"无数据"读成"无内存瓶颈"，
> 属**静默失败**（判定看似完成、结论却是假的）。
>
> 机器化执行：`python .agents/skills/profiling-collect/scripts/check_output.py --dir {ASCEND_PROFILER_OUTPUT} --start-marker {run_dir}/.start_epoch`
> （**六项检查各自独立报结论**：产物存在 / 执行序 / 单元利用率 / 判型列有值（抽样） /
> **数据表行数 > 0** / **产物新鲜度**；退出码 **0 = 全部通过 / 1 = 有检查判失败 / 2 = 前置缺失 /
> 3 = 无失败项但存在「无法判定」**）；**`--selftest` 覆盖 10 类夹具逐条核对**（陈旧产物 / 只有表头零数据行 /
> 无开始标记 / 标记不存在 / 缺利用率档 / 利用率全 N/A / 缺执行序 / 空目录 + 2 类合格），pre-commit 挂钩。
> **`3` 与 `0` 必须区别对待**：没给开始标记时新鲜度只能报「无法判定」，**不得当通过**（这正是要防的事）。

各框架接入/使能上下文与实测案例参考 framework-integration 的 references：
`lightx2v-enablement.md`（LightX2V 开启方式 + 采集相关坑）、`vllm-omni-enablement.md`
（vLLM-Omni HTTP 服务使能方法 + 采集 hook 探针）、`cache-dit-enablement.md`（cache-dit × vLLM-Omni
托管链开启方式）、`diffsynth-engine-enablement.md`（DiffSynth-Engine compile 接入/使能与验证），见文末 Reference Files。

### 运行时集合通信采集（collective shim 法 · 零仓库改动）

需要**通信分布**（哪些阶段、哪些 collective、多少字节）而非 CANN 默认输出时，用运行时 shim：

- **做法**：包装 `torch.distributed` 的 6 个 collective（`all_reduce` / `all_gather` /
  `all_to_all_single` / `all_to_all` / `broadcast` / `reduce_scatter`），逐 op 记录
  stage / op / shape / dtype / tensor_bytes / world / step。
- **阶段锚**：各阶段模型入口 forward（DiT / VAE 的 `forward` / `decode_latent`）；
  ⚠️ **text encoder 的入口常常是 `encode_ids` / `encode_prompt` 而不是 `forward`**——锚错方法会把
  该阶段通信落进 other。
- **钩子时机**：`torch.distributed` 在 `torch` 包导入期间就被加载 ⇒ 独立 meta-finder 拦截不稳；
  可靠做法 = meta-finder 拦 **`torch` 根导入**，加载完成后强制 `import torch.distributed` 再 wrap；
  阶段锚可在首个 collective 内**懒安装**（届时框架模型模块已导入）。
- **输入语义坑**：同 prompt 时 text encoder 输出被缓存（只有首请求有通信）⇒ 要测 encoder 通信必须
  **每请求不同 prompt**。
- **字节口径**：`tensor_bytes` = op 的**本 rank 载荷**；`moved~` 为**估算**（a2a/broadcast
  ≈ ×(w−1)/w、all_reduce ≈ ×2(w−1)/w、all_gather ≈ ×(w−1)），**不是 HCCL 硬件计数器**；
  rank 对称时整簇移动 ≈ 2×per-rank。
- **二次验证**：与 `msprof` 的 HCCL 计数器对拍（`--hccl=on`；output 目录须先建、`--rule` 不可与
  `--export` 同用、导出后查 `hccl.db`）。
- 方法细节与实测明细见 `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §6。
- **只读插桩范式（`PYTHONPATH` + `sitecustomize`）**：需要**喂给内核的入参 / 调用参数**这类
  CANN 默认输出看不到的东西时，用 `PYTHONPATH` 指到一个目录、其 `sitecustomize.py` 在导入期
  **包装目标函数**（本仓形态：包住模型入口的 `forward`）⇒ **不改生产树**即可插桩，
  收工只需撤掉环境变量 ✓。与上面的 collective shim 法同源，区别是包装点不同（模型入口 vs 通信原语）。

## Profiler 配置

使用 `torch_npu.profiler` 采集 level=l1 数据：

```python
import torch_npu

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(input_data)
    torch_npu.synchronize()

prof.export_chrome_trace("trace_view.json")
```

CANN Profiler 环境要求：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Warmup 配置

Profiling 采集时必须在 profiler 外部完成 warmup，确保分析数据不含 JIT 编译开销：

- Profiler 打开前先完成 warmup（口径 **5 步**；含 `torch.npu.synchronize()`）。**warmup 步数由被测脚本/补丁控制，本入口不接管**：自家脚本按其自身默认预热（如 `examples/dummy_run/wan_infer.py`），三方框架补丁用 `WARMUP`/`H3_WARMUP_STEPS` 环境变量（模板默认 5）。
- Profiler 仅在 warmup 之后开启 **capture ≥5 步** timed steps（自家脚本保守口径；三方框架补丁
  只采 1 步，见下「少步快速采集经验」）
- MindieSDBackend 编译场景：warmup 步数需同时覆盖 JIT 编译（最多 8 次，建议 ≥10 步）
- 口径与实现的差异须知：`collect_profile.py` 只透传 `--device_id` / `--profile`（+ `--compile`）给远端脚本，**未透传 warmup 步数**——预热由脚本/补丁侧负责；若需改预热步数，改脚本或补丁的环境变量，而不是采集入口的参数。

> profiling-analyze 会验证 warmup 是否已剔除，未剔除时标注 `WARMUP_NOT_STRIPPED` 异常。

### 少步快速采集经验（1 步预热 + 1 步采集）

**少 step 是 profiling 的手段，不是优化目标**：单轮 profile run 只需代表单步形态，把预热与
采集窗口压到最少即可显著缩短单轮采集与迭代周期（如 60 步 → 2 步 ≈ 30× 提速），从而更快地做
多次采集对比（算子序、kernel diff、off/on AB）。这与 S4「时间步优化」（少步作为产品特性交付）
是两回事——采集用的少步**只改采集配置、采完还原**。

- **极简配置：1 步预热 + 1 步采集**（`H3_WARMUP_STEPS=1`，补丁只采第 2 步）——适用 eager /
  图已编译稳定的重复采集（同图多次 run、热身过的 compile），单步即代表算子形态与 kernel 序；
  kernel diff 用 1 步数据足够（算子级耗时/出现与否在单步即成立）。
- **compile / 首次 JIT 场景不适用 1 步预热**：JIT 编译发生在 profiler 外的前几步（最多 8 次），
  预热必须覆盖编译完成（≥10 步）后再采，否则采到的是编译步形态。eager 场景可视稳定度
  用 1 步预热即可（不稳时回到默认 5）。
- 采集步数不必多：**采样 1 步已够分析**（模板默认只采第 `WARMUP+1` 步 1 步）；
  SKILL 默认 "capture ≥5 步" 是自家脚本的保守口径，三方框架补丁按 1 步采样执行即可。

## 前置验证

采集 profiling 前，先快速验证模型推理正确性：

```python
# 跑 1 步推理，检查输出非空、shape 合法
output = pipe("test prompt", num_inference_steps=1)
assert output.images[0].size is not None, "Output shape is invalid"
print(f"Pre-check OK: output shape={output.images[0].size}")
```

验证通过后再开启 profiler 采集。验证失败时中止，排查推理问题（参考 framework-integration）。

## 输出产物

回传的 `profile_l1.tar.gz` 解压后包含标准 CANN Profiler 输出：

| 文件 | 格式 | 说明 |
| ------ | ------ | ------ |
| `kernel_details.csv` | CANN Profiler CSV | 每算子耗时 (Name, Duration, Wait Time) |
| `trace_view.json` | Chrome Trace JSON | Host + Device 事件时间线 |
| `step_trace_time.csv` | CANN Profiler CSV | Step 级汇总 |
| `communication.json` | JSON | 通信算子详情（若开启） |

> 此格式直接对接 profiling-analyze 的 5 层递进分析（Layer 3 内含三层子分析）。

### 导出类操作一律「无新文件即失败」（本仓实测，3 次采集 / 2 种注入布局复现）

**事实**：在**服务进程内**做设备 profile 采集时，"进程内文本导出"可能**从不执行，且失败被静默吞掉**：

| 环节 | 表面现象 | 实际 |
|---|---|---|
| 该次运行自身 | 正常结束 | **没有产出** `kernel_details.csv` |
| `msprof --export=on` | 打印 `[INFO] Export all data in PROF_… done.` | **只写 MindStudio 布局**，仍无该文件 |
| `torch_npu.profiler.profiler.analyse()`（**在采集进程内**） | 打印 `analyse() returned OK` | **什么都不写**（**在独立进程里对已落盘 raw dump 调则正常**，见文末「交付件必须含 trace view 与 kernel 明细」） |

**根因（已定位到库内源码）**：`analyse_profiling_data()` 被 `@no_exception_func()` 装饰 ⇒ **异常被吞**，
那个"成功"是假的（库内 `analysis/_profiling_parser.py`）。

**处置（按序）**：

1. **断言"新"**：产物必须**带新时间戳**（本次采集之后生成）**且行数 > 0**；
   **不以日志里的 `done` / `OK` / 退出码 0 为准** ✗ —— 这与宿主侧 `no NEW … -- skipped`
   是**同一类陷阱的两种表现**（判据单点见 `../../perf-gate/references/measurement-discipline.md` §10.1）；
2. **怀疑被吞掉的异常**：遇到"报成功却不出文件"，去找 `@no_exception_func` 这类装饰器 / 全局 except；
3. **绕过包装层**：直接驱动底层解析器（本仓已验证形态）——
   `ProfilingParser(profdir, Constant.TENSORBORAD_TRACE_HANDLER, None, {}).analyse_profiling_data()`
   ⇒ **数十秒量级内产出完整一套** ✓（比在包装层里"再试一次"有效得多）。

## 数据流向

```text
profiling-collect ──→ profiling-analyze ──→ dit-perf-opt
       │                        │                        │
   采集数据                 5 层递进分析           选取最优方案
```

上游数据消费者：`profiling-analyze/SKILL.md` 的 `## 数据源` 章节。

## Bundled Scripts

- `scripts/collect_profile.py` — SSH连接 → 执行 profiling → 压缩 → 下载（mindiesd 自家脚本入口）；
  **已接上产物门禁**：采集**之前**在本地输出目录落开始标记（`.start_epoch`，可用 `--start-marker` 改路径）
  → 下载后**自动解包** → 调 `check_output.py --dir <解包目录> --start-marker <标记>`，
  并**显式三态**落地：`0` 通过 / `1` 有检查判失败 / `3` 无法判定（**不得当成通过**，也不静默吞掉）；
  三态分别打印且**退出码透传**（`--no-check-output` / `--no-extract` 记为"未验证"= 3；
  `--skip-profiling` 时**不传标记**⇒新鲜度只能报"无法判定"并写明原因）。
  自带 `--selftest`（零 SSH：3 类夹具逐条核对 0/1/3 三态，并断言"无法判定"报告写明原因）
- `scripts/collect_patch_template.py` — 三方框架采集补丁模板（顶层推理方法包装 + torchrun 多卡）
- `scripts/check_output.py` — **产出完成检查门禁**（fail-closed，**六项检查各自独立报结论**）：
  产物存在 / 按列名断言执行序与单元利用率档齐备 / 判型列非全 `N/A`（抽样 200 行）/
  **数据表行数 > 0**（`kernel_details*.csv`、`op_summary*.csv`，**只有表头判失败**）/
  **产物新鲜度**（`--start-marker <运行开始标记>` 或 `--start-epoch <秒>`，**无标记报「无法判定」**）。
  退出码 **0 通过 / 1 判失败 / 2 前置缺失 / 3 无法判定（≠ 通过）**；`--json` 时 stdout **只**给一个
  JSON 文档（人类可读行走 stderr）。采集结束即跑
  `python scripts/check_output.py --dir {ASCEND_PROFILER_OUTPUT} --start-marker {run_dir}/.start_epoch`；
  自带 `--selftest`（10 类夹具逐条报结论并核对退出码，pre-commit 的 `check-output-selftest` 已挂钩）

> 为什么要有"新鲜度 + 行数"两项：**「命令返回 0 / 日志里有 `done`·`OK`」都不是产物存在或本次产出的证据**
> —— 本环境实测导出/解析的异常会被装饰器吞掉（见上文「导出类操作一律『无新文件即失败』」）。
>
> 部署使用 env-install/scripts/deploy_to_remote.py，空闲卡检测使用 remote-access/scripts/pick_free_device.py。

## Reference Files

- 🔧 `scripts/collect_profile.py` — 自家脚本入口：SSH 采集 → 回传
- 🔧 `scripts/collect_patch_template.py` — 三方框架入口：补丁模板（warmup/rank0/trace handler）
- 🔗 `../profiling-analyze/SKILL.md` — 下游分析（统一 ASCEND_PROFILER_OUTPUT 输入）
- 🔗 `../env-install/SKILL.md` / `../remote-access/SKILL.md` — 部署与 SSH 工具
- 🔗 `../framework-integration/SKILL.md` — 使能异常时排查推理问题
- 📁 `references/profile-dir-isolation.md` — 加载时机: **并行跑多份采集**（多个 `*_infer.py` 并行 worker）时——`--profile` 默认目录会互相覆盖，须按 模型×配置 隔离输出目录（`--profile-dir` / `DUMMY_PROFILE_DIR`）；自 `dummy-run` 下沉，**该口径的单点在本文件**
- 🔗 `../framework-integration/references/lightx2v-enablement.md` — 加载时机:
  LightX2V 接入/采集前，参考其开启方式、采集相关坑与档位方向
- 🔗 `../framework-integration/references/cache-dit-enablement.md` — 加载时机:
  cache-dit（框架本体）× vLLM-Omni 托管链采集前，确认自研算子部署顺序与单 rank 单 forward
  kernel 采集 hook 姿势
- 🔗 `../framework-integration/references/vllm-omni-enablement.md` — 加载时机:
  vLLM-Omni HTTP 服务采集前，确认 transformer forward 采集点与服务预热姿势（单 step 采集 hook 见 §5，
  证据口径与日志判据见 §1.3/§3.6）
- 🔗 `../framework-integration/references/diffsynth-engine-enablement.md` — 加载时机:
  DiffSynth-Engine 使能/性能验证时，参考其使能判断与方向结论（§3.4/§5）
- 📁 `references/e2e-split-and-capture-traps.md` — 加载时机: **要出"某特性的端到端收益拆分"
  （收益落在哪一段、为何 device 省了 e2e 没省），或采集结果要作为交付件交出去**时——三档交付件
  契约（带特性 / 不带特性 / 换精度）、离线导出判据与"臂脚本强杀导致导出从不执行"、
  端到端分段仪器与闭包对账、DiT 段占比量级、「采集单步 ≠ wall 均值」、
  残留占卡与大文件分片两条采集陷阱（流程单点在 remote-access §9/§10）

> **何时读 `e2e-split-and-capture-traps.md`**（路由判据）：任务要求"给出某特性的端到端收益/占比"
> ⇒ 先读它定**三档交付件**（带特性 / 不带特性 / 换精度）并逐档留交付清单；采集要**交给下游**时按它
> 逐档核对`trace_view.json` + `kernel_details.csv` 两件套、并确认导出是**外层显式**做的；
> 只要算子级形态普查（哪族算子占多少）不必读它。**收益拆分结论须带本组合作用域，换组合重测占比。**

## 交付件必须含 trace view 与 kernel 明细：文件名对照与最小代价采集

**交付要求（用户 2026-09-19 明确）**：profiling 交付件**必须**包含
①**trace view JSON**（时间线）与 ②**kernel 级明细表**（每个 kernel 一行，含 shape/时长/占比）。
只有 `op_statistic` 这类**聚合**表**不算**交付完成。单个 step 即可。

### 唯一可靠的产出命令：离线 `analyse()`（零卡耗 / 不碰 NPU，代价为 CPU 侧分钟以内量级）

`torch_npu` 自带离线导出入口，**对已存在的 raw dump 直接跑，不碰 NPU、不需要活会话、
不需要 MindStudio Insight（本机根本没装 `msinsight`）**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
RAW={run_results_dir}/{raw_dump}_ascend_pt   # 传 *_ascend_pt 目录
{venv}/bin/python -c \
  "from torch_npu.profiler.profiler import analyse; analyse('$RAW', max_process_number=8)"
ls -la "$RAW"/ASCEND_PROFILER_OUTPUT/    # 必须看到 trace_view.json + kernel_details.csv
```

一次产出**完整一套**（对两份不同 dump 各跑一次均成立，判据 = `rc=0`；绝对耗时出库
`{run_results_dir}/archive/`）：`trace_view.json`、`kernel_details.csv`、`step_trace_time.csv`、
`op_statistic.csv`、`api_statistic.csv`、`task_time.csv`、`communication.json`、
`ascend_pytorch_profiler_<dev>.db`（原始 db 的副本，喂 MindStudio Insight 用）。

> 同一路径的**失败形态**要分清：在**采集进程内**调 `analyse()` 会被 `@no_exception_func()`
> 吞掉异常、什么都不写（见上文「导出类操作一律『无新文件即失败』」）；在**独立进程**里对
> **已落盘的 raw dump** 调则正常工作。故判据永远是"目录里有没有带新时间戳的这两个文件"，
> 不是日志里的 `done`/`OK`/退出码。绕过包装层的备选形态：
> `ProfilingParser(profdir, Constant.TENSORBORAD_TRACE_HANDLER, None, {}).analyse_profiling_data()`。

### 文件名对照（按本机实际导出物写，不要凭空造名字）

**两套导出器给同一份数据起不同名字**（映射随导出器版本变，现场用一次导出核对；本栈实测版本与
对照表见归档 `{run_results_dir}/archive/`）：

| 交付件 | torch_npu 导出（`ASCEND_PROFILER_OUTPUT/`，**推荐**） | `msprof --export=on` 产出（`mindstudio_profiler_output/`） |
| --- | --- | --- |
| **时间线 trace view** | `trace_view.json`（Chrome trace-event 数组） | `msprof_<ts>.json`（README.txt 称之为 *Timeline report*） |
| **kernel 级明细** | `kernel_details.csv` | `op_summary_<ts>.csv` |
| 聚合统计 | `op_statistic.csv`（按 OP Type 聚合） | `op_statistic_<ts>.csv` |
| step 汇总 | `step_trace_time.csv`（Computing/Comm/Free/Stage） | 无（只有 torch_npu 导出产生） |
| 逐 task 起止 | `task_time.csv` | `task_time_<ts>.csv` |
| 通信 / api | `communication.json` / `api_statistic.csv` | `communication_statistic_*.csv` / `api_statistic_*.csv` |

- **`kernel_details.csv` 就是 `op_summary_*.csv` 换 6 个表头**（本机实测：48 列、42 列同名、
  首行数据逐字节相同）：`Op Name→Name`、`OP Type→Type`、`Task Type→Accelerator Core`、
  `Task Start Time(us)→Start Time(us)`、`Task Duration(us)→Duration(us)`、
  `Task Wait Time(us)→Wait Time(us)`（映射源：`torch_npu/profiler/analysis/prof_common_func/
  _csv_headers.py::CsvHeaders.OP_SUMMARY_SHOW_HEADERS → OP_SUMMARY_KERNEL_BASE_HEADERS`，
  由 `_kernel_view_parser.py::KernelViewParser` 施加）。其余 `aic*`/`aiv*`/`cube_utilization(%)`
  等利用率列**一字不改** ⇒ 利用率档不会因换名而丢失。
- 本栈实测版本下全树**搜不到** `kernel_details` 字样，`trace_view` 只命中 `*viewer*` 类名 ⇒
  这两个文件名属于 **torch_npu 导出器**，不是 msprof 的；`msprof` 也没有任何按名选择 trace view 的开关
  （`--reports` 只能开关 timeline 子层）。
- 本仓旧文件 `msprof_step_trace_full.json` + `msprof_step_trace_full_mindstudio_insight_data.db`
  与 `profiles/prof_dit8_v2/` 的 `trace_view.json` + `trace_view_mindstudio_insight_data.db`
  **是同一件产物的不同别名**（都是 torch_npu 导出器出的），不是另一条采集链路。
- 服务化采集常见"缺这两件套"的**根因**：arm/服务脚本结尾 `pkill -9`，torch_npu 的
  **退出期导出**（`tensorboard_trace_handler`）没机会跑 ⇒ 只剩 raw dump +
  `msprof --export=on` 的 MindStudio 布局。补法就是上面那条离线 `analyse()`。

**最小代价采集（目标：一个 step 的 trace + kernel 明细，别为采集烧一整轮 A/B）**：

1. **先复用已有 dump**：对**已存在的 raw dump** 跑离线 `analyse()`（或退一步
   `msprof --export=on --output=<PROF 目录>`）是**零卡耗**的（CPU 侧分钟以内量级）；
   够用就**不要**重采——两份 8 卡服务化 dump 的完整两件套就是这样白捡的；
2. **必须重采时限定到单 step**：本链由 `H3_PROF_CALL=<k>` 控制落盘（第 k 次
   `MiniMaxH3DiTModel.forward`），配 `H3_PROF_OUT` / `H3_PROF_LOCK`，经
   `PYTHONPATH=<注入目录>` 的 `sitecustomize` 送进 spawn 出来的 worker。**采集窗口天然就是
   一个 step（应与 `step_trace_time.csv` 的 `Stage` 相等——这是窗口正确性的现场核对），
   不需要 `msprof --duration/--delay`；`--pid` 动态附着在本容器不可用**；
3. **请求数压到刚好覆盖第 k 次调用**：`num_inference_steps=4` 时 `H3_PROF_CALL=5` 落在
   第 2 个请求的第 2 个去噪步 ⇒ **1 冷 + 1 采 = 2 个请求即可**（臂默认跑 4 个）。
   注意**启动（权重加载）才是大头**，砍请求数省下的墙钟与卡时占比很小
   （绝对量级出库 `{run_results_dir}/archive/`）；
4. 采集完**立刻核对两件套**：`ASCEND_PROFILER_OUTPUT/` 里有**带新时间戳**的
   `trace_view.json` + `kernel_details.csv`，且行数 > 0（用 `check_output.py`）。
   注意两条实测坑：①`check_output.py` 的执行序断言原先只认 `Task Type`，而
   torch_npu 的 `kernel_details.csv` 把它改名为 `Accelerator Core` ⇒ **合格产物会被误判失败**；
   `ORDER_TYPE` 必须同时接受 `Task Type` 与 `Accelerator Core`（`scripts/check_output.py`；
   漏掉后者会把合格产物误判失败；`--selftest` 覆盖该断言的夹具）。
   ②`--start-epoch` 有**时钟域**问题：容器时钟按 UTC 走、Windows 主机按本地时区显示，
   用**容器侧**时间戳当 epoch 会让"产物新鲜度"在本地误判陈旧 ⇒ 起始标记/epoch 一律取
   **持有产物那一侧**的时钟；
5. 大文件**分片拉取**（一个 step 的 `trace_view.json` 可达 140–148 MB），交付清单里写清
   **md5 + 大小 + 是哪个 step**（哪个 `H3_PROF_CALL`、哪个 rank、span 多少）。

## 维护与更新

当 CANN Profiler 接口变更、torch_npu.profiler API 升级或 profiling 输出格式变化时，
按 dev-workflow 的复盘流程更新本 skill。
