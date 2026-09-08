---
name: profiling-collect
compatibility: torch_npu.profiler, CANN set_env.sh, paramiko（collect_profile.py）, tar
description: profiling 数据统一采集：远端昇腾采集 NPU 性能数据回传本地，供 profiling-analyze 分析。
             两条入口——mindiesd 自家脚本入口（collect_profile.py）与三方框架入口
             （collect_patch_template.py 补丁 + torchrun，含 vLLM-Omni HTTP 服务等分场景）；
             warmup 在 profiler 外（默认 5 步，compile ≥10）。
             只要用户想"开 profiler/采 profile/拿算子级数据"，自家脚本或 vllm、LightX2V 等框架都先
             用本入口；分析/瓶颈定位/优化选型/并行/基准请求分别属 profiling-analyze、
             performance-optimization、parallelism-strategy、benchmark-dev，本入口不承接。
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
  > {model_dir}/prof.log 2>&1 < /dev/null &
```

回传：远端打包 `ASCEND_PROFILER_OUTPUT` 目录为 tar.gz → 本地解压后直接喂 profiling-analyze。

三方框架采集常见坑：

- 卡 Health 会波动（`UB LINK ERROR` 等瞬时错误）→ 重跑即可；对比须同卡组
- 容器内 `ss -tlnp` 可能查不到 HCCL 端口（走 NPU RoCE 网卡）→ 端口"看似空闲"仍冲突时直接加端口范围
- Windows 编辑的脚本上传报 `$'\r'` → 先转 LF
- 远端 `pgrep` 等待不可靠 → 每次只启动一个 run，用日志行数外部轮询，完成后再启下一个

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
trace_view.json + step_trace_time.csv）→ 打包回传后喂同一 profiling-analyze 管道
（analyze_trace.py / compare_traces.py），无需按框架分化。

各框架接入/使能上下文与实测案例参考 framework-feature-enablement 的 references：
`lightx2v-mindiesd-case.md`（LightX2V 接入完整案例，含采集方法）、`vllm-omni-case.md`
（vLLM-Omni HTTP 服务使能方法）、`diffsynth-engine-case.md`（DiffSynth-Engine 使能与
验证），见文末 Reference Files。

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

- Profiler 打开前先执行 **5 步 warmup**（`--warmup-steps` 默认 5，含 `torch.npu.synchronize()`）
- Profiler 仅在 warmup 之后开启 **capture ≥5 步** timed steps（自家脚本保守口径；三方框架补丁
  只采 1 步，见下「少步快速采集经验」）
- MindieSDBackend 编译场景：warmup 步数需同时覆盖 JIT 编译（最多 8 次，建议 ≥10 步）
- `--warmup-steps` 参数（默认 5）控制 warmup 步数

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

验证通过后再开启 profiler 采集。验证失败时中止，排查推理问题（参考 framework-feature-enablement）。

## 输出产物

回传的 `profile_l1.tar.gz` 解压后包含标准 CANN Profiler 输出：

| 文件 | 格式 | 说明 |
| ------ | ------ | ------ |
| `kernel_details.csv` | CANN Profiler CSV | 每算子耗时 (Name, Duration, Wait Time) |
| `trace_view.json` | Chrome Trace JSON | Host + Device 事件时间线 |
| `step_trace_time.csv` | CANN Profiler CSV | Step 级汇总 |
| `communication.json` | JSON | 通信算子详情（若开启） |

> 此格式直接对接 profiling-analyze 的 5 层递进分析（Layer 3 内含三层子分析）。

## 数据流向

```text
profiling-collect ──→ profiling-analyze ──→ performance-optimization
       │                        │                        │
   采集数据                 5 层递进分析           选取最优方案
```

上游数据消费者：`profiling-analyze/SKILL.md` 的 `## 数据源` 章节。

## Bundled Scripts

- `scripts/collect_profile.py` — SSH连接 → 执行 profiling → 压缩 → 下载（mindiesd 自家脚本入口）
- `scripts/collect_patch_template.py` — 三方框架采集补丁模板（顶层推理方法包装 + torchrun 多卡）

> 部署使用 env-install/scripts/deploy_to_remote.py，空闲卡检测使用 remote-access/scripts/pick_free_device.py。

## Reference Files

- 🔧 `scripts/collect_profile.py` — 自家脚本入口：SSH 采集 → 回传
- 🔧 `scripts/collect_patch_template.py` — 三方框架入口：补丁模板（warmup/rank0/trace handler）
- 🔗 `../profiling-analyze/SKILL.md` — 下游分析（统一 ASCEND_PROFILER_OUTPUT 输入）
- 🔗 `../env-install/SKILL.md` / `../remote-access/SKILL.md` — 部署与 SSH 工具
- 🔗 `../framework-feature-enablement/SKILL.md` — 使能异常时排查推理问题
- 🔗 `../framework-feature-enablement/references/lightx2v-mindiesd-case.md` — 加载时机:
  LightX2V 接入/采集前，参考其实测案例与采集方法
- 🔗 `../framework-feature-enablement/references/vllm-omni-case.md` — 加载时机:
  vLLM-Omni HTTP 服务采集前，确认 transformer forward 采集点与服务预热姿势
- 🔗 `../framework-feature-enablement/references/diffsynth-engine-case.md` — 加载时机:
  DiffSynth-Engine 使能/性能验证时，参考其实测结论

## 维护与更新

当 CANN Profiler 接口变更、torch_npu.profiler API 升级或 profiling 输出格式变化时，
按 dev-workflow 的复盘流程更新本 skill。
