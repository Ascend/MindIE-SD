---
name: profiling-collection
description: 在远端昇腾 NPU 设备上采集性能 profiling 数据，打包回传本地供 performance-analysis 使用。
             统一采集流程：开启 Profiler → 运行推理 → 压缩 → 回传（部署由 ascend-deploy 完成）。
             当用户需要采集模型 profiling 数据、开启 profiler 跑推理、或为性能分析准备数据时使用此 skill。
             即使用户只提到"帮我采一下 profile"或"开 profiler 跑这个模型"，也应触发。
             通常由 model-verification、ascend-deploy 或 performance-evaluation 的 NPU 路径路由触发。
---

# Profiling 数据采集

在远端昇腾设备上采集性能 profiling 数据，为 performance-analysis 提供标准化输入。

## 核心流程

```text
部署代码（ascend-deploy） → 开启 Profiler → 运行推理 → 压缩 → 回传本地
```

> 部署由 ascend-deploy/scripts/deploy_to_remote.py 完成。本 skill 仅负责 profiling 采集。

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

- Profiler 打开前先执行 **2 步 warmup**（含 `torch.npu.synchronize()`）
- Profiler 仅在 warmup 之后开启 **capture ≥5 步** timed steps
- MindieSDBackend 编译场景：warmup 步数需同时覆盖 JIT 编译（最多 8 次，建议 2 步以上）
- `--warmup-steps` 参数（默认 2）控制 warmup 步数

> performance-analysis 会验证 warmup 是否已剔除，未剔除时标注 `WARMUP_NOT_STRIPPED` 异常。

## 前置验证

采集 profiling 前，先快速验证模型推理正确性：

```python
# 跑 1 步推理，检查输出非空、shape 合法
output = pipe("test prompt", num_inference_steps=1)
assert output.images[0].size is not None, "Output shape is invalid"
print(f"Pre-check OK: output shape={output.images[0].size}")
```

验证通过后再开启 profiler 采集。验证失败时中止，排查推理问题（参考 model-verification §B）。

## 输出产物

回传的 `profile_l1.tar.gz` 解压后包含标准 CANN Profiler 输出：

| 文件 | 格式 | 说明 |
|------|------|------|
| `kernel_details.csv` | CANN Profiler CSV | 每算子耗时 (Name, Duration, Wait Time) |
| `trace_view.json` | Chrome Trace JSON | Host + Device 事件时间线 |
| `step_trace_time.csv` | CANN Profiler CSV | Step 级汇总 |
| `communication.json` | JSON | 通信算子详情（若开启） |

> 此格式直接对接 performance-analysis 的三层递进分析。

## 数据流向

```text
profiling-collection ──→ performance-analysis ──→ performance-optimization
       │                        │                        │
   采集数据                 三层递进分析             选取最优方案
```

上游数据消费者：`performance-analysis/SKILL.md` 的 `## 数据源` 章节。

## Bundled Scripts

- `scripts/collect_profile.py` — SSH连接 → 执行 profiling → 压缩 → 下载（纯采集）

> 部署使用 ascend-deploy/scripts/deploy_to_remote.py，空闲卡检测使用 ascend-deploy/scripts/pick_free_device.py。

## 维护与更新

当 CANN Profiler 接口变更、torch_npu.profiler API 升级或 profiling 输出格式变化时，
按 dev-workflow 的复盘流程更新本 skill。
