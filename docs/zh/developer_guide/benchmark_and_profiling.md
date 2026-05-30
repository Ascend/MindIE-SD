# 性能分析

## Profiling 采集

使用 `torch_npu.profiler` 采集 NPU 算子级性能数据：

```python
import torch_npu

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=False,
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(input_data)
    torch_npu.synchronize()

prof.export_chrome_trace("trace_view.json")
```

CANN 环境要求：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Warmup 配置要点：

- Profiler 开启前先执行 **2 步** warmup（含 `torch.npu.synchronize()`），排除 JIT 编译开销
- Profiler 仅在 warmup 之后开启，采集 **5 步以上** timed steps

采集输出文件：

| 文件 | 格式 | 说明 |
|------|------|------|
| `kernel_details.csv` | CANN Profiler CSV | 每算子耗时 (Name, Duration, Wait Time) |
| `trace_view.json` | Chrome Trace JSON | Host + Device 事件时间线 |
| `step_trace_time.csv` | CANN Profiler CSV | Step 级汇总 |
| `communication.json` | JSON | 通信算子详情（多卡场景） |

## Profiling 数据分析

对采集到的 profiling 数据，按以下方法分析瓶颈。

**阶段分离**：将算子按推理阶段归类。

| 阶段 | 识别特征 | 典型算子 |
|------|---------|---------|
| DiT (Transformer) | attention_forward, MatMul, LayerNorm, RoPE | FlashAttention, Linear, RMSNorm |
| VAE | Conv2D, GroupNorm, Upsample | Conv2D, ResBlock |

**算子分类**：对每个阶段的算子按四类汇总。

| 分类 | 包含算子 |
|------|---------|
| FA | FlashAttention, SDPA, attention_forward, fused_attn_score |
| MatMul | Linear, MatMul, GEMM, DequantGEMM |
| Vector | 激活函数 (GELU/SiLU/ReLU), Norm (LayerNorm/RMSNorm), element-wise (Mul/Add/Div) |
| Comm | HCCL: all_gather, all_reduce, reduce_scatter, broadcast |

**瓶颈判定**：结合以下维度定位优化方向。

| 发现 | 阈值 | 优化方向 |
|------|------|----------|
| DiT MatMul 占比高 | >50% | MatMul 量化 |
| DiT FA 占比高 | >30% | Attention 优化（量化+稀疏） |
| DiT Vector 占比高 | >20% | 编译融合 |
| DiT Comm exposed | >30% | 通信掩盖优化 |

## 特性性能说明

各加速特性的技术原理和使用方式详见 `docs/zh/features/`：

| 特性 | 文档 |
|------|------|
| 稀疏 | [../features/sparse.md](../features/sparse.md) |
| 量化 | [../features/quantization.md](../features/quantization.md) |
| 以存代算 | [../features/cache.md](../features/cache.md) |
| CPU 卸载 | [../features/cpu_offload.md](../features/cpu_offload.md) |
| 显存共享 | [../features/share_memory.md](../features/share_memory.md) |
| 多卡并行 | [../features/parallelism.md](../features/parallelism.md) |
| 编译加速 | [../features/compilation.md](../features/compilation.md) |
| 核心加速 API | [../features/core_layers.md](../features/core_layers.md) |
| 动态专家负载均衡 | [../features/DyEPLB.md](../features/DyEPLB.md) |
