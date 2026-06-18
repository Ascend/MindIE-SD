# Performance Analysis

## Profiling Data Collection

Use `torch_npu.profiler` to collect NPU operator-level performance data:

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

CANN environment requirements:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Warmup configuration notes:

- Execute **2 steps** of warmup (including `torch.npu.synchronize()`) before enabling the profiler to exclude JIT compilation overhead
- Only enable the profiler after warmup, collecting **5 or more** timed steps

Collected output files:

| File | Format | Description |
| ------ | ------ | ------ |
| `kernel_details.csv` | CANN Profiler CSV | Per-operator duration (Name, Duration, Wait Time) |
| `trace_view.json` | Chrome Trace JSON | Host + Device event timeline |
| `step_trace_time.csv` | CANN Profiler CSV | Step-level summary |
| `communication.json` | JSON | Communication operator details (multi-card scenarios) |

## Profiling Data Analysis

Analyze bottlenecks in collected profiling data using the following methods.

**Stage Separation**: Categorize operators by inference stage.

| Stage | Identifying Features | Typical Operators |
| ------ | --------- | --------- |
| DiT (Transformer) | attention_forward, MatMul, LayerNorm, RoPE | FlashAttention, Linear, RMSNorm |
| VAE | Conv2D, GroupNorm, Upsample | Conv2D, ResBlock |

**Operator Classification**: Summarize operators in each stage into four categories.

| Category | Included Operators |
| ------ | --------- |
| FA | FlashAttention, SDPA, attention_forward, fused_attn_score |
| MatMul | Linear, MatMul, GEMM, DequantGEMM |
| Vector | Activation (GELU/SiLU/ReLU), Norm (LayerNorm/RMSNorm), element-wise (Mul/Add/Div) |
| Comm | HCCL: all_gather, all_reduce, reduce_scatter, broadcast |

**Bottleneck Identification**: Locate optimization targets using the following dimensions.

| Finding | Threshold | Optimization Direction |
| ------ | ------ | ---------- |
| DiT MatMul proportion high | >50% | MatMul quantization |
| DiT FA proportion high | >30% | Attention optimization (quantization + sparsification) |
| DiT Vector proportion high | >20% | Compilation fusion |
| DiT Comm exposed | >30% | Communication masking optimization |

## Feature Performance Reference

Technical principles and usage of each acceleration feature are detailed in `docs/zh/features/`:

| Feature | Documentation |
| ------ | ------ |
| Sparse | [../features/sparse.md](../features/sparse.md) |
| Quantization | [../features/quantization.md](../features/quantization.md) |
| Compute-via-Storage | [../features/cache.md](../features/cache.md) |
| CPU Offload | [../features/cpu_offload.md](../features/cpu_offload.md) |
| Shared Memory | [../features/share_memory.md](../features/share_memory.md) |
| Multi-Card Parallelism | [../features/parallelism.md](../features/parallelism.md) |
| Compilation Acceleration | [../features/compilation.md](../features/compilation.md) |
| Core Acceleration API | [../features/core_layers.md](../features/core_layers.md) |
| Dynamic Expert Load Balancing | [../features/DyEPLB.md](../features/DyEPLB.md) |
