# Benchmark 规范

## 计时方法

测试耗时对比必须使用正确的 benchmark 方法：

- 必须 warmup（至少 5 次），首次调用包含 JIT 编译
- 必须 `torch.npu.synchronize()` 确保 NPU 操作完成
- 多次迭代（≥10 次），取后 5 次平均
- 断言 `compiled_time < original_time`（融合后应更快）

```python
def benchmark(func, args, warmup=5, repeat=10):
    for _ in range(warmup):
        func(*args)
        torch.npu.synchronize()
    times = []
    for _ in range(repeat):
        torch.npu.synchronize()
        start = time.perf_counter()
        func(*args)
        torch.npu.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return sum(times[-5:]) / 5
```

## MindIE-SD 编译预热

`MindieSDBackend()` 编译时首次推理包含 JIT 编译耗时（默认最多 8 次尝试），
后续 replay 跳过编译。Benchmark 时必须排除编译开销：

- 执行 ≥ 5 步 warmup，确认不再触发 recompile
- 前 N 步不计入计时，从第 N+1 步开始统计
- 多卡场景各 rank 独立 warmup
- 编译预热也触发 CANN Profiler JIT，warmup 步数需同时满足 Profiler 预热要求

```python
# 正确: warmup 不计时，timed 从稳定步开始
for step in range(total_steps):
    output = model(...)
    torch.npu.synchronize()
    if step >= warmup_steps:
        timed_outputs.append(output)

# 错误: warmup 未同步或计入计时
for step in range(total_steps):
    start = time.perf_counter()
    output = model(...)
    end = time.perf_counter()
    times.append(end - start)  # 包含编译耗时
```

## Triton kernel launch 开销

Triton kernel 对逐元素小张量（如 32×8192）存在显著的 launch 开销（~0.2ms），而 torch 原生操作（如 `x * scale + y`）由 NPU 编译器直接融合（~0.03ms）。此场景下 Triton 路径可能慢 5-6 倍：

- **规则**：kernel 层测试仅验证正确性（`atol`），不要求耗时验证
- **规则**：pattern 集成测试的 `assertLess(compiled_time, original_time)` 对 sub-ms 级操作不适用，应移除或用更大张量（如 128×8192）重测
- **判断标准**：当原始操作耗时 < 1ms 时，一律不要求耗时断言

## 多场景对照原则

每次优化后必须与 baseline 对照，**不得改变测试负载**后宣称优化有效：

- 同模型、同分辨率、同帧数、同精度、同 NPU 数
- 仅改变优化变量（如开启/关闭量化、开启/关闭稀疏）
- 差异 < 3% 视为噪声，不宣称有效
