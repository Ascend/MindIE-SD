# 三方框架性能分析方法论（MindIE-SD 适配实战经验）

> 来源：LightX2V MiniMax-H3 接入 mindiesd 全流程（2026-08，Ascend 950PR ×4，USP4）。
> 目标：把"算子级性能分析"从凭感觉变成可复现、可审计的流程。

## 0. 单次结果不迁移（最重要的原则）

- 本方法论来自 LightX2V MiniMax-H3 单次工程实践；**任何具体结论**（某融合有效/无效、
  某修复必要/不必要）**只在该框架 + 并行配置下成立**。
- 换框架（DiffSynth-Engine / vLLM-Omni / 其他）、换模型、换并行策略（TP/CP/USP 混合）
  → 必须重新跑完整试验协议（假设→最小验证→kernel 归因→墙钟确认→记录）。
- **可迁移的是方法，不是结论**：rank0 统计口径、kernel diff 流程、三层证据、
  试验协议，这些对所有三方框架通用。

## 1. 分析流程总览（5 步）

```text
正确性验证 → 墙钟采集 → kernel 采集 → 算子级 diff → 归因与决策
   (视频/输出)  (完整30步)  (CANN profiler)  (compare_traces)  (采纳/回退)
```

每步都产出可审计证据；任何优化动作**先验证正确性，再评估性能**。

## 2. 墙钟采集与统计口径（最重要，最容易错）

### 2.1 多卡日志的 rank 陷阱

LightX2V/多卡 torchrun 日志中，同一指标每个 rank 打印一条，**值各不相同**：

```text
Run DiT cost  Rank1=148.8s  Rank2=149.2s  Rank3=150.1s  Rank0=152.4s   (compile_full1)
Run DiT cost  Rank2=145.3s  Rank1=147.4s  Rank3=147.6s  Rank0=152.0s   (compile_full2)
Run DiT cost  Rank2=149.5s  Rank3=150.2s  Rank1=150.3s  Rank0=155.2s   (runtime)
```

- rank 间差异可达 **5-10s**（同步/打印时序）
- `head -1` 取最先打印的 rank，跨 run 对比时两边 rank 不一致 → **收益被夸大**
  （实例：真实 -2% 被误报成 -6.5%）

### 2.2 正确口径

| 指标 | 正确做法 |
|---|---|
| Run DiT total | **固定 rank0**（4 卡同步收尾 rank），或全部取 max；表格注明口径 |
| per-step sum | 取 rank0 的 29 步序列求和 |
| p50 / avg | p50 代表稳态（avg 被首步编译拉高）；首步排除后看 steady |
| 对比要求 | 同窗口 + 同卡组 + 2-3 次复现取中位数 |

### 2.3 参考实现（parse_steps.py 逻辑）

```python
# 按 rank 分组解析 "Run Dit every step cost X seconds"
steps.setdefault(rank, []).append(x)
# rank0 的 29 步: min / p50 / max / sum / avg
# "Run DiT cost" 取 rank0 行
```

## 3. Kernel 级采集（CANN profiler）

### 3.1 标准采集（性能稳定、可对比）

```python
handler = torch_npu.profiler.tensorboard_trace_handler(PROF_OUT)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    record_shapes=True, on_trace_ready=handler,
) as prof:
    <框架推理 1 步>; torch.npu.synchronize()
```

- warmup ≥5 步在 profiler 外（compile 场景 ≥10 覆盖 JIT）
- **只 rank0 采集**；产出 `ASCEND_PROFILER_OUTPUT/`（kernel_details.csv + trace_view.json + step_trace_time.csv）
- ⚠️ 采集脚本若用 `infer_steps=N` 小步数配置，必须保证 profiled step ≤ N（否则永远不触发）

### 3.2 解析工具

- `analyze_trace.py`：5 层递进（阶段分离/分类占比/Host Bound/通信/融合机会）
  - `--profile-dir` 传 ASCEND_PROFILER_OUTPUT 的**父目录**
  - 无 `__main__` 块，需 `sys.argv` 包装调用
  - 报告里的 NOTIFY_WAIT_SQE/DAVID_EVENT_WAIT 是同步事件非计算 kernel，占比需剔除
- `compare_traces.py`：两次 run 的 kernel diff（baseline vs target）
  - New/Removed/Common 三表 + 自动 verdict

## 4. 算子级归因（收益从哪来）

### 4.1 三层证据（可靠性递增）

1. **图命中**（pattern matcher dump `graph.print_readable()`）：pattern 匹配 ≠ 运行期生效
2. **kernel diff**：融合 kernel 实际执行次数/耗时
3. **墙钟 rank0**：最终收益

### 4.2 反例：kernel 改善 ≠ 墙钟收益

gate-msa 残差 2D 融合：kernel 级 -0.7%（新增 2D kernel 15ms、消除 mul+add 19.8ms），
但墙钟 p50 3.673s vs 3.564s（+0.1s）——2D kernel 的 `.contiguous()` 拷贝 + triton
启动开销抵消了融合收益。**决策必须以墙钟为准，kernel 级只作解释**。

### 4.3 归因框架

| 收益类别 | 识别方法 | 实例 |
|---|---|---|
| 算子融合 | 新增单算子替代分解链 | RmsNorm 替代 pow+mean+rsqrt（-74% kernel 时间） |
| 通信重叠 | 通信总耗时下降（非 kernel 减少） | a2a 留 eager → 通信 -35% |
| 数据移动 | Cast/InplaceCopy 数量下降 | Cast 650→250（-99%，被单算子吸收） |
| 长序列瓶颈 | 分类占比随序列变长 | 15s 通信占 57.6%（vs 5s 17%） |

## 5. 常见陷阱清单

| 陷阱 | 症状 | 对策 |
|---|---|---|
| 混 rank 统计 | 收益数字异常大/小 | 固定 rank0 |
| 首步编译拉高 avg | avg >> p50 | 用 p50 / 排除首步 |
| 图命中但未生效 | kernel diff 无融合 kernel | 看运行期 kernel 而非图 |
| 跨窗口对比 | 收益波动 > 实际 | 同窗口同卡组，多次复现 |
| profiler 步数不匹配 | 无 CANN 输出 | 确认 profiled step ≤ infer_steps |
| 其他容器占卡 | OOM / 性能漂移 | 跑前 npu-smi 确认空闲 |
| 同步事件计入 kernel | NOTIFY_WAIT 占比虚高 | 从 kernel 统计中剔除 |
| 热降频污染长跑 | 30 步 run 后段步长 4→7-8s（~14-17 步起，83-86°C，与代码无关） | 用 **clean-window（steps 2-14）avg/p50** 或 p50；同窗口同卡组 |
| kernel-sum 跨流多计数 | head-parallel kernel-sum 2.1× bulk 但墙钟更快 | **墙钟/clean-window 为准**，kernel-sum 只解释（重叠流会重复计数） |
| 卡组/端口环境劣化 | 全组 ~10× 慢 / HCCL 端口 bind 泄漏（见 parallelism-strategy `ascend-topology-bandwidth-diag.md`） | 换卡组；确认 init 前 `set_device`；避免 SIGKILL 进行中多卡任务 |

## 6. 决策规则

- **采纳**：rank0 墙钟下降 ≥1% 且可复现（正确性先过）
- **回退**：kernel 改善但墙钟无收益 / 墙钟上升（如 gate2、dynF、FA disable）
- **记录**：每个尝试（成功/失败/回退）写入优化日志，附实测数据与归因
- 融合/并行收益用**融合阶梯**呈现（kernel 数/总耗时逐级：分解链→单算子→compile 融合），
  与单次 diff 相比更能说明收益来源与剩余空间
