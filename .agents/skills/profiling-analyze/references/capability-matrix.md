# 能力矩阵

## 后端 × 采集通道 支持矩阵

判据按「该后端是否走 MindieSDBackend 编译」与「目标设备是否具备 msprof / torch_npu profiler 采集通道」判定，
**不按具体芯片型号判定**——型号与通道能力用 `npu-smi` + 试采确认，不把型号写进判据。

| 后端 | 昇腾 NPU（msprof 可用） | 仅 torch.profiler 通道 | Profiler 工具 |
|------|------|------|-------------|
| MindIE-SD compiled | ✓（MindieSDBackend 依赖昇腾运行时） | — | msprof + trace.json |
| diffusers native | ✓ | ✓ | torch.profiler |
| PyTorch eager | ✓ | ✓ | torch_npu.profiler / torch.profiler |

## 三表接口要求

| 分析类型 | 最低要求 | 输入 |
|---------|---------|------|
| 单次推理三表（Single-trace） | 已有 profile 输出 | trace.json / kernel_details.csv |
| 编译 vs 原生对比（Two-trace） | 两次 profile 输出 | compiled trace + eager trace |

## 验证证据

验证证据（具体型号 / 日期 / 模型清单 / 实测读数）**归档于会话产物**，不在本文件登记——换设备与换模型都会作废。
计入「已验证」的最低条件（三条同时满足）：

1. 分析流程能对**同一次采集**输出三表（trace.json / kernel_details.csv / step_trace_time.csv）；
2. 编译态与原生态**各一份** trace，且编译态确实触发 MindieSDBackend 编译（未触发见下节）；
3. 结论带口径（同窗对比、warmup 在 profiler 之外）与**归档坐标**（`{run_results_dir}`）。

## 不可支持场景

- **未触发 MindieSDBackend 编译**的 trace：门控中止，先修复编译配置
- **diffusers 原生 fallback** 的 trace：标注为 baseline，不用于优化分析

## 维护与更新

当支持矩阵的**判据**（后端 × 采集通道）或三表接口要求变化时，按 dev-workflow 的复盘流程更新本文件；
新增验证证据写进会话产物并只回填归档坐标。
