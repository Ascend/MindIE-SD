# 能力矩阵

## 后端 × 硬件 支持矩阵

| 后端 | 910B | 910C | L20 | Profiler 工具 |
|------|------|------|-----|-------------|
| MindIE-SD compiled | ✓ | ✓ | — | msprof + trace.json |
| diffusers native | ✓ | ✓ | ✓ | torch.profiler |
| PyTorch eager | ✓ | ✓ | ✓ | torch_npu.profiler / torch.profiler |

## 三表接口要求

| 分析类型 | 最低要求 | 输入 |
|---------|---------|------|
| 单次推理三表（Single-trace） | 已有 profile 输出 | trace.json / kernel_details.csv |
| 编译 vs 原生对比（Two-trace） | 两次 profile 输出 | compiled trace + eager trace |

## 验证证据

以下为已验证据（确认分析流程可正常输出三表）：

| 模型 | 硬件 | 日期 | 结果 |
|------|------|------|------|
| Wan2.2-T2V-14B | 910B × 1 | 2026-05-09 | 已验证 (NC/C/torchair_ge/npugraph_ex 四模式, torchair_ge 消除 Copy) |
| FLUX.1-dev | 910B × 1 | 2026-05-09 | 已验证 (NC/C/torchair_ge/npugraph_ex 四模式, default(C) 最优) |
| FLUX.1-dev | L20 × 1 | 2024-03-08 | 已验证（见 evaluation_report.md） |

## 不可支持场景

- **未触发 MindieSDBackend 编译**的 trace：门控中止，先修复编译配置
- **diffusers 原生 fallback** 的 trace：标注为 baseline，不用于优化分析
