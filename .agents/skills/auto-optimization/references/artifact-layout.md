# 产物目录规范

每次 auto-optimization 闭环的产物目录结构：

```text
runs/YYYYMMDD_<model_slug>_optimization/
├── manifest.txt                    # 环境版本：CANN / PyTorch / torch_npu / MindIE-SD
├── step1_profile/
│   ├── profile_l1.tar.gz           # 原始 profiling 数据
│   └── collect_profile.log         # 采集日志
├── step2_analysis/
│   ├── profiling_report.md         # 5 层分析报告
│   ├── model_architecture_report.md
│   └── compare_traces.log          # 对比分析日志（Step 4）
├── step3_patches/
│   ├── patch_001_quantization.diff
│   ├── patch_002_compilation.diff
│   └── patch_003_sparse.diff
├── step4_validation/
│   ├── profile_l1.tar.gz           # 复验 profiling 数据
│   └── profiling_report.md         # 复验分析报告
├── final_report.md                 # 闭环报告
└── compare/                        # 对比产物
    ├── comparison_report.md
    └── kernel_diff.csv
```

## manifest.txt 格式

```text
device: ATLAS_800_A2_376T_64G
cann: 8.0.0
pytorch: 2.6.0
torch_npu: 2.6.0
mindiesd: <git_commit>
model: Wan2.2-T2V-14B
framework: Cache DiT + diffusers
precision: bfloat16
```

## final_report.md 模板

```markdown
# 优化闭环报告

## 环境
- 模型: {model}
- 硬件: {device}
- 框架: {framework}

## 基线
- DiT: xx ms, VAE: xx ms, 总计: xx ms
- 显存峰值: xx GB

## 优化方案（按实施顺序）
1. {P0}: {方案描述} → 改进: xx%
2. {P1}: {方案描述} → 改进: xx%

## 复验结果
- DiT: xx ms ({delta}%), VAE: xx ms ({delta}%), 总计: xx ms ({delta}%)
- 显存峰值: xx GB ({delta}%)

## 结论
{通过/部分通过/未通过}
- 目标达成 / 噪声范围 / 外部瓶颈 / 硬件瓶颈
```
