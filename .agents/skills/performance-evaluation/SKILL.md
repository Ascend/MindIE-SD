---
name: performance-evaluation
description: 使用 msmodeling 评估模型推理性能，支持多硬件平台和多模态模型。
             无 NPU 时在 CPU 上模拟各类 NPU/GPU 性能，完成初步评估；
             有 NPU 时可通过 profiling-collection 采集真实数据，对接 performance-analysis 做深度分析。
             当用户需要模拟或实测模型推理性能、对比不同硬件平台、建立性能基线时使用此 skill。
             即使用户只提到"测一下这个模型的速度"而未说 msmodeling，也应触发。
             用户必须明确指定模型规格、分辨率、设备类型和量化方式。评估过程中必须记录完整日志，评估结果按规范路径保存。
---

# 性能评估

使用msmodeling工具进行深度学习模型性能评估，提供算子级性能分析和跨平台对比能力。

⚠️ **重要**：每次评估都必须**重新执行**msmodeling工具，获取实时数据。禁止使用任何缓存或历史数据。

## ⚠️ 执行前确认清单

**必须向用户确认的必需参数**（无默认值）：

| 参数 | 说明 | 示例 |
|------|------|------|
| **模型规格** | 模型具体规格版本 | `T2V-14B`, `7B` |
| **设备类型** | 目标硬件设备名称 | `ATLAS_800_A2_376T_64G` |
| **分辨率** | 输入图像/视频分辨率 | `480x832`, `512x512` |
| **量化方式** | 模型量化策略 | `DISABLED`, `W8A8_DYNAMIC` |
| **视频帧数** | 视频模型必须指定 | `81`, `121` |

**可选参数**（有默认值）：

- `--seq-len`: 64（文本长度）
- `--dtype`: bfloat16
- `--batch-size`: 1
- `--world-size`: 1（卡数）
- `--sample-step`: 28

### 重要规则

1. **禁止自动推断**：不得猜测用户未明确指定的参数
2. **必须确认**：缺少必需参数时**必须暂停询问用户**
3. **多卡需明确策略**：world-size>1时必须明确并行策略
4. **记录所有选择**：用户指定的参数必须记录在日志中
5. **⚠️ 每次评估必须重新执行msmodeling**：禁止使用缓存数据或历史结果，每次都必须重新运行工具进行实时评估

## 核心流程

### Step 1: 环境准备

msmodeling 下载、安装及硬件支持列表检测详见 references/setup-guide.md。
简要流程：`git clone` → `pip install -e .` → 检查 `DeviceProfile.all_device_profiles.keys()` → 确定目标设备。

⚠️ 每次评估前必须重新执行硬件检测，获取最新信息。

### Step 2: 确认模型规格

如果用户未明确模型规格，列出可用选项供选择。若模型未在 msmodeling 中支持，按照 msmodeling 使用方法进行适配。

### Step 3: 确认硬件和参数

获取用户明确指定的设备类型、分辨率、量化方式、视频帧数（视频模型）。

**未知硬件处理**：详见 references/hardware-specs.md —— 常见硬件从公开资料预填充参考值，
明确必需参数（矩阵BF16算力、显存容量、显存带宽、多卡互联带宽）。

### Step 4: 确认并行策略（多卡时）

如果 world-size > 1：

```text
检测到使用多卡配置（world-size=4）
推荐并行策略：
- Ulysses并行: 4 (ulysses-size=4)
- 适用于: Wan2.2-T2V-14B
- 通信模式: all-gather + all-reduce
是否接受此配置？[Y/N]
```

**推荐策略**：

- 多模态生成：ulysses-size = world-size
- 支持CFG模型：cfg-parallel + ulysses-size = world-size/2

### Step 5: 执行评估并记录

⚠️ **关键**：**必须实际执行**msmodeling工具进行实时评估，禁止使用任何缓存数据或历史结果。

**执行要求**：

1. 使用`python -m cli.inference.video_generate`或`python -m cli.inference.text_generate`实际执行
2. 等待工具完成推理并输出性能数据
3. 捕获实时输出并保存到日志文件
4. **禁止**使用之前的评估结果或缓存数据

> 有 NPU 环境且需要深度分析时：评估后通过 profiling-collection skill 在真实设备上采集 profiling 数据，
> 交给 performance-analysis 做三层递进分析。

**路径命名规范**：

```text
results/<model>_<spec>_<device>_w<N>_u<N>_cfg<N>/
```

**示例**：

- `wan2.2_t2v-14b_a2-376t-64g_d1/`（单卡）
- `wan2.2_t2v-14b_a2-376t-64g_d4_u4/`（4卡Ulysses）
- `wan2.2_t2v-14b_a2-376t-64g_d4_u2_cfg2/`（4卡+CFG）

**必须记录到日志**（从实时执行输出捕获）：

- 执行配置（模型、设备、分辨率、量化、并行策略等）
- 算子分析（FlashAttention, MatMul, Vector, Comm）
- 通信算子详情（all_gather, all_reduce, reduce_scatter）
- 内存使用
- 执行时间

### Step 6: 生成报告

生成 `evaluation_report.md`，包含：

- 测试配置汇总
- 性能指标表格
- 算子分析
- 通信分析（多卡）
- 关键发现和优化建议
- 附录：msmodeling的执行命令

## 结果目录结构

```text
results/
├── wan2.2_t2v-14b_a2-376t-64g_w1_u1_cfg0/
│   ├── config.json              # 配置信息
│   ├── iteration_1.log          # 执行日志
│   ├── iteration_2.log
│   └── summary.json             # 汇总数据
├── evaluation_report.md         # 评估报告
└── compare/                     # 比较报告（多硬件时）
    └── comparison_*.md
```

## 算子分析说明

评估输出包含以下算子类别：

| 算子 | 说明 |
|------|------|
| **FlashAttention** | 注意力机制运算 |
| **MatMul** | 矩阵乘法（主要计算瓶颈） |
| **Vector** | 元素级运算（激活函数等） |
| **Comm** | 通信/内存操作开销 |

**多卡场景**：Comm包含通信算子（all_gather, all_reduce, reduce_scatter）

## Reference Files

- 📦 `references/setup-guide.md` — 加载时机: 首次使用 msmodeling，下载安装工具并分析硬件支持列表时
- 📋 `references/evaluation-guide.md` — 加载时机: 需要完整评估流程细节、参数详解时
- 🔧 `references/parameters.md` — 加载时机: 需要所有参数详细说明和约束条件时
- 💻 `references/hardware-specs.md` — 加载时机: 遇到未知硬件需收集规格，或查询算力/带宽/互联参数时
- ⏱️ `references/benchmark-guide.md` — 加载时机: 设置计时方法、判断 Triton launch 开销时
- 📝 `references/examples.md` — 加载时机: 需要参考典型评估场景的完整命令行和配置示例时
- ✅ `references/best-practices.md` — 加载时机: 评估操作规范、多卡配置、结果记录规范参考时

## Bundled Scripts

- `scripts/validate_results.py` — 验证评估结果是否符合 skill 规范（检查 config.json 字段完整性和路径命名）

## 维护与更新

当 msmodeling 版本更新、新硬件加入支持列表、评估参数规范变更或 Benchmark 方法改进时，
按 dev-workflow 的复盘流程更新本 skill。
