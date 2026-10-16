---
name: performance-optimization
description: 针对 performance-analysis 发现的性能瓶颈，从 mindiesd-features.md（唯一真相源）
             中选取最优 MindIE-SD 解决方案（量化/稀疏/并行/通信掩盖/缓存等）。
             5步闭环: baseline→分析→根因→修补→复验。
             即使用户只提到"这个模型怎么加速"而未说 benchmark，也应触发。
             当用户需要将分析结论转化为具体优化操作时使用此 skill。
             由 dev-workflow 的优化阶段触发。
---

# 性能优化

## 优化闭环

```text
建立基线 → 瓶颈分析 → 根因定位 → 保守修补 → 复验
   ↑                                              │
   └──────────────────────────────────────────────┘
```

### Step 1: 建立基线

使用 performance-evaluation 建立基线，记录模型 / 分辨率 / 帧数 / 精度 / NPU 数等配置。

### Step 2: 获取分析诊断

从 performance-analysis 的 5 层分析报告中获取：

- Layer 1: 瓶颈阶段（DiT vs VAE）
- Layer 2: 算子分类占比（FA/MatMul/Vector/Comm）
- Layer 3: Host Bound / 通信暴露 / 融合机会
- Layer 5: 优化方向（P0-P2 优先级 + 引用 features.md 章节）

分析报告给出的是**优化方向**（如"量化方向"、"通信掩盖方向"），具体方案在本 Step 选取。

### Step 3: 选取具体方案

基于分析报告的优化方向，查 mindiesd-features.md 确定具体 API 和参数：

```text
正例: "分析报告显示 MatMul 占 DiT 58%，优化方向→量化。
       查 mindiesd-features.md §MatMul量化，选取 W8A8_MXFP8"
反例: "感觉矩阵乘法比较慢，试试量化"
```

选择时需考虑：

- 硬件约束（features.md 中的硬件列）
- 模型兼容性（features.md 中的模型支持矩阵）
- 精度 vs 速度权衡
- 多方案时按优先级：MindIE-SD Pattern > 量化 > 稀疏 > 通信 > 通用

### Step 4: 实施 + 验证

优化方案从 mindiesd-features.md 中选取，详见 references/optimization-dimensions.md 的决策树。

| ✅ 允许 | ❌ 禁止 |
|---------|---------|
| 启用已有的、经验证的 kernel | 削弱输出正确性（cosine similarity 下降） |
| 修复遗漏的 fast path | 改变测试负载后宣称优化有效 |
| 减少不必要的同步/warmup | 仅为单框架/单硬件优化而破坏兼容性 |
| 添加有证据支撑的启发式配置 | 从单一 trace 数据得出普适结论 |

### Step 5: 复验

- 重新运行 performance-evaluation 的相同 benchmark
- 重新运行 performance-analysis 确认 5 层分析指标变化
- 差距 < 3% 视为噪声

## 优化维度

→ references/optimization-dimensions.md（决策树）
→ references/mindiesd-features.md（API/算法映射表，唯一真相源）

## 特性映射刷新规则

当性能优化过程中发现以下信号时，需检查 mindiesd-features.md 是否需要更新：

- 用户提到 MindIE-SD 新版本号（与 features.md 中记录的版本不一致）
- 建议的 API 在远端环境中不存在或签名不同
- 建议的量化/稀疏算法在目标硬件上不可用（与支持矩阵矛盾）

更新方式：

```bash
python scripts/refresh_features.py \
    --docs-dir <path/to/MindIE-SD/docs/zh/features> \
    --output references/mindiesd-features.md
```

## 停止条件

满足任一条件即停止优化循环：

1. **目标达成**: MindIE-SD compiled 在目标硬件上已满足性能预期
2. **噪声范围**: 与 baselines 差距 < 3%，继续优化无统计意义
3. **外部瓶颈**: 根因在 CANN / torch_npu / HCCL 而非 MindIE-SD 代码
4. **硬件瓶颈**: 已改善但受限于 NPU 物理显存 / 带宽上限

## Reference Files

- 📋 `references/optimization-dimensions.md` — 加载时机: 选择优化方向和决策逻辑时
- 🗺️ `references/mindiesd-features.md` — 加载时机: 确定具体 API/算法/硬件约束时（唯一真相源）

## Bundled Scripts

- `scripts/refresh_features.py` — 从 MindIE-SD docs 自动生成 mindiesd-features.md

## 维护与更新

当新的优化维度经验证有效、硬件平台升级或发现新的优化模式时，
按 dev-workflow 的复盘流程更新本 skill。
