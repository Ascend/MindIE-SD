---
name: auto-optimization
compatibility: 依赖 profiling-collection / performance-analysis / performance-optimization；NPU 设备
description: 端到端性能优化闭环，组合 profiling-collection、performance-analysis、performance-optimization
             三个下层 skill 完成从 profiling 采集到方案实施的完整流程。
             当用户需要完整优化模型推理性能、并确认优化效果时使用此 skill。
             前置条件：模型已通过 ascend-deploy 部署至 NPU 设备且已验证可通过。
             由 dev-workflow 的各阶段触发。
---

# 端到端性能优化闭环

## 依赖关系

```text
auto-optimization
    ↓
profiling-collection（采集 profiling 数据）
    ↓
performance-analysis（5 层递进分析 → 方向级建议）
    ↓
performance-optimization（查 features.md → 选取具体方案 → 实施）
    ↓
profiling-collection + performance-analysis（复验）
```

| 下层 skill | 在本闭环中的职责 |
|-----------|----------------|
| profiling-collection | Step 1: 采集 profiling 数据；Step 4: 复验时重新采集 |
| performance-analysis | Step 2: 5 层分析，输出方向级建议（P0-P2） |
| performance-optimization | Step 3: 基于分析建议，查 mindiesd-features.md 选具体 API/方案并实施 |

## 前置条件

- ascend-deploy 已完成：代码已编译安装、`import mindiesd` 成功
- 模型已部署并通过验证：用 framework-integration §1 确认推理正确
- NPU 设备可用且显存充足

## 核心流程

### Step 1: 采集 profiling 数据

调用 profiling-collection：

```bash
python profiling-collection/scripts/collect_profile.py \
    --host <IP> --user <用户名> --password <密码> \
    --container <容器名> --script wan_infer.py --device-id 0 \
    --warmup-steps 10
```

保障 warmup ≥10 步、profiler capture ≥5 步，采集完成后回传 `profile_l1.tar.gz` 到本地。

### Step 2: 分析 + 诊断

调用 performance-analysis：

```bash
python performance-analysis/scripts/analyze_trace.py \
    --profile-dir ./profile_l1 --output-dir ./analysis
```

关键产出：

- Layer 1: 瓶颈阶段（DiT vs VAE）
- Layer 2: 算子分类占比（FA/MatMul/Vector/Comm，分阶段）
- Layer 3: Host Bound / 通信暴露 / 融合机会（分阶段，含 similarity note）
- Layer 5: 方向级建议（P0-P2，每项指向 mindiesd-features.md §章节）

### Step 3: 选取方案 + 实施

基于 performance-analysis 输出的方向级建议：

1. 逐条加载 mindiesd-features.md 对应章节
2. 确认硬件约束、模型支持、精度需求
3. 选取具体 API/参数/开关
4. 实施优化（修改代码、配置开关、添加量化步骤）
5. 记录修改，准备复验

选取优先级：P0（MindIE-SD Pattern）→ P1（算子分类方向）→ P2（通用方案）

### Step 4: 复验

重新采集 + 分析，对比前后指标：

```bash
# 4.1 重新采集
python profiling-collection/scripts/collect_profile.py ...

# 4.2 重新分析
python performance-analysis/scripts/compare_traces.py \
    --baseline ./baseline/profile_l1/kernel_details.csv \
    --target ./validation/profile_l1/kernel_details.csv \
    --baseline-label "Before" \
    --target-label "After"
```

确认关键指标变化：

- DiT/VAE 阶段耗时变化
- FA/MatMul/Vector/Comm 占比变化
- Host Bound / 通信暴露率变化
- 总推理耗时变化

差距 < 3% 视为噪声，不宣称有效（阈值同 performance-optimization 单点维护）。

## 停止条件

同 `performance-optimization/SKILL.md` 的「停止条件」（目标达成 / 噪声范围 / 外部瓶颈 / 硬件瓶颈），
本 skill 只编排不重复定义。

## 产出规范

每次优化闭环的最终产物见 references/artifact-layout.md。
