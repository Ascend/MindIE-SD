# 示例场景

> **目录** · 场景1: 单卡视频模型评估 · 场景2: 多卡视频模型评估 · 场景3: 多卡+CFG视频模型评估 · 场景4: 同一场景不同硬件比较 · 场景5: 不同量化方式对比 · 场景6: 不同序列长度性能测试 · 日志文件示例 · 性能对比 · 总结

## 场景1：单卡视频模型评估（Wan2.2 on A2）

**步骤1：确认模型规格（T2V-14B）**

```text
模型 Wan2.2 有多个规格可用：
1. Wan2.2-T2V-14B (14B参数，文本到视频)
2. Wan2.2-I2V-14B (14B参数，图像到视频)
3. Wan2.2-T2V-1.3B (1.3B参数，轻量版)

请选择具体规格（输入编号或名称）： 1
已选择: Wan2.2-T2V-14B
```

**步骤2：确认分辨率（480×832）和帧数（81）**

```text
检测到视频模型：Wan2.2-T2V-14B

视频模型必须明确frame-num（视频总帧数）：
常见选择：
- 81帧（约5秒@16fps）
- 121帧（约5秒@24fps）

请输入frame-num: [81]
```

**步骤3：确认量化方式（DISABLED）**

```text
请选择量化方式：
1. DISABLED (无量化，BF16)
2. W8A8_DYNAMIC (动态INT8)
3. FP8

请输入选项: 1
```

**步骤4：执行评估**

```bash
python -m cli.inference.video_generate \
    Wan-AI/Wan2.1-T2V-14B \
    --device ATLAS_800_A2_376T_64G \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action DISABLED \
    --world-size 1 \
    --ulysses-size 1
```

**步骤5：确认结果路径**

结果保存路径：`results/wan2.2_t2v-14b_a2-376t-64g_d1/`

---

## 场景2：多卡视频模型评估（Wan2.2 on A2，4卡）

**步骤1：确认并行策略（Ulysses=4）**

```text
检测到使用多卡配置（world-size=4）

推荐并行策略：
- Ulysses并行: 4 (ulysses-size=4)
- 适用于: Wan2.2-T2V-14B
- 通信模式: all-gather + all-reduce

是否接受此配置？
[Y] 接受推荐配置
[N] 自定义配置
```

**步骤2：确认是否使用CFG（否）**

```text
Wan2.2-T2V-14B 不支持CFG，使用标准Ulysses并行策略。
```

**步骤3：执行评估**

```bash
python -m cli.inference.video_generate \
    Wan-AI/Wan2.1-T2V-14B \
    --device ATLAS_800_A2_376T_64G \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action DISABLED \
    --world-size 4 \
    --ulysses-size 4
```

**步骤4：确认结果路径**

结果保存路径：`results/wan2.2_t2v-14b_a2-376t-64g_w4_u4_cfg0/`

---

## 场景3：多卡+CFG视频模型评估（4卡，支持CFG）

**步骤1：确认CFG并行策略（CFG=2, Ulysses=2）**

```text
检测到使用多卡配置（world-size=4）
模型支持CFG: 是

推荐并行策略：
- CFG并行: 启用
- Ulysses并行: 2 (ulysses-size=2)
- 通信模式: all-gather + all-reduce + broadcast

是否接受此配置？
[Y] 接受推荐配置
[N] 自定义配置
```

**步骤2：执行评估**

```bash
python -m cli.inference.video_generate \
    Wan-AI/Wan2.1-T2V-14B \
    --device ATLAS_800_A2_376T_64G \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action DISABLED \
    --world-size 4 \
    --cfg-parallel \
    --ulysses-size 2
```

**步骤3：确认结果路径**

结果保存路径：`results/wan2.2_t2v-14b_a2-376t-64g_w4_u2_cfg1/`

---

## 场景4：同一场景不同硬件比较

**步骤1：在A2上执行**

```bash
python -m cli.inference.video_generate \
    Wan-AI/Wan2.1-T2V-14B \
    --device ATLAS_800_A2_376T_64G \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action DISABLED \
    --world-size 1 \
    --ulysses-size 1
```

结果路径：`results/wan2.2_t2v-14b_a2-376t-64g_w1_u1_cfg0/`

**步骤2：在A3上执行（相同配置）**

```bash
python -m cli.inference.video_generate \
    Wan-AI/Wan2.1-T2V-14B \
    --device ATLAS_800_A3_752T_128G_DIE \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action DISABLED \
    --world-size 1 \
    --ulysses-size 1
```

结果路径：`results/wan2.2_t2v-14b_a3-752t-128g_w1_u1_cfg0/`

**步骤3：生成比较报告**

```bash
python scripts/generate_comparison.py \
    --scenario "wan2.2-480p-standard" \
    --baseline results/wan2.2_t2v-14b_a2-376t-64g_w1_u1_cfg0/ \
    --target results/wan2.2_t2v-14b_a3-752t-128g_w1_u1_cfg0/ \
    --output compare/comparison_wan2.2_480p_20240307.md
```

**步骤4：查看比较报告**

比较报告包含：

- 各硬件性能指标对比表
- 加速比分析
- 算子级差异分析
- 通信开销对比（多卡场景）
- 性价比分析

---

## 场景5：量化方式对比

**步骤1：FP16基准**

```bash
python -m cli.inference.video_generate \
    model \
    --device TEST_DEVICE \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action DISABLED
```

**步骤2：W8A8动态量化**

```bash
python -m cli.inference.video_generate \
    model \
    --device TEST_DEVICE \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action W8A8_DYNAMIC
```

**步骤3：W8A8静态量化**

```bash
python -m cli.inference.video_generate \
    model \
    --device TEST_DEVICE \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action W8A8_STATIC
```

**步骤4：FP8量化**

```bash
python -m cli.inference.video_generate \
    model \
    --device TEST_DEVICE \
    --batch-size 1 \
    --seq-len 64 \
    --height 480 \
    --width 832 \
    --frame-num 81 \
    --sample-step 28 \
    --dtype bfloat16 \
    --quantize-linear-action FP8
```

---

## 场景6：不同序列长度性能测试（LLM）

```bash
for len in 128 512 1024 2048; do
    python -m cli.inference.text_generate \
        meta-llama/Llama-2-7b-hf \
        --device TEST_DEVICE \
        --num-queries 1 \
        --query-length $len \
        --decode
done
```

---

## 日志文件示例

### config.json

```json
{
  "config_name": "wan2.2_t2v-14b_a2-376t-64g_w1_u1_cfg0",
  "model_name": "wan2.2",
  "model_spec": "t2v-14b",
  "model_params": "14B",
  "device": "ATLAS_800_A2_376T_64G",
  "device_spec": "a2-376t-64g",
  "model_path": "./models/Wan2.1-T2V-14B",
  "height": 480,
  "width": 832,
  "frame_num": 81,
  "sample_step": 28,
  "seq_len": 64,
  "dtype": "bfloat16",
  "quantization": "DISABLED",
  "world_size": 1,
  "ulysses_size": 1,
  "cfg_parallel": false,
  "parallel_strategy": "single-card",
  "iterations": 3,
  "timestamp": "2024-03-07T14:30:00",
  "user_specified": {
    "model_spec": "t2v-14b",
    "resolution": "480x832",
    "frame_num": 81,
    "device": "ATLAS_800_A2_376T_64G",
    "quantization": "DISABLED"
  },
  "default_used": {
    "seq_len": 64,
    "dtype": "bfloat16",
    "batch_size": 1,
    "world_size": 1,
    "ulysses_size": 1,
    "sample_step": 28
  }
}
```

### iteration_1.log

```text
Evaluation Configuration: wan2.2_t2v-14b_a2-376t-64g_w1_u1_cfg0
Model: wan2.2-t2v-14b
Device: ATLAS_800_A2_376T_64G
Device Spec: a2-376t-64g
Resolution: 480x832
Frames: 81
Sample Steps: 28
Sequence Length: 64
Dtype: bfloat16
Quantization: DISABLED
World Size: 1
Ulysses Size: 1
CFG Parallel: false
Parallel Strategy: single-card
Iterations: 3
Timestamp: 2024-03-07T14:30:00

Model compilation and execution time: 145.32s

+---------------+----------+----------+
| Op Type       | Time(ms) | Percent  |
+---------------+----------+----------+
| FlashAttention| 5234.5   | 36.0%    |
| MatMul        | 6128.3   | 42.2%    |
| Vector        | 2156.2   | 14.8%    |
| Comm          | 1013.2   | 7.0%     |
+---------------+----------+----------+

Communication Operators:
+-------------------+----------+----------+
| Op Name           | Time(ms) | Calls    |
+-------------------+----------+----------+
| all_gather        | N/A      | N/A      |
| all_reduce        | N/A      | N/A      |
| reduce_scatter    | N/A      | N/A      |
+-------------------+----------+----------+
Total communication time: 0ms (0.0%)
Note: Single-card mode, no communication operators

Peak memory: 38.5 GB
Memory bandwidth utilization: 62.3%

Per-step average time: 5.19s

Summary:
- Total execution time: 145.32s
- Average per-step time: 5.19s
- Peak memory: 38.5GB
- Operator breakdown:
  * FlashAttention: 5234.5ms (36.0%)
  * MatMul: 6128.3ms (42.2%)
  * Vector: 2156.2ms (14.8%)
  * Comm: 1013.2ms (7.0%)
- Communication breakdown:
  * all_gather: N/A (single-card)
  * all_reduce: N/A (single-card)
  * reduce_scatter: N/A (single-card)
```

### 比较报告示例

```markdown
# 硬件性能比较报告

**场景**: wan2.2-480p-standard
**比较硬件**: Atlas A2 vs Atlas A3
**时间**: 2024-03-07

## 性能对比

| 指标 | Atlas A2 | Atlas A3 | 加速比 |
|------|----------|----------|--------|
| 总推理时间 | 145.4s | 98.2s | **1.48×** |
| 单步时间 | 5.19s | 3.51s | **1.48×** |
| 峰值内存 | 38.5GB | 38.2GB | 0.99× |

## 算子级对比

| 算子 | A2耗时 | A3耗时 | 加速比 |
|------|--------|--------|--------|
| FlashAttention | 5249ms | 3542ms | **1.48×** |
| MatMul | 6138ms | 4123ms | **1.49×** |
| Comm | 1013ms | 892ms | 1.14× |

## 结论

Atlas A3相比A2在Wan2.2视频生成场景下性能提升约**48%**...
```
