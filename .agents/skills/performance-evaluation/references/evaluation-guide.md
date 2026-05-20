# 详细评估指南

> **目录** · 第一步: 下载msmodeling+分析硬件 · 第二步: 准备模型确认规格 · 第三步: 确认硬件和参数 · 第四步: 未知硬件处理 · 第五步: 执行评估 · 第六步: 日志记录和保存 · 第七步: 生成报告 · 第八步: 同一场景不同硬件比较

完整的性能评估流程指南。

## 使用流程

### 第一步：下载 msmodeling 并分析硬件支持

⚠️ **这是评估的第一步，必须在每次评估时执行**。

#### 1.1 下载和安装 msmodeling

```bash
# 下载 msmodeling 工具
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling

# 安装依赖
pip install -r requirements.txt

# 安装工具
pip install -e .
```

#### 1.2 验证安装

```bash
# 验证安装是否成功
python -c "import tensor_cast; print('msmodeling 安装成功')"
```

#### 1.3 通过 msmodeling 获取支持的硬件列表

**执行命令**：

```bash
python -c "from tensor_cast.device import DeviceProfile; print(list(DeviceProfile.all_device_profiles.keys()))"
```

**预期输出示例**：

```text
['TEST_DEVICE', 'ATLAS_800_A2_376T_64G', 'ATLAS_800_A2_313T_64G', 'ATLAS_800_A3_752T_128G_DIE']
```

**重要**：

- 此列表是**实时获取**的，必须在每次评估前重新执行
- **禁止使用缓存**的硬件列表
- 即使之前执行过，也必须重新执行此命令

#### 1.4 确定目标设备

询问用户要评估的硬件，并验证是否在 msmodeling 支持列表中：

```text
当前环境通过 msmodeling 检测到的可用硬件：
1. ATLAS_800_A2_376T_64G (华为昇腾A2, 376 TFLOPS BF16, 64GB HBM)
2. ATLAS_800_A3_752T_128G_DIE (华为昇腾A3, 752 TFLOPS BF16, 128GB HBM)
3. TEST_DEVICE (测试设备)

请选择要评估的硬件（输入编号或设备名）：
```

详细步骤参见[安装和初始分析](setup-guide.md)。

### 第二步：准备模型并确认规格

**步骤1：获取模型列表**

如果用户只提供了模型名称，需要查询可用规格：

```text
模型 Wan2.2 有多个规格可用：
1. Wan2.2-T2V-14B (14B参数，文本到视频，480p-720p)
2. Wan2.2-I2V-14B (14B参数，图像到视频，480p-720p)
3. Wan2.2-T2V-1.3B (1.3B参数，轻量版，480p)

请选择具体规格（输入编号或名称）： 1
已选择: Wan2.2-T2V-14B
```

**步骤2：下载模型**

```bash
# 使用Hugging Face模型ID
python -m cli.inference.text_generate meta-llama/Llama-2-7b-hf --device TEST_DEVICE --num-queries 1 --query-length 64

# 使用本地模型路径
python -m cli.inference.video_generate /path/to/local/model --device TEST_DEVICE --batch-size 1 --seq-len 64 --height 480 --width 832
```

### 第三步：确认硬件和参数

获取用户明确指定的评估参数：

- 设备类型（如 ATLAS_800_A2_376T_64G）
- 分辨率（如 480x832）
- 量化方式（如 DISABLED）
- 视频帧数（视频模型，如 81）

**设备验证流程**：

1. 询问用户目标硬件设备名称
2. 验证该设备是否在 Step 1 中获取的 msmodeling 支持列表中
3. 如果在列表中：直接使用该设备
4. 如果不在列表中：进入未知硬件处理流程

**未知硬件处理**：如果设备不在支持列表中，**必须暂停**收集硬件规格。详见[硬件规格说明](hardware-specs.md)。

### 第四步：配置硬件

**如果硬件已在支持列表中**：直接使用 `--device <设备名>`

**如果硬件不在支持列表中**：

⚠️ **必须按[硬件规格说明](hardware-specs.md)收集规格信息**。

**收集流程**：

1. **尝试预填充参考值**：对于常见硬件（NVIDIA A100/H100, AMD MI300等），从公开资料获取参考规格
2. **用户确认或修改**：显示参考值供用户核对
3. **明确必需参数**：
   - ✅ **矩阵运算BF16算力**（必填）→ FP16默认相同，INT8/FP8自动×2
   - ⚪ **向量运算BF16算力**（可选）→ 影响FA评估准确性
   - ✅ **显存容量**（必填）
   - ✅ **显存带宽**（必填）
   - ✅ **互联带宽**（多卡时必填）

**示例交互**：

```text
检测到未支持的硬件: NVIDIA_A100_40GB

正在查询公开资料获取参考规格...
✓ 找到参考信息（来源：NVIDIA官方规格表）

参考规格（请核对并修改）：
1. 矩阵运算BF16算力: 312 TFLOPS ← 必填
2. 向量运算BF16算力: 19.5 TFLOPS ← 可选（影响FA评估）
3. 显存容量: 40 GB ← 必填
4. 显存带宽: 1555 GB/s ← 必填
5. 内部互联带宽: 600 GB/s (NVLink) ← 多卡时必填

请确认或修改以上数值：
```

**创建自定义设备配置**：

```python
from tensor_cast.device import DeviceProfile, CommGrid, InterconnectTopology, StaticCost
import torch

MY_DEVICE = DeviceProfile(
    name="用户指定的设备名",
    vendor="用户指定的厂商",
    mma_ops={
        torch.bfloat16: xx.x * 1e12,   # BF16矩阵算力（用户提供，必填）
        torch.float16: xx.x * 1e12,    # FP16: 同BF16（默认）
        torch.int8: xx.x * 2e12,       # INT8: BF16×2（自动推断）
    },
    gp_ops={
        torch.bfloat16: xx.x * 1e12,   # BF16向量算力（用户提供，可选）
        torch.float32: xx.x * 0.5e12,  # FP32: BF16÷2（自动推断）
    },
    memory_size_bytes=xx * (1024**3),              # 显存容量（GB，必填）
    memory_bandwidth_bytes_ps=xxx * (1024**3),     # 显存带宽（GB/s，必填）
    internal_interconnect_bw=xxx * (1024**3),      # 内部互联带宽（卡数<8时必填）
    cross_node_interconnect_bw=xxx * (1024**3),    # 机间互联带宽（卡数≥8时必填）
    compute_efficiency=0.75,
    memory_efficiency=0.65,
    comm_grid=CommGrid(...),
    static_cost=StaticCost(),
)
```

### 第五步：执行评估

#### 评估前确认检查清单

执行前必须确认：

- [ ] Step 1 已完成：已下载并安装 msmodeling
- [ ] Step 1 已完成：已执行查询获取支持的硬件列表（实时）
- [ ] Step 1 已完成：已从 msmodeling 支持列表中确定目标设备
- [ ] 模型规格已明确指定（无默认值）
- [ ] 设备类型已明确指定（支持列表中或已收集规格）
- [ ] 未知硬件已收集规格：矩阵BF16算力、显存、带宽
- [ ] 多卡时已收集：互联带宽（内部/机间）
- [ ] 分辨率已明确指定（多模态模型，无默认值）
- [ ] 量化方式已明确指定（无默认值）
- [ ] 视频模型frame-num已明确指定（无默认值）
- [ ] 多卡时并行策略已明确（world-size > 1）
- [ ] seq_len使用默认值64或用户明确指定
- [ ] ⚠️ **已重新执行msmodeling获取实时数据**（禁止使用缓存）
- [ ] 最终配置参数已记录到日志

#### 文本生成评估 (LLM)

⚠️ **关键**：**必须实际执行**以下命令，获取实时评估数据。禁止使用缓存。

**基础命令**：

```bash
python -m cli.inference.text_generate \
    <model_id> \
    --device <device_name> \
    --num-queries 1 \
    --query-length 64 \
    [--context-length 0] \
    [--decode] \
    [--dtype bfloat16] \
    [--tp-size 1] \
    [--dp-size 1] \
    [--ep-size 1]
```

**注意**：执行后等待工具完成推理，捕获实时输出并保存到日志文件。

#### 视频生成评估 (Diffusion)

⚠️ **关键**：**必须实际执行**以下命令，获取实时评估数据。禁止使用缓存。

**基础命令**：

```bash
python -m cli.inference.video_generate \
    <model_path> \
    --device <device_name> \
    --batch-size 1 \
    --seq-len 64 \
    --height <必须指定> \
    --width <必须指定> \
    --frame-num <必须指定> \
    [--sample-step 28] \
    [--dtype bfloat16] \
    [--world-size 1] \
    [--ulysses-size 1] \
    [--cfg-parallel]
```

**注意**：执行后等待工具完成推理，捕获实时输出并保存到日志文件。

### 第六步：日志记录与结果保存

#### 路径命名规范（必须遵守）

**结果目录命名格式**：

```text
results/<model_name>_<model_spec>_<device_spec>_w<world_size>_u<ulysses_size>_cfg<cfg_flag>/
```

**各字段说明**：

- `model_name`: 模型名称（如wan2.2, llama）
- `model_spec`: 模型规格（如14b, 7b, t2v-14b）
- `device_spec`: 设备规格（从device名称提取，如a2-376t-64g）
- `world_size`: 卡数
- `ulysses_size`: Ulysses并行大小
- `cfg_flag`: 是否启用CFG（0或1）

**命名示例**：

```text
# 单卡示例
wan2.2_t2v-14b_a2-376t-64g_w1_u1_cfg0/

# 多卡示例（4卡，Ulysses并行）
wan2.2_t2v-14b_a2-376t-64g_w4_u4_cfg0/

# 多卡+CFG示例（4卡，CFG并行+Ulysses）
wan2.2_t2v-14b_a2-376t-64g_w4_u2_cfg1/
```

#### 日志记录规范

**每次评估必须记录以下内容到日志文件**：

1. **执行配置信息**（必须记录）

   ```text
   Evaluation Configuration: [配置名]
   Model: [模型名]-[规格]
   Device: [设备名]
   Device Spec: [设备规格]
   Resolution: [高度]x[宽度]
   Frames: [帧数] (视频模型)
   Sample Steps: [步数]
   Sequence Length: [seq_len]
   Dtype: [数据类型]
   Quantization: [量化方式]
   World Size: [卡数]
   Ulysses Size: [Ulysses并行大小]
   CFG Parallel: [是否CFG并行]
   Parallel Strategy: [并行策略描述]
   Iterations: [迭代次数]
   Timestamp: [ISO时间戳]
   ```

2. **算子分析表**（必须记录）

   ```text
   +---------------+----------+----------+
   | Op Type       | Time(ms) | Percent  |
   +---------------+----------+----------+
   | FlashAttention| 5234.5   | 36.0%    |
   | MatMul        | 6128.3   | 42.2%    |
   | Vector        | 2156.2   | 14.8%    |
   | Comm          | 1013.2   | 7.0%     |
   +---------------+----------+----------+
   ```

3. **通信算子详情**（多卡时必须记录）

   ```text
   Communication Operators:
   +-------------------+----------+----------+
   | Op Name           | Time(ms) | Calls    |
   +-------------------+----------+----------+
   | all_gather        | 523.4    | 28       |
   | all_reduce        | 312.8    | 28       |
   | reduce_scatter    | 177.0    | 14       |
   +-------------------+----------+----------+
   Total communication time: 1013.2ms (7.0%)
   ```

4. **最终结果汇总**（必须记录）

   ```text
   Summary:
   - Total execution time: [时间]s
   - Average per-step time: [时间]s
   - Peak memory: [内存]GB
   - Operator breakdown:
     * FlashAttention: [耗时]ms ([占比]%)
     * MatMul: [耗时]ms ([占比]%)
     * Vector: [耗时]ms ([占比]%)
     * Comm: [耗时]ms ([占比]%)
   - Communication breakdown:
     * all_gather: [耗时]ms ([占比]%)
     * all_reduce: [耗时]ms ([占比]%)
     * reduce_scatter: [耗时]ms ([占比]%)
   ```

### 第七步：分析结果

工具输出包含：

1. **性能汇总**：单次前向时间、总推理时间、吞吐量
2. **算子分析**：FA(Flash Attention)、MM(MatMul)、Vector、Comm各算子耗时和占比
3. **通信分析**（多卡时）：all_gather、all_reduce、reduce_scatter等通信算子详情
4. **内存分析**：显存使用情况、峰值内存
5. **带宽分析**：内存带宽利用率
6. **Chrome Trace**：可视化时间线（当使用 `--chrome-trace` 时）

### 第八步：同一场景不同硬件比较

**当需要在同一场景下比较不同硬件性能时**：

1. **在各自硬件上执行评估**（按上述流程）
2. **收集各硬件的性能数据**
3. **生成比较报告**：

```bash
# 生成比较报告
python scripts/generate_comparison.py \
    --results results/wan2.2_t2v-14b_a2-376t-64g_w1_u1_cfg0/ \
    --results results/wan2.2_t2v-14b_a3-752t-128g_w1_u1_cfg0/ \
    --output compare/comparison_wan2.2_480p_$(date +%Y%m%d_%H%M%S).md
```

**比较报告内容**：

- 各硬件性能指标对比表
- 加速比分析
- 算子级差异分析
- 通信开销对比（多卡场景）
- 性价比分析

**比较报告保存位置**：`compare/comparison_<scenario>_<timestamp>.md`
