# 硬件规格说明

> **目录** · 硬件规格收集流程 · 矩阵算力(BF16 MMA) · 向量算力(BF16 GP) · 显存容量 · 多卡互联带宽 · 硬件规格详细说明 · 常见设备配置示例 · 快速参考表

## 硬件规格收集流程

### 1. 已知硬件（支持列表中）

如果设备在支持列表中，直接使用：`--device <设备名>`

当前支持的设备：

- `ATLAS_800_A2_376T_64G` - 华为Atlas 800 A2 (376 TFLOPS BF16, 64GB HBM2e)
- `ATLAS_800_A2_313T_64G` - 华为Atlas 800 A2 (313 TFLOPS BF16, 64GB)
- `ATLAS_800_A2_280T_64G` - 华为Atlas 800 A2 (280 TFLOPS BF16, 64GB)
- `ATLAS_800_A3_752T_128G_DIE` - 华为Atlas 800 A3 (752 TFLOPS BF16, 128GB)
- `ATLAS_800_A3_560T_128G_DIE` - 华为Atlas 800 A3 (560 TFLOPS BF16, 128GB)

### 2. 未知硬件（不在支持列表中）

如果设备不在支持列表中，**必须**收集以下规格信息。

#### 硬件规格查询和预填充

**步骤1：尝试从公开资料获取**

对于常见硬件（如NVIDIA A100/H100, AMD MI300等），可以从官方规格表或技术文档获取参数，作为参考预填充：

```text
检测到未支持的硬件: NVIDIA_A100_40GB

正在查询公开资料获取参考规格...
✓ 找到参考信息（来源：NVIDIA官方规格表）

参考规格（请核对并修改）：
1. 设备厂商: NVIDIA
2. 矩阵运算BF16算力: 312 TFLOPS ← 参考值
3. 向量运算BF16算力: 19.5 TFLOPS ← 参考值（可选）
4. 显存容量: 40 GB ← 参考值
5. 显存带宽: 1555 GB/s ← 参考值
6. 内部互联带宽: 600 GB/s (NVLink) ← 参考值（卡数<8时需要）

请确认或修改以上数值：
```

**常见硬件参考值**（来源：官方技术规格）：

| 硬件 | 矩阵BF16 | 向量BF16 | 显存 | 带宽 | 互联带宽 |
|------|---------|---------|------|------|---------|
| NVIDIA A100 40GB | 312 TFLOPS | 19.5 TFLOPS | 40 GB | 1555 GB/s | 600 GB/s |
| NVIDIA A100 80GB | 312 TFLOPS | 19.5 TFLOPS | 80 GB | 2039 GB/s | 600 GB/s |
| NVIDIA H100 80GB | 989 TFLOPS | 67 TFLOPS | 80 GB | 3350 GB/s | 900 GB/s |
| AMD MI300X | 1300 TFLOPS | - | 192 GB | 5300 GB/s | 896 GB/s |

**步骤2：用户确认或修改**

如果无法从公开资料获取，或用户需要修改参考值：

```text
检测到未支持的硬件: [设备名]

请提供以下规格信息：

## 必需参数（矩阵运算）
1. 矩阵运算BF16算力（TFLOPS）: [必填]
   → 注：FP16算力默认与BF16相同
   → 注：INT8/FP8算力 = BF16算力 × 2（自动推断）

## 可选参数（向量运算）
2. 向量运算BF16算力（TFLOPS）: [可选，回车跳过]
   → 注：如不填写，FlashAttention性能评估可能不准确
   → 注：FP32算力 = BF16算力 ÷ 2（自动推断）

## 必需参数（内存）
3. 显存容量（GB）: [必填]
4. 显存带宽（GB/s）: [必填]

## 多卡必需参数（互联）
5. 内部互联带宽（GB/s）: [卡数<8时必填]
   → 示例：NVLink 600GB/s, HCCS 392GB/s

   机间互联带宽（GB/s）: [卡数≥8时必填]
   → 示例：InfiniBand 200Gbps (25GB/s)
```

## 硬件规格详细说明

### 矩阵运算算力（MMA Ops）

#### 必需参数

**BF16矩阵运算算力**（`mma_ops[torch.bfloat16]`）

- **必填**，单位：TFLOPS
- 表示硬件进行BF16矩阵乘法的峰值算力
- **推断规则**：
  - FP16算力默认与BF16相同（`mma_ops[torch.half] = mma_ops[torch.bfloat16]`）
  - INT8/FP8算力 = BF16算力 × 2（`mma_ops[torch.int8] = mma_ops[torch.bfloat16] * 2`）

```python
mma_ops = {
    torch.bfloat16: xxx * 1e12,  # BF16算力（用户提供，必填）
    torch.float16: xxx * 1e12,   # FP16算力（默认与BF16相同）
    torch.int8: xxx * 2e12,      # INT8算力（BF16的2倍，自动推断）
}
```

#### 可选参数

**FP32矩阵运算算力**（`mma_ops[torch.float32]`）

- 可选，如不填写使用默认值
- 通常约为BF16算力的1/2到1/4

### 向量运算算力（GP Ops）

#### 可选参数

**BF16向量运算算力**（`gp_ops[torch.bfloat16]`）

- **可选**，单位：TFLOPS
- 表示硬件进行BF16元素级运算（如激活函数）的峰值算力
- **影响**：如不填写，FlashAttention性能评估可能不够准确
- **推断规则**：
  - FP32向量算力 = BF16向量算力 ÷ 2

```python
gp_ops = {
    torch.bfloat16: xxx * 1e12,  # BF16向量算力（用户提供，可选）
    torch.float32: xxx * 0.5e12, # FP32向量算力（BF16的一半，自动推断）
}
```

**注意**：向量运算算力通常远小于矩阵运算算力（例如A100：矩阵312 TFLOPS vs 向量19.5 TFLOPS）

### 内存规格

#### 必需参数

**显存容量**（`memory_size_bytes`）

- **必填**，单位：GB
- 表示单卡可用的显存/HBM容量

**显存带宽**（`memory_bandwidth_bytes_ps`）

- **必填**，单位：GB/s
- 表示显存/HBM的峰值带宽
- 影响内存密集型算子（如FlashAttention）的性能

```python
memory_size_bytes = xxx * (1024**3)           # 显存容量（GB）
memory_bandwidth_bytes_ps = xxx * (1024**3)   # 显存带宽（GB/s）
```

### 互联带宽（多卡场景）

#### 多卡时必须提供

**内部互联带宽**（卡数 < 8时）

- **必填**（当world-size > 1且卡数<8时）
- 单位：GB/s
- 表示单机内多卡之间的互联带宽
- 常见技术：NVLink, HCCS, Infinity Fabric

**机间互联带宽**（卡数 ≥ 8时）

- **必填**（当world-size ≥ 8时）
- 单位：GB/s
- 表示跨机多卡之间的网络带宽
- 常见技术：InfiniBand, RoCE

```python
# 卡数<8时
internal_interconnect_bw = xxx * (1024**3)  # 内部互联带宽（GB/s）

# 卡数≥8时
cross_node_interconnect_bw = xxx * (1024**3)  # 机间互联带宽（GB/s）
```

#### 常见互联带宽参考

| 互联技术 | 带宽 | 适用场景 |
|---------|------|---------|
| NVLink 3.0 | 600 GB/s | NVIDIA GPU单机多卡 |
| NVLink 4.0 | 900 GB/s | NVIDIA H100单机多卡 |
| HCCS (华为) | 392 GB/s | 华为昇腾单机多卡 |
| InfiniBand HDR | 200 Gbps (25 GB/s) | 跨机互联 |
| InfiniBand NDR | 400 Gbps (50 GB/s) | 跨机互联 |
| RoCE v2 | 100 Gbps (12.5 GB/s) | 跨机互联 |

### 其他说明

参考`msmodeling\tensor_cast\device_profiles`路径下的README.md了解更多的配置信息要求。

## 完整设备配置示例

### 示例1：单卡配置（NVIDIA L20）

```python
"""NVIDIA L20 Device Profile."""

import torch

from ..device import DeviceProfile, CommGrid, InterconnectTopology, InterconnectType, StaticCost
from ..utils import DTYPE_FP8, DTYPE_FP4


# NVIDIA L20 Device Profile
# Specs: 48GB HBM3, ~60 TFLOPS FP32, ~239 TFLOPS FP16/BF16
L20_DEVICE = DeviceProfile(
    name="NVIDIA_L20",
    vendor="NVIDIA",
    mma_ops={                          # 此处代表矩阵计算的算力
        torch.float32: 59.8 * 1e12,    # FP32: BF16/2（自动推断）
        torch.bfloat16: 119.5 * 1e12,  # BF16: 312 TFLOPS（用户提供）
        torch.half: 119.5 * 1e12,      # FP16: 同BF16（自动推断）
        torch.int8: 239 * 1e12,        # INT8: BF16×2（自动推断）
        DTYPE_FP8: 239 * 1e12,         # FP8:  同INT8（自动推断）
    },
    gp_ops={                           # 此处代表向量计算的算力
        torch.float32: 29.9 * 1e12,    # FP32: BF16/2（自动推断）
        torch.bfloat16: 59.8 * 1e12,   # BF16: 59.8 TFLOPS（用户提供）
        torch.half: 59.8 * 1e12,       # FP16: 同BF16（自动推断）
    },
    memory_size_bytes=48 * (1024**3),               # 显存: 48GB（用户提供）
    memory_bandwidth_bytes_ps=0.864 * (1024**4),    # 带宽: 864 GB/s（用户提供）
    compute_efficiency=0.75,
    memory_efficiency=0.7,
    static_cost=StaticCost(
        mma_op_cost_s=5 * 1e-6,
        gp_op_cost_s=2 * 1e-6,
    ),
)
```

### 示例2：多卡配置（NVIDIA L20）

```python
"""NVIDIA L20 Device Profile."""

import torch

from ..device import DeviceProfile, CommGrid, InterconnectTopology, InterconnectType, StaticCost
from ..utils import DTYPE_FP8, DTYPE_FP4


# L20 uses PCIe interconnect for single node
L20_INTERCONNECT = CommGrid(
    grid=torch.arange(8).reshape(8),
    topologies={
        0: InterconnectTopology(       # 此处代表机内互联带宽，多卡配置时使用
            bandwidth_bytes_ps=64 * 1e9,  # PCIe Gen4 x16
            latency_s=0.2 * 1e-6,
            comm_efficiency=0.7,
        ),
    },
)

# NVIDIA L20 Device Profile
# Specs: 48GB HBM3, ~60 TFLOPS FP32, ~239 TFLOPS FP16/BF16
L20_DEVICE = DeviceProfile(
    name="NVIDIA_L20",
    vendor="NVIDIA",
    mma_ops={                          # 此处代表矩阵计算的算力
        torch.float32: 59.8 * 1e12,    # FP32: BF16/2（自动推断）
        torch.bfloat16: 119.5 * 1e12,  # BF16: 312 TFLOPS（用户提供）
        torch.half: 119.5 * 1e12,      # FP16: 同BF16（自动推断）
        torch.int8: 239 * 1e12,        # INT8: BF16×2（自动推断）
        DTYPE_FP8: 239 * 1e12,         # FP8:  同INT8（自动推断）
    },
    gp_ops={                           # 此处代表向量计算的算力
        torch.float32: 29.9 * 1e12,    # FP32: BF16/2（自动推断）
        torch.bfloat16: 59.8 * 1e12,   # BF16: 59.8 TFLOPS（用户提供）
        torch.half: 59.8 * 1e12,       # FP16: 同BF16（自动推断）
    },
    memory_size_bytes=48 * (1024**3),               # 显存: 48GB（用户提供）
    memory_bandwidth_bytes_ps=0.864 * (1024**4),    # 带宽: 864 GB/s（用户提供）
    compute_efficiency=0.75,
    memory_efficiency=0.7,
    comm_grid=L20_INTERCONNECT,        # 配置内部互联关系
    static_cost=StaticCost(
        mma_op_cost_s=5 * 1e-6,
        gp_op_cost_s=2 * 1e-6,
    ),
)
```

## 快速参考表

### 矩阵运算算力推断

| 用户输入 | 自动推断FP16 | 自动推断INT8/FP8 |
|---------|-------------|-----------------|
| BF16: 312 TFLOPS | FP16: 312 TFLOPS | INT8: 624 TFLOPS |
| BF16: 376 TFLOPS | FP16: 376 TFLOPS | INT8: 752 TFLOPS |
| BF16: 989 TFLOPS | FP16: 989 TFLOPS | INT8: 1978 TFLOPS |

### 向量运算算力推断

| 用户输入BF16 | 自动推断FP32 |
|-------------|-------------|
| BF16: 19.5 TFLOPS | FP32: 9.75 TFLOPS |
| BF16: 23.5 TFLOPS | FP32: 11.75 TFLOPS |
| BF16: 67 TFLOPS | FP32: 33.5 TFLOPS |

### 多卡互联带宽要求

| 卡数 | 必需参数 | 示例 |
|------|---------|------|
| 1 | 无 | - |
| 2-7 | 内部互联带宽 | NVLink 600 GB/s |
| ≥8 | 内部互联带宽 + 机间互联带宽 | NVLink 600 GB/s + IB 25 GB/s |
