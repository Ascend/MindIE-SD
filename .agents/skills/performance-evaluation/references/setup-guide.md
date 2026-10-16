# msmodeling 安装和初始分析

## Step 1: 下载和安装 msmodeling

### 1.1 克隆仓库

```bash
# 下载 msmodeling 工具
git clone https://gitcode.com/Ascend/msmodeling.git
cd msmodeling
```

### 1.2 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 msmodeling 包
pip install -e .
```

### 1.3 验证安装

```bash
# 检查安装是否成功
python -c "import tensor_cast; print('msmodeling 安装成功')"
```

## Step 2: 分析 msmodeling 支持的硬件规格

### 2.1 获取支持的硬件列表

**执行命令**：

```bash
python -c "from tensor_cast.device import DeviceProfile; print(list(DeviceProfile.all_device_profiles.keys()))"
```

**预期输出示例**：

```text
['TEST_DEVICE', 'ATLAS_800_A2_376T_64G', 'ATLAS_800_A2_313T_64G', 'ATLAS_800_A3_752T_128G_DIE', ...]
```

**重要**：此列表是**实时获取**的，必须在每次评估前重新执行。

### 2.2 获取特定设备的详细规格

**执行命令**：

```python
python << 'EOF'
from tensor_cast.device import DeviceProfile

device_name = "ATLAS_800_A2_376T_64G"
device = DeviceProfile.all_device_profiles.get(device_name)

if device:
    print(f"设备名称: {device.name}")
    print(f"厂商: {device.vendor}")
    print(f"矩阵运算算力 (BF16): {device.mma_ops.get('bfloat16', 'N/A')} TFLOPS")
    print(f"显存容量: {device.memory_size_bytes / (1024**3)} GB")
    print(f"显存带宽: {device.memory_bandwidth_bytes_ps / (1024**3)} GB/s")
else:
    print(f"设备 {device_name} 不在支持列表中")
EOF
```

## Step 3: 确定目标设备

### 3.1 设备匹配流程

1. **获取用户目标设备**：询问用户要在哪个硬件上评估

   ```text
   请提供目标硬件设备名称（如 ATLAS_800_A2_376T_64G）：
   ```

2. **检查设备是否在支持列表中**：

   ```bash
   python -c "from tensor_cast.device import DeviceProfile; print('ATLAS_800_A2_376T_64G' in DeviceProfile.all_device_profiles)"
   ```

3. **如果在列表中**：直接使用该设备进行后续评估

4. **如果不在列表中**：进入未知硬件处理流程（见[硬件规格说明](hardware-specs.md)）

### 3.2 设备选择示例

```text
当前环境通过 msmodeling 检测到的可用硬件：
1. ATLAS_800_A2_376T_64G (华为昇腾A2, 376 TFLOPS BF16, 64GB HBM)
2. ATLAS_800_A3_752T_128G_DIE (华为昇腾A3, 752 TFLOPS BF16, 128GB HBM)
3. TEST_DEVICE (测试设备)

请选择要评估的硬件（输入编号或设备名）：
```

## Step 4: 记录设备信息

在配置文件中记录从 msmodeling 获取的设备信息：

```json
{
  "device_info_from_msmodeling": {
    "name": "NVIDIA_L20",
    "vendor": "NVIDIA",
    "mma_ops_bf32_tflops": 59.8,
    "mma_ops_bf16_tflops": 119.5,
    "gp_ops_bf32_tflops": 29.9,
    "gp_ops_bf16_tflops": 59.8,
    "memory_size_gb": 48,
    "memory_bandwidth_tbs": 0.864,
    "interconnect_bandwidt_gbs": 64,
    "queried_at": "2024-03-08T10:30:00"
  }
}
```

## 注意事项

1. **每次评估前必须重新执行**：硬件支持列表可能会随环境变化，必须在每次评估前重新执行查询命令。

2. **禁止使用缓存的硬件信息**：即使之前查询过，也必须重新执行命令获取最新信息。

3. **记录查询时间**：在配置中记录何时从 msmodeling 获取了设备信息。

4. **多设备环境**：如果环境中有多个可用设备，必须明确选择要评估的目标设备。
