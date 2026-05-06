# CPU 卸载

## 通用原理

在 DiT 模型推理中，所有层（block）的权重需要常驻 NPU 显存。当模型规模超出单卡显存容量时，需要将部分层的权重暂时卸载到 CPU 内存，在计算到该层时再搬运回 NPU。这种技术称为 offload。

在同步 offload 模式中，GPU 计算完一层后需要停止，等待下一层的权重从 CPU 搬运到 NPU，加载完成后再继续计算。这会造成 GPU 在大部分时间处于空闲等待状态，利用率降低。

## 技术特点

本仓库采用**异步 Offload**方案来解决同步模式的效率问题。

其核心原理是：通过异步流水线设计，将计算和权重加载并行化。GPU 在计算第 N 层时，第 N+1 层的权重已经在后台搬运。当第 N 层计算完成时，第 N+1 层的权重也已加载就绪，计算耗时掩盖了搬运耗时，GPU 空闲时间显著减少。

下图展示了同步 offload 和异步 offload 的流程对比：

![](../figures/offload%E6%B5%81%E7%A8%8B-image.png) ![](../figures/%E5%BC%82%E6%AD%A5offload-image.png)

具体通过以下机制实现：

- **独立的拷贝流**：`h2d_stream`（Host 到 Device）和 `d2h_stream`（Device 到 Host）与计算流分离，实现拷贝与计算并行。
- **前向预 Hook**：在 block 执行前，异步加载后续 block 的权重到 NPU。
- **前向 Hook**：在 block 执行后，将已用完的权重从 NPU 卸载，释放显存。
- **预留 block 数**：通过 `min_reserved_blocks_count` 参数控制始终保留在 NPU 上的 block 数量，其余 block 动态换入换出。

## 接口说明

```python
from mindiesd.offload import enable_offload
```

### 函数签名

```python
enable_offload(model, blocks, min_reserved_blocks_count=2)
```

### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | `torch.nn.Module` | 是 | - | 需要启用 offload 的目标模型 |
| `blocks` | `ModuleList` | 是 | - | 模型中按顺序排列的 block 列表 |
| `min_reserved_blocks_count` | `int` | 否 | `2` | 始终保留在 NPU 上的 block 数量 |

### 返回值

`None`：原地修改，不返回任何值。

### 使用示例

```python
from mindiesd.offload import enable_offload

# 创建模型
model = DiTModel(...)

# 启用 offload，保留 2 个 block 在 NPU
enable_offload(model, model.blocks, min_reserved_blocks_count=2)

# 将模型移动到 NPU
model.to("npu")

# 正常执行推理，框架自动管理权重的异步换入换出
with torch.no_grad():
    output = model(x)
```

### 注意事项

- 与 [DyEPLB.md](DyEPLB.md) 同时使用时可能存在带宽争抢，需自行调整执行时机避免相互阻塞。
