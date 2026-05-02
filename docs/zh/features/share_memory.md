# 显存共享

- **核心问题**

  多实例场景下，在同一个NPU设备中，多个模型使用了相同的权重（如下图所示），可使用显存共享降低消耗。

    ![](../figures/%E6%98%BE%E5%AD%98%E5%85%B1%E4%BA%AB-image-1.png)
- **理论支撑**

  利用相同的NPU物理地址和偏移构建不同Tensor，可同时访问同一片内存。
- **设计思路**

  使用进程间共享的内存管理器管理内存，不同进程使用内存管理器分配的内存进行共享。
- **实现流程**

    ![](../figures/%E6%98%BE%E5%AD%98%E5%85%B1%E4%BA%AB-image-2.png)
  1. 进程0统计所需的内存大小offset，通过进程间共享的NPU Allocator申请内存。
  2. NPU Allocator将申请的物理内存地址data_ptr返回给进程0。
  3. 进程0将实际物理内存地址data_ptr通过进程间通信传给进程1。
  4. 进程0触发内存拷贝，将CPU的内存拷贝到实际NPU物理地址上。
  5. 进程0和进程1通过物理内存地址data_ptr和offset构建Tensor。

## 接口说明

```python
from mindiesd.share_memory import init_share_memory, share_memory
```

### init_share_memory

初始化进程间共享内存管理器。

```python
init_share_memory(instance_world_size, instance_id, master_addr="127.0.0.1", base_port=5555)
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `instance_world_size` | `int` | 是 | - | 总实例数 |
| `instance_id` | `int` | 是 | - | 当前实例 ID（0 为主实例） |
| `master_addr` | `str` | 否 | `"127.0.0.1"` | ZMQ 通信主地址 |
| `base_port` | `int` | 否 | `5555` | ZMQ 基础端口 |

### share_memory

将模型迁移到共享 NPU 显存。

```python
share_memory(module, device=None, dtype=None)
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `module` | `torch.nn.Module` | 是 | - | 待迁移的模型实例 |
| `device` | `str` / `torch.device` | 否 | `None` | 目标设备，如 `"npu:0"` |
| `dtype` | `torch.dtype` | 否 | `None` | 目标数据类型 |

### 使用示例

主实例（加载权重并共享）：

```python
from mindiesd.share_memory import init_share_memory, share_memory

init_share_memory(instance_world_size=2, instance_id=0)
model = ModelClass().to("npu")
model = share_memory(model, device="npu:0")
```

从实例（接收共享内存）：

```python
from mindiesd.share_memory import init_share_memory, share_memory

init_share_memory(instance_world_size=2, instance_id=1)
model = ModelClass()  # 不加载权重
model = share_memory(model, device="npu:0")  # 通过共享句柄构建 Tensor
```

主实例将权重所在 NPU 物理地址通过 ZMQ 广播给从实例，从实例通过同一物理地址构建 Tensor，实现多进程共享同一片显存。
