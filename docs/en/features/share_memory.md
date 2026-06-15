# Shared Memory

- **Core Problem**

  In multi-instance scenarios, multiple models on the same NPU device share the same weights (as shown below). Shared memory can reduce memory consumption.

    ![](../../figures/memory_share_image_1.png)
- **Theoretical Basis**

  Different tensors constructed with the same NPU physical address and offset can access the same memory region simultaneously.
- **Design Approach**

  Use an inter-process shared memory manager to manage memory; different processes use memory allocated by the manager for sharing.
- **Implementation Flow**

    ![](../../figures/memory_share_image_2.png)
  1. Process 0 calculates the required memory size offset and allocates memory through the inter-process shared NPU Allocator.
  2. The NPU Allocator returns the allocated physical memory address data_ptr to Process 0.
  3. Process 0 transmits the actual physical memory address data_ptr to Process 1 via inter-process communication.
  4. Process 0 triggers memory copy, copying CPU memory to the actual NPU physical address.
  5. Process 0 and Process 1 construct tensors using the physical memory address data_ptr and offset.

## API Reference

```python
from mindiesd.share_memory import init_share_memory, share_memory
```

### init_share_memory

Initialize the inter-process shared memory manager.

```python
init_share_memory(instance_world_size, instance_id, master_addr="127.0.0.1", base_port=5555)
```

| Parameter | Type | Required | Default | Description |
| ------ | ------ | ------ | -------- | ------ |
| `instance_world_size` | `int` | Yes | - | Total number of instances |
| `instance_id` | `int` | Yes | - | Current instance ID (0 is the primary instance) |
| `master_addr` | `str` | No | `"127.0.0.1"` | ZMQ communication master address |
| `base_port` | `int` | No | `5555` | ZMQ base port |

### share_memory

Move a model to shared NPU memory.

```python
share_memory(module, device=None, dtype=None)
```

| Parameter | Type | Required | Default | Description |
| ------ | ------ | ------ | -------- | ------ |
| `module` | `torch.nn.Module` | Yes | - | Model instance to be moved |
| `device` | `str` / `torch.device` | No | `None` | Target device, e.g. `"npu:0"` |
| `dtype` | `torch.dtype` | No | `None` | Target data type |

### Usage Example

Primary instance (loads weights and shares):

```python
from mindiesd.share_memory import init_share_memory, share_memory

init_share_memory(instance_world_size=2, instance_id=0)
model = ModelClass().to("npu")
model = share_memory(model, device="npu:0")
```

Secondary instance (receives shared memory):

```python
from mindiesd.share_memory import init_share_memory, share_memory

init_share_memory(instance_world_size=2, instance_id=1)
model = ModelClass()  # No weight loading
model = share_memory(model, device="npu:0")  # Build tensors via shared handles
```

The primary instance broadcasts the NPU physical address of weights to secondary instances via ZMQ. Secondary instances construct tensors using the same physical address, enabling multiple processes to share the same memory region.
