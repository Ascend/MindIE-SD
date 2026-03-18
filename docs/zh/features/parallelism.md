> 本章节主要介绍`多卡并行`相关加速特性
> - [张量并行](#张量并行)
> - [Ring Sequence Parallel](#ring-sequence-parallel)
> - [Ulysses Sequence Parallel](#ulysses-sequence-parallel)
> - [CFG并行](#cfg并行)


## 张量并行
随着模型规模的扩大，单卡显存容量无法满足大模型的需求。张量并行会将模型的张量计算（如矩阵乘法、卷积等）分散到多个设备上并行执行 ，从而降低单个设备的内存和计算负载。本章节以一次矩阵乘法为例，介绍张量并行的原理。
设输入数据为X，参数为W，X的维度 = (b, s, h)，W的维度 = (h, h')，一次矩阵乘法如下图所示。其中：
b：batch_size，表示批次大小
s：sequence_length，表示输入序列的长度
h：hidden_size，表示每个token向量的维度。
h'：参数W的hidden_size。

![](../figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-1.png)

优化方法：

- 按行切分：按照权重W的行切分，以N=2为例，将矩阵按照虚线切分

    ![](../figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-2.png)

    下图展示了切分后的结果，从一个矩阵乘转换为两个矩阵乘，分别在不同的NPU上运算，通过卡间通信将各个结果进行加法运算得到完整结果

    ![](../figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-3.png)

- 按列切分：按照权重W的列切分，以N=2为例，将矩阵按照虚线切分

    ![](../figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-4.png)

    下图展示了切分后的结果，从一个矩阵乘转换为两个矩阵乘，分别在不同的NPU上运算，通过卡间通信将各个结果进行拼接得到完整结果

    ![](../figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-5.png)

---
## Ring Sequence Parallel

将 Q 切分发给每个设备，计算时各设备计算完当前KV对后，将持有的KV对发送给下一设备，并接收前一设备的 KV对，形成一个环状的通信结构。当卡间通讯时间≤计算时间，通讯开销可被计算时间掩盖。

![](../figures/ring.png)

---
## Ulysses Sequence Parallel
把每个样本在序列维度上进行分割，分配给不同设备。在进行注意力计算之前，对分割后的 Q、K 和 V 进行 AlltoAll。各设备和其他所有设备交换信息，每个设备都能收到注意力头的非重叠子集。各设备并行计算不同注意力头，计算后再通过 AlltoAll 收集计算结果。

![](../figures/ulysses.png)

- 未使用并行的例子：

```python
import torch
import torch_npu
from mindiesd import attention_forward
torch.npu.set_device(0)
batch, seqlen, hiddensize = 1, 4096, 512
head = 8
x = torch.randn(batch, seqlen, hiddensize, dtype=torch.float16).npu()
x = x.reshape(batch, seqlen, head, -1)
out = attention_forward(x, x, x, opt_mode="manual", op_type="prompt_flash_attn", layout="BSND")
x = out.reshape(batch, seqlen, hiddensize)
```

- 使用ulysess并行

```python
import os
import torch
import torch.distributed as dist
import torch_npu
from mindiesd import attention_forward

batch, seqlen, hiddensize = 1, 4096, 512
head = 8
x = torch.randn(batch, seqlen, hiddensize, dtype=torch.float16).npu()

def init_distributed(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "hccl"
):
    dist.init_process_group(
        backend=backend,
        init_method=distributed_init_method,
        world_size=world_size,
        rank=rank,
    )
    torch.npu.set_device(f"npu:{os.environ['LOCAL_RANK']}")
# 1、初始化分布式环境
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["LOCAL_RANK"])
init_distributed(world_size, rank)

# 2、对seqlen维度按照world_size进行切分
x = torch.chunk(x, world_size, dim=1)[rank] # 序列切分
seqlen_chunk = x.shape[1]
x = x.reshape(batch, seqlen_chunk, head, -1)

# 3、调用all_to_all使能ulysess并行
in_list =  [t.contiguous() for t in torch.tensor_split(x, world_size, 2)]
output_list = [torch.empty_like(in_list[0]) for _ in range(world_size)]
dist.all_to_all(output_list, in_list)
x = torch.cat(output_list, dim=1).contiguous()
att_out = attention_forward(x, x, x, opt_mode="manual", op_type="prompt_flash_attn", layout="BSND")
in_list =  [t.contiguous() for t in torch.tensor_split(att_out, world_size, 1)]
output_list = [torch.empty_like(in_list[0]) for _ in range(world_size)]
dist.all_to_all(output_list, in_list)
x = torch.cat(output_list, dim=2).contiguous()
x = x.reshape(batch, seqlen_chunk, hiddensize)

# 4、对seqlen维度进行all_gather操作
x = dist.all_gather(x, dim=1)
```


---
## CFG并行

对于一个带噪声的图像和文本提示词，模型需要执行两次推理，分别计算正样本和负样本，该计算过程为串行过程，导致每个去噪步骤都需要两次前向传播，增加了推理时间。CFG并行可以将正样本和负样本分别在不同的设备上计算，将两次串行计算合并为一次并行计算，显著提升推理速度。

![](../figures/CFG%E5%B9%B6%E8%A1%8C.png)
