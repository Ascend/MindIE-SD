# Multi-Card Parallel Acceleration Features

## Tensor Parallel

As models grow larger, the memory capacity of a single card becomes insufficient. Tensor parallelism distributes tensor operations such as matrix multiplication and convolution across multiple devices so that each device carries only part of the memory and compute load. This section uses a matrix multiplication example to explain the principle.

Assume the input is `X`, the parameter is `W`, `X` has shape `(b, s, h)`, and `W` has shape `(h, h')`, as shown below:

- `b`: batch size
- `s`: sequence length
- `h`: hidden size of each token vector
- `h'`: hidden size of parameter `W`

![](../../zh/figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-1.png)

Two common strategies are used:

- Row-wise split: split the weight matrix `W` by rows. Using `N=2` as an example:

  ![](../../zh/figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-2.png)

  The original matrix multiplication becomes two matrix multiplications executed on different NPUs. The partial results are then reduced by inter-device communication to obtain the full result.

  ![](../../zh/figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-3.png)

- Column-wise split: split the weight matrix `W` by columns. Using `N=2` as an example:

  ![](../../zh/figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-4.png)

  The original matrix multiplication becomes two matrix multiplications executed on different NPUs. The partial outputs are concatenated through inter-device communication to form the full result.

  ![](../../zh/figures/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C-image-5.png)

---

## Ring Sequence Parallel

In Ring Sequence Parallel, the query tensor `Q` is partitioned across devices. After each device finishes computing with its current `K` and `V` pair, it sends the local `K` and `V` pair to the next device and receives the previous device’s `K` and `V` pair, forming a ring communication topology. When inter-device communication time is less than or equal to compute time, the communication overhead can be hidden behind computation.

![](../../zh/figures/ring.png)

---

## Ulysses Sequence Parallel

Each sample is split along the sequence dimension and distributed across devices. Before attention is computed, the partitioned `Q`, `K`, and `V` tensors are exchanged by `AlltoAll`. Each device receives a non-overlapping subset of attention heads from every other device, computes attention in parallel, and then uses `AlltoAll` again to gather the results.

![](../../zh/figures/ulysses.png)

- Example without Ulysses Sequence Parallel:

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

- Example with Ulysses Sequence Parallel:

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
  # 1. initialize the distributed environment
  world_size = int(os.environ["WORLD_SIZE"])
  rank = int(os.environ["LOCAL_RANK"])
  init_distributed(world_size, rank)

  # 2. split the sequence dimension by world_size
  x = torch.chunk(x, world_size, dim=1)[rank]
  seqlen_chunk = x.shape[1]
  x = x.reshape(batch, seqlen_chunk, head, -1)

  # 3. enable Ulysses through all_to_all
  in_list = [t.contiguous() for t in torch.tensor_split(x, world_size, 2)]
  output_list = [torch.empty_like(in_list[0]) for _ in range(world_size)]
  dist.all_to_all(output_list, in_list)
  x = torch.cat(output_list, dim=1).contiguous()
  att_out = attention_forward(x, x, x, opt_mode="manual", op_type="prompt_flash_attn", layout="BSND")
  in_list = [t.contiguous() for t in torch.tensor_split(att_out, world_size, 1)]
  output_list = [torch.empty_like(in_list[0]) for _ in range(world_size)]
  dist.all_to_all(output_list, in_list)
  x = torch.cat(output_list, dim=2).contiguous()
  x = x.reshape(batch, seqlen_chunk, hiddensize)

  # 4. all_gather across the sequence dimension
  x = dist.all_gather(x, dim=1)
  ```

---

## CFG Parallel

For a noisy image and a text prompt, the model traditionally runs two serial inference passes to compute the positive and negative branches. That means every denoising step requires two forward passes, increasing latency. CFG Parallel dispatches the positive and negative branches to different devices and merges the two serial passes into one parallel execution, significantly improving inference speed.

![](../../zh/figures/CFG%E5%B9%B6%E8%A1%8C.png)
