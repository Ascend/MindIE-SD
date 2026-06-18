# FA_Power_Cap Technology

As multimodal video generation moves toward longer durations and higher resolutions, sequence lengths can grow to the million-token scale or beyond. As sequence length increases, Flash Attention (FA) takes a larger share of runtime, and AI processor power consumption also rises. If the power required by long-sequence workloads exceeds the PCIe card TDP (Thermal Design Power) limit, the PMC (Power Management Controller) power protection mechanism may trigger frequency throttling to reduce power and keep average power within the PCIe card TDP budget. For long-duration high-definition video generation, `FA_Power_Cap` reduces end-to-end model average power consumption and improves end-to-end performance.

## Overview

`FA_Power_Cap` splits the FA operator appropriately and inserts the communication before FA between the split FA executions. This increases the interval between consecutive FA executions and interrupts sustained high-density compute on the AI processor, giving the processor a short gap before the next high-density compute segment. By splitting FA and reordering FA and communication, the technique reduces end-to-end model average power consumption and improves end-to-end performance.

This guide uses Wan2.2 as an example to explain how to enable `FA_Power_Cap`.

This guide is intended for users who can already run a Wan2.2 model repository. It explains how to manually add the `FA_Power_Cap` `insertcomm` and `blockattn` single-case optimizations to a Wan model repository that has not integrated this feature yet. The only public switch introduced in this flow is one `--comm_type` argument.

`FA_Power_Cap` targets Wan Ulysses sequence parallelism. Within each card, it further splits attention and optimizes the execution order of FA and communication. It keeps `attention_forward` as the FA compute entry point and does not change the output projection path or the overall attention semantics. The optimization only changes where `A2A` communication, RoPE/rotation, quantization, and FA computation are placed in the head-chunk or Q-block loops.

As shown in the diagram below, baseline keeps the original QKV projection, RoPE, `A2A QKV`, rotation/quantization, and FA path; `insertcomm` moves `A2A -> rotation/quantization -> FA -> A2A` into a per-attention-head-chunk loop; `blockattn` further splits Q into blocks, repeats rotation, quantization, and FA for each Q block, then concatenates the Q-block and head results. During integration, `attnlayer.py` only needs to dispatch by `comm_type` among baseline, `_run_insertcomm`, and `_run_blockattn`.

## When to use this feature

`FA_Power_Cap` is only meaningful when Wan inference already uses Ulysses sequence parallelism. The runtime command must use `ulysses_size > 1`, and the attention head count must be divisible by the Ulysses world size.

Conceptually, the three modes form a progressive relationship, but runtime use selects only one mode:

| Mode | Argument | Behavior |
| --- | --- | --- |
| Baseline | `--comm_type 0` | Keep the original baseline path: one `AlltoAll -> FA -> AlltoAll`. |
| InsertComm | `--comm_type 1` | Within each card, split by attention heads, then run `AlltoAll -> FA` chunk by chunk. |
| BlockAttn | `--comm_type 2` | Within each card, split by attention heads first, then split the local query sequence into 2 FA blocks. |

## Pipeline diagram

![FA_Power_Cap technology pipeline](../../figures/fa_power_cap_pipeline.png)

The diagram compares baseline, InsertComm, and BlockAttn paths. Its legend marks the input processing, communication, matrix multiplication, quantization, attention computation, concatenation, and output projection stages.

## Step 1: Add comm_type to generate.py

In the Wan model repository, find the argparse section in `generate.py` and add:

```python
parser.add_argument(
    "--comm_type",
    type=int,
    default=0,
    choices=[0, 1, 2],
    help="FA_Power_Cap attention communication mode: 0 disables it, 1 enables insertcomm, 2 enables block attention.")
```

Then add this validation:

```python
if args.comm_type != 0 and args.ulysses_size <= 1:
    raise ValueError("comm_type optimization requires ulysses_size > 1.")
```

This prevents users from enabling the optimization on single-card runs or runs without Ulysses sequence parallelism.

## Step 2: Pass args into attention blocks

After creating the Wan transformer, attach `args` to every block. Handle plain models and FSDP-wrapped models separately:

```python
if args.dit_fsdp:
    for block in transformer._fsdp_wrapped_module.blocks:
        block._fsdp_wrapped_module.args = args
else:
    for block in transformer.blocks:
        block.args = args
```

If Wan2.2 uses both high-noise and low-noise transformers, set both models:

```python
for model in (transformer_high, transformer_low):
    if args.dit_fsdp:
        for block in model._fsdp_wrapped_module.blocks:
            block._fsdp_wrapped_module.args = args
    else:
        for block in model.blocks:
            block.args = args
```

## Step 3: Read comm_type in the attention layer

In the attention class initializer, read the communication mode from the `args` object passed into the block:

```python
self.comm_type = int(getattr(self.args, "comm_type", 0))
```

Use one branch layout:

```python
if self.fa_alltoall_overlap:
    ...
elif self.comm_type == 1:
    ...
elif self.comm_type == 2:
    ...
else:
    ...
```

## Step 4: Implement comm_type=1 insertcomm

`insertcomm` splits `Q`/`K`/`V` by attention heads before Ulysses `AlltoAll`, so each chunk can enter FA earlier and each collective call is smaller.

The core shape is:

```python
_, _, hc, _ = query.shape
world_size = dist.get_world_size(group=self.ulysses_pg)
heads_per_npu = hc // world_size
loop_time = 10
heads_per_chunk = heads_per_npu // loop_time
global_chunk_size = heads_per_chunk * world_size

q_chunks = query.split(global_chunk_size, dim=2)
k_chunks = key.split(global_chunk_size, dim=2)
v_chunks = value.split(global_chunk_size, dim=2)
output_chunks = []

for chunk_id in range(loop_time):
    query = all_to_all_4D(q_chunks[chunk_id], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
    key = all_to_all_4D(k_chunks[chunk_id], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
    value = all_to_all_4D(v_chunks[chunk_id], scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

    out = attention_forward(query, key, value, opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
    output = all_to_all_4D(out, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)
    output_chunks.append(output)

output = torch.cat(output_chunks, dim=2)
```

When integrating this into real Wan code, keep the original ring, padding, RainFusion, FA backend selection, and quantization branches. Do not replace the complete attention implementation with this simplified snippet.

## Step 5: Implement comm_type=2 blockattn

`blockattn` first splits by attention heads, then splits the local query sequence into 2 blocks. `K` and `V` are not split by sequence; each query block attends to the full local `K/V`.

The core shape is:

```python
block_count = 2
query_blocks = torch.tensor_split(query_layer, block_count, dim=1)
block_outputs = []

for block_id in range(block_count):
    query_block = query_blocks[block_id]
    out = attention_forward(query_block, key_layer, value_layer, opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
    block_outputs.append(out)

out = torch.cat(block_outputs, dim=1)
```

Keep `block_count` fixed to 2. Users only need to select `--comm_type 2`; they do not need another tuning parameter.

## Complete attnlayer.py modification example

The example below shows how to organize baseline, `insertcomm`, and `blockattn` in one attention class inside `attnlayer.py`. When applying it to a real Wan repository, keep and merge the target code's real class name, `all_to_all_4D` import path, projections around attention, RoPE, padding, ring, RainFusion, and quantization branches.

```python
import torch
import torch.distributed as dist

from mindiesd import attention_forward
# from your_wan_sequence_parallel_module import all_to_all_4D


class FAPowerCapAttention:
    def __init__(self, args, ulysses_pg, fa_alltoall_overlap=False):
        self.args = args
        self.ulysses_pg = ulysses_pg
        self.fa_alltoall_overlap = fa_alltoall_overlap
        self.comm_type = int(getattr(self.args, "comm_type", 0))
        if self.comm_type not in (0, 1, 2):
            raise ValueError(f"comm_type must be 0, 1, or 2, but got {self.comm_type}.")

    def _attention_forward(self, query, key, value):
        return attention_forward(
            query,
            key,
            value,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BNSD")

    def _run_baseline(self, query, key, value):
        query_layer = all_to_all_4D(query, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        key_layer = all_to_all_4D(key, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        value_layer = all_to_all_4D(value, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

        out = self._attention_forward(query_layer, key_layer, value_layer)
        return all_to_all_4D(out, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)

    def _split_qkv_by_head(self, query, key, value):
        _, _, head_count, _ = query.shape
        world_size = dist.get_world_size(group=self.ulysses_pg)
        if head_count % world_size != 0:
            raise ValueError(
                f"head_count must be divisible by ulysses world size, "
                f"but got head_count={head_count}, world_size={world_size}.")

        heads_per_rank = head_count // world_size
        loop_time = 10
        if heads_per_rank % loop_time != 0:
            raise ValueError(
                f"heads_per_rank must be divisible by loop_time, "
                f"but got heads_per_rank={heads_per_rank}, loop_time={loop_time}.")

        global_chunk_heads = heads_per_rank // loop_time * world_size
        return (
            query.split(global_chunk_heads, dim=2),
            key.split(global_chunk_heads, dim=2),
            value.split(global_chunk_heads, dim=2),
        )

    def _run_insertcomm(self, query, key, value):
        q_chunks, k_chunks, v_chunks = self._split_qkv_by_head(query, key, value)
        output_chunks = []

        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            query_layer = all_to_all_4D(q_chunk, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
            key_layer = all_to_all_4D(k_chunk, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
            value_layer = all_to_all_4D(v_chunk, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

            out = self._attention_forward(query_layer, key_layer, value_layer)
            output = all_to_all_4D(out, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)
            output_chunks.append(output)

        return torch.cat(output_chunks, dim=2)

    def _run_blockattn(self, query, key, value):
        q_chunks, k_chunks, v_chunks = self._split_qkv_by_head(query, key, value)
        output_chunks = []

        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            query_layer = all_to_all_4D(q_chunk, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
            key_layer = all_to_all_4D(k_chunk, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
            value_layer = all_to_all_4D(v_chunk, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

            block_outputs = []
            for query_block in torch.tensor_split(query_layer, 2, dim=1):
                block_outputs.append(self._attention_forward(query_block, key_layer, value_layer))

            out = torch.cat(block_outputs, dim=1)
            output = all_to_all_4D(out, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)
            output_chunks.append(output)

        return torch.cat(output_chunks, dim=2)

    def forward(self, query, key, value):
        # Keep the target Wan repository's existing qkv projection, RoPE,
        # norm, padding, and other pre-attention logic.
        if self.fa_alltoall_overlap:
            output = self._run_fa_alltoall_overlap(query, key, value)
        elif self.comm_type == 1:
            output = self._run_insertcomm(query, key, value)
        elif self.comm_type == 2:
            output = self._run_blockattn(query, key, value)
        else:
            output = self._run_baseline(query, key, value)

        # Keep the target Wan repository's existing output reshape, O_proj,
        # dropout, and other post-attention logic.
        return output
```

Do not create a new attention class unrelated to the original model. A safer migration is to add helpers such as `_run_insertcomm`, `_run_blockattn`, and `_split_qkv_by_head` to the original attention class in `attnlayer.py`, then replace only the original parallel attention branch in `forward` with the `comm_type` dispatch.

## Step 6: Pass comm_type from the shell script

Keep only one script variable:

```bash
COMM_TYPE=${COMM_TYPE:-0}
```

Validate it:

```bash
case "${COMM_TYPE}" in
  0|1|2) ;;
  *) echo "COMM_TYPE must be 0, 1, or 2"; exit 1 ;;
esac
```

Pass it to `generate.py`:

```bash
torchrun --nproc_per_node=8 generate.py \
  --task t2v-14B \
  --ulysses_size 8 \
  --comm_type "${COMM_TYPE}"
```

The shell script only needs this one `COMM_TYPE` variable, which is forwarded to `generate.py` through `--comm_type`.

## Run examples

Baseline:

```bash
COMM_TYPE=0 bash infer_t2v.sh
```

InsertComm:

```bash
COMM_TYPE=1 bash infer_t2v.sh
```

BlockAttn:

```bash
COMM_TYPE=2 bash infer_t2v.sh
```

## Validation and troubleshooting

- Confirm `ulysses_size > 1`; otherwise `comm_type=1/2` should not be enabled.
- Confirm the attention head count is divisible by `ulysses_size`.
- Confirm the attention layer gets `self.comm_type` from `args.comm_type`.
- Confirm the runtime command contains `--comm_type 0/1/2`.
- Run all three modes with the same prompt, seed, resolution, and step count, then check that generated videos are valid.
- Compare latency logs, MSPTI kernel logs, and HCCL logs to verify that `AlltoAll` and FA are chunked as expected.

## Benefit reference

The following data was collected with the Wan2.2 customer-distilled Lite model, which uses only 4 inference steps. The test configuration is `W16A16F8`, no CFG, no VAE parallelism, and `SP4`. End-to-end time covers the full inference path. The table only includes valid scenarios where baseline finishes normally.

| Resolution and frames | Baseline time (s) | InsertComm time (s) | InsertComm gain | BlockAttn time (s) | BlockAttn gain | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 720p161 frames | 108.35 | 90.36 | 16.60% | 90.34 | 16.62% | Baseline finished normally; gains are computed as `(baseline - optimized) / baseline`. |
