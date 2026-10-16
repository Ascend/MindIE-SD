# USP Attention Acceleration API

`usp_attention` is a single-call acceleration API for diffusion-model attention
layers. It combines Ulysses sequence parallelism, KV-AllGather, communication
quantization, and MindIE SD FA backends. Parallel groups and execution parameters
are supplied explicitly by the caller.

The API accepts tensors, scalars, enums, and standard PyTorch process groups. It
does not accept generic metadata, config, or options containers.

> `kv_gather_group` means KV-AllGather, not Ring Attention. KV-AllGather
> materializes global K/V on every rank before one local FA call. True Ring
> Attention circulates KV blocks and merges partial results using the softmax LSE
> returned by every hop; that path is not implemented by this API.

## Features

- Executes parallel communication, FA, and output resharding in one call, with no
  public `prepare()` or `plan()` lifecycle.
- Supports four-dimensional Q/K/V in `BSND` and `BNSD` layouts.
- Supports strict Ulysses: exchanges local sequence shards for local head shards
  before FA and performs the reverse All-to-All after FA.
- Supports KV-AllGather: gathers K/V sequence shards while keeping local Q in
  place.
- Supports combined Ulysses and KV-AllGather with two-dimensional process groups.
- Supports head-cut execution and an NPU dual-stream pipeline that overlaps
  forward communication, FA, and reverse communication.
- Supports FP8 E4M3 and MXFP8-style communication codecs with separate payload
  and scale collectives.
- Disables communication quantization by default. When enabled, the default
  `exposed` scope quantizes only the first forward and last reverse boundary of
  an overlap pipeline; `all` quantizes every selected communication.
- Supports native NPU FA and block-FP8 quant FA backends.
- Supports joint K/V, caller-owned output buffers, output dtype selection, and
  query chunks.
- Raises specific errors for unsupported shapes, topologies, workspaces, and
  execution combinations instead of silently changing the execution strategy.

## Call Contract

Before calling `usp_attention`:

- Select local, Ulysses, KV-AllGather, or combined parallel execution.
- Create and own process groups.
- Supply the current rank's local Q/K/V shards and any mask or effective sequence
  lengths.
- Select communication quantization, chunking, and FA backend options.

During the call, MindIE SD:

- Validate tensor, layout, head divisibility, group, and buffer contracts.
- Execute forward and reverse Ulysses All-to-All.
- Execute K/V AllGather with optional communication quantization.
- Slice and concatenate joint K/V heads.
- Select and invoke a MindIE SD FA backend.
- Restore the input layout, sequence shard, and output dtype.

The function does not create process groups, generate a parallel plan, or parse
generic metadata, config, or options containers.

## Execution Flow

```text
Local sequence-sharded Q/K/V
        │
        ├─ Ulysses group > 1
        │      optional quant → All-to-All → full sequence / local heads
        │
        ├─ KV gather group > 1
        │      optional quant → K/V AllGather → global K/V
        │
        ├─ joint K/V head slice and concatenate
        │
        ├─ native FA or quant FA
        │
        └─ reverse Ulysses All-to-All → original sequence shard
```

The supplied groups determine the parallel mode:

| `ulysses_group` | `kv_gather_group` | Mode |
|---|---|---|
| `None` or world size 1 | `None` or world size 1 | Local FA |
| world size > 1 | `None` or world size 1 | Ulysses |
| `None` or world size 1 | world size > 1 | KV-AllGather |
| world size > 1 | world size > 1 | Ulysses + KV-AllGather |

## API

```python
from mindiesd.layers.usp import usp_attention

output = usp_attention(
    q,
    k,
    v,
    ulysses_group=None,
    kv_gather_group=None,
    scatter_dim=2,
    gather_dim=1,
    seq_lens=None,
    chunk_size=None,
    head_chunk_size=None,
    layout="BSND",
    joint_k=None,
    joint_v=None,
    attn_mask=None,
    comm_dtype="none",
    comm_tensors=("k", "v", "out"),
    comm_quant_scope="exposed",
    q_block_size=128,
    kv_block_size=256,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    overlap=False,
    backend="auto",
    out_dtype=None,
    workspace=None,
    out=None,
    return_lse=False,
)
```

### Main Parameters

| Parameter | Description |
|---|---|
| `q`, `k`, `v` | Local sequence shards on the current rank; `layout` defines their layout |
| `ulysses_group` | Ulysses All-to-All process group; `None` disables it |
| `kv_gather_group` | K/V AllGather process group; `None` disables it |
| `scatter_dim` | Ulysses head dimension; v1 supports only `2` in normalized BSND |
| `gather_dim` | Ulysses sequence dimension; v1 supports only `1` in normalized BSND |
| `seq_lens` | Effective KV lengths for the batch, with `int32` or `int64` dtype |
| `chunk_size` | FA query chunk size; `None` computes the entire local query at once |
| `head_chunk_size` | Global Q-head chunk size; enables synchronous head-cut execution, or controls overlap granularity |
| `layout` | Q/K/V layout, either `"BSND"` or `"BNSD"` |
| `joint_k`, `joint_v` | Replicated joint K/V; both values must be supplied together |
| `comm_dtype` | Communication format: `"none"`, `"fp8_e4m3"`, or `"mxfp8"` |
| `comm_tensors` | Tensor names to quantize for communication: `"q"`, `"k"`, `"v"`, and reverse `"out"` |
| `comm_quant_scope` | `"exposed"` quantizes overlap boundaries; `"all"` quantizes every selected communication |
| `overlap` | Enable the NPU communication-stream and FA-stream pipeline; defaults to `False` |
| `backend` | FA backend: `"auto"`, `"npu_fa"`, or `"quant_fa"` |
| `out` | Optional contiguous output tensor with the final result shape and dtype |
| `return_lse` | Return softmax LSE; currently supported only by `quant_fa` |

## Usage

### Local FA

```python
from mindiesd.layers.usp import usp_attention

output = usp_attention(q, k, v, layout="BSND", backend="npu_fa")
```

### Ulysses

The caller creates the Ulysses group and supplies local sequence shards. The Q
and KV head counts must be divisible by the Ulysses world size.

```python
output = usp_attention(
    q,
    k,
    v,
    ulysses_group=ulysses_process_group,
    layout="BSND",
    backend="npu_fa",
)
```

### Quantized KV-AllGather

This call encodes K/V as FP8 before communication and gathers payload and scale
separately. With `backend="quant_fa"`, the FA block-quantized K/V and scales are
consumed directly by fused quant FA without dequantizing and quantizing them a
second time. Other backends restore the original FA input dtype after the
collective.

```python
output = usp_attention(
    q,
    k,
    v,
    kv_gather_group=kv_gather_process_group,
    comm_dtype="fp8_e4m3",
    comm_tensors=("k", "v"),
    kv_block_size=256,
    backend="npu_fa",
)
```

### Head-cut and communication-compute overlap

`head_chunk_size` alone selects the synchronous head-cut path. With
`overlap=True`, MindIE SD uses a communication stream and the current FA stream,
connected by per-chunk events. If `head_chunk_size` is omitted, one local head
per rank is used for each pipeline chunk.

```python
output = usp_attention(
    q,
    k,
    v,
    ulysses_group=ulysses_process_group,
    kv_gather_group=kv_gather_process_group,
    head_chunk_size=None,
    overlap=True,
    comm_dtype="fp8_e4m3",
    comm_tensors=("k", "v", "out"),
    comm_quant_scope="exposed",
    backend="quant_fa",
)
```

Communication quantization remains disabled unless `comm_dtype` is explicitly
set. In synchronous execution every communication is exposed, so an enabled
codec applies to all selected tensors. In overlap execution, `exposed` applies
it to the first Q/K/V forward boundary and the last output reverse boundary;
middle communication remains unquantized because it is intended to be hidden by
FA. Only names present in `comm_tensors` are affected. Use
`comm_quant_scope="all"` when measurements show that middle
communication is not fully hidden.

### Combined Ulysses and KV-AllGather

```python
output = usp_attention(
    q,
    k,
    v,
    ulysses_group=ulysses_process_group,
    kv_gather_group=kv_gather_process_group,
    comm_dtype="mxfp8",
    comm_tensors=("q", "k", "v"),
    backend="quant_fa",
    out_dtype=q.dtype,
)
```

## Errors

| Error | Meaning |
|---|---|
| `USPNotSupported` | Inputs are valid, but the current executor or FA backend does not support the combination |
| `USPTopologyError` | Process-group world size conflicts with Q/K/V head partitioning |
| `USPShapeError` | Tensor, layout, dtype, scale, or length violates the contract |
| `USPWorkspaceError` | Workspace or output buffer violates the contract |

Catch specific USP errors only when the caller has an explicit recovery policy.
Avoid catching every exception and silently changing the execution strategy.

## Current Limitations

- The current implementation uses strict Ulysses; head counts must be divisible
  by the Ulysses world size.
- KV-AllGather requires equal local K/V shapes across all ranks in its group.
- `seq_lens` currently validates the contract and does not produce variable
  collective split sizes.
- `chunk_size` cannot be combined with `attn_mask`, `head_chunk_size`, or
  `overlap=True`.
- `workspace` is validated for dtype, device, and contiguity but does not yet
  replace all temporary allocations.
- Direct FA-quant reuse currently applies to FP8 E4M3 K/V AllGather with
  `backend="quant_fa"` when each rank's post-Ulysses K/V sequence is aligned to
  `kv_block_size`. Other shapes, MXFP8, and native FA use the communication
  codec and restore the FA input dtype after the collective.
- True Ring Attention, per-hop K/V P2P, and partial output/LSE merging are not
  supported.
