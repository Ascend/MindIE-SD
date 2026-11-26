# USP Attention Acceleration API

`usp_attention` is a single-call acceleration API for diffusion-model attention
layers. `usp_attention` combines {term}`Ulysses sequence parallelism`, KV-AllGather, communication
quantization, and MindIE SD Flash Attention (FA) backends. Parallel groups and execution parameters
are supplied explicitly by the caller.

The API accepts tensors, scalars, enums, and standard PyTorch process groups. It
does not accept generic metadata, config, or options containers.

> `kv_gather_group` means KV-AllGather, not {term}`Ring Attention`. KV-AllGather
> materializes global K/V on every rank before one local FA call. True Ring
> Attention circulates KV blocks and merges partial results using the softmax LSE
> returned by every hop; that path is not implemented by this API.

## Terminology

```{glossary}
Ulysses sequence parallelism
    Converts sequence shards to attention-head shards with All-to-All, computes attention, then restores sequence shards. See [DeepSpeed-Ulysses](https://www.deepspeed.ai/tutorials/ds-sequence/).

Ring Attention
    Circulates K/V blocks around ranks and combines partial attention using softmax log-normalizers (LSE), avoiding materialized global K/V. This API does not implement it. See the [Ring Attention paper](https://arxiv.org/abs/2310.01889).
```

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
- Execute KV-AllGather with optional communication quantization.
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
        ├─ KV-AllGather group > 1
        │      optional quant → KV-AllGather → global K/V
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

| Parameter | Type | Required | Meaning, default and range (units) |
| --- | --- | --- | --- |
| `q`, `k`, `v` | Tensor | Yes | Local 4-D shards; same device/dtype; K/V shapes match, Q/K head dimensions match; Q heads divisible by KV heads; backend must support dtype. |
| `ulysses_group`, `kv_gather_group` | ProcessGroup / None | No | Caller-initialized groups; None (default) or size=1 disables that communication; Ulysses size must divide Q/KV head counts. |
| `scatter_dim`, `gather_dim` | int | No | Fixed at 2 (heads) and 1 (sequence) in normalized BSND, also the defaults. |
| `seq_lens` | Tensor / None | No | None by default; int32/int64, shape=[batch], token counts in [1, global KV length], including joint KV. |
| `chunk_size` | int / None | No | Positive Q chunk size in tokens; None by default; excludes head_chunk_size, overlap and attn_mask. |
| `head_chunk_size` | int / None | No | Positive global Q-head chunk; divides Q heads and is divisible by Ulysses size; mapped KV chunks must also divide KV heads and match topology; None by default, overlap defaults to group size. |
| `layout` | str | No | BSND (default) or BNSD. |
| `joint_k`, `joint_v` | Tensor / None | No | None by default; supply both, matching shape/dtype/device and KV batch/heads/head dimension; replicated across ranks, not gathered. |
| `attn_mask` | Tensor / None | No | None by default; shape/dtype must match the FA backend; masked query chunks require caller-sliced Q/masks and separate calls without chunk_size. |
| `comm_dtype` | str | No | none (default), fp8_e4m3 or mxfp8; none excludes communication scale overrides. |
| `comm_tensors` | tuple[str, ...] | No | Only q/k/v/out; defaults to (k,v,out); only selected tensors are communication-quantized. |
| `comm_quant_scope` | str | No | exposed (default) or all; synchronous communication is all exposed; overlap exposed selects first/last boundaries. |
| `q_block_size` | int | No | Defaults to 128, positive; FA quantization groups Q tokens (operator supports 128/256/512); Q/out communication codec groups flattened elements. |
| `kv_block_size` | int | No | Defaults to 256, positive; FA quantization groups KV tokens (128/256/512); KV communication groups flattened elements; need not equal q_block_size. |
| `q_scale`, `k_scale`, `v_scale` | Tensor / None | No | None by default; forward communication scale overrides, converted to FP32; scalar or [C, ceil(flat_width/block_size)], where C counts leading chunks and flat_width counts flattened elements per chunk, positive and finite; not quant_fa internal scales. |
| `overlap` | bool | No | False by default; True requires NPU streams and excludes chunk_size. |
| `backend` | str | No | auto (default), npu_fa or quant_fa; quant_fa requires the block-FP8 operator. |
| `out_dtype` | torch.dtype / None | No | None preserves Q dtype; requested dtype must be backend-supported; out must match the final dtype. |
| `workspace` | Tensor / None | No | None by default; reserved buffer must be uint8, contiguous and on Q device; current paths do not guarantee its use. |
| `out` | Tensor / None | No | None by default; contiguous, on Q device, matching final result shape/dtype. |
| `return_lse` | bool | No | False by default; True requires backend=quant_fa and returns (output, softmax LSE). |

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
it to the first Q/K/V forward boundary and the last output reverse boundary.
Middle communication remains unquantized because it is intended to be hidden by
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
- Direct FA-quant reuse currently applies to FP8 E4M3 KV-AllGather with
  `backend="quant_fa"` when each rank's post-Ulysses K/V sequence is aligned to
  `kv_block_size`. Other shapes, MXFP8, and native FA use the communication
  codec and restore the FA input dtype after the collective.
- True Ring Attention, per-hop K/V P2P, and partial output/LSE merging are not
  supported.

## RF-v2 / RF-v3: CP × Ulysses Sparse Attention

`mindiesd.parallel.distributed_sparse_attention` is separate from ordinary FA USP; its shared SDK contract is provided by Core PR #910, and CP consumers additionally require #912. Sequence worlds 8/16/32 use Ulysses size 4 or 8, with CP size=sequence world/Ulysses size. Contiguous U8 rank groups have CP sizes 1/2/4 and matching-head KV groups [r], [r,r+8] or [r,r+8,r+16,r+24]. H3's 56 heads form seven chunks per U8 owner, without requiring divisibility by the sequence world.

The caller creates orthogonal groups in the same order on all ranks, selects physical devices/HCCL options, rearranges Q/K/V into global RF order, then shards by contiguous logical rank. Logical ranks are not physical device IDs.

```python
from mindiesd.parallel import ParallelState, distributed_sparse_attention

state = ParallelState(group=ulysses_group, cp_group=cp_group, bsa_backend="eagle")
output, stats = distributed_sparse_attention(
    q_local, k_local, v_local, global_spans,
    state=state, used_len=global_used_len,
    precision="mix", sparse_type="rf_v3", pre_rearranged=True, return_stats=True,
)
```

Q/K/V share shape [1,S_local,H,128], with H divisible by Ulysses size. used_len is the globally valid token count excluding padding. Pure Ulysses sets State(cp_group=None) and uses pre_rearranged=False/True for original/RF order. CP supplies its actual group and requires pre_rearranged=True. A State cannot change communicator or CP phase.

| Operator | sparse_type | precision | State.bsa_backend | NPU |
| --- | --- | --- | --- | --- |
| RainFusionAttention | rf_v2 | bf16/fp16 | cann | A2/A3 |
| Eagle BSA | rf_v3 | bf16/fp16 | eagle | A2/A3 |
| EagleQBSA | rf_v3 | mix | eagle | A5 |
| Native MXFP8 BSA | rf_v3 | mxfp8 | cann | A5 |

Omit bsa_backend or use None for table-based selection. Only fp16 requires FP16 inputs; other modes require BF16. Native MXFP8 requires CANN 9.2.0 beta 2 or later, matching operators and the PyTorch BSA/metadata extension loaded on every rank.

| precision | Forward Q/K | Forward V |
| --- | --- | --- |
| bf16 / fp16 | Corresponding native dtype | Corresponding native dtype |
| mix | INT8, FP32 scale per head/64 RF rows | E4M3, global FP32 scale per head/channel |
| mxfp8 | E4M3, E8M0 scale per D32 head elements | E4M3, E8M0 scale per S32 global RF rows |

KV-AllGather/Q-edge exchange transmits packets unchanged, using shared scales for cross-rank blocks; BSA consumes packets/scales directly. Reverse All-to-All uses input dtype. All paths construct RainFusion masks from pre-quantization Q/K pools, global K ranking and protected regions. `stats["plan"]["mask_algorithm"]` is rainfusion; `stats["mask"]` retains its historical execution label.

The public entry points are distributed_sparse_attention and ParallelState. Communicators go only in State; execution rejects group/cp_group. Optional world/rank check its Ulysses group; other arguments are grouped by precision/numerical policy, sparsity, geometry/RF order and diagnostics. Legacy layout/group helpers remain lazy compatibility imports. State.quantized_comm defaults to False (BF16 communication followed by consumer quantization); True retains quantized communication. The boolean is fixed for the state lifetime and must agree across ranks; precision still selects the BSA.

All four paths support pure Ulysses and CP with the following options. Device acceptance is recorded separately; CPU checks do not establish NPU acceptance.

| Optimization | Configuration/conditions |
| --- | --- |
| Communication/compute overlap | State.overlap_comm=True (explicit opt-in; False by default), separate compute/communication streams |
| RF rearrangement before sharding | Global rearrangement then sharding, pre_rearranged=True |
| Packet preparation | Prepare all head chunks before exchange, including quantized payload/scales |
| Pool Q/K before concatenation | optimize_attention_pipeline=True, separate pooling/aggregation |
| Common reverse storage | optimize_attention_pipeline=True, NPU serial/overlap chunks share receive storage and one layout restore |

mix/MX retain baseline geometry, chains and quantization state with B disabled; float serial B-off execution retains at most four geometry maps and rebuilds each chain. B controls extra packing, pooling, protection and receive storage; payload/amax/pool/scales refresh each call. Serial NPU calls use a tail event without a communication stream; full diagnostics are built on demand.

CP queues all forward Q/K/V All-to-All transfers, then KV-AllGather and necessary Q-edge exchange. Per-chunk ready events limit compute waits; BSA-done/returned events protect reverse transfers and final merging, without per-operation device synchronization. Each call has 4×chunks Ulysses All-to-All operations, 2×chunks KV-AllGather operations, and chunks Q-edge gathers for a crossed 128-token boundary.

All CP phases use global geometry for edge participation, including padding-only phases. Exchange at most 128 Q-edge rows per group, align global query blocks, then crop to owned rows. Short groups may borrow the full Q window, reported by `stats["hybrid"]["full_query_gather"]`. The CP2×U8 example with 52282 valid/52288 stored tokens and S_local=3268 borrows 96/32 rows; protection caches only each phase's rectangular Q rows.

`stats["hybrid"]` records group sizes, phase, global rank, windows and actual gathers. boundary_exchange describes local borrowing; boundary_collective describes group-wide participation. GQA, Ring Attention and CP inputs without global RF reorder are rejected before tensor collectives. Failed communication invalidates State, with no silent fallback or replay.

## Revision History

| Document version | Publication date | Changes |
| --- | --- | --- |
| 02 | 2026-11-23 | Main Parameters: types and constraints; Terminology: definitions and references; Sparse Attention: CP/MXFP8 integration. |
| 01 | 2026-09-03 | Initial formal publication. |
