# USP 注意力加速接口

`usp_attention` 是面向扩散模型注意力层的单次调用加速接口。它在一个调用中组合
Ulysses 序列并行、KV-AllGather、通信量化和 MindIE SD FA 后端。并行组和执行参数由
调用方显式传入。

该接口接收张量、标量、枚举和标准 PyTorch process group，不接收通用 metadata、
config 或 options 容器。

> `kv_gather_group` 表示 KV-AllGather，不表示 Ring Attention。KV-AllGather 会先在每个
> rank 上物化完整 K/V，再执行一次本地 FA；真正的 Ring Attention 需要逐跳传递 KV，
> 并根据每一跳返回的 softmax LSE 合并局部结果，当前接口尚未实现该路径。

## 特性

- 单次调用完成并行通信、FA 计算和输出分片恢复，不提供公开 `prepare()` 或 `plan()`。
- 支持 `BSND` 和 `BNSD` 两种四维 Q/K/V 布局。
- 支持严格 Ulysses：FA 前将本地序列分片转换为本地 head 分片，FA 后执行反向
  All-to-All。
- 支持 KV-AllGather：聚合各 rank 的 K/V 序列分片，本地 Q 保持不变。
- 支持 Ulysses 与 KV-AllGather 组合，用于二维并行组。
- 支持按 head 切分的同步执行，以及用 NPU 双流掩盖前向通信、FA 和反向通信的流水执行。
- 支持 FP8 E4M3 和 MXFP8 风格的通信编解码，量化 payload 和 scale 分开通信。
- 通信量化默认关闭；显式开启后默认使用 `exposed` 范围，只量化流水首段前向通信和末段反向通信，
  也可用 `all` 量化全部选中通信。
- 支持普通 NPU FA 和 block FP8 quant FA 后端。
- 支持 joint K/V、调用方 output buffer、输出 dtype 和 query chunk。
- 对不支持的 shape、拓扑、workspace 和执行组合抛出细分异常，不静默改变执行策略。

## 调用约定

调用 `usp_attention` 前：

- 选择单卡、Ulysses、KV-AllGather 或组合并行方案。
- 创建并持有 process group。
- 提供当前 rank 的本地 Q/K/V 分片，以及可选的 mask 和真实序列长度。
- 选择通信量化、chunk 和 FA 后端参数。

调用过程中，MindIE SD：

- 验证 tensor、layout、head 可分性、group 和 buffer 契约。
- 执行 Ulysses 前向和反向 All-to-All。
- 执行 K/V AllGather 及可选通信量化。
- 处理 joint K/V head 切分与拼接。
- 选择并调用 MindIE SD FA 后端。
- 恢复输入 layout、sequence shard 和输出 dtype。

该函数不创建 process group、不生成并行方案，也不解析通用 metadata、config 或
options 容器。

## 执行流程

```text
本地 sequence shard Q/K/V
        │
        ├─ Ulysses group > 1
        │      quant（可选）→ All-to-All → full sequence / local heads
        │
        ├─ KV gather group > 1
        │      quant（可选）→ K/V AllGather → global K/V
        │
        ├─ joint K/V head slice + concatenate
        │
        ├─ native FA 或 quant FA
        │
        └─ reverse Ulysses All-to-All → 原始 sequence shard
```

并行模式由传入的 group 决定：

| `ulysses_group` | `kv_gather_group` | 执行模式 |
|---|---|---|
| `None` 或 world size 1 | `None` 或 world size 1 | 本地 FA |
| world size > 1 | `None` 或 world size 1 | Ulysses |
| `None` 或 world size 1 | world size > 1 | KV-AllGather |
| world size > 1 | world size > 1 | Ulysses + KV-AllGather |

## 接口

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

### 主要参数

| 参数 | 说明 |
|---|---|
| `q`, `k`, `v` | 当前 rank 的本地序列分片，布局由 `layout` 指定 |
| `ulysses_group` | Ulysses All-to-All process group；`None` 表示不启用 |
| `kv_gather_group` | K/V AllGather process group；`None` 表示不启用 |
| `scatter_dim` | Ulysses 被切分的 head 维；当前仅支持标准化 BSND 的 `2` |
| `gather_dim` | Ulysses 聚合的 sequence 维；当前仅支持标准化 BSND 的 `1` |
| `seq_lens` | batch 的真实 KV 长度，类型为 `int32` 或 `int64` |
| `chunk_size` | FA query chunk 大小；`None` 表示一次计算完整本地 query |
| `head_chunk_size` | 全局 Q head 的 chunk 大小；用于同步切头，或控制双流流水粒度 |
| `layout` | Q/K/V 布局，支持 `"BSND"`、`"BNSD"` |
| `joint_k`, `joint_v` | 在各 rank 复制的 joint K/V，必须同时传入 |
| `comm_dtype` | 通信格式：`"none"`、`"fp8_e4m3"` 或 `"mxfp8"` |
| `comm_tensors` | 启用通信量化的张量名称，允许 `"q"`、`"k"`、`"v"` 和反向输出 `"out"` |
| `comm_quant_scope` | `"exposed"` 只量化流水边界；`"all"` 量化全部选中通信 |
| `overlap` | 启用 NPU 通信流与当前 FA 流组成的流水；默认值为 `False` |
| `backend` | FA 后端：`"auto"`、`"npu_fa"` 或 `"quant_fa"` |
| `out` | 可选的连续输出张量，shape 和 dtype 必须与最终结果一致 |
| `return_lse` | 是否返回 softmax LSE；当前仅限 `quant_fa` |

## 使用方法

### 本地 FA

```python
from mindiesd.layers.usp import usp_attention

output = usp_attention(q, k, v, layout="BSND", backend="npu_fa")
```

### Ulysses

调用方先创建 Ulysses group，并把本地 sequence shard 传入接口。Q head 和 KV head 数必须
能被 Ulysses world size 整除。

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

### 量化 KV-AllGather

以下调用在通信前把 K/V 编码成 FP8，并分别聚合 payload 和 scale。使用
`backend="quant_fa"` 时，FA block quant 后的 K/V 和 scale 会直接送入 fused quant FA，
不会先反量化再做第二次量化；其他后端会在 collective 后恢复 FA 输入 dtype。

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

### 切头与通算掩盖

只设置 `head_chunk_size` 时使用同步切头分支。再设置 `overlap=True` 后，MindIE SD 使用通信流和
当前 FA 流，并通过逐 chunk event 串联两条流。未设置 `head_chunk_size` 时，每个流水 chunk 默认
对应每 rank 一个本地 head。

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

只有显式设置 `comm_dtype` 才会启用通信量化。同步执行中的通信都无法被 FA 掩盖，因此启用后会
量化全部选中张量。流水执行中，`exposed` 量化第一个 chunk 中 `comm_tensors` 选中的 Q/K/V
前向边界和最后一个 chunk 的输出反向边界；中间通信保持非量化，交由 FA 掩盖。如果实测中间通信不能完全掩盖，可改用
`comm_quant_scope="all"`。

### Ulysses 与 KV-AllGather 组合

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

## 异常

| 异常 | 含义 |
|---|---|
| `USPNotSupported` | 参数合法，但当前 executor 或 FA backend 不支持该组合 |
| `USPTopologyError` | group world size 与 Q/K/V head 分片不一致 |
| `USPShapeError` | tensor、layout、dtype、scale 或长度不符合契约 |
| `USPWorkspaceError` | workspace 或 output buffer 不符合契约 |

仅在调用方有明确恢复策略时捕获对应的 USP 异常，不建议捕获所有异常后静默改变执行策略。

## 当前限制

- 当前实现是严格 Ulysses，head 数必须被 Ulysses world size 整除。
- KV-AllGather 要求 group 内各 rank 的本地 K/V shape 相同。
- `seq_lens` 当前用于契约校验，不会生成变长 collective split sizes。
- `chunk_size` 不能与 `attn_mask`、`head_chunk_size` 或 `overlap=True` 同时启用。
- `workspace` 当前只做类型、device 和连续性检查，尚未用于替代全部临时分配。
- FA 量化结果直接复用当前覆盖 `backend="quant_fa"` 且各 rank Ulysses 后 K/V 序列长度按
  `kv_block_size` 对齐的 FP8 E4M3 K/V AllGather；其他 shape、MXFP8 和普通 FA 使用通信
  codec，并在 collective 后恢复 FA 输入 dtype。
- 当前不支持真正的 Ring Attention、逐跳 K/V P2P 和 partial output/LSE 合并。
