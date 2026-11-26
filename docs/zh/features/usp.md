# USP 注意力加速接口

`usp_attention` 是面向扩散模型注意力层的单次调用加速接口。`usp_attention` 在一个调用中组合
{term}`Ulysses 序列并行`、KV-AllGather、通信量化和 MindIE SD Flash Attention（FA）后端。进程组和执行参数由
调用方显式传入。

该接口接收张量、标量、枚举和标准 PyTorch 进程组（process group），不接收通用 metadata、
config 或 options 容器。

> `kv_gather_group` 表示 KV-AllGather，不表示 {term}`Ring Attention <Ring Attention（环形注意力）>`。KV-AllGather 会先在每个
> rank 上物化完整 K/V，再执行一次本地 FA；真正的 Ring Attention 需要逐跳传递 KV，
> 并根据每一跳返回的 softmax LSE 合并局部结果，当前接口尚未实现该路径。

## 术语

```{glossary}
Ulysses 序列并行
    将序列分片经 All-to-All 转换为 head 分片以计算注意力，再反向恢复序列分片的策略。参见 [DeepSpeed-Ulysses](https://www.deepspeed.ai/tutorials/ds-sequence/)。

Ring Attention（环形注意力）
    按环逐跳传递 K/V 块，以 softmax 对数归一化量（LSE）合并局部注意力结果，避免物化完整 K/V。本接口未实现该策略。参见 [Ring Attention 论文](https://arxiv.org/abs/2310.01889)。
```

## 特性

- 单次调用完成并行通信、FA 计算和输出分片恢复，不提供公开 `prepare()` 或 `plan()`。
- 支持 `BSND` 和 `BNSD` 两种四维 Q/K/V 布局。
- 支持严格 Ulysses：FA 前将本地序列分片转换为本地 head 分片，FA 后执行反向
  All-to-All。
- 支持 KV-AllGather：聚合各 rank 的 K/V 序列分片，本地 Q 保持不变。
- 支持 Ulysses 与 KV-AllGather 组合，用于二维进程组。
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
- 创建并持有进程组。
- 提供当前 rank 的本地 Q/K/V 分片，以及可选的 mask 和真实序列长度。
- 选择通信量化、chunk 和 FA 后端参数。

调用过程中，MindIE SD：

- 验证 tensor、layout、head 可分性、group 和 buffer 契约。
- 执行 Ulysses 前向和反向 All-to-All。
- 执行 KV-AllGather 及可选通信量化。
- 处理 joint K/V head 切分与拼接。
- 选择并调用 MindIE SD Flash Attention（FA）后端。
- 恢复输入 layout、sequence shard 和输出 dtype。

该函数不创建进程组、不生成并行方案，也不解析通用 metadata、config 或
options 容器。

## 执行流程

```text
本地 sequence shard Q/K/V
        │
        ├─ Ulysses group > 1
        │      quant（可选）→ All-to-All → full sequence / local heads
        │
        ├─ KV-AllGather group > 1
        │      quant（可选）→ KV-AllGather → global K/V
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

| 参数 | 类型 | 必填 | 含义、默认值与范围（单位） |
| --- | --- | --- | --- |
| `q`, `k`, `v` | Tensor | 是 | 本地四维分片；同设备/dtype，K/V等形，Q/K头维相同；Q头数可整除KV头数，dtype须被后端支持。 |
| `ulysses_group`, `kv_gather_group` | ProcessGroup / None | 否 | 调用方已初始化的进程组；默认None或size=1不启用对应通信；Ulysses需Q/KV头数整除组大小。 |
| `scatter_dim`, `gather_dim` | int | 否 | 标准化BSND下固定为2（head）和1（sequence），也是默认值。 |
| `seq_lens` | Tensor / None | 否 | 默认None；int32/int64，shape=[batch]，token数范围[1, 全局KV长度]，长度包括joint KV。 |
| `chunk_size` | int / None | 否 | Q块大小，单位token，须>0；默认None；不能与head_chunk_size、overlap或attn_mask组合。 |
| `head_chunk_size` | int / None | 否 | 全局Q头数块，须>0、整除Q头数且被Ulysses size整除；映射后的KV头块也须兼容；默认None，双流默认组大小。 |
| `layout` | str | 否 | BSND（默认）或BNSD。 |
| `joint_k`, `joint_v` | Tensor / None | 否 | 默认None；必须同时提供，shape/dtype/device一致且匹配KV的batch/head/head维；跨rank复制而非gather。 |
| `attn_mask` | Tensor / None | 否 | 默认None；shape/dtype需符合所选FA后端，带mask的query分块需调用方自行切分Q/mask并分次调用，不能设chunk_size。 |
| `comm_dtype` | str | 否 | none（默认）、fp8_e4m3或mxfp8；none时不可提供通信scale。 |
| `comm_tensors` | tuple[str, ...] | 否 | 只允许q/k/v/out，默认(k,v,out)；仅选中张量参与通信量化。 |
| `comm_quant_scope` | str | 否 | exposed（默认）或all；同步时全部选中通信暴露，双流exposed仅首/末边界。 |
| `q_block_size` | int | 否 | 默认128，须>0；FA量化沿Q序列按token分块（算子支持128/256/512）；Q/out通信codec按展平元素分块。 |
| `kv_block_size` | int | 否 | 默认256，须>0；FA量化沿KV序列按token分块（128/256/512）；KV通信codec按展平元素分块，无需等于q_block_size。 |
| `q_scale`, `k_scale`, `v_scale` | Tensor / None | 否 | 默认None；仅对应前向通信量化的scale覆盖，转换为FP32；标量或[C, ceil(flat_width/block_size)]，C为该阶段首维chunk数、flat_width为每chunk展平元素数，须正且有限；不是quant_fa内部scale。 |
| `overlap` | bool | 否 | 默认False；True需NPU流运行时且不能指定chunk_size。 |
| `backend` | str | 否 | auto（默认）、npu_fa或quant_fa；quant_fa需block FP8算子。 |
| `out_dtype` | torch.dtype / None | 否 | 默认None保持Q dtype；指定值必须被后端支持；out须匹配最终dtype。 |
| `workspace` | Tensor / None | 否 | 默认None；预留缓冲须uint8、连续并与Q同设备；不保证当前执行路径使用该缓冲。 |
| `out` | Tensor / None | 否 | 默认None；连续、与Q同设备，shape/dtype与最终结果一致。 |
| `return_lse` | bool | 否 | 默认False；True仅支持backend=quant_fa，返回(output, softmax LSE)。 |

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
  `kv_block_size` 对齐的 FP8 E4M3 KV-AllGather；其他 shape、MXFP8 和普通 FA 使用通信
  codec，并在 collective 后恢复 FA 输入 dtype。
- 当前不支持真正的 Ring Attention、逐跳 K/V P2P 和 partial output/LSE 合并。

## RF-v2 / RF-v3：CP × Ulysses 稀疏注意力

`mindiesd.parallel.distributed_sparse_attention` 是独立于普通 FA USP 的稀疏接口（共享SDK接口由Core PR #910提供，CP消费另需#912），序列world可为8/16/32，Ulysses size为4或8，CP size=序列world/Ulysses size。 U8的连续rank组对应CP size为1/2/4，KV组为同head owner的[r]、[r,r+8]或[r,r+8,r+16,r+24]。 H3的56个heads在每个U8 owner上分7个chunk，无需整除序列world。

调用方在所有rank按同一顺序创建正交进程组、选择物理设备及HCCL options，并先完成全局RF重排，再按连续logical rank分片。 logical rank不等于物理卡号。

```python
from mindiesd.parallel import ParallelState, distributed_sparse_attention

state = ParallelState(group=ulysses_group, cp_group=cp_group, bsa_backend="eagle")
output, stats = distributed_sparse_attention(
    q_local, k_local, v_local, global_spans,
    state=state, used_len=global_used_len,
    precision="mix", sparse_type="rf_v3", pre_rearranged=True, return_stats=True,
)
```

Q/K/V等形为[1,S_local,H,128]，H可被Ulysses size整除；used_len是全局有效token数，排除padding。 纯Ulysses设置State(cp_group=None)，原始/RF行序分别用pre_rearranged=False/True；CP传实际cp_group且必须pre_rearranged=True。 同一State不能换进程组或CP phase。

| 算子 | sparse_type | precision | State.bsa_backend | NPU |
| --- | --- | --- | --- | --- |
| RainFusionAttention | rf_v2 | bf16/fp16 | cann | A2/A3 |
| Eagle BSA | rf_v3 | bf16/fp16 | eagle | A2/A3 |
| EagleQBSA | rf_v3 | mix | eagle | A5 |
| 原生MXFP8 BSA | rf_v3 | mxfp8 | cann | A5 |

省略bsa_backend或传None按表选择；仅fp16要求FP16输入，其他档输入BF16。 原生MXFP8需CANN 9.2.0 beta 2或以上及匹配的算子包、PyTorch BSA/metadata扩展，在每个rank加载。

| precision | Q/K前向报文 | V前向报文 |
| --- | --- | --- |
| bf16 / fp16 | 对应原生dtype | 对应原生dtype |
| mix | INT8；每head的64个RF行共享FP32 scale | E4M3；每head/channel共享全序列FP32 scale |
| mxfp8 | E4M3；沿head维D32共享E8M0 scale | E4M3；沿全局RF行序S32共享E8M0 scale |

KV-AllGather/Q边界交换原样传递量化报文，跨rank块共享scale，BSA直接消费报文和scale；反向All-to-All用输入dtype。 所有分路按量化前Q/K均值构造RainFusion mask，进行全局K排名及保护区域保留；`stats["plan"]["mask_algorithm"]`为rainfusion，`stats["mask"]`保留旧执行标签。

公开入口为distributed_sparse_attention和ParallelState；进程组仅传State，执行器不接受group/cp_group。 可选world/rank校验State的Ulysses组；其余参数按精度/数值、稀疏、输入几何/RF行序、诊断分区。 旧布局/建组helper保留惰性兼容。State.quantized_comm默认False，BF16通信后在消费端量化；True保留量化通信。该布尔值在State生命周期内固定，所有rank必须一致；precision仍决定BSA。

四条分路均支持纯Ulysses与CP，优化由以下选项控制。 设备验收与功能支持分别记录，CPU不能替代NPU验收。

| 优化 | 配置与条件 |
| --- | --- |
| 双流通算掩盖 | State.overlap_comm=True（显式开启，默认False），计算流和通信流 |
| RF重排前移 | 全局重排后分片，pre_rearranged=True |
| 通信预打包 | 交换前准备所有head chunk，量化时含payload/scale |
| Q/K先池化后拼接 | optimize_attention_pipeline=True，分别池化/汇总Q/K统计 |
| 公共反向缓冲 | optimize_attention_pipeline=True，NPU串行/双流共用接收存储并统一恢复布局 |

mix/MX的基础几何、链和量化状态即使B关闭也复用；浮点单流B关闭仅保留最多4份几何映射，逐调用重建链。 B控制额外打包、池化、保护区域和接收缓冲；payload/amax/pool/scale逐调用刷新。 NPU单流用尾事件保护复用，不建通信流，完整诊断按需生成。

CP通信流先提交全部正向Q/K/V All-to-All，再提交KV-AllGather及必要Q边界交换。 每chunk就绪事件限定计算等待，BSA完成/返回事件保护反向通信与最终合并，无逐操作设备同步。 每调用为4×chunks次Ulysses All-to-All、2×chunks次KV-AllGather，跨128-token块时增加chunks次Q边界gather。

各CP phase按全局几何决定边界collective，padding-only phase也参与；每组交换至多128行Q边缘，补齐全局Q块后裁回owned区域。 短组可能借用整个Q窗口，`stats["hybrid"]["full_query_gather"]`据实报告。 52282有效/52288存储、S_local=3268的CP2×U8示例分别借96/32行；保护mask只缓存本phase矩形Q行。

`stats["hybrid"]`记录组大小、phase、global rank、Q窗口和实际gather次数；boundary_exchange表示本phase借用，boundary_collective表示全组通信。 GQA、Ring Attention、未全局RF重排的CP输入均在tensor collective前拒绝；通信失败后State不能复用，不静默回退或重放。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 02 | 2026-11-23 | 主要参数：补全类型与约束；术语：定义与引用；稀疏并行：CP/MXFP8接入说明。 |
| 01 | 2026-09-03 | 第一次正式发布。 |
