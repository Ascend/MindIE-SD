# 状态携带切分：递归解码器的精确切法

适用：网络含“上一帧反馈”的记忆块（`MemBlock` / `past` / `state`），候选轴为时间轴。目标：**在 CPU 上与整段解码逐位等价**，同时把每次算子调用的 batch 降下来（省显存、绕开大 batch 设备缺陷）。

> **作用域（强制）**：文中耗时 / 显存 / 帧级读数的来源环境为 Ascend 950PR 8 卡 · CANN 25.7.rc1.6 ·
> torch_npu · 容器内 vLLM-Omni 0.28 + MindIE-SD（离线单卡 fp32 口径见 §6 表注）。推导与骨架可迁移，
> **数字换环境即重测**。

## 1. 为什么只有记忆块需要状态

| 层类型 | 跨帧？ | 处理 |
|---|---|---|
| `Conv2d` / `Conv3d`(时间核=1) / `Relu` / `Clamp` / `Cast` | 否 | 直接切 |
| `Upsample`(nearest/bilinear，空间) / `pixel_shuffle` | 否 | 直接切 |
| 时间上采样 `TGrow`（1×1 conv 把 stride 展开到时间轴） | 否（逐帧映射，只改帧数） | 直接切（切分点须落在该级帧边界） |
| `MemBlock(x, past)`：`cat([x, past]) -> 3 convs + skip(x)` | **是**（`past` = 上一帧在该块**输入**激活） | **需携带 `state[i]`** |

关键语义：`past` 取的是**该块输入的上一帧**，不是该块输出的上一帧。搬运时别搬错。

## 2. 整段实现的语义（对照基准）

逐帧独立层之上，整段实现通常写成“把 (N,T,C,H,W) 展平成 (N*T,C,H,W)，再对每个 MemBlock 造一个 `mem`”：

```python
view = flat.reshape(n, tt, cc, hh, ww)
mem  = F.pad(view, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :tt].reshape(flat.shape)  # 前移一帧、首帧补 0
flat = block(flat, mem)
```

即 `mem[t] = view[t-1]`，`mem[0] = 0`。分段实现必须与此**逐位一致**。

## 3. 分段实现（骨架，可抄）

```python
def apply_sliced(model, x, patch_size, slices, state=None):
    """x: (N, T, C, H, W)。state[i] = 上一段末帧在 MemBlock i 输入处的激活。"""
    n_t = int(x.shape[1])
    if slices < 2 or n_t < 2 * slices:
        return apply_whole(model, x, patch_size)        # 退化：整段
    bounds = [(i * n_t) // slices for i in range(slices + 1)]
    state = {} if state is None else state
    outs = []
    for i in range(slices):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        part, state = apply_whole_with_state(model, x[:, lo:hi], patch_size, state)
        outs.append(part)
    return torch.cat(outs, dim=1)
```

`apply_whole_with_state` 与整段实现的唯一差别就是 mem 的来源：

```python
prev = state.get(block_index)
if prev is None:                                     # 首段：与整段同语义
    mem = F.pad(view, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :tt]
else:                                                # 后续段：接上一段末帧
    mem = torch.cat([prev.to(view.dtype), view[:, : tt - 1]], dim=1)
nxt[block_index] = view[:, -1:].detach().clone()      # 本段末帧在“该块输入”处
```

要点：

1. `state` 存的是**该块输入**的末帧（`view` 是进入该块之前的张量），别存块输出；
2. `prev` 要 `to(view.dtype)`（跨段可能 dtype 不同）；
3. `.detach().clone()`：避免把整段计算图/显存留住（推理下也要 clone，否则切片视图会绑住整块内存）；
4. 段间**顺序执行**，不要并行（第 k 段依赖第 k−1 段状态）。

## 4. 切分点必须落在潜帧边界（推导）

- 每级时间上采样都是 2 的整数幂（本案例 `decoder_time_upscale=(False,True,True)` ⇒ `time_upscale = 2^2 = 4`）；
- 因此“潜帧索引 j”在所有层的帧索引都是整数（`j*4`、`j*2`、`j*1`）⇒ **潜帧边界在所有层都是合法帧边界**；
- 若网络里有 stride 不是整数倍的层（例如 stride=3 的时间池化），必须取这些 stride 的**最小公倍数**对齐，或改用“最粗级边界”。

反例提醒：不要按“输出帧数 ÷ 段数”去切——输出帧与潜帧之间隔着时间上采样与裁剪，切错级别会让段的边界落在半帧位置。

## 5. 判定必须与 rank 无关（否则死锁）

```python
# 对：env + shape 推导，所有 rank 同结论
pieces = int(os.environ.get("MY_DECODER_SLICES", "2"))     # 全 rank 同值
ok = pieces >= 2 and world >= 2 and n_frames >= 2 * pieces
if ok: ...进入分段路径（若含通信，则所有 rank 一起进）...
else:  ...整段路径...
```

- 分段本身在单卡内顺序执行，**不必然含通信**；一旦含（例如段间状态广播、或空间分块的 halo 交换），必须全员进入；
- `rank < pieces` 的选举式参与**只适用于“输出 shape 可由算数独立推出”的场景**（例如逐帧网按片长切：每片长度可由总长算），否则接收方无法为别的 rank 预分配缓冲 ⇒ 必须全员参与。

## 6. 与同环境实测一致的结果

| 项 | 结果 |
|---|---|
| CPU 等价性 | 整段 vs 2 段 vs 4 段 **max\|d\| = 0（bitwise）** |
| NPU 正确性 | 2 段后尾段塌陷消失，16 个 24 帧桶的 hf/均值/块方差与 CPU 真值逐桶一致 |
| NPU 耗时/峰值显存 | 耗时降到约四成、峰值显存约减半；段数继续增加只带来小幅耗时收益与更低的显存（该耗时为单卡离线 fp32 口径；绝对数字见归档 `{run_results_dir}/archive/`） |
| 与设备缺陷的关系 | 该网络第 3 次 `Upsample` 单次 batch = 潜帧数×2 = 220，触发 CANN 缺陷；2 段后降到 110 ⇒ 干净 |

## 7. 何时不用状态携带切分

| 情形 | 应改用的做法 |
|---|---|
| 无记忆块（逐帧网） | 直接按轴等分（更简单、可并行） |
| 需要**并行**（不是省显存/绕缺陷） | 状态携带是顺序的，给不了并行度 ⇒ 只有“无跨轴耦合”时才可能并行 |
| 跨轴是空间卷积 | 空间分块 + halo（halo ≥ 感受野），可并行 |
| 交换量超标 | 先算交换预算（见 `verification-and-budget.md`），再决定切不切 |

## 8. 落地顺序（照做即可）

1. 抄 `scripts/sliced_state_decode.py`，替换两个 `ADAPTER` 钩子（哪些层是记忆块、整段实现什么样）；
2. 跑 `scripts/shard_equivalence_check.py`：CPU 上整段 vs 2/4 段 **必须 max|d| = 0**；
3. 加 env 开关（默认段数、`0/1` 回退整段），并保证**判定与 rank 无关**；
4. 上卡：端到端 4 请求比 md5（`scripts/frame_health.py` 看是否有尾段异常）；
5. 记录耗时/峰值显存，确认“更快或持平、更省显存”。
