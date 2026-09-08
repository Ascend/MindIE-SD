# eagle_block_sparse_attention 单算子测试

本目录是 910B / 910_93 上 `eagle_block_sparse_attention` 的三层套件：小 shape 精度、大 shape 精度、大 shape `msprof op`。layout 为 BNSD (Batch, Num heads, Seq, Dim, 批、头、序列、头维)。`block_shape` 覆盖 `[128, 128]` 与 `[128, 64]`（BlockK 即 `block_shape[1]`，910B 上须为 16 的倍数）。`inner_precise=0`。造数走开源 50% $U[-5,5]$ + 50% 正态，seed 固定为 `20260811`。选卡脚本不在本目录，见仓根 `tests/tools/select_npu_device.py`。

被测 DUT (Device Under Test, 被测实现) 是仓内 `mindiesd.layers._custom_ops.eagle_block_sparse_attention`。整体门禁用 cosine (cosine similarity, 余弦相似度) 与模长比 (norm ratio, 欧氏范数比)。

## 三层对照

| 层 | 文件 | shape | 对照 | 尺子 |
|----|------|-------|------|------|
| 小 shape 精度 | `test_eagle_block_sparse_attention_accuracy.py` + `eagle_block_sparse_attention_accuracy_common.py` | Q/KV `[1,2,256,128]`（整 tile）与 `[1,2,257,128]`（$256+1$ 尾块）；FP16 与 BF16；`block_shape` `[128,128]` 与 `[128,64]` | CPU 上按块做 `matmul` + masked softmax + `matmul`，最后一档落到计算 dtype | 按计算 dtype 查混合容差；cosine $\ge 0.99$；模长比 $\in [0.9,1.1]$ |
| 大 shape 精度 | `check_eagle_block_sparse_attention_prod_accuracy.py` + `run_eagle_block_sparse_attention_prod_accuracy.sh` | Q `[1,32,32768,128]`，KV `[1,4,32768,128]`；FP16 与 BF16；`block_shape` `[128,128]` 与 `[128,64]` | 同设备 NPU 小算子拼接（按 Q tile 切开，禁止物化 $S\times S$）；无 CPU 逐步模拟 | 同精度，仍按计算 dtype 查混合容差；cosine $\ge 0.99$；模长比 $\in [0.9,1.1]$ |
| 大 shape 性能 | `profile_eagle_block_sparse_attention_prod.py` + `run_eagle_block_sparse_attention_msprof_op.sh` | 与大 shape 精度同一条（含 BlockK=64） | 先采 CANN 桥接 `block_sparse_attention`（`--kernel-name` `BlockSparseAttention`，仅 BlockK=128），再采本地 `eagle_block_sparse_attention`（`EagleBlockSparseAttention`，BlockK=128 与 64）；`cann_bsa` 的 Duration 为基线 | `msprof op` Duration |

$257=2\times 128+1$，用来覆盖完整 128-token 块之外的尾块。BlockK=64 时同一条序列再按 64 切 KV 块：256 为整块，257 为 $4\times 64+1$ 尾块。大 shape 的 Q 头数与 KV 头数是 GQA (Grouped Query Attention, 分组查询注意力) $32/4$。Q/KV 张量维与 BlockK=128 用例相同，只改 `block_shape[1]` 与 mask 第 4 维（`ceil(kv_seq / BlockK)`）。

没有同语义的 `torch_npu` 块稀疏算子时，大 shape 对照用 NPU 上的 `matmul` + `softmax` + `matmul`，不要退回 CPU。

## 文件

| 文件 | 用途 |
|------|------|
| `eagle_block_sparse_attention_accuracy_common.py` | 造数、块稀疏 mask、CPU / NPU compose golden、混合容差、cosine / 模长比 |
| `test_eagle_block_sparse_attention_accuracy.py` | pytest 小 shape；设备固定 `npu:0` |
| `check_eagle_block_sparse_attention_prod_accuracy.py` | 大 shape argparse |
| `run_eagle_block_sparse_attention_prod_accuracy.sh` | 选卡后跑上一行 |
| `profile_eagle_block_sparse_attention_prod.py` | 大 shape 单实现、单 dtype、单次 forward，供 `msprof op`；`--impl cann_bsa` 或 `eagle_bsa` |
| `run_eagle_block_sparse_attention_msprof_op.sh` | 选卡后先采 `cann_bsa`、再采 `eagle_bsa`；每个 impl × dtype 单独进程 |

## 怎么跑

在仓根、已 `source` CANN `set_env.sh` 且本仓 `mindiesd` 可 import 的环境：

```bash
unset ASCEND_RT_VISIBLE_DEVICES

# 小 shape 精度（256 与 257，FP16 与 BF16，block_shape [128,128] 与 [128,64]；设备固定 npu:0）
pytest tests/ops/eagle_block_sparse_attention/test_eagle_block_sparse_attention_accuracy.py -s

# 大 shape 精度（默认 FP16/BF16 × BlockK 128/64）
bash tests/ops/eagle_block_sparse_attention/run_eagle_block_sparse_attention_prod_accuracy.sh
bash tests/ops/eagle_block_sparse_attention/run_eagle_block_sparse_attention_prod_accuracy.sh \
  --device-id 3 -- --dtype all --block-kv all

# 大 shape 性能（先 CANN 桥接 BlockK=128，再 eagle BlockK=128 与 64；各 dtype 单独采）
bash tests/ops/eagle_block_sparse_attention/run_eagle_block_sparse_attention_msprof_op.sh
bash tests/ops/eagle_block_sparse_attention/run_eagle_block_sparse_attention_msprof_op.sh \
  --device-id 3 --impls cann_bsa,eagle_bsa --dtypes float16,bfloat16 --block-kvs 128,64
```

**不要**设置 `ASCEND_RT_VISIBLE_DEVICES`。小 shape pytest 固定 `npu:0`。大 shape 换卡用 `--device-id`（`npu-smi info` 的 NPU ID），Python 内 `torch.npu.set_device`。两条实现都传 `inner_precise=0`（与精度套件一致；不使用 Python 封装 `block_sparse_attention` 的缺省 4），且不传 dequant scale。对照算子 `--kernel-name` 为上板前缀 `BlockSparseAttention`（可用 `CANN_BSA_KERNEL_NAME` 覆盖；前缀匹配不到时可改成 `block_sparse`）；本地算子为 `EagleBlockSparseAttention`（`EBSA_KERNEL_NAME` / `KERNEL_NAME`）。默认采集顺序是 `cann_bsa` 再 `eagle_bsa`，每条单独起进程，避免记录混合。CANN 桥接 BSA 只采 BlockK=128；`eagle_bsa` 额外采 BlockK=64。产物在 `--output` 的绝对路径下（默认 `logs/msprof_eagle_block_sparse_attention_<stamp>/<impl>/<dtype>/k<BlockK>/`），以工具自己的 Operator Basic Information 为准。`cann_bsa` 的 Duration 是性能基线：

```text
speedup = Duration(cann_bsa) / Duration(eagle_bsa)
speedup_percent = (speedup - 1) * 100%
```
