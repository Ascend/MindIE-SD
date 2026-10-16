# eagle_quant_block_sparse_attention 单算子测试

本目录是 Ascend 950 上 `eagle_quant_block_sparse_attention` 的三层套件：小 shape 精度、大 shape 精度、大 shape `msprof op`。精度被测路径只有 mix：量化前 BF16；Q/K 按序列块 INT8（block $=64$）；V 按通道 FP8 E4M3（INT8 存储）；输出 BF16；`inner_precise=4`；`block_shape` 为 `[128, 128]`。大 shape 性能在同一条 Q/KV 张量维上另采 CANN 桥接的 950 FP8 BSA 作 Duration 基线。layout 为 BNSD (Batch, Num heads, Seq, Dim, 批、头、序列、头维)。造数走开源 50% $U[-5,5]$ + 50% 正态，seed 固定为 `20260811`。选卡脚本不在本目录，见仓根 `tests/tools/select_npu_device.py`。

被测 DUT (Device Under Test, 被测实现) 是仓内 `mindiesd.layers._custom_ops.eagle_quant_block_sparse_attention`。Cube MMA (Matrix Multiply Accumulate, 矩阵乘加累加) 在 CPU 上只能用 FP32 `matmul` 代替，小 shape 逐元素门禁查 FLOAT8 E4M3 混合容差。整体门禁用 cosine (cosine similarity, 余弦相似度) 与模长比 (norm ratio, 欧氏范数比)。

## 三层对照

| 层 | 文件 | shape | 对照 | 尺子 |
|----|------|-------|------|------|
| 小 shape 精度 | `test_eagle_quant_block_sparse_attention_accuracy.py` + `eagle_quant_block_sparse_attention_accuracy_common.py` | Q/KV `[1,2,256,128]`（整 tile）与 `[1,2,257,128]`（$256+1$ 尾块） | DUT 与 golden 共用量化后的 Q/K/V 与 scale；CPU 上先反量化再按块做 `matmul` + masked softmax + `matmul`，最后一档落到 BF16 | 按计算 dtype FLOAT8 E4M3 查混合容差（$rtol=0.25$，$atol=0.0625$，匹配率 $\ge 0.99$，$\max\lvert a-g\rvert \le 4$）；cosine $\ge 0.99$；模长比 $\in [0.9,1.1]$ |
| 大 shape 精度 | `check_eagle_quant_block_sparse_attention_prod_accuracy.py` + `run_eagle_quant_block_sparse_attention_prod_accuracy.sh` | Q `[1,32,2304,128]`，KV `[1,4,30757,128]`；GQA (Grouped Query Attention, 分组查询注意力) $32/4$ | 同设备 NPU 上对未量化 BF16 做小算子拼接（按 Q tile 切开，禁止物化 $S\times S$）；无 CPU 逐步模拟 | 量化 DUT 对未量化对照不上逐元素硬门禁；打印 `max_abs`；cosine $\ge 0.99$；模长比 $\in [0.9,1.1]$ |
| 大 shape 性能 | `profile_eagle_quant_block_sparse_attention_prod.py` + `run_eagle_quant_block_sparse_attention_msprof_op.sh` | 与大 shape 精度同一条 Q/KV 张量维 | 先采 CANN (Compute Architecture for Neural Networks, 昇腾计算架构) 桥接 950 FP8 BSA (Block Sparse Attention, 块稀疏注意力)（`--kernel-name` `BlockSparseAttention`，Hadamard (Hadamard rotation, 哈达玛旋转) 后块量化 Q/K/V，`block_shape` `[128, 256]`），再采 mix `eagle_quant_block_sparse_attention`（`EagleQuantBlockSparseAttention`，`block_shape` `[128, 128]`）；`cann_fp8_bsa` 的 Duration 为基线 | `msprof op` Duration |

$257=2\times 128+1$，用来覆盖完整 128-token 稀疏块之外的尾块。Q 量化块为 64：256 为整块，257 为 $4\times 64+1$ 尾块。大 shape 的 KV 长 $30757=240\times 128+37$，同样带尾块。CANN FP8 BSA 的 BlockK 必须是 256 的倍数（tiling 约束），与 mix 路径共用同一份 128-token mask，再按 KV 维 any-merge 到 256。

没有同语义的 `torch_npu` 量化块稀疏算子时，大 shape 对照用 NPU 上的 `matmul` + `softmax` + `matmul`，不要退回 CPU。

## 文件

| 文件 | 用途 |
|------|------|
| `eagle_quant_block_sparse_attention_accuracy_common.py` | 造数、块稀疏 mask、mix 量化、CANN FP8 量化、CPU / NPU compose golden、混合容差、cosine / 模长比 |
| `test_eagle_quant_block_sparse_attention_accuracy.py` | pytest 小 shape；设备固定 `npu:0` |
| `check_eagle_quant_block_sparse_attention_prod_accuracy.py` | 大 shape argparse |
| `run_eagle_quant_block_sparse_attention_prod_accuracy.sh` | 选卡后跑上一行 |
| `profile_eagle_quant_block_sparse_attention_prod.py` | 大 shape 单实现、单次 forward，供 `msprof op`；`--impl cann_fp8_bsa` 或 `eqbsa` |
| `run_eagle_quant_block_sparse_attention_msprof_op.sh` | 选卡后先采 `cann_fp8_bsa`、再采 `eqbsa`；每个 impl 单独进程 |

## 怎么跑

在仓根、已 `source` CANN `set_env.sh` 且本仓 `mindiesd` 可 import 的 950 环境：

```bash
unset ASCEND_RT_VISIBLE_DEVICES

# 小 shape 精度（256 与 257；设备固定 npu:0）
pytest tests/ops/eagle_quant_block_sparse_attention/test_eagle_quant_block_sparse_attention_accuracy.py -s

# 大 shape 精度
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_prod_accuracy.sh
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_prod_accuracy.sh \
  --device-id 3

# 大 shape 性能（先 CANN FP8 BSA，再 mix eagle_quant_block_sparse_attention；各 impl 单独采）
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_msprof_op.sh
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_msprof_op.sh \
  --device-id 3 --impls cann_fp8_bsa,eqbsa
```

**不要**设置 `ASCEND_RT_VISIBLE_DEVICES`。小 shape pytest 固定 `npu:0`。大 shape 换卡用 `--device-id`（`npu-smi info` 的 NPU ID），Python 内 `torch.npu.set_device`。两条实现都传 `inner_precise=4`。对照算子 `--kernel-name` 为上板前缀 `BlockSparseAttention`（可用 `CANN_BSA_KERNEL_NAME` 覆盖；前缀匹配不到时可改成 `block_sparse`）；被测算子为 `EagleQuantBlockSparseAttention`（`EQBSA_KERNEL_NAME` / `KERNEL_NAME`）。默认采集顺序是 `cann_fp8_bsa` 再 `eqbsa`，每条单独起进程，避免记录混合。产物在 `--output` 的绝对路径下（默认 `logs/msprof_eagle_quant_block_sparse_attention_<stamp>/<impl>/`），以工具自己的 Operator Basic Information 为准。`cann_fp8_bsa` 的 Duration 是性能基线：

```text
speedup = Duration(cann_fp8_bsa) / Duration(eqbsa)
speedup_percent = (speedup - 1) * 100%
```
