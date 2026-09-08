# quantization 测试

本目录包含量化模块单元测试，以及 `FP8RotateQuantFA` 大 shape `msprof op` 采集入口。`msprof op` 是 CANN 的单算子性能采集工具，用来量 FIA (Fused Infer Attention Score, 融合推理注意力) 上板 kernel 耗时。

## 单元测试

```bash
pytest tests/quantization -q
```

## FP8RotateQuantFA 大 shape `msprof op`

场景与 FIA 大 shape 精度一致（DiT-Prof.xlsx / 0825-eaglefia-tiling512 / 第 34 行）：GQA (Grouped Query Attention, 分组查询注意力)，Q 32 头、K/V 4 头。`FP8RotateQuantFA` 按 K 的头数向 FIA 传入 `num_key_value_heads`。

| 张量 | shape | dtype |
|------|-------|-------|
| Q | `[1,32,2304,128]` | BF16，层内量化为 FP8 E4M3 |
| K/V | `[1,4,30757,128]` | BF16，层内量化为 FP8 E4M3 |
| attention_out | `[1,32,2304,128]` | BF16 |

layout 为 `BNSD`。上板 kernel 前缀为 `EagleFusedInferAttentionScore`。

| `--mode` | 说明 |
|----------|------|
| `HIGH_PRECISION` | 原高精度路径，性能基线 |
| `C8V16_TILING512` | C8V16 + V tiling 512（`value_quant_mode=12`，V 块 512×64） |

### 文件

| 文件 | 用途 |
|------|------|
| `profile_fp8_rotate_quant_fa.py` | 按 `--mode` 构造一份输入，调用一次 `FP8RotateQuantFA` |
| `run_fp8_rotate_quant_fa_msprof_op.sh` | 先采 `HIGH_PRECISION`，再采 `C8V16_TILING512`；各起一个 `msprof op` |
| `../tools/select_npu_device.py` | 未指定卡号时选择空闲物理 NPU (Neural Processing Unit, 神经网络处理器) |

### 入口

```bash
unset ASCEND_RT_VISIBLE_DEVICES

bash tests/quantization/run_fp8_rotate_quant_fa_msprof_op.sh

bash tests/quantization/run_fp8_rotate_quant_fa_msprof_op.sh \
  --device-id 3 \
  --output-dir /data/fp8_rotate_quant_fa_msprof
```

默认 `warm-up=10`、`launch-count=5`。可用环境变量覆盖：

```bash
MSPROF_WARMUP=5 MSPROF_LAUNCH_COUNT=20 \
bash tests/quantization/run_fp8_rotate_quant_fa_msprof_op.sh --device-id 3
```

不要设置 `ASCEND_RT_VISIBLE_DEVICES`。脚本会清除该变量；未传 `--device-id` 时调用 `tests/tools/select_npu_device.py` 选物理卡，Python 内使用 `torch.npu.set_device(device_id)`。

### 产物

默认写到 `logs/msprof_fp8_rotate_quant_fa_<时间戳>/`。各模式子目录为 `HIGH_PRECISION` 与 `C8V16_TILING512`。以子目录中 `OpBasicInfo.csv` 的 FIA `Duration(us)` 为准。相对高精度加速比为：

```text
speedup = Duration(HIGH_PRECISION) / Duration(C8V16_TILING512)
```
