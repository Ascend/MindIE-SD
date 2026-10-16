# fused_infer_attention_score 测试

本目录按“小 shape CPU Golden、大 shape NPU Golden、大 shape `msprof op`”三层验证 FIA FP8 量化路径。四条被测路径共用 `fia_quant_common.py`，防止精度与性能脚本的 quant mode、block 或 `inner_precise` 不一致。

## 被测路径

| 路径 | Q/K/V quant mode | `inner_precise` | V 量化块 | 说明 |
|------|------------------|-----------------|----------|------|
| `original` | `7/7/7` | `0` | `256x128` | K256/V256 原始路径，性能基线 |
| `c8v16` | `7/7/7` | `4` | `256x128` | K256/V256 C8V16 |
| `v512` | `7/7/11` | `4` | `512x128` | K256/V512x128 C8V16 |
| `v512_d64` | `7/7/12` | `4` | `512x64` | K256/V512x64 C8V16 |

V512 两条路径必须使用 `inner_precise=4`。

## 文件

| 文件 | 用途 |
|------|------|
| `fia_quant_common.py` | 四条路径的 quant mode、token/channel block 和 `inner_precise` 唯一定义 |
| `fia_accuracy_common.py` | snapshot 造数、C8V16 CPU Golden、FP8 混合容差、cosine/max_abs/norm_ratio |
| `test_fused_infer_attention_score_v2.py` | API 冒烟和小 shape 四路径精度 pytest |
| `check_fia_dit_accuracy.py` | 大 shape 四路径对比 `torch_npu.npu_fusion_attention` |
| `run_fia_dit_accuracy.sh` | 选物理卡并启动大 shape 精度测试 |
| `profile_fia_dit_quant.py` | 单路径、单次 forward，供 `msprof op` 采集 |
| `run_fia_msprof_op.sh` | 四条路径分别启动 `msprof op` |
| `run_fia_arch35_ut.sh` | arch35 tiling C++ UT |
| `tests/tools/select_npu_device.py` | 精度和性能脚本共用的物理 NPU 选卡工具 |

原有 `profile_fia_dit.py` 和 `profile_fia_dit_inner_precise4.py` 保留为 PR550 的单路径入口；四条量化路径测试统一使用 `profile_fia_dit_quant.py`。

## 第一层：小 shape CPU Golden

- layout：BNSD
- Q：`[1,8,128,128]`
- K/V：`[1,2,513,128]`
- 量化前：BF16；量化后：FP8 E4M3；输出：BF16
- `513` 同时覆盖完整 V512 tile 和尾块
- DUT 与 Golden 共用量化后的 Q/K/V 和 scale
- CPU Golden 按 C8V16 的 FP16 online-softmax、P 的 FP8 RNA 和 LastDiv 阶段计算

FP8 Cube MMA 无法由 CPU FP32 `matmul` 位级复刻，因此逐元素门禁使用 FLOAT8 E4M3 混合容差：`rtol=0.25`、`atol=0.0625`、匹配率至少 `0.99`、全张量最大绝对误差不超过 `4`。同时要求 cosine ≥ `0.99`、norm_ratio ∈ `[0.9,1.1]`。

```bash
unset ASCEND_RT_VISIBLE_DEVICES
pytest \
  tests/ops/fused_infer_attention_score/test_fused_infer_attention_score_v2.py::test_fused_infer_attention_score_v2_fp8_small_vs_cpu_golden \
  -s
```

## 第二层：大 shape NPU Golden

与 DiT 生产大 shape 一致：

| 张量 | shape | 输入 dtype |
|------|-------|------------|
| Q | `[1,32,2304,128]` | BF16，随后量化为 FP8 E4M3 |
| K/V | `[1,4,30757,128]` | BF16，随后按各路径量化为 FP8 E4M3 |
| attention_out | `[1,32,2304,128]` | BF16 |

Golden 是同一份未量化 BF16 输入上的 `torch_npu.npu_fusion_attention`。量化 DUT 对未量化 Golden 不使用逐元素容差作为硬门禁，只打印 max_abs；硬门禁为 cosine ≥ `0.99`、norm_ratio ∈ `[0.9,1.1]`。

```bash
unset ASCEND_RT_VISIBLE_DEVICES

# 自动选空闲物理卡，默认测试全部四条路径
bash tests/ops/fused_infer_attention_score/run_fia_dit_accuracy.sh

# 指定物理卡；额外 Python 参数放在 -- 后
bash tests/ops/fused_infer_attention_score/run_fia_dit_accuracy.sh \
  --device-id 3 -- --paths original,c8v16,v512,v512_d64 --enhance-mode 2.0
```

## 第三层：大 shape `msprof op`

性能使用与大 shape 精度完全相同的 shape、dtype、造数和路径参数。上板 kernel 前缀为 `EagleFusedInferAttentionScore`。每条路径单独启动一个 Python 进程和一个 `msprof op`，避免记录混合；`original` 的 Duration 是性能基线。

```bash
unset ASCEND_RT_VISIBLE_DEVICES

bash tests/ops/fused_infer_attention_score/run_fia_msprof_op.sh \
  --device-id 3 \
  --output-dir /data/fia_msprof_v_quant \
  --paths original,c8v16,v512,v512_d64
```

默认 `warm-up=10`、`launch-count=5`。也可通过环境变量覆盖：

```bash
MSPROF_WARMUP=5 MSPROF_LAUNCH_COUNT=20 \
bash tests/ops/fused_infer_attention_score/run_fia_msprof_op.sh --device-id 3
```

产物分别位于 `<output-dir>/original`、`c8v16`、`v512`、`v512_d64`。以各目录中 `OpBasicInfo.csv` 的 FIA `Duration(us)` 为准。相对原始路径加速比为：

```text
speedup = Duration(original) / Duration(path)
speedup_percent = (speedup - 1) * 100%
```

## 选卡约束

不要设置 `ASCEND_RT_VISIBLE_DEVICES`。两个 shell 入口会清除此变量，并在未传 `--device-id` 时调用 `tests/tools/select_npu_device.py` 选择物理卡；Python 内使用 `torch.npu.set_device(device_id)`。
