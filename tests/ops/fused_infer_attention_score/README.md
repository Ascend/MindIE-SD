# fused_infer_attention_score 测试

在 CANN + `torch_npu` + 本仓 `mindiesd` 环境里跑本算子的功能冒烟、arch35 tiling UT，以及单算子 `msprof op` 采集。

## 文件

| 文件 | 用途 |
|------|------|
| `profile_fia_dit_tiling512.py` | DiT Excel 场景，默认 `inner_precise=0` 的 Python 单次 / 采集 forward |
| `profile_fia_dit_tiling512_inner_precise4.py` | 同一场景，`inner_precise=4`（C8V16） |
| `run_fia_msprof_op.sh` | 选卡后先采 `inner_precise=0`，再采 `inner_precise=4` |
| `tests/tools/select_npu_device.py` | 仓内共用：优先无任务卡，否则 Util=0 且 HBM 最小 |
| `test_fused_infer_attention_score_v2.py` | pytest：API mock + NPU smoke + 小 shape 量化 FIA vs C8V16 CPU golden（默认 0 与 `inner_precise=4` 各一条） |
| `fia_accuracy_common.py` | snapshot 造数、C8V16 CPU golden、opbase 混合容差、cosine / max_abs / norm_ratio |
| `check_fia_dit_tiling512_accuracy.py` | 大 shape：FP8 FIA vs 未量化 `npu_fusion_attention`（默认 0 与 `inner_precise=4` 各一条） |
| `run_fia_dit_tiling512_accuracy.sh` | 选卡后跑上述大 shape 精度比较 |
| `run_fia_arch35_ut.sh` | 编译并跑 arch35 tiling C++ UT |

## 性能采集场景

对齐 `DiT-Prof.xlsx` sheet `0825-eaglefia-tiling512` 第 34 行。layout **BNSD**，QKV 为 FP8 E4M3FN per-block（`quant_mode=7/7/7`），Q block=128，KV block=256，输出 bf16。

| 张量 | shape | dtype |
|------|-------|-------|
| Q | `[1, 32, 2304, 128]` | float8_e4m3fn |
| K / V | `[1, 4, 30757, 128]` | float8_e4m3fn |
| attention_out | `[1, 32, 2304, 128]` | bfloat16 |

Excel Duration 参考值 **2314.363 µs**（`OpBasicInfo` / aicore_time）。KV 分块与表内 dump 不完全相同，该数字不作硬门禁。

`--kernel-name` 必须是上板前缀 `EagleFusedInferAttentionScore`，不要写成 Excel 类型名 `FusedInferAttentionScore`。

## 怎么跑

在仓根、已 `source` CANN `set_env.sh` 且本仓 `mindiesd` 可 import 的环境：

```bash
unset ASCEND_RT_VISIBLE_DEVICES

# 自动选卡（无任务卡优先；没有则选 Util=0 且 HBM 最小，并打屏提示）
# 同一条 sh 先采 inner_precise=0，再采 inner_precise=4
bash tests/ops/fused_infer_attention_score/run_fia_msprof_op.sh

# 指定 npu-smi 上的物理卡号
bash tests/ops/fused_infer_attention_score/run_fia_msprof_op.sh --device-id 3
```

产物默认在仓根：

- `inner_precise=0`：`logs/msprof_fia_tiling512_<时间戳>/`，路径写在 `logs/last_fia_msprof.path`
- `inner_precise=4`：`logs/msprof_fia_tiling512_<时间戳>_inner_precise4/`，路径写在 `logs/last_fia_msprof_inner_precise4.path`

看各自目录里 `OpBasicInfo.csv` / `PipeUtilization.csv` 的 FIA 行。

只跑 Python（不采 profile）时同样传物理卡号：

```bash
python3 tests/ops/fused_infer_attention_score/profile_fia_dit_tiling512.py --device-id 0
python3 tests/ops/fused_infer_attention_score/profile_fia_dit_tiling512_inner_precise4.py --device-id 0
```

**不要**设置 `ASCEND_RT_VISIBLE_DEVICES`。该变量与 `msprof` / `msprof op` 不兼容：脚本能跑在目标卡上，分析侧会找不到 `device_*`。换卡用 `--device-id`（`npu-smi info` 的 NPU ID），Python 内 `torch.npu.set_device`。

## 功能 / tiling UT

```bash
pytest tests/ops/fused_infer_attention_score/test_fused_infer_attention_score_v2.py
bash tests/ops/fused_infer_attention_score/run_fia_arch35_ut.sh
```

## 精度

被测一律是仓里 FP8 7/7/7 的 `fused_infer_attention_score_v2`。输入按 snapshot mean/std/clamp 生成，**enhance-mode=2.0**。

小 shape（pytest，需要 NPU）：GQA `Q[1,8,128,128]` / `KV[1,2,256,128]`。Q/K/V 先 `fa_block_quant_preprocess`（7/7/7）。默认 `inner_precise=0` 与 `inner_precise=4` 各一条。CPU golden 仿 C8V16 kernel 的向量侧（FP16 softmax、P 的 RNA、LastDiv），输出 BF16。逐元素对拍用 [opbase experimental_standard](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) 的混合容差，查 **FLOAT8 E4M3** 行（算子是 FP8 FullQuant，不是输出 BF16 的 $2^{-7}$ 双千分之五）：$|a-g|\le 0.0625+0.25\cdot|g|$，匹配率 $\ge 0.99$，且 $\max|a-g|\le \max(1,32\cdot 2^{-3})=4$。另外打印 cosine / max_abs / norm_ratio，硬门禁仍要求 cosine $\ge 0.99$ 且 norm_ratio $\in [0.9,1.1]$。

大 shape（NPU）：DiT tiling512 row34，`Q[1,32,2304,128]` / `KV[1,4,30757,128]`。对照是未量化 `torch_npu.npu_fusion_attention`，默认 `inner_precise=0` 与 `inner_precise=4` 各比一次。指标对齐 `FIA_c8v16_super_test`：cosine、max_abs_error、norm_ratio。cosine 看整体方向，对整体乘常数不敏感；norm_ratio 是 $\|FIA\|_2/\|ref\|_2$，用来抓缩放；max_abs_error 是逐元素最大绝对差。量化对未量化不用双千分之五。max_abs 只打印。硬门禁：cosine ≥ 0.99 且 norm_ratio ∈ [0.9, 1.1]。

```bash
unset ASCEND_RT_VISIBLE_DEVICES
bash tests/ops/fused_infer_attention_score/run_fia_dit_tiling512_accuracy.sh
bash tests/ops/fused_infer_attention_score/run_fia_dit_tiling512_accuracy.sh --device-id 3
python3 tests/ops/fused_infer_attention_score/check_fia_dit_tiling512_accuracy.py --device-id 7 --enhance-mode 2.0
```
