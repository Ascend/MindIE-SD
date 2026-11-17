# QuantBatchMatmulV3 + SwiGLU + DynamicMxQuant

CATLASS Python-DSL 融合算子（Ascend 950 / DAV_3510）：MXFP8/MXFP4 输入的量化
matmul，接 SwiGLU，再输出 Dynamic MX 量化的 MXFP8/MXFP4。

## 语义

```text
C32 = dequant(A) @ dequant(B)                 # fp32 累加，MX 逐 32 元素共享 E8M0
S   = silu(C32[:, :N/2]) * C32[:, N/2:]       # FP32 激活（与官方融合算子一致）
S16 = round_bf16_rne(S)                       # BF16 中间类型
Y, SY = dynamic_mx_quant(S16)                 # 32 元素一块，输出 MXFP8 或 MXFP4
```

激活输入为 FP32 累加结果，激活计算 FP32，量化输入 BF16；scale 为 E8M0。

## 张量契约

所有张量均为 byte tensor（benchmark/ATK 的 dtype 集合没有 fp8/fp4 项）。

| 名称 | FP8 模式 | FP4 模式 |
| --- | --- | --- |
| `a` | `uint8 [M, K]` E4M3FN | `uint8 [M, K/2]` E2M1 沿 K 打包 |
| `scale_a` | `uint8 [M, K/64, 2]` E8M0 | 同左 |
| `b` | `uint8 [K, N]` E4M3FN | `uint8 [N, K/2]`，逻辑 `[K, N]` 列主序 |
| `scale_b` | `uint8 [K/64, N, 2]` E8M0 | 同左 |
| `in_fp4` / `out_fp4` | `int` 0/1 选择输入/输出格式 | |
| `y` | `uint8 [M, N/2]`（out_fp4=0）或 `[M, N/4]`（out_fp4=1） | |
| `scale_y` | `uint8 [M, N/128, 2]` E8M0 | |

约束：`K % 64 == 0`，`N % 128 == 0`。`N` 是 matmul 输出宽度；SwiGLU 将宽度减半，
随后每 32 个输出共享一个 E8M0 scale，因此每行有 `N/64` 个 scale 字节，每两个
字节一对，共 `N/128` 对。

入口会在分配输出和启动 kernel 之前集中校验：四个输入必须为同设备的 `uint8`
连续 byte tensor，shape 必须与 FP8/FP4 各自的布局一致，且 `in_fp4` / `out_fp4`
只能取 0 或 1；不满足时直接抛出 `ValueError`。

## 使用

```python
from mindiesd.layers.qbmm_swiglu_mxquant import run

y, sy = run(a_npu, scale_a_npu, b_npu, scale_b_npu, in_fp4=0, out_fp4=0)
```

需要 CANN CATLASS Python DSL（`import catlass.tla`）与 `torch_npu`。

CPU golden 位于 `tests/ops/qbmm_swiglu_mxquant/reference.py`，随算子测试一起维护。
其 `matmul_bf16=False` 为融合/官方语义，`matmul_bf16=True` 为小算子拼接
（QBMM 先写 BF16）语义。

## 实现要点

- 参照 CATLASS `mx_matmul` 与官方 `QuantMatmulActivationQuant` 的尾块模式：
  M/N 只拷贝有效 extent，K 尾块在片内 `fill(0)`；E8M0 scale 的 K 组数在 host
  补齐为 127（`fill` 只支持 0，且 0 不是 E8M0 的中性值）。
- 小 M 使用 `TILE_M=16/32`，减少 pad 行的 cube/fixpipe 工作量。
- 代码内置可选的两 kernel split-K 路径（`_ENABLE_SPLITK`，默认关闭），用于
  M≤32、K≥2048、N_out≤1024 的长 K 链形状；默认走融合单 kernel。

## 验证

| 项 | 结果 |
| --- | --- |
| 200 例泛化 sweep | 200/200 通过，y 匹配率 min 0.999783，scale 匹配率 min 1.0 |
| ATK 4 例 | 4/4，100%，Pass |
| ATK 500 例泛化 | 500/500，Pass |
| 与官方 `quant_matmul_activation_quant` 60 例 mxFP8 对比 | 60/60，cross y/scale 匹配率 1.0 |

性能（Ascend950DT，msprof kernel）：

| 用例 | kernel | 对小算子拼接基线 |
| --- | ---: | ---: |
| model1 FP8→FP8（13640×5376×28672） | 5458.8 us | 1.359x |
| model2 FP8→FP8（56240×5376×28672） | 23270.3 us | 1.303x |
| model1 FP4→FP4 | 3241.3 us | 2.288x |

200 例泛化 kernel 中位 12.62 us；small/medium/large 分档中位 9.62 / 19.64 / 5458.0 us。
