# QuantBatchMatmulV3 + SwiGLU + DynamicMxQuant

CATLASS Python-DSL fused operator (Ascend 950 / DAV_3510): a quantized matmul
with MXFP8/MXFP4 inputs, followed by SwiGLU and Dynamic MX quantization to
MXFP8/MXFP4.

## Semantics

```text
C32 = dequant(A) @ dequant(B)                 # fp32 accumulate, E8M0 per 32 elements
S   = silu(C32[:, :N/2]) * C32[:, N/2:]       # FP32 activation (matches the official fused operator)
S16 = round_bf16_rne(S)                       # BF16 intermediate
Y, SY = dynamic_mx_quant(S16)                 # 32-element blocks, MXFP8 or MXFP4 output
```

The activation consumes the FP32 accumulator and is computed in FP32; only the
quantization input is BF16. Scales are E8M0.

## Tensor Contract

All tensors are byte tensors (the benchmark/ATK dtype set has no fp8/fp4 entry).

| Name | FP8 mode | FP4 mode |
| --- | --- | --- |
| `a` | `uint8 [M, K]` E4M3FN | `uint8 [M, K/2]` E2M1 packed along K |
| `scale_a` | `uint8 [M, K/64, 2]` E8M0 | same |
| `b` | `uint8 [K, N]` E4M3FN | `uint8 [N, K/2]`, logical `[K, N]` column-major |
| `scale_b` | `uint8 [K/64, N, 2]` E8M0 | same |
| `in_fp4` / `out_fp4` | `int` 0/1 selecting the input/output format | |
| `y` | `uint8 [M, N/2]` (out_fp4=0) or `[M, N/4]` (out_fp4=1) | |
| `scale_y` | `uint8 [M, N/128, 2]` E8M0 | |

Constraints: `K % 64 == 0`, `N % 128 == 0`. `N` is the matmul output width;
SwiGLU halves it, and every 32 outputs share one E8M0 scale, so each row
contains `N/64` scale bytes, arranged in `N/128` pairs.

Before allocating outputs or launching the kernel, the entry point validates
that all four inputs are contiguous `uint8` byte tensors on the same device,
that their shapes match the FP8/FP4 layout, and that `in_fp4` / `out_fp4` are
0 or 1. It raises `ValueError` on any violation.

## Usage

```python
from mindiesd.layers.qbmm_swiglu_mxquant import run

y, sy = run(a_npu, scale_a_npu, b_npu, scale_b_npu, in_fp4=0, out_fp4=0)
```

Requires the CANN CATLASS Python DSL (`import catlass.tla`) and `torch_npu`.

The CPU golden lives with the operator tests in
`tests/ops/qbmm_swiglu_mxquant/reference.py`. Its `matmul_bf16=False` mode is the
fused/official semantics; `matmul_bf16=True` is the small-op stitching semantics
(QBMM writes a BF16 intermediate first).

## Implementation Notes

- Tail handling follows the CATLASS `mx_matmul` and official
  `QuantMatmulActivationQuant` pattern: only the valid M/N extent is copied,
  the K tail is zero-filled on chip, and the E8M0 K-group count is padded to a
  multiple of the tile on the host with the neutral value 127 (`fill` only
  supports 0, which is not the E8M0 neutral value).
- Small M uses `TILE_M=16/32` to reduce pad-row cube/fixpipe work.
- An optional two-kernel split-K path (`_ENABLE_SPLITK`, off by default) targets
  long-K shapes with M<=32, K>=2048, and N_out<=1024; the default is the fused
  single kernel.

## Validation

| Item | Result |
| --- | --- |
| 200-case generalization sweep | 200/200, min y match ratio 0.999783, min scale match ratio 1.0 |
| ATK 4 cases | 4/4, 100%, Pass |
| ATK 500-case generalization | 500/500, Pass |
| 60-case mxFP8 comparison with official `quant_matmul_activation_quant` | 60/60, cross y/scale match ratio 1.0 |

Performance (Ascend950DT, msprof kernel):

| Case | Kernel | vs. small-op stitching baseline |
| --- | ---: | ---: |
| model1 FP8->FP8 (13640x5376x28672) | 5458.8 us | 1.359x |
| model2 FP8->FP8 (56240x5376x28672) | 23270.3 us | 1.303x |
| model1 FP4->FP4 | 3241.3 us | 2.288x |

200-case generalization kernel median: 12.62 us; small/medium/large buckets:
9.62 / 19.64 / 5458.0 us.
