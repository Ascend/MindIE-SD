# 950PR&950DT products sparse_block_estimate tests

The operator returns an `int8` mask and an `int32` sparse count. The mask gate is a business ruler for that integer output: `diff_ratio < 0.02`, `omitted_ratio < 0.01`, plus count-table consistency and zero padding. It is not a floating-point mixed-tolerance check.

Q tile is 256 and K tile is 512. Layout is BNSD. Compute dtypes are FP16 and BF16. The production shape is Q `[1,32,2304,128]`, K `[1,4,30757,128]`.

| Layer | Entry | Shape | Golden | Gate |
|-------|-------|-------|--------|------|
| Small-shape accuracy | `test_sparse_block_estimate_accuracy.py` on `npu:0` | `smoke_align` Q256/K2048, `smoke` Q256/K2049, `smoke_q_tail` Q257/K2049, `multitile` Q512/K4097 | CPU matmul + FP32 softmax + block sum | mask diff / omitted ratio |
| Large-shape accuracy | `run_sparse_block_estimate_prod_accuracy.sh` | production shape above | same formula with NPU matmul and softmax | same mask gate |
| Large-shape performance | `run_sparse_block_estimate_msprof_op.sh` | production Q `[1,32,2304,128]`; K seq 30757, 65536 (64K), 425984 (416K); BF16 | none | three concatenation wall-time runs, then three `msprof op` runs, kernel prefix `SparseBlockEstimate` |

A single accuracy run that does not pass `--data` draws 50% `U[-5,5]` and 50% normal values (`mu` in `[-5,5]`, `sigma` in `[0.1,2]`), seed `2026`. Passing `--data` uses that snapshot. Do not set `ASCEND_RT_VISIBLE_DEVICES`. Omit `--device-id` and the shell selects an idle card with `tests/tools/select_npu_device.py`. Shell entry points are only the two below: large-shape accuracy and large-shape performance. Small shape is pytest on `npu:0`.

Small-shape pytest:

```bash
pytest tests/ops/sparse_block_estimate/test_sparse_block_estimate_accuracy.py -s
```

Large-shape accuracy:

```bash
bash tests/ops/sparse_block_estimate/run_sparse_block_estimate_prod_accuracy.sh
bash tests/ops/sparse_block_estimate/run_sparse_block_estimate_prod_accuracy.sh --device-id 4
```

Large-shape performance. One BF16 dtype, six runs. The shell first times the small-op concatenation (`torch_npu_baseline.py` via `run_sbe_performance.py`) at K = 30757, 65536 and 425984, then runs `msprof op` on the fused estimate kernel at those same lengths. Q stays `[1,32,2304,128]`. Concatenation prints a synchronized wall-time median. Estimate forwards once per process; `msprof op` owns warm-up and launch count.

```bash
bash tests/ops/sparse_block_estimate/run_sparse_block_estimate_msprof_op.sh
bash tests/ops/sparse_block_estimate/run_sparse_block_estimate_msprof_op.sh --device-id 4
```

CPU-only golden smoke, including the aligned tile and the Q/K tails:

```bash
python tests/ops/sparse_block_estimate/run_a5_correctness.py \
  --reference-only --preset smoke_q_tail --dtype both
```
