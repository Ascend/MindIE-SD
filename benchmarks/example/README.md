# Example（样例脚本）

核心算子（FA / MM / GMM / BSA）benchmark 的样例脚本，参数由 `mindie_bench` 的 `--op` / `--config` 内联表达。

## `run_benchmark.sh`

```bash
# 序列扫描（FA/MM/GMM/BSA 四算子，1k→256k）+ HTML 报告
bash benchmarks/example/run_benchmark.sh

# 自定义报告目录
bash benchmarks/example/run_benchmark.sh /tmp/rep
```

脚本三段：

1. **模型负载**（默认注释）：Wan2.2 / MiniMax-H3 / 图片模型的 packed 序列命令，去掉 `#` 启用
2. **序列扫描**（默认启用）：一条命令四算子，seqlen 共享扫描轴
3. **HTML 报告**：`mindie_bench report` 合并生成

## 常用变体

```bash
# 单算子参数扫描
python benchmarks/scripts/mindie_bench.py run --op "{fa: {}}" \
    --config "{seqlen: [1024, 4096, 16384, 65536, 262144], dtype: [bf16, mxfp8], timeout: 300, peak_flops: 377.78}" \
    --report-dir reports/fa_sweep

# BSA 稀疏度扫描
python benchmarks/scripts/mindie_bench.py run --op "{bsa: {}}" \
    --config "{seqlen: [8192, 32768, 131072], dtype: [bf16], sparse: [0.6, 0.8, 0.95, 0.99], timeout: 300, peak_flops: 377.78}" \
    --report-dir reports/bsa_sparse

# MM 量化档位
python benchmarks/scripts/mindie_bench.py run --op "{mm: {}}" \
    --config "{seqlen: [4096, 65536], quant: [NO_QUANT, W8A8, W8A8_MXFP8, W4A4_MXFP4], timeout: 300, peak_flops: 377.78}" \
    --report-dir reports/mm_quant

# 多 run 合并为单个 HTML（各 run 独立目录，report 指向父目录）
python benchmarks/scripts/mindie_bench.py report --report-dir reports --baseline-dir baselines
```

## 注意

- **峰值必填**：`--config {..., peak_flops: <实测>, peak_bw: <实测>}`——代码不内置峰值，MFU/MBU 以输入峰值为分母；示例值 377.78 为 A310 类 CUBE 峰值（425/9×8），`peak_bw` 缺失时 MBU 显示 n/a
- **长序列必配 `timeout`**（`--config {..., timeout: 300}`）：默认 5s 会跳过大档位
- **`--op` / `--config` 值必须加双引号**（bash / PowerShell 均需）：不带引号时 bash 会对含逗号的 `{...}` 做花括号展开/分词，把参数拆碎导致 argparse 报错
- **复现**：`--config {..., seed: 42}` 固定输入张量
- 完整 CLI 说明见 `benchmarks/README.md`
