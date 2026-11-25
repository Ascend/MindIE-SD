# SigmoidMulDynamicMxQuant 测试

输入 `x` 和 `gate` 为同形 BF16 矩阵。算子先计算 `x * sigmoid(gate)`，再对乘积做 MX 量化。输出为 E4M3FN 矩阵和按尾轴每 32 个元素共享的 E8M0 scale。torch_npu 以 uint8 承载 scale，形状为 `[rows, ceil(cols/64), 2]`。模式 0 用 FP32 完成 Sigmoid 和乘法，模式 1 将 Sigmoid 舍入为 BF16，再以 BF16 相乘。可选参数 `scale_alg` 默认为 0，按组内最大指数确定 scale。设为 1 时，根据组内最大绝对值与 E4M3FN 最大有限值 448 的比值向上选取 E8M0 scale。两种算法均支持两档门控精度。

## 精度

从装有本次 wheel 的环境运行。若源码目录也在 Python 搜索路径上，须确认 `mindiesd` 解析到包含本次插件和 OPP 的目录。需要先初始化设备的应用可在启动进程前设置 vendor 路径。

```bash
vendor="$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("mindiesd").origin).parent / "ops/vendors/aie_ascendc")')"
export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
```

在能解析 `tests` 包的仓根运行以下命令。

```bash
python3 -m pytest --test-mode=NPU tests/ops/sigmoid_mul_dynamic_mx_quant/test_sigmoid_mul_dynamic_mx_quant.py -s
bash tests/ops/sigmoid_mul_dynamic_mx_quant/run_sigmoid_mul_dynamic_mx_quant_prod_accuracy.sh --device-id 0
```

小 shape 对照独立的 CPU 分组量化实现，覆盖 32 元素分组、奇数 scale 补位、64/256 元素边界、非连续输入、空输入、默认精度、跨列分片及参数校验。随机输入按用例选择均匀或正态分布。每个输入分别采样正态分布参数，seed 和参数在日志中可见。E4M3FN 使用 0.125 的相对与绝对容差，匹配率至少 0.99，同时检查最大误差、余弦相似度和模长比。scale 按字节精确比较。边界用例用门控乘积 448 和 480 检查两种 scale 选择。

大 shape 使用 `[8068, 1536]`。模式 0 用正态分布，模式 1 用均匀分布。每档精度均运行两种 scale 算法，与相同精度、相同算法的 NPU Sigmoid、Mul、DynamicMxQuant 对照。大 shape 要求余弦相似度至少 0.99，模长比在 0.9 至 1.1 之间，记录 FP8 逐字节匹配率并要求 scale 完全相同。默认再对固定输入连续运行 1000 轮，每轮逐字节比较两项输出。使用 `--mode accuracy` 或 `--mode stress --stress-runs N` 可单独运行。

## 性能

```bash
bash tests/ops/sigmoid_mul_dynamic_mx_quant/run_sigmoid_mul_dynamic_mx_quant_msprof_op.sh --device-id 0
```

性能覆盖模式 1 的生产场景。两条路径使用相同 seed 生成的两份 BF16 正态输入，形状均为 `[8068, 1536]`。每轮复用同一份设备输入，CPU 造数和输入搬运在计时区间外。

- `kernel` 采集融合算子。msprof op 每轮预热 20 次，再采集一次 Task Duration。
- `native` 采集完整的 BF16 三算子链。Sigmoid 作用于 `gate`，结果与 `x` 做 BF16 乘法，乘积直接送入 DynamicMxQuant。msprof 每轮先运行整链 5 次，再用独立的 NPU Event 对 10 次完整调用逐次计时，取本轮最小值。

950 默认先完成融合算子的 3 轮，再完成三算子链的 3 轮。默认测 `scale_alg=0`。用 `--scale-alg 1` 在另一批次测量算法 1，融合和完整调用使用相同算法。每条路径分别取其 3 轮中的最小值。输出目录中的 `final_summary.txt` 和 `summary.json` 给出两种指标、各自的最短耗时、对应轮次及数据目录。

## 保存和续跑

大 shape 精度及性能入口接受 `--output DIR` 或 `--resume DIR`。小 shape 精度可用环境变量 `MX_SMALL_OUTPUT` 指定新批次，续跑时取消该变量并在 pytest 命令后加 `--resume DIR`。保持原设备和参数，续跑会验证计划、代码、软件版本和已完成结果。性能入口用 `--dry-run` 预览计划，用 `--path kernel` 或 `--path native` 选择单项，默认运行两项。`--path native --no-msprof` 直接采集整链 Event。每轮的原始 Event 样本保存在 `event_samples.csv`。若融合 kernel 的三轮耗时波动超过 10%，可在新目录提高其预热次数后复测。
