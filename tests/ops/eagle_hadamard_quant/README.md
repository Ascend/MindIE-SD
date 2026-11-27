# EagleHadamardQuant 测试

从仓库根目录运行。设备测试需要在 Ascend 950 环境安装匹配的 OPP 和 PyTorch 插件。

```bash
python -m pytest -q --test-mode=CPU tests/UT/ops_reference/test_eagle_hadamard_quant_small_shape.py
MINDIE_TEST_DEVICE_ID=5 python -m pytest -q --test-mode=NPU tests/ops/eagle_hadamard_quant/test_eagle_hadamard_quant.py
```

950 小 shape 用例通过真实算子检查 MXFP4、MXFP8 E4M3 及 E8M0 scale 的形状、类型和全部字节。
覆盖空输入、共享和逐块权重、ND/NZ、Hadamard 和置换矩阵。步长为 2 的输入和权重覆盖 K32/K64 分组条件。
图捕获分别对两种格式执行 fullgraph compile 和 export，同一张图运行 3 行、5 行输入。

原有九组生产配置和运行命令见[测试报告](eagle_hadamard_quant_test_report.md)。
共享 ND 对照使用相同种子生成的 BF16 激活和一份 K128 权重。
两份 OPP 使用同一份内核源码。关闭复用时在构建前设置 `MINDIESD_EAGLE_REUSE_L0B=0`，开启复用时使用默认值。
运行时分别将 `ASCEND_CUSTOM_OPP_PATH` 和 `ASCEND_CACHE_PATH` 指向相应的 OPP 和独立缓存。
先在关闭复用版本保存两个输出，再在开启版本读取并逐字节比较。两次 profiling 分别指定独立输出目录。

```bash
cd tests/ops/eagle_hadamard_quant
bash run_eagle_hadamard_quant_prod_accuracy.sh --case shared-k128-fp4 --mode accuracy --device-id 5 --save-outputs /tmp/eagle-off
bash run_eagle_hadamard_quant_prod_accuracy.sh --case shared-k128-fp4 --mode accuracy --device-id 5 --compare-outputs /tmp/eagle-off
bash run_eagle_hadamard_quant_msprof_op.sh --case shared-k128-fp4 --device-id 5 --repeats 10 --output /tmp/eagle-profile
bash run_eagle_hadamard_quant_msprof_op.sh --case shared-k128-fp4 --device-id 5 --metrics MemoryL0 --repeats 1 --output /tmp/eagle-pmu
```
