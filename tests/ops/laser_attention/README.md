# laser_attention 单算子测试

本目录在 910B / 910_93 上直接测试 `torch.ops.mindiesd.la`，包含小 shape 精度、大 shape 精度和大 shape `msprof op`。输入从造数开始就是算子接收的连续 BNSD (Batch, Num heads, Seq, Dim，批、头、序列、头维) 张量。测试不调用业务 layer，也不对被测输入做转置、补齐或 dtype 转换；向 NPU 传输时保持 shape、dtype 和元素值不变。

| 路径 | Q/K dtype | V dtype | tiling key | 原始输出 dtype |
|------|-----------|---------|------------|----------------|
| fp16 | FP16 | FP16 | 0 | FP32 |
| native-bf16 | BF16 | FP16 | 1 | BF16 |

Q、K、V 分别按上表的目标 dtype 生成。大小 shape 的每组 shape / dtype / 属性都分别执行 `uniform` 与 `normal` 两条独立精度用例，各占 50%。`uniform` 的 Q/K/V 全部从 `U[-5,5]` 采样；`normal` 对 Q/K/V 分别随机采样均值 `[-5,5]`、标准差 `[0.1,2]` 后生成完整张量，不钳位正态样本。seed 默认为 `20260921`，日志打印分布、seed 和各输入实际正态参数。小 shape 共 8 条 NPU 精度用例，大 shape 共 4 条；逐条判定门禁，不合并指标。性能测试保留单个 Tensor 内逐元素 50/50 混合造数。输出直接按算子返回的 BNSD shape 和 dtype 检查、比较，不裁剪、不转换成业务存储 dtype。

## 场景与对照

| 层 | 实际 Q shape | 实际 K/V shape | 属性与对照 |
|----|--------------|----------------|------------|
| 小 shape 全序列 | `[1,2,256,128]` | `[1,2,512,128]` | 两种 dtype 路径；`pre_tokens=2147483647`，全部 KV 有效；CPU 按 kernel 数值阶段模拟 |
| 小 shape 尾部屏蔽 | `[1,2,512,128]` | `[1,2,768,128]` | 两种 dtype 路径；显式 `pre_tokens=255`，屏蔽最后 255 个 KV 位置，有效 KV 长度 513；CPU 按 kernel 数值阶段模拟 |
| 大 shape 精度 | `[1,24,5376,128]` | `[1,24,8960,128]` | 两种 dtype 路径；`pre_tokens=2147483647`，全部 KV 有效；NPU FP32 分块组合 golden |
| 大 shape 性能 | `[1,24,5376,128]` | `[1,24,8960,128]` | shape、dtype、属性与大 shape 精度一致，造数保留逐元素 50/50 混合；采集 `LaserAttention` 的 Duration |

序列长度必须是 256 的倍数，头维必须是 128 的倍数。命令行指定非对齐尺寸时直接报错，不自动修正。小 shape 的尾部屏蔽通过算子属性表达，Q/K/V 自始至终保持上表的合法物理 shape；不借助 layer 将非法 shape 补齐后伪装成单算子尾块测试。

两档 golden 消费被测算子的同一份 Q/K/V 元素值和屏蔽属性，最后落到算子原始输出 dtype。小 shape 的 CPU golden 按 `CubeForward.h` / `VectorForward.h` 模拟：QK 用 FP32 matmul 近似 Cube 累加后落 FP16，scale 转 FP16，乘 scale、减行最大值、exp 各阶段落 FP16；屏蔽位置填为内核使用的 -60000。指数值按 128 列块补零、折半相加，每次相加落 FP16，行和落 FP16 后升 FP32；未归一化的 FP16 指数值与 FP16 V 做 FP32 PV，最后乘 FP32 行和倒数。CPU golden 限于 KV 不超过 4096 的单 section，覆盖本目录两组小 shape；Cube 累加、exp 和 vcadd 内部归约仍是 CPU 近似，不保证逐位一致。

大 shape 保持 NPU FP32 `Q @ K.T → softmax → P @ V` 组合 golden。native-bf16 使用 BF16 Q/K 和 FP16 V，NPU 组合 golden 分别将其无损提升到 FP32，避免为统一参考算子的输入 dtype 而再次舍入 V。大 shape 按 128 个 Q 行分块控制分数矩阵占用，全部计算留在 NPU；不调用 CPU golden。

小 shape 的硬门禁为按 Q/K 计算 dtype 查表的混合容差、余弦相似度不低于 0.99、模长比处于 `[0.9,1.1]`。大 shape 的精度硬门禁仅为上述余弦相似度与模长比；逐元素匹配率和最大绝对误差作为软约束，继续计算并打印阈值、实测值及通过状态，未达标不导致整体测试失败。shape、输出 dtype 和 NaN/Inf 检查仍保留。

混合容差按 [opbase 远端标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)（2026-10-09 核对）：FP16 的 atol/rtol 均为 `2**-10`，BF16 均为 `2**-7`，匹配率至少 99%；两者最大绝对误差兜底值均为 0.1。最大绝对误差阈值取 `max(兜底值, 32 * ULP(g_low))`，其中 `g_low` 是最大有限绝对误差点的 golden 按实际输出 dtype 收窄后的值，ULP 随该值域计算。FP16 路径输出 FP32，因此 ULP 使用 FP32 格式；BF16 路径使用 BF16 格式。最大误差并列时取展平后的首个点。大 shape 的 FP32 组合 golden 与 kernel 的 FP16 中间阶段仍存在差异，整体通过不表示混合容差达标。两条路径均比较完整原始输出。

日志在调用前从实际张量读取 Q/K/V 的 shape、dtype、device、stride 和连续性，同时打印 layout、head_num、scale、pre_tokens、next_tokens、is_highPrecision 和预期输出状态。覆盖命令行参数后，日志仍对应实际输入；不使用固定业务 shape 作为场景标识。大 shape 的 5376 / 8960 均为有效长度，不能将其结果解释为带补齐屏蔽的 5280 / 8800 场景。

A5 / 950 上该算子不可用，本目录 NPU 用例会 skip 或直接退出。

## 文件与执行

| 文件 | 用途 |
|------|------|
| `laser_attention_accuracy_common.py` | BNSD 造数、输入检查与日志、直接算子调用、CPU / NPU golden、精度门禁 |
| `test_laser_attention_accuracy.py` | 八条小 shape NPU 精度用例（固定 `npu:0`），每条均调用真实 LA 算子 |
| `check_laser_attention_prod_accuracy.py` | 大 shape 精度，依次验证 fp16 与 native-bf16 |
| `run_laser_attention_prod_accuracy.sh` | 自动选卡并启动大 shape 精度 |
| `profile_laser_attention_prod.py` | 一次进程直接调用一次算子，shape、dtype、属性与大 shape 精度一致，造数保留逐元素混合 |
| `run_laser_attention_msprof_op.sh` | 自动选卡并分别采集两条 dtype 路径 |

在仓根、已加载 CANN 环境且本仓 `mindiesd` 可导入时运行：

```bash
unset ASCEND_RT_VISIBLE_DEVICES
pytest tests/ops/laser_attention/test_laser_attention_accuracy.py -s
bash tests/ops/laser_attention/run_laser_attention_prod_accuracy.sh
bash tests/ops/laser_attention/run_laser_attention_msprof_op.sh
bash tests/ops/laser_attention/run_laser_attention_msprof_op.sh --path fp16
bash tests/ops/laser_attention/run_laser_attention_msprof_op.sh --path native-bf16
```

两个 shell 入口均支持 `--device-id 3` 指定物理卡号；未指定时使用仓根 `tests/tools/select_npu_device.py` 自动选卡。Python 内调用 `torch.npu.set_device`，不要设置 `ASCEND_RT_VISIBLE_DEVICES`。shape 等参数通过 `-- --query-seq-len 5376 --kv-seq-len 8960` 传给 Python，数值表示实际算子维度。

`--kernel-name` 默认 `LaserAttention`。若工具报 `No profiling data matched`，用 `KERNEL_NAME=laser_attention` 重试。默认两条路径的性能产物分别写入输出目录的 `fp16/` 与 `native_bf16/`；只采一条时直接写入该输出目录。读取工具 Operator Basic Information 中目标 kernel 的 Duration。`--path cast-fp16` 仅作为 `--path fp16` 的兼容别名，实际造数、日志和产物目录均使用 fp16。

性能入口在同一张选定的卡上，对每条 dtype 路径独立启动 20 次 `msprof op`，每次默认 `--warm-up=100`、`--launch-count=1`。可用 `MSPROF_WARMUP` 覆盖预热次数；`MSPROF_LAUNCH_COUNT` 控制单次采集上限，测试轮数固定为 20。每条路径目录下的 `run_01/` 至 `run_20/` 分别保存采集产物和完整回显 `msprof.log`。脚本从每次回显提取 `Task Duration(us)`，全部完成后按路径、测试序号逐条打屏，并保存到输出根目录的 `task_duration_summary.txt`。默认两条路径共 40 个结果，指定单条路径时为 20 个结果。采集失败、耗时缺失或出现多条耗时时，脚本报错退出，可查看对应轮次的日志。

部分机型在不同测试中可能出现耗时增加；逐轮表格后按场景分别打印全部 20 组的最小耗时作为最终性能结果，代表该实现在当前测试条件下观测到的最佳性能，不代表波动已消除。预热仍默认 100 次。
