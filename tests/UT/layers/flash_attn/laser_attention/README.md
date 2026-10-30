# LaserAttention layer 对照测试

本目录测试完整 layer：从原始输入进入函数，到返回最终输出，包含 dtype 转换、布局转换、补齐、连续化、LA 算子、输出裁剪和布局恢复。输入创建和传入设备在计时外。当前路径直接调用生产代码 `AscendLaserAttention.forward_attn_bnsd`；旧路径保存在 `LegacyLaserAttention`，对应提交 `07b8d842` 的转换行为。生产 layer 不为本测试增加分支。

## 场景与计时边界

业务输入采用 BSND（批、序列、头数、头维）布局，Q 为 `[1,5280,24,128]`，K/V 为 `[1,8800,24,128]`，三者均为 BF16。输出为 `[1,5280,24,128]`、BF16。`mask=None`，`scale=128**-0.5`，当前 layer 的非空 mask 行为不在本测试范围内。

| 路径 | 计时内输入转换 | LA 算子输出 | 计时内输出转换 |
|------|----------------|-------------|----------------|
| current | V：BF16 → FP16；Q/K 保持 BF16 | BF16 | 无 dtype 转换 |
| legacy | Q/K/V：BF16 → FP16 | FP32 | FP32 → FP16 → BF16 |

两条路径都在 layer 内转为 BNSD，Q 序列补齐至 5376，K/V 补齐至 8960，传入 `pre_tokens=160` 屏蔽补齐的 KV。最终恢复有效长度和 BSND。禁止在计时外先转换 QKV，也不对返回值额外调用 `contiguous()`。

造数使用固定 seed `20260921`，按 50% 均匀分布 `U[-5,5]` 与 50% 正态分布混合；正态均值取 `[-5,5]`、标准差取 `[0.1,2]`。两条路径使用相同业务输入，独立采集进程使用相同 seed 重建输入。

## 执行

在已编译当前算子、plugin，且可导入本仓 `mindiesd` 的 910B / 910_93 环境，从仓根运行：

```bash
# 自动选卡；先 current 20 轮，再 legacy 20 轮。
bash tests/layers/flash_attn/laser_attention/run_laser_attention_layer_msprof.sh

# 指定物理卡、每轮正式计时次数、每路径独立采集轮数和空输出目录。
bash tests/layers/flash_attn/laser_attention/run_laser_attention_layer_msprof.sh \
  --device-id 3 --iterations 10 --runs 20 --output /tmp/la_layer_comparison

# 直接运行 Python，保持相同预热、计时次数、轮数和场景顺序。
bash tests/layers/flash_attn/laser_attention/run_laser_attention_layer_msprof.sh \
  --device-id 3 --no-msprof --output /tmp/la_layer_direct
```

每轮对应一个独立的 msprof 被测进程，只运行一条路径。固定手动预热 **5 次**，随后正式调用默认 **10 次**，每次分别用 NPU Event 测量完整 layer，取这 10 个耗时的最小值作为本轮结果。先完成 current 的全部轮次，再完成 legacy 的全部轮次，默认共 40 次采集。没有额外的无 profiler 对照进程。

`--iterations` 控制每轮的正式计时次数，`--runs` 控制每条路径的独立采集轮数，默认值分别为 10 和 20、必须为正整数。预热固定为 5，不提供 `--warmup` 参数。

`--no-msprof` 将每轮替换为直接启动 Python，不调用或要求安装 msprof，也不检查 PROF、timeline 或工具导出的 CSV。默认模式仍仅执行 msprof 采集，不额外插入直接运行。

sh 引用仓内 `tests/tools/select_npu_device.py`，只选卡一次，全部进程固定使用同一物理 ID；采集前清除 `ASCEND_RT_VISIBLE_DEVICES`。Python 在导入 MindIE-SD / layer helper 前设置目标设备。输出目录转为绝对路径，已有非空目录会被拒绝。`PYTHON` 环境变量可指定解释器。

## Event 计时

为每次正式调用预先创建独立的 start/end Event，预热前各记录一次以初始化底层资源。预热后依次提交“start[i] → 完整 layer 调用 → end[i]”，全部正式调用提交完毕后，再遍历 Event 对读取各次耗时。提交循环内不读取耗时或显式等待，避免每次下发都等待上一调用完成。各调用仍在同一当前流中按序执行；跨流业务须保证所有被测任务依赖汇合到对应结束 Event 之前。

`start.elapsed_time(end) * 1000` 将毫秒转换为微秒。此协议依赖目标 torch_npu 的 `elapsed_time()` 内部等待两个事件完成后读取耗时，见[官方实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUEvent.cpp)；不额外添加显式 synchronize、CPU 墙钟或范围标记。输入准备在计时外，最后一次输出的 shape、dtype 和有限值检查在全部计时完成后进行。

每轮结果是单次完整 layer Event 耗时的最小值，不是 10 次总耗时除以 10，也不来自 CSV 的 LA Task Duration。Event 包含完整 layer 的设备任务及可能的等待间隙，复用同一输入形成热缓存场景；汇总明确标识是否带 msprof。

采集命令使用 `msprof`，不使用 `msprof op` 的 kernel 重放参数：

```bash
msprof --output=/absolute/path/current --task-time=on \
  --ascendcl=on --runtime-api=on \
  python3 tests/layers/flash_attn/laser_attention/profile_laser_attention_layer.py \
  --device-id 3 --path current --iterations 10 --msprof-mode \
  --output /absolute/path/current/profiled_timing.json
```

## 汇总与原始产物

终端与 `summary.txt` 的逐轮表格列出以下四列，先完成仅转换 V 的场景，再完成转换 QKV 和输出的场景：

| 列 | 含义 |
|------|------|
| Scenario | 含输入 cast、LA 输出类型和输出 cast 的完整场景名称 |
| Test | 该路径第几轮独立采集 |
| Layer Duration(us) | 本轮正式 Event 样本的最小值，完整 layer 耗时 |
| DataDir | msprof 模式为对应 `PROF_*` 绝对路径；直接模式为该轮结果目录 |

场景名称为 `CastV(BF16->FP16)+LA(BF16-out)`（current）与 `CastQKV(BF16->FP16)+LA(FP32-out)+CastOut(FP32->FP16->BF16)`（legacy）。二者均接收、返回 BF16，并包含布局转换、补齐和输出裁剪；名称展示两者的 cast 差异。

逐轮表格后，每个场景各打印一行最终性能结果，取该场景全部组耗时的最小值。部分机型在不同测试中可能出现耗时增加，因此以此最小值作为当前测试条件下观测到的最佳性能，不代表波动已消除。`summary.json` 的 `best_results` 保存最小值、场景名及对应轮次、数据目录；`rows` 保留全部逐轮结果。覆盖 `--runs` 时按实际组数汇总。

每轮原始产物位于 `run_NN/current/` 或 `run_NN/legacy/`，含 `msprof.log`、`profiled_timing.json`、`PROF_*`。计时 JSON 保存 `warmup=5`、`iterations`、`samples`、全部 `event_samples_us` 和最小值 `event_us`，以及进程、设备、版本和输入配置。汇总 JSON 保留各轮全部 Event 样本及 `profile_dir`。

直接模式改为保存 `timing.log`、`timing.json`，`profiled=false`，不生成 PROF 目录。两种模式均保存 `event_samples.csv`，按执行顺序逐条列出 `Path,Sample,Layer Duration(us)`，默认每轮 10 条正式样本（不含预热）。汇总 JSON 的 `data_dir` 和 `timing_file` 指向本轮数据；直接模式 `profile_dir` 为 null。

汇总器只读取 Python 计时 JSON，检查模式、样本数、有限正耗时、最小值和配置一致性，不读取 CSV 耗时。msprof 模式额外定位唯一的 `PROF_*` 目录，sh 检查非空 `op_summary*.csv` 与 `msprof_*.json`，供查看原始任务和 timeline。每个进程默认执行 5 次预热和 10 次正式 layer 调用；初始化、传输、预热和输出检查也可能出现在采集文件中，不能全表求和当成 layer 耗时。

`environment.log` 及各轮 `device_before.log` / `device_after.log` 保存工具和设备快照，查询不支持时保留告警。这些是进程外快照，不能代表 kernel 运行期间的频率。多轮取最小值用于形成约定的性能结果，不代表已解决跨启动波动。

已有相同协议的采集结果可以重新汇总：

直接模式的结果需在以下命令后追加 `--no-msprof`，不能混用两种模式的数据。

```bash
python3 tests/layers/flash_attn/laser_attention/summarize_laser_attention_layer.py \
  --output /tmp/la_layer_comparison --runs 20
```

## 验证

```bash
# 无 torch/NPU 依赖的协议和汇总测试。
python3 -m unittest discover -s tests/layers/flash_attn/laser_attention -p test_measurement_protocol.py -v
python3 -m unittest discover -s tests/layers/flash_attn/laser_attention -p test_run_summary.py -v

# CPU 契约与真实 NPU 数值测试。
MINDIE_TEST_MODE=CPU pytest tests/layers/flash_attn/laser_attention -q
MINDIE_TEST_MODE=ALL DEVICE_ID=3 pytest \
  tests/layers/flash_attn/laser_attention/test_laser_attention_layer.py -q -s
```

契约测试覆盖 FP16/BF16、BSND/BNSD、KV 补齐 0/1/128/255、D=64 补齐、dtype 传递和输出裁剪，并用双重舍入样例锁定旧路径。NPU 数值测试对新旧完整 layer 分别与 FAS（Fused Attention Score，融合注意力计算）比较，使用余弦相似度 ≥0.99、模长比 `[0.9,1.1]`，不要求新旧路径逐位相同。性能脚本只检查输出 shape、dtype 和有限值；精度测试独立运行。
