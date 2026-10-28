# eagle_indexed_gate_norm_modulate 测试

本目录直接调用注册算子 torch.ops.mindiesd.eagle_indexed_gate_norm_modulate，包含小 shape 精度、大 shape 精度与稳定性压测、单算子性能采集。运行前安装支持该算子的 MindIE-SD，并加载 CANN 环境。

该实现仅支持 Ascend 910B，采用单缓冲，要求 D≤7168。五份逐行输入均整行搬入，较大尺寸按 3072、2048 或 1024 元素容量分段进行 FP32 计算；branch/scale、residual/shift 分别复用一份 FP32 计算缓存。weight 每核加载并转 FP32 一次，后续行只读复用，updated 与归一化工作区整行驻留。indices 通过 GlobalTensor::GetValue 由 Scalar 直接读取 GM。默认保留 8 KiB API 空间和 256 B 系统空间，不添加禁用预留 UB 的编译选项。Host 按实际显式缓存检查容量，整行计算辅助区为 288 B、分段计算辅助区为 1056 B；D=7168 时显式分配 185376 B，距可用 188160 B 余 2784 B。

计算分段先确定段数，再将完整 128 元素块按商余数均分，非对齐尾部并入最后一段；各段最多相差 128 个有效元素。D≤3968 使用整行计算，D=3969～6144 使用 3072 容量，D=6145～6656 使用 2048 容量，D=6657～7168 使用 1024 容量，D>7168 拒绝。5376 分为 2688＋2688，6656 分为 四段 1664，7168 分为七段 1024。搬运长度与计算分段独立，每行五份输入各搬运一次；所有路径均无显式 MTE2 barrier。

## 输入与场景

平方和在整行加载路径采用两级 WholeReduceSum，在分段路径采用三级归约；倒均方根在 UB 内通过 Sqrt 与 Div 计算，避免基础 Rsqrt 的近似误差。随后由 Brcb 广播，再通过 Mul 完成归一化，数值路径不读取 Scalar 归约结果。分段路径将 weight 原始加载空间复用为 residual_out 暂存，最终 modulated_out 在 FP32 工作区原地收窄；需通过多行精度与稳定性测试检查这些空间的复用。索引读取及循环控制仍由 Scalar 执行。

residual、branch 为 [rows,D]，gate、shift、scale 为 [K,D]，weight 为 [D]，indices 为 [rows]。六个浮点输入共用 FP16 或 BF16，输出 dtype 独立选择 FP16 或 BF16，两个输出依次为 residual_out、modulated_out，均连续。indices 支持 int32 或 int64，元素必须处于 [0,K)。eps 默认 1e-6，计算中间值始终为 FP32，仅两个输出分别收窄；normalized 使用未收窄的 updated。

注册算子与 Python 接口的参数顺序为 residual、gate、branch、shift、scale、indices、weight、bias、norm_type、eps、output_dtype。weight 在接口中声明为可选且默认 None；当前仅支持 norm_type=0（带权重 RMSNorm），因此省略 weight 或传入 None 均报错，其他 norm_type 也报错。bias 为后续归一化模式预留，默认 None，当前传入任何 Tensor 均报错。norm_type 默认 0，output_dtype 默认沿用输入 dtype。调用时建议为可选参数指定名称：

```python
residual_out, modulated_out = torch.ops.mindiesd.eagle_indexed_gate_norm_modulate(
    residual, gate, branch, shift, scale, indices,
    weight=weight, bias=None, norm_type=0, eps=1e-6, output_dtype=torch.bfloat16,
)
```

小 shape 接口测试包含省略 eps/norm_type/output_dtype、默认或显式 bias=None 的有效调用，以及缺失 weight、非法 norm_type 和 Tensor bias 的拒绝检查。

| 层 | 场景 | 对照与门禁 |
|---|---|---|
| 小 shape | D=1、63、64、65、127、128、129、255、3968、3969、4095、4096、4097、5376、6144、6145、6655、6656、6657、7167、7168；rows=0、1、3、81；四种输入/输出 dtype 组合；两种索引 dtype；D=255、6656、6655、7167 非连续输入；D=7169 超限拒绝；默认输出 dtype、不同 eps 与非法接口检查 | CPU 按 FP32 数值阶段计算；逐输出混合容差、cosine 和模长比均为硬门禁 |
| 大 shape 精度 | (rows,K)=(13640,3) 或 (5672,12)，D=2048、5376、6656 或 7168；BF16→BF16 及两种跨 dtype 转换，连续 ND，int64 indices | 同设备 NPU 小算子拼接；cosine 和模长比为硬门禁，混合容差为打印的软约束 |
| 大 shape 压测 | 与大 shape 精度相同，每组固定输入，首次 baseline 之外默认重复 1000 次 | 两个输出逐位一致，包括正负零；核对输入未变；单独 stress 通过不代表 golden 精度通过 |
| 性能 | 八组大 shape，仅 BF16→BF16，只使用正态数据 | msprof op 采集 EagleIndexedGateNormModulate 前缀，指标为 Task Duration(us) |

小 shape 每组独立执行 uniform 和 normal 两条用例；大 shape 精度与压测各使用下表的 10 条，不再展开分布。uniform 使用 U[-5,5]；normal 对六个浮点输入逐个随机抽取 mean∈[-5,5]、std∈[0.1,2]，再生成整个张量，不钳位样本。seed=20261024，日志记录每个输入的实际参数及 shape、dtype、device、stride。索引采用 row%K，大 shape 覆盖首末索引与重复索引。性能各轮使用相同 seed、输入顺序和参数复现数据。

大 shape 包含 (rows,K)=(13640,3) 和 (5672,12)，hidden_size 均取 2048、5376、6656、7168，共 8 组 shape，覆盖四条计算路径；非对齐尾段由小 shape 覆盖。性能仅运行 BF16→BF16，共 8 个正态场景。精度与压测各选 10 条：八组 shape 均有 BF16→BF16 对照，另选两条跨 dtype 转换，正态和均匀各五条；FP16→FP16 及完整组合由小 shape 覆盖。

| rows | K | hidden_size | 输入→输出 | 分布 |
|---|---|---|---|---|
| 13640 | 3 | 2048 | BF16→BF16 | normal |
| 13640 | 3 | 5376 | BF16→BF16 | uniform |
| 13640 | 3 | 6656 | BF16→BF16 | normal |
| 13640 | 3 | 7168 | BF16→BF16 | uniform |
| 5672 | 12 | 2048 | BF16→BF16 | uniform |
| 5672 | 12 | 5376 | BF16→BF16 | normal |
| 5672 | 12 | 6656 | BF16→BF16 | uniform |
| 5672 | 12 | 7168 | BF16→BF16 | normal |
| 5672 | 12 | 6656 | FP16→BF16 | uniform |
| 13640 | 3 | 7168 | BF16→FP16 | normal |

两种入口均可用 `--rows 5672` 或 `--rows 13640` 筛选，对应 K 固定为 12 或 3。精度的尺寸和 dtype 参数只筛选上表，不生成额外组合；无匹配项时返回错误。默认 all 模式为 10 条精度加 10 条压测，每条压测默认重复 1000 次。

两个输出按实际输出格式选择验收精度档位。FP16 的 $atol=rtol=2^{-10}$，BF16 为 $2^{-7}$，匹配率要求 0.99，最大绝对误差上限为 $\max(0.1,32\times\mathrm{ULP})$。ULP (Unit in the Last Place, 末位单位精度) 使用最大误差位置的 golden 值在实际输出格式中的间距。cosine≥0.99，模长比∈[0.9,1.1]；两边全零时两项记为 1。大小 shape 的指标统一在 CPU FP64 计算；大 shape 仅将已完成的 DUT 和 NPU golden 输出搬到 CPU 统计，不在 CPU 计算 golden。日志分别标明每个门禁的结果及软硬性质。

压测在 NPU 上额外保留一份全部输入快照和两个独立输出 baseline，日志打印快照字节数；每轮输出在比较后释放，不累积保存。

## 运行精度与压测

在仓库根目录执行小 shape 测试，固定使用物理 0 卡；在导入 MindIE-SD 前选卡：

```bash
python3 -m pytest -q -s tests/ops/eagle_indexed_gate_norm_modulate/test_eagle_indexed_gate_norm_modulate_accuracy.py
```

大 shape 默认先完成全部单次精度，再执行每组 1000 次压测。sh 未指定设备时使用仓内 tests/tools/select_npu_device.py 选择空闲卡；所有 NPU 入口先校验可见设备变量与物理 ID，再选卡及导入注册包。不要设置 ASCEND_RT_VISIBLE_DEVICES。

```bash
bash tests/ops/eagle_indexed_gate_norm_modulate/run_eagle_indexed_gate_norm_modulate_prod_accuracy.sh --device-id 3
bash tests/ops/eagle_indexed_gate_norm_modulate/run_eagle_indexed_gate_norm_modulate_prod_accuracy.sh --device-id 3 --mode accuracy
bash tests/ops/eagle_indexed_gate_norm_modulate/run_eagle_indexed_gate_norm_modulate_prod_accuracy.sh --device-id 3 --mode stress --stress-runs 1000 --hidden-size 7168 --input-dtype bf16 --output-dtype fp16
```

全部所选项目成功后末尾逐项打印 PASS，压测行包含实际重复次数；任一失败非零退出。默认日志写入仓根 log/eagle_indexed_gate_norm_modulate/accuracy_时间戳/accuracy.log，支持通过 `--output-dir DIR` 改写。

## 运行性能采集

唯一性能入口默认运行八个 BF16→BF16 fused 场景，支持 `--path all` 或 `--path fused`，可通过尺寸参数筛选。性能脚本的输入与输出 dtype 参数仅接受 bf16；大 shape 精度参数筛选上述十条用例，小 shape 覆盖全部 dtype 组合。kernel 前缀可用 `--kernel-name PREFIX` 覆盖，以实际编译产物为准。

```bash
bash tests/ops/eagle_indexed_gate_norm_modulate/run_eagle_indexed_gate_norm_modulate_msprof_op.sh --device-id 3
bash tests/ops/eagle_indexed_gate_norm_modulate/run_eagle_indexed_gate_norm_modulate_msprof_op.sh --device-id 3 --path fused --hidden-size 2048 --input-dtype bf16 --output-dtype bf16
bash tests/ops/eagle_indexed_gate_norm_modulate/run_eagle_indexed_gate_norm_modulate_msprof_op.sh --device-id 3 --hidden-size 5376
bash tests/ops/eagle_indexed_gate_norm_modulate/run_eagle_indexed_gate_norm_modulate_msprof_op.sh --device-id 3 --hidden-size 6656
bash tests/ops/eagle_indexed_gate_norm_modulate/run_eagle_indexed_gate_norm_modulate_msprof_op.sh --device-id 3 --hidden-size 7168
```

默认每轮 `msprof op --warm-up=20`，独立采集 20 轮，每个 Python 进程仅调用一次目标算子，工具采用 `--launch-count=1`。可用 MSPROF_WARMUP 和 TEST_COUNT 覆盖次数。命令参数语义见 [msOpProf 官方参考](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md#命令参考)。

默认产物位于仓根 log/eagle_indexed_gate_norm_modulate/perf_时间戳/，每场景保存 run_NN/msprof.log 和工具分析文件；深入分析使用各轮工具导出的指标 CSV。采集日志必须恰好含一个有效且为正的 Task Duration(us)，缺失、多条或采集失败均立即退出。成功后连续展示全部逐轮明细及各场景最小值，保存相同内容到 task_duration_summary.txt 与 final_summary.txt，明细包含绝对 DataDir。

最终值取各场景 20 轮的最小耗时，表示当前条件下观测到的最佳性能。汇总保留最大值与波动比例 (max-min)/min；严格超过 10% 时，考虑将预热增至 50 或 100，在相同卡号与输入配置下换新目录完整重测 20 轮。频率尚未充分提升只是可能原因；重测后仍波动时保留样本与警告，不能认为取最小值消除了波动。
