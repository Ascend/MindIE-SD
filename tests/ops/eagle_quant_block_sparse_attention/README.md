# EQBSA qmode1 测试

测试直接调用注册算子 `torch.ops.mindiesd.eagle_quant_block_sparse_attention`。
需要 Linux、Ascend 950、与当前源码匹配的 CANN / torch_npu / MindIE-SD 构建。
本目录只覆盖 qmode1（INT8 Q/K、FP8 E4M3 V），不测试业务 layer。

## 场景及输入

- 小 shape 共 68 条，包含 58 条数值场景、2 条真实算子非法 LSE flag 检查，以及
  BNSD/TND 各 4 条注册算子 FakeTensor 检查。FakeTensor 对 0/1 核对 O、LSE 的元数据，对 -1/2 核对报错。
  覆盖 BNSD/TND、D=64/128、O=FP16/BF16、256/257 尾块、KV=512+1、
  多 batch/GQA、dense、全空/部分空稀疏行及零 logits。
  每条数值场景同时调用 LSE=0/1，按字节比较 O（含正负零）。
  TND 多 batch 使用实际长度 Q=[257,256]、KV=[513,512]，检查非对齐前序 batch 的 scale 偏移；
  长度列表是各 batch 长度，不是累计长度。
  增加 Q 尾行 1/31/32/33/63/64/65/95/96/97/127/128、空尾块、
  D=64 下四轮 KV tile 的流水回绕，以及 64 个 Q head 下空行/非空行交替的多任务覆盖。
  KV=497/511 覆盖补齐到 512 列的 Softmax 尾块，分别使用负 logits 回归种子和零 logits。
  两种布局各覆盖 3 组稀疏 Q 块高与 Q 长度组合，即 (32,128)、(96,225)、(65,137)。
  偶数量化组的 Q 源数据乘以 1/8，分别检验从组内起步、跨组和尾行的 O 与 LSE。
- 大 shape：Q=[1,32,2304,128]，K/V=[1,4,30757,128]，BNSD，O=BF16。
  LSE=0/1 各有 uniform/normal，共 4 条精度及压测场景。
  默认性能测试采集同一组 normal 输入上的 CANN FP8 BSA、EQBSA LSE=0、EQBSA LSE=1。
- EQBSA 默认稀疏块为 [128,128]，Q scale 寻址回归使用上述三种块高。inner_precise=4，scale_value=1/sqrt(D)。
  Q/K 每 64 token 做对称 absmax/127、RNE INT8 编码；V 按通道 absmax/448、
  RNE FP8 E4M3 编码，以 INT8 位模式存储。三个 scale 均为 FP32。
  全零量化分组使用 scale=1，编码值为 0。
  O 保持 Q 的 shape；LSE 的 shape 为 Q.shape[:-1]+[1]，dtype=FP32。
- BF16 源数据在目标物理 layout 上直接生成，随后在 CPU 完成上述输入编码。
  整个 uniform 用例采用 U[-5,5]；normal 用例为 Q/K/V 分别抽取
  μ~U[-5,5]、σ~U[0.1,2]，生成整个张量，不钳位正态样本。
  固定 seed（默认 20260811）复现参数和位模式，日志记录每个输入实际 μ/σ。
  稀疏 mask 约半数块可见，按每个 batch 的实际长度强制首末有效 KV 块和末有效 Q 块连通；
  超出该 batch 有效块数的物理 mask 行、列必须为 0，dense 等特殊用例也仅在有效区域生成。
  例如 TND KV=[513,512] 的 mask 有 5 列，但第二个 batch 的第 5 列必须为 0；
  Mask2Idx 会扫描整个物理 mask，非零填充列会产生越界的 KV 索引。
- 调用日志从实际 Tensor 读取 shape/dtype/device/stride/contiguous，并记录全部语义属性。
  无隐藏 pad/transpose/cast；reference 内部的逻辑视图不改变 DUT 输入。

## Golden 和判定

小 shape CPU golden 按源码数值阶段模拟：
INT32 QK → Fixpipe scale + FP16 → FP16 Q/K scale 乘积及反量化 →
FP16 max/exp/DN32 分组归约 → FP32 在线累计和；
O 分支再模拟 FP8 P 的位级 RNA 编码、FP32 PV/在线 rescale、V scale/448 和最终输出转换。
LSE 用最终 gm+log(gl)。空稀疏行期望 O=0、LSE=-inf；
零 logits 另外对照独立解析结果 log(有效 key 数)。

大小 shape 的参考均先按 Arch3510 QUANT_PRE 编码处理 Fixpipe 系数：
FP32 位模式只保留 bits 13..31，低 13 位无效，即 1 位符号、8 位指数、10 位尾数。
例如 D=128 时，传入约 0.0883883476，实际参与计算的系数为 0.08837890625；
D=64 的 0.125 可精确表示。Q/K scale 乘积的 FP16 转换不采用这条位截断规则。
依据见 [Ascend C 随路量化的 Arch3510 编码表](https://asc.gitcode.com/api/SIMD-API/c_api/cube_datamove/cube_store_key_features/accompanying_quantization.html)。
此修正仅使参考对齐硬件系数，不修改 DUT 输入、kernel 或精度门禁。

Fixpipe DEQF16 使用饱和 FP16 转换，而不是普通 IEEE 溢出得到 Inf 的 cast。
接口说明见 [Ascend 950 asc_copy_l0c2ub](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/C-API/cube_datamove/asc_copy_l0c2ub.md)，
饱和模式含义见 [CANN 环境变量参考](https://www.hiascend.com/doc_center/source/zh/canncommercial/700/reference/envvar/CANN%207.0.0%20%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E5%8F%82%E8%80%83%2001.pdf)。
CPU 采用软件 exp/log 模拟数值阶段，不声称复刻设备指令每一个 bit。

大 shape 的 golden 全部在同一张 NPU 上拼接计算：
同一份实际 INT8 Q/K → FP32 matmul → Fixpipe 饱和/FP16 logits/反量化 →
FP32 masked softmax、FP32 PV、logsumexp。
它保留 QK 的约定低精度阶段，P 和归约使用高精度对照，未回退 CPU attention。

| 输出 | 验收档位及原因 | atol=rtol | 最大误差兜底 |
| --- | --- | --- | --- |
| O | FP8 E4M3：P 再量化及 FP8 PV 路径 | 2^-3 | 1 |
| LSE | FP16：logits、exp、局部归约舍入；不经过 P/V 量化 | 2^-10 | 0.1 |
| 零 logits 解析 LSE | FP32 log(整数计数) | 2^-13 | 0.01 |

逐输出匹配率要求 0.99；最大误差上限为 max(兜底,32×ULP)。
ULP 取最大有限误差点 golden 收窄到**实际输出格式**后在该值域的间距。
cosine 在 FP64 统计，要求 ≥0.99，模长比要求 [0.9,1.1]。
小 shape 三项均为硬门禁；大 shape 匹配率/最大误差为明确标注的软约束，
cosine/模长比和原始输出契约是硬门禁。NaN 不通过，合法同号 -inf 单独校验。
每个输出分别记录阈值、实测值、最大误差坐标、ULP 和各门禁状态。

LSE=0 的返回 LSE 缓冲区未写入：检查其数量/shape/dtype/连续性，
数值及压测 baseline 只包含 O；LSE=1 同时验收和逐位压测 O、LSE。
这不是 LSE=0 数值正确性的声明。

本轮 kernel 复核同时修正了旧路径中影响上述验收的错误：TND 的 O 行间距、
TND scale 的逐 batch 向上取整偏移和 K scale head stride、D=64 的 rescale 列数，
以及空稀疏行 O 未写入的问题。这些修复在两个 LSE 分支一致生效；
开关逐位对比针对修复后的同一输入，不能将旧路径的错误或未初始化输出作为正确基线。
原有 D=128 的三个 rescale VF 算术函数保持不变；性能仍以实机对比为准。

KV 尾块在反量化后、Max/Exp 之前将无效 DN32 lane 选为 `-inf`，避免原始 `-60000`
填充值乘 scale 后高于真实负 logits，污染 gm/gl。该修复同时用于 LSE=0/1。
有 padding 的 tile 使用带编译期 `HAS_PADDING` 的通用 VF，包括补齐到 512 列的尾块；
无 padding 的通用 VF 裁剪掩码逻辑，完整 512 列仍使用原有 fused VF。
不增加 UB 分区、同步事件或 pingpong 缓冲；尾块路径的性能需重新上板测量。

LSE 仅在最后一次 rescale 中计算。BNSD 读取尾块 Softmax 暂存的连续 FP16 行最大值，
并从相邻重复的 FP32 gl 提取每行的和。TND 读取当前 AIV 的 gm[4*i]（FP16）和 gl[2*i]（FP32）。
TND 写回偏移为 `(此前 batch 的 Q token 总数 + q) * N + h`；
BNSD 写回偏移为 `(b * N + h) * S + q`，均以 FP32 元素计。
LSE UB 位于字节偏移 243200，最大占用 2048 字节，不与 O、gm/gl 或 scale 重叠。
单份缓冲区使用原 MTE3_V(EVENT_ID4) 保护复用、V_MTE3(EVENT_ID1) 保护写出，
最终退出仍等待 EVENT_ID4。关闭模式下 kernel 内 LSE 的 UB 绑定、计算、
有效地址构造和拷贝受 if constexpr 裁剪；接口仍返回前述未写入的 LSE 输出缓冲区。

## 入口

在仓根执行以下命令。两个 sh 也可以从任意目录按路径调用。
性能批次由 `run_eagle_quant_block_sparse_attention_msprof_op.sh` 调度，负责选卡、逐轮采集、续跑和汇总。
每次采集由 `msprof op` 调用 `profile_eagle_quant_block_sparse_attention_prod.py` 执行一次前向。
先 source 正确的 CANN 环境，或通过 ASCEND_ENV_SH 显式指定 set_env.sh。

```bash
# 仅真实 NPU 小用例；仓库 pytest 默认 CPU 模式会排除 tests/ops
python -m pytest --test-mode=NPU tests/ops/eagle_quant_block_sparse_attention/test_eagle_quant_block_sparse_attention.py -s

# 预览不访问 NPU，不产生正式结果；--help 列出完整默认 tuple 和可复制筛选参数
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_prod_accuracy.sh --dry-run --device-id 1 3
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_msprof_op.sh --help

# 全部大精度通过后，每场景固定输入重复 1000 次；baseline 不计入
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_prod_accuracy.sh --device-id 1 3 --mode all

# 可以只选一个阶段或直接按实际参数筛选
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_prod_accuracy.sh --device-id 3 --mode accuracy --softmax-lse-flag 1 --distribution normal
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_prod_accuracy.sh --device-id 3 --mode stress --stress-runs 1000

# 性能固定一张卡，对比 CANN 与 EQBSA，并测量 EQBSA 的 LSE 开销
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_msprof_op.sh --device-id 3 --impls cann_fp8_bsa,eqbsa
# 只采 EQBSA 开关对照，可附加 --softmax-lse-flag 0 或 1
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_msprof_op.sh --device-id 3 --impls eqbsa
```

小用例通过公共 device_support 选择可见逻辑卡，默认选 0，可由 MINDIE_TEST_DEVICE_ID 指定。
支持 ASCEND_RT_VISIBLE_DEVICES 映射，批次记录主机、对应板卡和芯片身份。
已支持设备上缺依赖/执行失败不得变为 skip。
大精度支持多个互异物理卡号，每卡一个进程，按完整场景分配；父进程不初始化 NPU。
同一场景本次的精度、golden、压测使用同一卡。
新批次未指定卡时，调度入口调用仓内 `tests/tools/select_npu_device.py` 自动选一张卡。
Python worker 先 set_device，再导入 MindIE-SD。sh 清除 ASCEND_RT_VISIBLE_DEVICES；
独立大精度和性能 Python 入口发现该变量非空会报错。性能只接受一张卡。

本目录没有独立 CPU UT；golden/造数/门禁是辅助函数。
仓库 `tests/scripts/run_suites.py --npu_only` 只收集唯一 test 文件，
不会自动启动独立大精度、1000 轮压测或 msprof。
pytest 通过不代表另外三种执行已通过。

## 性能口径

默认生产 shape 支持三条采样路径，使用相同 seed、BF16 原始 Q/K/V 和起始稀疏 mask。
EQBSA 的开、关两路使用相同的编码输入，目标 kernel 为 `EagleQuantBlockSparseAttention`。
CANN 基线适用于 B=1、BNSD、BF16 输出，复用原有的旋转、FP8 量化及 `BlockSparseAttention` 路径。
它将相邻两列 [128,128] 稀疏块取并集，生成 [128,256] 的 mask。默认 shape 的 241 列变为
121 列，因此 CANN 与 EQBSA 实际可见的 KV token 数可能不同。
每轮在输入准备后同步，以目标 kernel 名及 `--launch-count=1 --kill=off` 采集一次前向。

两边的计划样本齐全后，汇总分别报告以下指标。

- 跨实现加速比为 CANN 最小耗时 / EQBSA 关闭 LSE 的最小耗时。
- EQBSA 的 LSE 开销为开启减去关闭的最小耗时差，以及该差值占关闭耗时的百分比。

Ascend 950 上每个场景默认 **3 轮独立采集**（`--test-count 3`）。Attention 预热按 max(Q,KV) 分档：
<30000 为 20 次、30000～49999 为 2 次、≥50000 为 1 次；
当前 30757 对应 **2 次**。`--warm-up` 可覆盖，包括 0；`--test-count` 可指定任意正整数轮数。
全部采集完成后集中连续打印逐轮 Task Duration 表，再打印各场景最小值。
同时保存最大值与 (max-min)/min×100%；严格 >10% 时提示考虑增加预热，
固定卡及输入配置，在新目录按所选 `--test-count` 完整复测。最小值仅是当前条件下观测的最佳值，
不代表消除了波动；没有实测数据时不作性能结论。

## 产物、中断及续跑

默认在仓根 `logs/eagle_quant_block_sparse_attention/accuracy_时间戳/` 或
`perf_时间戳/` 新建独立批次，重名加唯一后缀；显式 `--output-dir` 拒绝覆盖旧目录。
计划记录源码/算子产物摘要、软件版本、场景/seed/门禁/目标轮数；
执行分段记录物理卡、主机、设备稳定标识和场景分配。
输入参数及位模式摘要、baseline、失败 Tensor、日志均保存在同一批次内。

每个小用例/大精度场景/压测轮次/性能轮次完成校验后立即 fsync 并原子提交；
排他锁及采集进程组记录阻止另一个执行者接管活任务。
Ctrl+C、SIGTERM、可处理错误会停止子任务，非零退出并打印/保存部分汇总，
零结果也保留表头。SIGKILL 无法当场汇总，下次从已提交记录恢复。
提交前被终止的单元可重做；已提交单元（包括 FAIL）不重测、不覆盖。

```bash
python -m pytest --test-mode=NPU tests/ops/eagle_quant_block_sparse_attention/test_eagle_quant_block_sparse_attention.py --resume /absolute/old/accuracy_batch -s
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_prod_accuracy.sh --resume /absolute/old/accuracy_batch
# 大精度/压测允许显式换卡，原输入/baseline/历史设备记录保留
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_prod_accuracy.sh --resume /absolute/old/accuracy_batch --device-id 1 5
bash tests/ops/eagle_quant_block_sparse_attention/run_eagle_quant_block_sparse_attention_msprof_op.sh --resume /absolute/old/perf_batch
```

未指定参数从旧计划恢复；冲突参数、损坏记录/产物、源码/软件/算子变更均拒绝合并。
性能恢复原物理卡和设备身份，小用例核对逻辑卡和对应主机、板卡、芯片身份。大精度未指定卡时沿用上一分段卡列表。
已完成批次只重建汇总，不重复 DUT。小用例续跑请单独指定本算子的 test 文件；
全仓 pytest 的公共 NPU 可用性检查仍会执行。

大精度日志为 accuracy.log；压测在首轮、每 10 轮、末轮，以及轮末距上次报告
≥30 秒时刷新已提交进度。每个长阶段提前提示，计时拆成 DUT+同步、校验、提交主机时间，
不把它们当作 kernel 性能。没有单轮内部的定时心跳。
baseline 全部有效输出独立保存，恢复时核对 baseline 和重建输入摘要；
每轮均检查输出位模式及输入未改变。跨进程续跑是累计稳定性，不称单进程连续压测。

性能各轮/重试目录为 attempts/<场景_轮次_唯一后缀>/，保留 msprof.log、
OpBasicInfo*.csv、dump 数据及摘要；仅唯一有效 Task Duration 和目标产物校验通过才提交。
所有入口保存 final_summary.txt；性能另存 task_duration_summary.txt。
日志/目录存在或子进程返回 0 本身不算采集成功。

## 本地验证边界

本次在 Windows 上完成语法、静态规则、CPU 造数/参考预检及调度状态验证。
当前可隔离收集 68 条小用例。58 条数值场景通过 CPU 造数和参考输出契约检查，
其中原有 52 条的输入位模式与参考输出均与本次检视前一致。注册 FakeTensor 在 CPU 调度验证中覆盖 8 个组合。
尾块修复的 CPU 分阶段复现精确重现服务器失败坐标的 O 和 LSE；屏蔽无效 logits 后
该 Q 分块与原 golden 逐位一致，并枚举检查了 KV tile 长度 1～512 的 DN32 掩码边界。
另将实际 epilogue 源码放入宿主 API 逻辑模型，检查 24576 次调用，覆盖 1～128 行、
D=64/128、BNSD/TND、空行及多轮 rescale，验证写回边界、gm/gl 只读和开关 O 一致。
额外注入 gm/gl 步长、TND 连续写回、旧 D=64 VF、空行不写 O 五类错误，模型均检出。
该模型不模拟设备 FP16/BF16 舍入或异步执行时序，不是 CANN 编译或 NPU 精度验收。
本地调度替身覆盖 pytest 中断/续跑/完成后不重跑，性能 30→40 轮接续、零结果中断、
强制终止后保留 10 条提交记录，以及压测原 baseline 恢复、换卡分段、损坏产物和 FAIL 拒绝覆盖。
替身验证没有运行真实算子，不构成精度通过或性能测量；Linux 多进程及信号回收仍待实机检查。
没有 Ascend 950 / CANN / torch_npu：真实 NPU pytest、大 shape 精度、
1000 轮稳定性、非 0 卡进程归属和 msprof 采集仍需上板验收。
