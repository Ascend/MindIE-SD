# Eagle QuantBatchMatmul Swiglu 测试

eagle_quant_batch_matmul_swiglu 融合量化批矩阵乘法、FP32 反量化与 SwiGLU（Swish Gated Linear Unit，Swish 门控线性单元）。将反量化结果沿 N 维拆成左右两半 A、B，输出为 [batch, M, N/2]，batch 可省略。计算直到最终输出转换前均保持 FP32：

$$
C = \operatorname{Cast}_{\mathrm{BF16/FP16}}\left(\frac{A}{1+\exp(-A)}\cdot B\right).
$$

源码来自 ops-nn 提交 a5a8fbf3e02d7632ffa2c56b29e8755f1aa62720 的 QuantBatchMatmulV3，迁移 910B kernel、tiling 和 shape 推导。注册名为 EagleQuantBatchMatmulSwiglu。plugin 对 ND 权重调用自动生成的 aclnnEagleQuantBatchMatmulSwiglu，对 NZ 权重调用 aclnnEagleQuantBatchMatmulSwigluWeightNz；后者按原生 WeightNz 方式整理物理 NZ shape 与逻辑 ND shape，通过 L0 launcher 启动同一算子。两个入口的参数顺序一致，更新时须一起重新编译安装算子库与 plugin。

## 接口与运行条件

在 Linux Ascend 910B 环境使用与所迁移源码兼容的 CANN 9 系列 SDK、torch、torch_npu 和已编译安装的 MindIE-SD。当前只完成本地静态检查，具体 SDK 版本的编译兼容性及上板数值、性能待验证。构建过滤排除 CANN 8.x，不代表所有 9.x 版本均已验证。

plugin 接受 INT8 x1、x2，维数为 2～6；x1 使用 ND，x2 可使用 ND 或 FRACTAL_NZ，逻辑张量须连续。支持两侧转置、相同 batch 或整矩阵广播；交错 batch 广播不支持。output_dtype 省略时为 BF16，也接受 FP16。scale 必须为 FP32 [N]，pertoken_scale 必须为 FP32 [M]，跨 batch 共用。ND 权重要求 N 为正偶数；NZ 权重还要求 N/2 在 transpose_x2=True 时对齐 16，否则对齐 32，以保证左右半边均从分形边界开始。group_size 为 0，offset 和 bias 必须为 None；非 None bias 在 plugin 和统一 host tiling 入口报错。

Cube 根据 shape、转置及权重格式自动选择 baseM/baseN。UB 列跨度为 baseN 向上对齐 16，行数取 5632 除以列跨度的整数部分，上限为 255；256 列对应 22 行，512 列对应 11 行。两个 AIV 按 UB 行块分配 Cube 输出行；左右反量化结果和 SwiGLU 暂存均保持 FP32，列尾块使用对齐后的行跨度，仅将有效列写出。ND/NZ 的 UB 反量化路径均使用 L2 容量的 45/192 作为切分阈值，该阈值不代表实际缓存占用比例。

910B 部分 tiling 依赖 CANN 的 libophost_comm_legacy.so。运行前 source CANN 的 set_env.sh，使 ASCEND_OPP_PATH 指向有效 opp；库需提供 LegacyTilingPrepareForOpCache、LegacyTilingParsePrepareForOpCache、LegacyGenTbeMatmulTiling、LegacyCheckSupportConditionQbmm。内部 legacy 查询保留 QuantBatchMatmulV3 名称。host 的 C++17 选项只用于本算子，UB 可分配预算限制为 188160 字节；22×256 分配 159936 字节，11×512 分配 161920 字节，均为每 AIV 开销。

从仓根构建安装的示例：

```bash
export ASCEND_OP_NAME=eagle_quant_batch_matmul_swiglu
export ASCEND_COMPUTE_UNIT=ascend910b
python -m build --wheel --no-isolation
python -m pip install --force-reinstall dist/mindiesd-*.whl
```

通用依赖见仓内[编译指导](../../../docs/zh/developer_guide/build_guide.md)。测试直接调用 torch.ops.mindiesd.eagle_quant_batch_matmul_swiglu；进程先选物理卡，再导入 MindIE-SD 完成注册。

## 场景与判据

| 层级 | 场景 | 对照和判据 |
| --- | --- | --- |
| 小 shape | 无 bias，交叉覆盖转置、FP16/BF16 输出和整矩阵 batch 广播；左右不同 channel scale 的配对用例覆盖 22/44/66/128 行附近及 M=1/2049、N=4096/8192，检查分行边界、空闲 AIV 和多个列块 | CPU INT32 累加、FP32 反量化与 SwiGLU、最后舍入到输出 dtype；混合容差、cosine、模长比均为硬门禁 |
| 列尾块与 NZ | ND 半 N 覆盖 1/7/8/9/15/16/17/63/65/80/96/129/193/257/513；NZ 覆盖四种转置组合、短 M、带 K padding 的 batch 和两种输出 dtype | CPU golden 检查 UB 行跨度、部分向量 mask、NZ 物理 batch 步长及左右分支地址 |
| 接口 | 非法输入和 FakeTensor 用例；INT32/FP32/BF16/FP16 bias 拒绝用例先执行 bias=None，再传入非空 bias | 检查 dtype、K、group_size、连续性、batch、offset、bias、缺少 token scale、奇数 N 及 NZ 半边未对齐的拒绝，以及 FakeTensor 的 N/2 输出 |
| 中间精度 | 构造正负 A 与不同 B；另用小 shape 比较 NPU INT32 基准与 CPU 基准 | 融合输出相对 FP32 基准的平方误差必须小于提前转为 BF16/FP16 后计算的误差的 10% |
| 大 shape 精度 | x1 [5672,5376]、x2 [5376,28672] 均为 INT8；scale [28672]、pertoken_scale [5672] 为 FP32；无 bias/offset/转置；输出 [5672,14336] BF16；uniform、normal 共 2 条 | 同一份输入调用 torch_npu.npu_quant_matmul 获取 INT32 累加结果，再以 FP32 反量化与计算 SwiGLU；cosine、模长比为硬门禁，混合容差只报告软约束 |
| 稳定性 | 复用上述 2 条，每条默认 1000 轮 | 首次独立 CPU baseline，随后输出逐位相同（含正负零），每轮核对输入未改变 |
| 融合性能 | 上述生产尺寸，仅 normal，权重 ND 和预转换 NZ 各一组 | 每组 20 次独立 msprof op 采集，每次 warm-up=20、launch-count=1；报告 Task Duration(us) |
| 原生组合性能 | 同一生产尺寸和输入，torch_npu.npu_quant_matmul 接 torch_npu.npu_swiglu，权重 ND 和预转换 NZ 各一组 | 每组 20 次独立 msprof 采集；固定预热 5 次，再独立记录 10 次完整调用的 Event，报告最小 Layer Duration(us) |

INT4、NZ 激活、packed scale、INT8 输出和无 token scale 场景不支持。CPU 对照依次乘 channel scale 和 token scale，再计算 SwiGLU；FP32 中间结果不舍入到 BF16/FP16。大 shape 基准要求安装版本的 torch_npu.npu_quant_matmul 支持 INT32 输出；不支持时测试报错，不退回带中间精度转换的基准。INT32 输出支持情况参见[官方 API 文档](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md)。

随机精度用例均匀与正态各半。均匀范围为 [-5,5]；正态为每个普通输入独立抽取均值 [-5,5]、标准差 [0.1,2]，不钳位样本。INT8 在造数阶段按 absmax/127 对称量化，使用最近偶数舍入；逐行和逐列 scale 由量化结果生成。FP16 随机小 shape 的 channel scale 额外乘 0.01，以使最终乘积处于有限范围，该倍率单独记入日志。日志记录每个输入的分布参数、seed、scale 范围、实际 shape/dtype/stride/device。性能各轮使用相同 seed 和正态参数。

BF16 输出按 BFLOAT16 档使用 atol=rtol=2^-7，FP16 输出按 FLOAT16 档使用 2^-10；两档匹配率至少 0.99、兜底值均为 0.1。最大绝对误差上限取兜底值与最大误差位置 golden 对应实际输出格式的 32 倍 ULP（Unit in the Last Place，末位间距）中的较大者。整体 cosine 至少 0.99，模长比在 [0.9,1.1]，输出必须符合 shape/dtype/连续性并为有限值。大 shape 的 PASS 仅表示硬门禁通过，日志单独列出混合容差结果。

## 执行

以下命令从仓根运行。shell 入口也可从其他目录用绝对路径启动，会自行定位仓根。算子脚本的 --help 和 --dry-run 不初始化设备、不创建批次。M/K/N 可覆盖为正整数，N 必须为偶数；逻辑输入 shape 为 [M,K]、[K,N]、[N]、[M]，输出为 [M,N/2]；默认值固定为上述生产尺寸。性能选用 NZ 时 N 必须是 64 的倍数。大 shape 精度仍只测 ND，NZ 功能由小 shape 用例覆盖。

```bash
unset ASCEND_RT_VISIBLE_DEVICES
pytest --test-mode=NPU tests/ops/eagle_quant_batch_matmul_swiglu/test_eagle_quant_batch_matmul_swiglu.py -s

bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_prod_accuracy.sh --dry-run
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_prod_accuracy.sh --device-id 1 3 --mode all
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_prod_accuracy.sh --device-id 1 --mode accuracy --distribution normal --m 5672 --k 5376 --n 28672
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --device-id 1
```

小 shape 默认物理 NPU 0，可通过 MINDIE_TEST_DEVICE_ID 指定，并按 OpDef 注册限制在 910B 执行。检查点要求取消 ASCEND_RT_VISIBLE_DEVICES，续跑保持设备身份。大 shape 未指定卡时使用仓内 select_npu_device.py 自动选择一张卡；多卡按完整场景分配，各场景独立造数和判定。all 模式先完成全部精度场景，再执行压测；也可选择 accuracy 或 stress。单独 stress 通过只说明重复执行稳定，不证明相对 golden 的精度。

测试通过 tests/scripts/run_suites.py 统一调度，CPU 和 NPU 分别在独立子进程中运行。精度、接口和通过真实注册 torch.ops 调用的 FakeTensor 用例保留在 test_eagle_quant_batch_matmul_swiglu.py，参数化静态展开合计 256 条，归入 NPU 套件。2 条独立检查点回归位于 tests/UT/tools/test_qbmm_state.py，归入 CPU 套件，两套合计 258 条。名称含 cpu_golden 的精度用例仍需 NPU，因为被测对象是真实算子。直接 pytest 默认 CPU 模式，运行算子主文件必须显式选择 NPU。

```bash
export MINDIE_TEST_RESULTS="$PWD/log/tests"
python tests/scripts/run_suites.py --all
python tests/scripts/run_suites.py --npu_only tests/ops/eagle_quant_batch_matmul_swiglu
python tests/scripts/run_suites.py --cpu_only tests/UT/tools/test_qbmm_state.py
```

全量 CPU 选择通过后执行逐文件分支覆盖率门禁，默认阈值 80%；局部选测不执行这项全仓门禁。--incremental 根据相对 --base-branch（默认 dev）的变更选择测试，可加 --dry-run 只查看选择。统一入口支持完整目录或文件选择，生成 CPU/NPU 独立 JUnit XML 和合并覆盖率报告。

性能入口只接受一张卡；未指定时运行一次设备选择器并保存报告。默认按 qbmm/ND、qbmm/NZ、torch_npu/ND、torch_npu/NZ 顺序各采集 20 轮，共 80 轮。--path 单选 qbmm 或 torch_npu，--weight-format 单选 ND 或 NZ；省略则运行对应全部选项。--runs 控制各组独立进程数，--iterations 控制原生组合每轮 Event 样本数，--warm-up 只控制融合路径。输入造数、传输与 NZ 权重转换均在计时之外，正式调用复用同一输入，属于热缓存测量。

启动时 PLAN 列出全部所选组及总采集数；默认显示 4 组、80 次。四组顺序执行，完成融合 ND/NZ 的 40 次采集后才开始原生组合。汇总列出每组已完成/计划数量，未运行组标为 PENDING，未跑满组标为 PARTIAL；只有全部所选采集成功才显示 COMPLETE。续跑恢复原批次的路径和格式，单路径旧批次不会因省略 --path 而扩成四组。

从仓根预览与选择路径：

```bash
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --dry-run
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --path qbmm --device-id 4
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --path torch_npu --device-id 4
```

只传 --weight-format NZ 仍选择融合与原生组合两条 NZ 路径；只传 --device-id 不筛选路径。中途失败时保存部分汇总并非零退出，不能以融合部分通过代替组合部分完成。

原生组合的 QBMM 输出 BF16，再传入原生 SwiGLU，最终输出 BF16；融合实现的中间结果保留 FP32。原生组合仅调用这两个算子，不使用精度基准中的 INT32 输出与手工反量化链。融合 Task Duration 和整段组合 Layer Duration 分别报告，不混算速度比；性能流程不计算 golden，仅在 Event 计时后检查最后一次输出的 shape、dtype 和有限值。

```bash
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --dry-run
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --path qbmm --weight-format NZ --device-id 1
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --path torch_npu --runs 20 --iterations 10 --device-id 1
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --path torch_npu --no-msprof --device-id 1
```

--no-msprof 只允许单选原生组合路径，使用同一 Event 协议直接计时；不查找或调用 msprof。默认组合采集保存 profiled_timing.json、msprof.log、event_samples.csv 和唯一 PROF 目录中的 op_summary CSV/timeline；直接模式保存 timing.json、timing.log 和 event_samples.csv。组合固定预热 5 次，不接受其他预热次数。

精度显示输入准备、DUT、golden、指标比较和 baseline 恢复的阶段进度与耗时。压测在本次首轮、每 10 轮、末轮及距上次报告满 30 秒的轮末刷新，显示已完成/目标轮数、调用同步、校验和提交耗时；30 秒刷新在当前轮结束后发生。

融合性能 workload 独立模块如下，每个进程只调用一次真实算子，msprof op 负责回放预热和采集；workload 内不计算精度指标。

```bash
python -m tests.ops.eagle_quant_batch_matmul_swiglu.profile_eagle_quant_batch_matmul_swiglu_prod --device-id 1 --m 5672 --k 5376 --n 28672
```

## 结果与续跑

默认结果位于仓根 `log/eagle_quant_batch_matmul_swiglu/`，每次新建 small、accuracy 或 perf 加时间戳的批次。可用 --output-dir 指定大 shape 与性能的新目录；已有目录拒绝覆盖。accuracy.log 保存实时精度日志，各 worker 的独立日志及失败张量位于 attempts 子目录。final_summary.txt 汇总历史与本次结果；性能还保存 task_duration_summary.txt 和每轮原始 msprof 产物。

检查点中的非有限诊断数值保存为字符串 nan、inf、-inf，保证标准 JSON 序列化和校验和读取可用；用例的 PASS/FAIL 判定保持不变。检查点回归仅操作临时目录中的 CPU 数据，无需加载算子或初始化 NPU，可从仓根单独运行：

```bash
python -m pytest --test-mode=CPU tests/UT/tools/test_qbmm_state.py
```

算子改名后须重新编译安装算子库与 plugin，并启动新测试进程和新批次。旧名称的批次保留用于排查，不能跨代码及产物指纹使用 --resume。

```bash
pytest --test-mode=NPU tests/ops/eagle_quant_batch_matmul_swiglu/test_eagle_quant_batch_matmul_swiglu.py -s --resume /absolute/path/to/small_batch
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_prod_accuracy.sh --resume /absolute/path/to/accuracy_batch --device-id 3 5
bash tests/ops/eagle_quant_batch_matmul_swiglu/run_eagle_quant_batch_matmul_swiglu_msprof_op.sh --resume /absolute/path/to/perf_batch
```

续跑校验不可变计划、代码/软件/已安装算子产物指纹、已提交记录及必要产物的校验和。小 shape 和性能保持原物理设备身份；大 shape 精度允许显式换卡，省略时恢复上次卡列表。已提交硬失败会报告并拒绝重测；未提交的中断尝试保留，使用新 attempt 重新执行。压测加载原 baseline，并按原生成器重建输入后核对摘要；跨进程累计轮数不等同于单进程连续压测。所有项目已完成时只重建报告，不访问设备。

Ctrl+C 和 SIGTERM 停止调度、回收本次子进程、保存部分报告，分别返回 130 和 143；SIGKILL 依赖逐单元持久记录在下次恢复。批次排他锁防止并发续跑，性能采集子进程继承锁。融合结果从 Operator Basic Information 提取唯一有效 Task Duration，组合结果读取经校验的 Event JSON；同时保存并校验原始产物，缺结果不会按 0 补齐。final_summary.txt 连续列出全部逐轮耗时，紧接各场景最小值；summary.json 保存逐轮样本、最小值及其轮次和产物路径。最小值表示当前条件下观测到的最佳性能。仅融合路径计算跨轮波动；超过 10% 时提示增加 --warm-up，并在新批次完整重测 20 轮。
