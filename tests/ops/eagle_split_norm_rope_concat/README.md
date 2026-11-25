# EagleSplitNormRopeConcat

`eagle_split_norm_rope_concat` 面向 Ascend 910B、910C（Atlas A3）和 Ascend 950，将 packed QKV 拆分、Q/K RMSNorm、全维或部分维度 RoPE 和输出布局变换融合为一个算子。Q、K、V 已完成投影，输入沿头轴依次排列为全部 Q 头、全部 K 头、全部 V 头。

## 接口

```python
query, key, value = torch.ops.mindiesd.eagle_split_norm_rope_concat(
    qkv, norm_query_weight, norm_key_weight, rope_cos, rope_sin, [8, 2, 2],
    norm_type=3, rope_type=1,
)
```

示例将 qkv 的 12 个头拆为 8 个 Q 头、2 个 K 头和 2 个 V 头，对完整 head_dim 做带权重 RMSNorm 和 HALF RoPE，cos/sin 的最后一维可为 head_dim 或 head_dim/2，输出默认为 BNSD。部分维度 RoPE 须显式传入 `rotary_dim`；需要保留输入的 SND/BSND 布局时传入 `output_bnsd=False`。也可从 `mindiesd.layers._custom_ops` 导入同名函数。接口用于推理，不提供反向实现。

| 参数 | 约束 |
| --- | --- |
| qkv | 连续 ND，FP16 或 BF16，[S,Nq+Nk+Nv,D] 或 [B,S,Nq+Nk+Nv,D] |
| `norm_query_weight`、`norm_key_weight` | `norm_type=3` 时两者必需，连续 `[D]`，与 qkv 同 dtype、同 NPU；`norm_type=2` 时不参与计算，可传 `None`，若提供 Tensor 仍须满足上述约束 |
| `rope_cos`、`rope_sin` | 两者同形状，连续，与 qkv 同 dtype、同 NPU；形状为 `[S,W]`、`[Bc,S,W]` 或 `[Bc,S,1,W]`；Bc 为 1 或 B，W 为 R 或 R/2，R 为实际参与 RoPE 的维数 |
| head_counts | int[] 参数，长度为 3，值为 [Nq,Nk,Nv]；依次表示 Q/K/V 头数，均为正且和等于输入头轴。910B/910C 要求 Nk=Nv、Nq 是 Nk 的整数倍；950 三路独立，各自不超过 128，总和可以超过 128 |
| `norm_type` | 必填，无默认值；当前仅支持 2：RMSNorm、3：带权重 RMSNorm，其余模式报错 |
| `rope_type` | 必填，无默认值；0：INTERLEAVE，相邻偶奇位置配对；1：HALF，前 R/2 与后 R/2 配对；不提供跳过 RoPE 的模式 |
| `rotary_dim` | 参与 RoPE 的前缀长度 R，默认 `None` 表示完整 D；显式值须为 32 的倍数，32 ≤ R ≤ D |
| `eps` | 转为 FP32 后仍为有限正数，默认 1e-5 |
| `output_bnsd` | 默认 True，返回 BNSD，3D 输入时 B=1；False 保留 SND/BSND |

本算子的 `norm_type` 不含跳过归一化的模式，编号如下。除 2、3 外，当前实现均在接口、形状推导及 tiling 校验中拒绝。

| 编号 | 含义 | 当前支持 |
| --- | --- | --- |
| 0 | LAYER_NORM | 否 |
| 1 | LAYER_NORM_AFFINE | 否 |
| 2 | RMS_NORM，逐 head 在 D 上归一化 | 是 |
| 3 | RMS_NORM_AFFINE，逐 head 在 D 上归一化并乘权重 | 是 |
| 4 | LAYER_NORM_ACROSS_HEADS | 否 |
| 5 | LAYER_NORM_AFFINE_ACROSS_HEADS | 否 |
| 6 | RMS_NORM_ACROSS_HEADS | 否 |
| 7 | RMS_NORM_AFFINE_ACROSS_HEADS | 否 |
| 8 | L2_NORM | 否 |

D 仅支持 128，其它值在接口和 Host 校验中报错；R 支持 32、64、96、128。不接受空 Tensor，每个浮点输入元素数不超过 INT32_MAX。输入不连续或尺寸不满足约束时直接报错，不做隐式 padding。

Q/K 分别在完整 D 上执行 FP32 RMSNorm，倒平方根使用 Sqrt + Div；norm_type=3 还在 FP32 下乘对应权重。归一化结果直接进入 FP32 RoPE，仅最终输出舍入到输入的 FP16 或 BF16。设 x 为一个 head 的输入，weight 在无权重模式下取 1，则归一化结果为：

$$
z_d=x_d\cdot\frac{1}{\sqrt{\frac{1}{128}\sum_{j=0}^{127}x_j^2+\varepsilon}}\cdot\mathrm{weight}_d.
$$

前 R 维计算 $y_d=z_d\cdot\mathrm{cos}_d+\mathrm{rotate}(z)_d\cdot\mathrm{sin}_d$，其余维度取 $y_d=z_d$，最后整行转换为输出 dtype。HALF 的旋转项为前 R 维的前后半区交换并将前半区取负；INTERLEAVE 每对位置满足 $\mathrm{rotate}(z)_{2i}=-z_{2i+1}$、$\mathrm{rotate}(z)_{2i+1}=z_{2i}$。半宽 cache 在 UB 内展开：HALF 将整个半区重复两遍，INTERLEAVE 将每个元素相邻重复两遍；全宽 cache 按原值使用，不要求成对系数相同。V 按原始位模式复制，布局由 output_bnsd 决定。

rotary_dim 省略时仍取 D=128，不从 cache 宽度推断。例如末维 64 的 cache 配默认 R 表示旋转 128 维，配显式 R=64 表示旋转 64 维。R=96 的半宽 cache 末维为 48，后 32 维仍只做 Norm。

三个输出均连续、与输入同 dtype，头数分别为 Nq/Nk/Nv。GQA 不复制 KV 头。concat 指 RoPE 前缀和归一化尾部在特征轴上的拼接。

输出形状由 qkv 的 shape 和整数列表 `head_counts` 直接推导，FakeTensor/Meta 无需读取 Tensor 数据，也不需要数据依赖输出形状的 `dynamic_output_shape` 标记或 `capture_dynamic_output_shape_ops` 开关。B/S 维继续由输入 shape 传递。Plugin 通过 `at::IntArrayRef` 将头数传入现有的 CANN `aclIntArray` 接口；CANN Host 继续按值依赖元数据读取头数并写入 tiling，设备内核使用 tiling 中的头数。

## 实现与 910B/910C 边界

Host 注册、形状推导、tiling 和内核位于 `csrc/ops/eagle_split_norm_rope_concat/`；CANN OpType 为 `EagleSplitNormRopeConcat`，入口为 `aclnnEagleSplitNormRopeConcat`。内核在 AIV 间划分独立任务，直接写入目标布局，不使用 Cube 或核间同步。

CANN 层的 `norm_type`、`rope_type` 同样为必填属性，`output_bnsd` 默认 true。CANN 整数属性用 `rotary_dim=0` 表示未指定；PyTorch 接口用 `None` 表示未指定，并在调用 CANN 前解析为实际 D。PyTorch 显式传入 0 会报错。

HALF 与 INTERLEAVE 的数学计算仍按单 token 的连续 head 分块，容量上限64。Host 按 `align_up(min(Hq,64),8)` 选择实际容量：56-head 场景分配56，55-head 分配56以满足 Brcb 的8行读取与对齐要求，57及以上仍使用64容量。输入双缓冲、FP32 x/tmp 和 reduction 按单 token 的该容量分配。Nk=Nv，V 与 K 保持相同任务归属和尾块；权重、Cos/Sin 语义和数学逻辑不变。

UB 足够时，同一 batch 的相邻两个 token 分到同一核。Q/K 使用两个独立输出队列，每个队列容纳一对 token，两个槽跨 Q/K 阶段轮换。56-head 下每槽 28 KiB，两槽共 56 KiB。原 64-head 单 token 双缓冲的输出区共 32 KiB。每对完成后提交一次 Store，各槽独立保留 Vector→MTE3 同步和 MTE3→Vector 复用保护。

BNSD 在最后一次 FP32→FP16/BF16 Cast 时通过 repeat stride 直接形成 `[head,token,128]`，无需额外转置区。完整对的搬出参数为 `blockCount=有效head数`、`blockLen=512B`、GM段间空隙 `(S-2)*256B`；单 token 尾组使用256B和 `(S-1)*256B`，每个batch单独收尾。BSND 保持 `[token,head,128]`，一次多段DMA提交本对的有效行，兼容head尾块。

V 将原两个单 token buffer重组为一个双 token绑定队列buffer，总空间不变。BSND 一次多段MTE2搬入本对V；BNSD 分别对两个token进行多段MTE2搬入，UB目标行间距为 `有效token数*256B`，直接得到 `[head,token,128]`。V 每对只提交一次Store，仍先于本对K计算提交；绑定队列保留 MTE2→MTE3 和 MTE3→MTE2 复用保护。V不再跨对预取，BNSD的MTE2分段方式也发生变化，需与Task Duration一起实测。

若实际 UB 不能容纳两个双 token输出槽，保留原单 token任务、一个双缓冲输出队列和V双缓冲，不拒绝原先可运行的64容量场景。输入一直保持单 token双缓冲，逐token计算时预取下一token，包括跨token对的预取。

Host 从查询到的物理 UB（至多192 KiB）扣除8 KiB API预留和256 B系统预留，标准可用预算为188160 B。保留 x/tmp、参数区的256 B bank 间隔；各缓冲尺寸按32 B对齐，BNSD仅改变写回地址和步长，不额外分配 UB。

56-head 相对原64容量，input、原output、V、x、tmp各省4096 B，合计20480 B；reduction再省320 B，总计20800 B。双 token输出双槽相对原单 token双缓冲output增加 `2*56*128*2=28672 B`。标准物理 UB=192 KiB 时：

| RoPE/cache | 原64容量、单 token UB | 新56容量、双 token输出双槽 UB | 可用预算剩余 |
| --- | --- | --- | --- |
| HALF Full | 170496 B | 178368 B | 9792 B |
| HALF Compact | 171008 B | 178880 B | 9280 B |
| INTERLEAVE Full | 179200 B | 187072 B | 1088 B |
| INTERLEAVE Compact | 179712 B | 187584 B | 576 B |

INTERLEAVE 继续使用16×128个uint32字节偏移和128个FP32符号值；Gather按最多16个head分组，最后一组只处理有效行。R=96的64/32列mask、归一化尾部和最终dtype收窄保持不变。Compact额外使用128个uint32展开索引；先在tmp中转换系数，再Gather到参数区，继续按实际cache宽度寻址GM。

Host 日志 `EAGLE910_TILE` 记录head容量、tokenTile、独立输出队列数、每队列buffer数、每buffer的token数、完整组BNSD blockLen及UB占用。56-head应显示 `tokenTile=2 outputQueues=2 buffersPerQueue=1 tokensPerBuffer=2 bnsdBlockBytes=512`；该Host日志不是设备执行证据。B=1、S=5672、Q/K/V均56-head时，源码搬出调用总数由17016降为8508，每路2836次；实际设备指令数、MTE3/Vector重叠及性能收益仍须通过编译产物、simulator和同卡实测确认。

构建环境就绪后，在仓根通过 whl 构建并安装。安装命令中的文件名须替换为本次生成的 whl 文件名：

```bash
unset MINDIESD_SKIP_OPS_BUILD SKIP_ALL_OPS_PLUGIN_BUILD
python setup.py bdist_wheel
python -m pip install --force-reinstall --no-deps "dist/本次生成的文件名.whl"
```

`setup.py` 会调用仓内构建脚本，将算子及对应 plugin 一并打入 whl；安装该 whl 即完成部署。`--force-reinstall` 用于替换已安装的同版本包，`--no-deps` 保留现有依赖环境。A3 的构建目标为 `ascend910_93`。

## Ascend 910C（Atlas A3）构建与选卡

在已加载 CANN 和 torch_npu 的环境中，从仓库根目录执行。A3 复用 910B 的 D=128 向量内核和运行时 UB 规划。56-head 客户场景走双 Token 写回，950 专属的独立 Head 场景只在 950 上运行。

```bash
export ASCEND_COMPUTE_UNIT=ascend910_93
unset MINDIESD_SKIP_OPS_BUILD SKIP_ALL_OPS_PLUGIN_BUILD
python3 setup.py bdist_wheel
# 安装本次生成的 wheel 后再运行下列命令
suite=tests/ops/eagle_split_norm_rope_concat
python3 -m pytest "$suite/test_eagle_split_norm_rope_concat.py" --test-mode=NPU -s
npu-smi info -m
# A3_LOGIC_ID 设为实际已分配且健康的 Chip Logic ID
bash "$suite/run_eagle_split_norm_rope_concat_prod_accuracy.sh" --device-id "${A3_LOGIC_ID:?}" --mode accuracy
# msprof 采集需在可直接寻址该已分配 die、且未设置 ASCEND_RT_VISIBLE_DEVICES 的环境中执行
bash "$suite/run_eagle_split_norm_rope_concat_msprof_op.sh" --device-id "$A3_LOGIC_ID" --path fused --runs 2
```

小 shape pytest 默认使用当前进程的设备，`MINDIE_TEST_DEVICE_ID` 可显式指定进程内可见序号。生产脚本的 `--device-id` 在 A3 上表示驱动的 Chip Logic ID。pytest、大 shape 精度与压测、`--path composed --no-msprof` 的 Event 计时均保留可见性映射。例如已分配设备的可见列表为 `8,12` 时，pytest 的可见序号 1 和生产脚本的 `--device-id 12` 选中同一 die。实际 NPU ID 和 Chip ID 可从 `npu-smi info -m` 查询，并用 `npu-smi info -t board -i <NPU ID> -c <Chip ID>` 核对。

未设置 `ASCEND_RT_VISIBLE_DEVICES` 时，A3 入口用 CANN 查询进程序号对应的物理设备 ID，再用 `npu-smi info -t phyid-remap -p <物理设备 ID>` 查询 Chip Logic ID，并与 `npu-smi info -m` 的 NPU ID、Chip ID 核对。生产脚本据此将分配的 Chip Logic ID 反查为进程序号，再执行选卡。例如驱动号为 8、12 的两个 die，即使管理表调整行序，`--device-id 12` 仍绑定逻辑号 12。若无法核对唯一映射，入口报错。精度和 Event 计时可使用管理员已确认对应关系的可见性掩码，msprof 采集需修复无掩码环境的映射。[CANN 设备 ID 转换](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2286.html) · [A3 驱动映射查询](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3npu/npusmia3_0062.html)

A3 大 shape 精度和压测可传多个已分配逻辑号（`--device-id ID1 ID2`），按完整场景并行。`accuracy.log` 的批次结论由父进程在所有设备结束后输出。只要一个分片失败，脚本就返回非零，并按设备、阶段、场景和输入分布列出失败或未完成的项。性能批次使用一个逻辑号，`device.json` 记录应用选中的设备。设置上述构建目标后，两个脚本均可用 `--dry-run` 离线预览 A3 的 8 个场景。

## 三层测试

910B/910C 大 shape 测试使用 BSND packed QKV `[1,5672,168,128]`、头数 `[56,56,56]`、BF16，固定 `norm_type=3`、`eps=1e-5`。权重各为 `[128]`，cos/sin 为 `[1,5672,1,W]`。场景表交叉选择 RoPE、输出布局、cache 宽度与 R=96/128，共 8 组；不展开 16 组笛卡尔积。每个 R 均覆盖两种 RoPE 与全宽/半宽。

| 内部场景键 | RoPE | 输出布局 | R | W | 精度/压测分布 |
| --- | --- | --- | --- | --- | --- |
| `half-bsnd` | HALF | BSND | 96 | 96 | uniform |
| `half-bnsd-r128` | HALF | BNSD | 128 | 128 | normal |
| `interleave-bsnd-r128` | INTERLEAVE | BSND | 128 | 128 | normal |
| `interleave-bnsd` | INTERLEAVE | BNSD | 96 | 96 | uniform |
| `half-bsnd-compact-r128` | HALF | BSND | 128 | 64 | normal |
| `half-bnsd-compact` | HALF | BNSD | 96 | 48 | uniform |
| `interleave-bsnd-compact` | INTERLEAVE | BSND | 96 | 48 | uniform |
| `interleave-bnsd-compact-r128` | INTERLEAVE | BNSD | 128 | 64 | normal |

Q/K/V 的 BSND 输出均为 `[1,5672,56,128]`，BNSD 为 `[1,56,5672,128]`。精度与压测各 8 条，分布按表交叉分配，不再乘两种分布；直接 Python 的 `--distribution all` 使用该表。原八场景的性能统一正态分布，融合与配套的完整 QKV 拼接逐项对应；同一配对使用相同输入配置、CPU 造数后端和 seed。半宽 cache 的 BF16 展开包含在拼接完整调用计时内。

950 另增加 `half-bsnd-asymmetric`：输入 `[1,5672,257,128]`、头数 `[1,128,128]`、BF16、HALF、BSND、全宽 W=R=96、norm_type=3、eps=1e-5、seed=20261017，精度与压测分别覆盖 normal 和 uniform。输出分别为 `[1,5672,1,128]`、`[1,5672,128,128]`、`[1,5672,128,128]`。生产精度与稳定性合计各 10 条（原八条 + 新 shape 的两种分布）；msprof op 仍对该 shape 使用 normal，性能记录数不变。910B 不选择此项。

半宽专项用例覆盖四种 R、两种 RoPE、两种 dtype、两种 Norm、两种输出布局、SND/BSND、三种 cache rank 及 batch 共享/独立，并与显式展开的全宽调用逐位比较。Gather 边界、head tile 边界、流水及常驻 cache 复用、FP32 中间精度和默认 R 用例同时覆盖两种宽度；真实入口、FakeTensor/Meta 验证宽度约束，拒绝既不是 R 也不是 R/2 的宽度及 cos/sin 宽度不一致。

| 层级 | 入口 | 对照与判定 |
| --- | --- | --- |
| 小 shape 与接口 | `test_eagle_split_norm_rope_concat.py` | CPU 分阶段 golden；Q/K 按 FLOAT16 或 BFLOAT16 档的混合容差、cosine、模长比均为硬门禁；V 精确相等；同时检查 dispatcher schema、默认值、整数头数校验、FakeTensor/Meta 形状与布局 |
| 生产精度 | `run_eagle_split_norm_rope_concat_prod_accuracy.sh` | 同一 NPU 上用小算子组合 golden；cosine、模长比为硬门禁，混合容差仅诊断；V 精确相等 |
| 稳定性压测 | 同一精度入口，`--mode stress` | 固定输入，每种分布默认重复 1000 轮，Q/K/V 相对首次输出逐位一致 |
| 生产性能 | `run_eagle_split_norm_rope_concat_msprof_op.sh` | 融合算子用 msprof op；完整 QKV 小算子链用 msprof + Event；默认全跑，`--path` 可选路径 |

910B 与 950 共用的原有矩阵保留 100 条参数化用例：38 条数值/语义回归，62 条接口检查。用例通过显式参数元组交叉选择，不对 dtype、Norm、布局、分布继续展开笛卡尔积。主要随机精度矩阵共 26 条，均匀、正态各 13 条；其余为默认值、权重语义和特殊值等专项回归。正态用例逐输入采样 mean∈[-5,5]、std∈[0.1,2]，不钳位，固定 seed 并打印参数。

| 测试组 | 展开条数 | 交叉覆盖与合并检查 |
|--------|----------|--------------------|
| 半宽/全宽等价 | 8 | R=32/64/96/128 × 两种 RoPE；交叉 dtype、Norm、布局、cache rank 和 batch 共享方式；同时比较 CPU golden 与两种编码的输出比特 |
| 基础精度 | 4 | MHA/GQA/MQA、S=1/16/17、B=1/2、SND/BSND；使用不要求两半重复的全宽 cache |
| INTERLEAVE R=96 边界 | 6 | head=15/16/17/63/64/65；交叉 dtype、布局、缓存宽度和分布；单独核对未旋转后缀 |
| 流水复用 | 4 | S=1/81/129、head=56、GQA=[130,65,65]、MQA=[512,1,1]；两种 seed、同输入重复执行及 V 比特检查 |
| 常驻 cache 复用 | 4 | 两种 RoPE × 两种宽度，交叉共享/独立 batch cache 与分布 |
| 数值及接口默认语义 | 12 | FP32 连续计算 2、零输入 1、V 负零 2、后缀不旋转 2、默认 R/布局 2、无权重模式 3 |
| 真实入口拒绝非法参数 | 30 | 通用负例 26、非法 D 2、非法 Norm 2 |
| 缓存宽度接口 | 9 | Fake/Meta 合法宽度 4，真实/Fake/Meta 非法或不一致宽度 5 |
| 其余接口 | 23 | 权重必填 2、schema 1、Fake 非法 D/eps 5、显式头数 4、默认 R 2、模式必填 2、Fake 非法 Norm/RoPE/头数 4、Meta 布局 2、符号维度 1 |

上表仅统计共用矩阵；同场景重复 seed、同输入重复调用及等价关系比较不另算独立场景。950 专项与上述用例由同一 pytest 文件收集，见文末。代表组合覆盖各维取值和关键依赖，不表示覆盖全部两两组合。

默认值用例直接调用公开接口，验证省略 rotary_dim 等价于显式 D、默认输出 BNSD，以及两种 RMSNorm 在 D=128 的 HALF 和 INTERLEAVE 批处理路径上的精度。无权重模式额外比较传 None 与提供权重的输出逐位一致；带权重模式缺失任一权重、未实现的归一化模式及非法 RoPE 编号均须报错。真实调用代表性检查 D=64/127/129，FakeTensor 检查 D=32/256 的拒绝行为。接口测试同时检查两个模式参数必填、MHA/GQA/MQA 的具体输出头数，以及未开启数据依赖形状支持时的 FakeTensor/Meta 推导。

批处理边界并入 Gather 与流水用例，检查完整 tile、尾 tile、Q/K tile 数不同的 GQA、两种输出布局，并在流水用例中要求同一输入重复执行的输出逐位一致。INTERLEAVE R=96 检查 15/16/17-head Gather 分组边界及 63/64/65-head 批次边界，交叉 FP16/BF16、两种宽度和两种输出布局，后 32 维单独比较。910B 的 HALF 多 tile 与尾块由 [130,65,65] 的流水场景覆盖；950 对该超限配置检查报错。

流水用例使用 S=1/81/129、R=32/64/96/128，覆盖双缓冲多轮复用、只有一个 head 的尾 tile、部分核没有 K/V 任务的 GQA，以及相邻调用更换输入数据；FP16/BF16、两种分布和输出布局均检查 Q/K 精度及 V 的逐位保持。

910B 常驻 RoPE 复用用例使用 B=2、S=1、头数 [4096,1,1]（950 检查超限报错），检查同核多个 head tile、Q/K 切换以及共享和独立 batch cache。单次收窄回归用例令输入和权重为 1、eps=2、cos=1.375、sin=0，要求 Q/K 前缀等于 BF16(1.375/√3)=0.79296875，直接检查 FP32 norm 到 RoPE 的精度边界。

负零用例先确认输入 V 的 BF16 位模式为 `0x8000`，再在 CPU 上将输出按 int16 视图逐位比较，同时检查 shape、dtype 和连续性。精度检查保留最大误差坐标、实际值、参考值和失败指标，便于复现对应测试点。

Q/K 的约定数值路径全程 FP32 计算，仅最终输出舍入到输入 dtype，分别按下表取整套混合容差参数。内部 FP32 计算及指标统计使用的 dtype 不改变验收档位；V 只做复制和布局变换，始终要求位模式精确一致。

| Q/K 实际输出 dtype | 验收精度档位 | atol | rtol | 最低匹配率 | 最大误差兜底值 |
| --- | --- | --- | --- | --- | --- |
| FP16 | FLOAT16 | $2^{-10}$ | $2^{-10}$ | 0.99 | 0.1 |
| BF16 | BFLOAT16 | $2^{-7}$ | $2^{-7}$ | 0.99 | 0.1 |

最大绝对误差上限为 $\max(0.1,32\times\mathrm{ULP})$。ULP 按最大误差点的 golden 值及实际输出 dtype 计算：FP16 输出用 FP16 间距，BF16 输出用 BF16 间距。cosine 至少 0.99，模长比在 [0.9,1.1]。小 shape 的三项指标均为硬门禁；生产精度仅 cosine、模长比为硬门禁，混合容差为软约束。小 shape 的指标在 CPU FP64 计算；生产指标留在 NPU，以 FP32 计算，仅打印标量结果。

逐输出日志记录 `tolerance_dtype`、选档依据、实际 `output_dtype`、容差与兜底值、匹配率、最大误差点的坐标和 golden、ULP、32 倍 ULP、最终上限，以及各项通过状态；V 单独打印精确比较结果。

## 精度运行

先 source CANN 环境，确保 torch_npu 和当前 MindIE-SD 算子产物已安装。小 shape 的 pytest 默认使用当前进程的 NPU 逻辑设备。如需换卡，用 `MINDIE_TEST_DEVICE_ID` 指定当前可见范围内的逻辑编号；两种方式都保留已有可见性映射。以下命令在仓库根目录运行：

```bash
python -m pytest --test-mode=NPU tests/ops/eagle_split_norm_rope_concat -s

# 显式选卡：使用分配给本进程的逻辑编号
MINDIE_TEST_DEVICE_ID="${ALLOCATED_LOGICAL_ID:?set allocated logical device ID}" \
  python -m pytest --test-mode=NPU tests/ops/eagle_split_norm_rope_concat -s

# 正式大 shape 验收：由 launcher 显式传入已分配的逻辑卡号
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id "${ALLOCATED_LOGICAL_ID:?set allocated logical device ID}"
```

生产精度入口默认 `--mode all`：950 按分配表执行 10 条单次精度测试，再对这 10 组固定输入分别压测 1000 轮；910B 保留原 8 条。使用 `--shape B S Htotal D`、`--heads Q K V`、`--rope-type all|half|interleave`、`--cache-width all|full|half` 、`--output-layout all|bsnd|bnsd` 和 `--rotary-dim all|96|128` 直接选择，无需查场景 ID。默认逐场景选择 shape/heads，原八场景为 `1 5672 168 128` / `56 56 56`，新增场景为 `1 5672 257 128` / `1 128 128`；四个语义选择项默认 all，指定多项时取交集。每组压测在循环外固定输入，以独立 CPU 副本保存首次输出；1000 轮不包含基线调用。每轮同步后检查全部 Q/K/V 的 shape、dtype、设备、连续性及原始 BF16 位模式，包含正负零；首次输出必须有限，输入摘要在基线前后、结束及输出异常时复核。循环内释放当轮结果，每 100 轮和最后一轮打印进度，不重新计算 golden。

```bash
# 只跑单次精度（950 十条，910B/910C 八条）
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id "${ALLOCATED_LOGICAL_ID:?set allocated logical device ID}" --mode accuracy

# 只测相邻旋转、BNSD 全宽 cache 场景（分配表为 uniform）
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id "${ALLOCATED_LOGICAL_ID:?set allocated logical device ID}" --mode accuracy --rope-type interleave --cache-width full --output-layout bnsd

# 自定义 BSND shape 与 GQA 头数，检查 INTERLEAVE + 半宽 cache + BNSD
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh \
  --device-id "${ALLOCATED_LOGICAL_ID:?set allocated logical device ID}" --mode accuracy --shape 2 1024 12 128 --heads 8 2 2 \
  --rope-type interleave --cache-width half --output-layout bnsd

# 无设备访问地预览选择结果；只选半宽 cache 时得到 4 个配置
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --cache-width half --dry-run

# 只跑 R=128 精度，覆盖 4 组代表组合
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id "${ALLOCATED_LOGICAL_ID:?set allocated logical device ID}" --mode accuracy --rotary-dim 128

# 只跑压测，覆盖默认轮数
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id "${ALLOCATED_LOGICAL_ID:?set allocated logical device ID}" --mode stress --stress-runs 2000
```

精度 Python 入口接受相同的 shape、heads 和四个语义选择参数；`--dry-run` 由 shell 提供，不启动 Python/NPU、不创建输出目录。自定义 shape 用于真实输入和输出契约，要求各维为正 int32、输入总元素数不超过 2147483647、D=128、Q+K+V=Htotal；950 每路 Head 在 1～128 内，910B/910C 要求 K=V、Q 能被 K 整除。当前大 shape 精度固定 BF16、norm_type=3、eps=1e-5；R 按代表场景取 96 或 128，cos/sin 为 `[B,S,1,R]`（full）或 `[B,S,1,R/2]`（half）。R=32/64 及 FP16 由小 shape 覆盖；小 shape 已包含 R=128 的两种 RoPE 半宽/全宽逐位等价与默认 R 回归。自定义 shape 与 heads 须匹配。shell 须同时传入两者；直接 Python 对 257-head 的生产 shape 默认使用 [1,128,128]。一般自定义 shape 运行原八个组合，精确选择新增 shape/heads 时运行该场景的 normal 和 uniform 两条用例。选择项只过滤表内代表配置，未匹配时报错，不自动补齐未列组合。`--stress-runs` 必须为正整数；`--help` 查看参数。精度与压测默认在脚本所属仓库创建 `log/eagle_split_norm_rope_concat/accuracy_YYYYMMDD_HHMMSS/`，实时输出同时保存为该批次的 `accuracy.log`；`--output DIR` 可指定空目录。同秒批次重名时拒绝覆盖，可稍后重试或指定新目录，不依赖启动目录。压测遵循本算子的逐位确定性要求，任意一位变化即失败，同时报告轮次、分布、输出、差异数量、最大绝对/相对误差和差异位置。异常输出与基线保存到本次批次的 `stress-*/mismatch.pt`。直接运行 Python 精度入口时，`--output` 指定失败张量目录，未指定则使用同样的时间戳默认路径。只跑压测通过仅表示重复执行稳定，不表示已通过 golden 精度验收。

某个测试失败时，该设备的分片停止，其余分片继续。日志末尾汇总每项结果和整个批次的结论。默认 950 模式为 10 项精度与 10 项压测，910B/910C 为 8+8 项。以下是两个 A3 设备并行时的部分汇总行：

```text
FAIL device=8 mode=accuracy case=half-bsnd distribution=uniform exit=1
PASS device=12 mode=accuracy case=half-bnsd-r128 distribution=normal
NOT RUN device=8 mode=accuracy case=interleave-bsnd-r128 distribution=normal
```

在 910B 检查 BF16、S=129、头数 [130,65,65]、R=32 的流水用例和负零复制时，可运行以下命令（选中的流水场景为 HALF、SND 输出、全宽 cache、均匀分布）：

```bash
python -m pytest --test-mode=NPU tests/ops/eagle_split_norm_rope_concat/test_eagle_split_norm_rope_concat.py -k 'bf16-gqa-r32 or value_signed_zero' -s
```

## 生产性能采集

性能入口为 `run_eagle_split_norm_rope_concat_msprof_op.sh`，默认运行 fused 与 composed 两条路径。950 使用九组输入配置，两条路径均为每组独立采集 3 轮，各 27 条记录，完整默认计划共 54 次采集；910B 使用原八组，两条路径均为每组 20 轮，各 160 条记录，共 320 次采集。所有性能输入为 normal，默认 seed=20261017。

调用 `msprof op`（fused）或 `msprof`（composed）前，launcher 检查 `ASCEND_RT_VISIBLE_DEVICES`。设置了该变量时采集入口报错退出。请在可直接寻址已分配目标卡或 die 的环境中采集；仅有掩码环境时，可运行 `--path composed --no-msprof` 测量完整调用的 Event 耗时。[CANN 9.1 环境变量说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0028.html)

910B 和 950 在未设掩码时可用 `--device-id` 或 `DEVICE_ID` 指定物理 NPU ID，省略时由 `tests/tools/select_npu_device.py` 自动选卡一次。在有掩码的 Event 计时中，显式传入进程内可见序号。A3 始终传已分配的 Chip Logic ID。`--runs N` 指定所有所选路径的采集轮数，`--path fused` 或 `--path composed` 只采集对应路径。

融合 profile 在 CPU 上生成 BF16 输入，再复制到目标卡；进程只提交一次目标算子调用，由 `msprof op --kernel-name=EagleSplitNormRopeConcat --warm-up=100 --launch-count=1 --kill=off` 负责预热与采集，并让程序自然完成同步及元数据检查。数值精度由精度入口验收。每轮提取唯一正 Task Duration，输出所有采集记录、最小值、最大值和波动比例；波动超过 10% 时保留警告，可通过 MSPROF_WARMUP 调整预热，并使用全新的结果目录采集完整轮数。最小值表示当前条件下观测到的最佳性能。

```bash
# 910B/950 显式使用物理卡 3，默认采集融合及组合路径；省略 device-id 时自动选卡
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh \
  --device-id 3

# 950 仅新增的 [1,128,128]-head 融合场景，共 3 条记录
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh \
  --device-id 3 --path fused --shape 1 5672 257 128

# 仅原八场景；原 RoPE/layout/cache/R 筛选继续可用
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh \
  --device-id 3 --shape 1 5672 168 128

# 无卡预览：分别显示 910B 和 950 的场景、路径、轮数与总采集数
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh --dry-run
```

组合路径使用完整 QKV Slice/RmsNorm/RoPE/Concat 小算子链作为性能参照，按每个场景的实际 Q/K/V 头数切片。每组默认独立采集轮数为 950 上 3 轮、910B 上 20 轮；每轮包含 5 次进程内预热、默认 10 个 Event 样本，取该轮最小值。`--no-msprof` 仅适用于显式选择 composed 的运行。它在算子间传递 BF16、Norm 后会舍入，不能替代 FP32 Norm → FP32 RoPE → 最终收窄的精度 golden。半宽展开、V 切片及 BNSD 转置均包含在完整调用内。融合与拼接使用相同 CPU 造数、seed 和输入规格；两类计时分别报告，不把各 kernel 时间相加当作完整调用耗时。

`--output DIR` 要求空目录；默认在仓内 `log/eagle_split_norm_rope_concat/perf_时间戳/` 写入记录。汇总只读取本次所选场景与路径，保留逐条 Task Duration 或 Event 样本和对应数据目录。`--shape` 接受上述两种生产 shape，`--rope-type`、`--cache-width`、`--output-layout`、`--rotary-dim` 继续按交集选择。直接 profile Python 使用 `--case half-bsnd-asymmetric` 或 `--shape 1 5672 257 128` 可选择新增场景。

生产 Python 入口及 helper 使用完整测试包路径。独立运行时从仓根使用模块方式，例如 `python -m tests.ops.eagle_split_norm_rope_concat.check_eagle_split_norm_rope_concat_prod_accuracy --help`。两个 shell 入口可从仓外启动，由脚本定位仓根；msprof op 的独立文件入口由启动器将仓根加入 PYTHONPATH。手动进行仿真并直接传入 profile 文件时，也须从仓根设置 `PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`，不添加算子子目录。

组合路径的 Q/K/V 切片和部分 RoPE 的前缀、尾部切片使用 `Tensor.narrow(...).clone(memory_format=torch.contiguous_format)`，每次都生成连续的独立副本，切片拷贝保留在 Event 计时范围内。该路径不依赖缺少 950 ACLNN 实现的 `torch_npu.npu_slice`。采集协议为 `eagle_snrc_layer_event_v7`，汇总器拒绝混入旧切片实现的采集记录。仅重新采集组合路径时，使用 `--path composed` 并指定新的空输出目录。

## Ascend 950 实现与测试

950 按 Head 数动态规划 Token Tile。共享的形状推导允许独立头数进入 Host Tiling，由 950 的每路 1～128 限制或 910B 的 GQA 约束最终校验。910B 调度字段 `outputQueues910`、`tokenTile910` 追加在 950 字段之后，950 路径将两者置 0。部署时用同一提交重建 Host、kernel 和完整 wheel。

Host 按 C=128、127……98 降序搜索，分别计算 Q/K/V 的 `T=max(1,round(C/H))`（正数半整数向上取整），各分支单独增加 T 直到 `100*T*H >= 55*C`。选择满足完整队列和至少一个缓存 token 的最大 C；不能容纳则报错。正常 Tile 不统一，batch/核区间尾部以及缓存子窗口按有效数量缩小。

Host 从查询到的物理 UB（上限 256 KiB）扣除 8 KiB 栈/API 预留，再按 32 B 对齐。令 `Q=Tq*Hq*256`、`K=Tk*Hk*256`、`V=Tv*Hv*256`，两个共用 Q/K 输入槽各占 `max(Q,K)`，三个独立输出槽也各占 `max(Q,K)`；这是共享槽生命周期所需容量，不会将三路的 Token Tile 统一。V 通过单槽绑定队列直接形成目标布局，同一物理缓冲承担搬入和写回，计 V 一次。Weight 占 512 B、Maps 占 1536 B。

RoPE 有两个独立 VECIN 缓存槽，cos、sin 在各槽中分别连续。每槽末尾保留 `align32(2*(128-W))` 字节的掩码读取保护区，因为现有 VF 对半宽/部分宽系数仍发出完整 128 元素的读取。扣除全部输入、输出、固定区和两个保护区后，单槽容量为 `floor(UB_remain/(8*W))` 个 token。FP16 与 BF16 均为两字节。除了不足一个缓存 token 的余数，不再固定保留 4 KiB 或固定分配 244 KiB。

以下为物理 UB=256 KiB、W=96 的规划示例；实际以 `EAGLE950_TILE` Host 日志为准：

| Hq/Hk/Hv | C | Tq/Tk/Tv | 每槽 cos+sin 的 token 数 | 总 UB（B） |
| --- | --- | --- | --- | --- |
| 56/56/56 | 128 | 2/2/2 | 103 | 253312 |
| 1/128/128 | 128 | 128/1/1 | 71 | 253312 |
| 43/65/86 | 128 | 3/2/1 | 82 | 253568 |
| 85/85/85 | 127 | 1/1/1 | 157 | 253312 |

每核负责连续 token 区间，并在 batch 边界截断；cache 窗口也不跨 batch。完整窗口和尾窗口都只提交一条多段 cos DMA、一条多段 sin DMA，段长为 W 个元素、GM/UB 段间空隙均为零。Q/K 各自推进消费者游标，只读一份系数；两者都消费完成才释放当前槽。下一槽提前搬入，与当前槽读者隔离；队列承担 MTE2→Vector 和 Vector→MTE2 复用保护。若 Tile 大于缓存容量，就拆为多个窗口处理。

输出仍按三个独立单槽队列 1→2→3→1 轮转，每次 Q/K 子任务占一槽。三路 Tile 不同时，Q/K 任务数量可能不同；某一路耗尽后继续处理另一路，不强行补齐或统一 Tile。V 使用自己的 Tv 和 Hv，沿用 BSND 多段搬入或 BNSD 按 token 搬入布局变换，以及 MTE2→MTE3 的绑定队列。K 轮先预取首个 V Tile，再计算并提交 K 写回，随后提交 V；当 V Tile 更小时，继续搬运本轮 K 已推进范围内的其余 V Tile。V 单槽重用仍等待 MTE3。

小 shape 和接口测试共用 `test_eagle_split_norm_rope_concat.py` 一个 pytest 入口，共 **215 项**。在原有 175 项基础上，910B 增加 40 项成对输出轮转回归，将 8 组参数与序列长度 1、2、3、128、129 组合。覆盖 56-head、55-head 对齐、57-head 回退、GQA/MQA、R=32/64/96/128、batch 内双 token 与尾组、共享和独立 batch 系数、重复调用及 V 位模式。910B 生产精度沿用八个大 shape 场景。

950 流水场景使用 [1,128,128] 和 [128,65,43]，检查不对称 Tile、系数更换和 V 位模式。共用用例中的 130/512/4096-head 配置在 910B 做数值检查，在 950 检查超限错误。HALF/INTERLEAVE、BSND/BNSD、Full/Compact 均覆盖 R=32/64/96/128，并检查 Q/K 混合容差、cosine、模长比及 V 位模式。

```bash
pytest test_eagle_split_norm_rope_concat.py --test-mode=NPU
```

在命令前设置 `EAGLE_CHECK_HOST_ROUTE=1 ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1` 可核对 Host 路由，`EAGLE950_TILE` 另外记录 C、Tq/Tk/Tv、每槽缓存容量和 UB 用量。Host 日志和离线 CPU 回放均不能证明目标设备对象执行。目标 CANN 编译、实板精度/性能、simulator 未取得执行证据前标为待测；跨版本性能对照须使用同一张已分配健康卡和同一场景、seed、实际加载产物。
