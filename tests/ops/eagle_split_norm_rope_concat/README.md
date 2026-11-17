# EagleSplitNormRopeConcat

`eagle_split_norm_rope_concat` 面向 Ascend 910B，将 packed QKV 拆分、Q/K RMSNorm、全维或部分维度 RoPE 和输出布局变换融合为一个算子。Q、K、V 已完成投影，输入沿头轴依次排列为全部 Q 头、全部 K 头、全部 V 头。

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
| head_counts | int[] 参数，长度为 3，值为 [Nq,Nk,Nv]；依次表示 Q/K/V 头数，均为正，Nk=Nv，Nq 是 Nk 的整数倍，和等于输入头轴；直接传整数列表 |
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

## 实现与 910B 边界

Host 注册、形状推导、tiling 和内核位于 `csrc/ops/eagle_split_norm_rope_concat/`；CANN OpType 为 `EagleSplitNormRopeConcat`，入口为 `aclnnEagleSplitNormRopeConcat`。内核在 AIV 间划分独立任务，直接写入目标布局，不使用 Cube 或核间同步。

CANN 层的 `norm_type`、`rope_type` 同样为必填属性，`output_bnsd` 默认 true。CANN 整数属性用 `rotary_dim=0` 表示未指定；PyTorch 接口用 `None` 表示未指定，并在调用 CANN 前解析为实际 D。PyTorch 显式传入 0 会报错。

HALF 与 INTERLEAVE 均使用批处理，每个任务处理同一 token 的最多 64 个连续 head，覆盖 R=32/64/96/128。Q/K 权重在核内缓存；相邻任务命中同一 RoPE token 时复用有效的 cos/sin 及 FP32 转换结果。平方、按行归约、倒数、广播和 RoPE 均在 Vector 完成，输入、输出和 V 使用双缓冲。Nk=Nv，V 与 K 使用相同的任务划分。FP16/BF16 分别显式匹配 tiling key 11/12。

INTERLEAVE 复用 16×128 个 uint32 字节偏移，将每行相邻元素通过 Gather 交换到已有临时区。每批最多四次 Gather，最后一组按实际 head 数处理；随后使用 repeats 跨 head 执行乘加。R=96 的两个列段分别使用 mask=64 和 mask=32，仅更新前 96 维，后 32 维保留 FP32 归一化结果，最后整行收窄。

HALF 单核 UB 分配为 170496 B（166.5 KiB）；INTERLEAVE 额外分配 8192 B 索引和 512 B 符号向量，共 179200 B（175 KiB）。两者均包含 FP32 工作区及参数区各 256 B 的间隔；间隔改变相对 bank 位置，性能作用需实测。Host 从至多 192 KiB 物理容量中扣除 8 KiB API 预留和 256 B 系统预留，默认预算为 188160 B；资源不足直接报错，没有逐 head 回退路径。BNSD 输出还检查 DMA 目标步长的 uint32 范围。UB 分区基址及 GM 行长度按 32 B 对齐，Gather 表内的 FP32 元素偏移按 4 B 对齐；队列承担流水间同步，显式事件通过 TPipe 获取。

上述 UB 数字对应全宽 cache。半宽模式额外常驻 128 个 uint32 展开索引（512 B），HALF/INTERLEAVE 总量分别为 171008/179712 B，均在预算内。cache 更新时，先将半宽 cos/sin Cast 到暂未用于平方的 tmp 区，再 Gather 到现有 FP32 参数区；后续复用展开结果，不按 head 重复展开。Host tiling 显式传递实际 cache 宽度，GM token 步长使用该宽度。

构建环境就绪后，在仓根通过 whl 构建并安装。安装命令中的文件名须替换为本次生成的 whl 文件名：

```bash
unset MINDIESD_SKIP_OPS_BUILD SKIP_ALL_OPS_PLUGIN_BUILD
python setup.py bdist_wheel
python -m pip install --force-reinstall --no-deps "dist/本次生成的文件名.whl"
```

`setup.py` 会调用仓内构建脚本，将算子及对应 plugin 一并打入 whl；安装该 whl 即完成部署。`--force-reinstall` 用于替换已安装的同版本包，`--no-deps` 保留现有依赖环境。910B 注册不代表其它架构可用。

## 三层测试

910B 大 shape 测试使用 BSND packed QKV `[1,5672,168,128]`、头数 `[56,56,56]`、BF16，固定 `norm_type=3`、`eps=1e-5`。权重各为 `[128]`，cos/sin 为 `[1,5672,1,W]`。场景表交叉选择 RoPE、输出布局、cache 宽度与 R=96/128，共 8 组；不展开 16 组笛卡尔积。每个 R 均覆盖两种 RoPE 与全宽/半宽。

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

Q/K/V 的 BSND 输出均为 `[1,5672,56,128]`，BNSD 为 `[1,56,5672,128]`。精度与压测各 8 条，分布按表交叉分配，不再乘两种分布；直接 Python 的 `--distribution all` 使用该表。性能统一正态分布，融合 8 条和完整 QKV 拼接 8 条逐项对应，共 16 条，不超过 20 条；同一配对使用相同输入配置、CPU 造数后端和 seed。半宽 cache 的 BF16 展开包含在拼接完整调用计时内。

半宽专项用例覆盖四种 R、两种 RoPE、两种 dtype、两种 Norm、两种输出布局、SND/BSND、三种 cache rank 及 batch 共享/独立，并与显式展开的全宽调用逐位比较。Gather 边界、head tile 边界、流水及常驻 cache 复用、FP32 中间精度和默认 R 用例同时覆盖两种宽度；真实入口、FakeTensor/Meta 验证宽度约束，拒绝既不是 R 也不是 R/2 的宽度及 cos/sin 宽度不一致。

| 层级 | 入口 | 对照与判定 |
| --- | --- | --- |
| 小 shape 与接口 | `test_eagle_split_norm_rope_concat.py` | CPU 分阶段 golden；Q/K 按 FLOAT16 或 BFLOAT16 档的混合容差、cosine、模长比均为硬门禁；V 精确相等；同时检查 dispatcher schema、默认值、整数头数校验、FakeTensor/Meta 形状与布局 |
| 生产精度 | `run_eagle_split_norm_rope_concat_prod_accuracy.sh` | 同一 NPU 上用小算子组合 golden；cosine、模长比为硬门禁，混合容差仅诊断；V 精确相等 |
| 稳定性压测 | 同一精度入口，`--mode stress` | 固定输入，每种分布默认重复 1000 轮，Q/K/V 相对首次输出逐位一致 |
| 生产性能 | `run_eagle_split_norm_rope_concat_msprof_op.sh` | 融合算子用 msprof op；完整 QKV 小算子链用 msprof + Event；默认全跑，`--path` 可选路径 |

小 shape 与接口共 100 条参数化展开用例，集中在唯一的 `test_eagle_split_norm_rope_concat.py` 中：38 条数值/语义回归，62 条接口检查。用例通过显式参数元组交叉选择，不对 dtype、Norm、布局、分布继续展开笛卡尔积。主要随机精度矩阵共 26 条，均匀、正态各 13 条；其余为默认值、权重语义和特殊值等专项回归。正态用例逐输入采样 mean∈[-5,5]、std∈[0.1,2]，不钳位，固定 seed 并打印参数。

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

上述数量按静态参数化展开核算；重复 seed、同输入重复调用及等价关系比较不另算独立场景。当前未执行 pytest 收集或 NPU 实测；环境可用时可用 `pytest --collect-only -q tests/ops/eagle_split_norm_rope_concat/test_eagle_split_norm_rope_concat.py` 核对 100 条。代表组合覆盖各维取值和关键依赖，不表示覆盖全部两两组合。

默认值用例直接调用公开接口，验证省略 rotary_dim 等价于显式 D、默认输出 BNSD，以及两种 RMSNorm 在 D=128 的 HALF 和 INTERLEAVE 批处理路径上的精度。无权重模式额外比较传 None 与提供权重的输出逐位一致；带权重模式缺失任一权重、未实现的归一化模式及非法 RoPE 编号均须报错。真实调用代表性检查 D=64/127/129，FakeTensor 检查 D=32/256 的拒绝行为。接口测试同时检查两个模式参数必填、MHA/GQA/MQA 的具体输出头数，以及未开启数据依赖形状支持时的 FakeTensor/Meta 推导。

批处理边界并入 Gather 与流水用例，检查完整 tile、尾 tile、Q/K tile 数不同的 GQA、两种输出布局，并在流水用例中要求同一输入重复执行的输出逐位一致。INTERLEAVE R=96 检查 15/16/17-head Gather 分组边界及 63/64/65-head 批次边界，交叉 FP16/BF16、两种宽度和两种输出布局，后 32 维单独比较。HALF 多 tile 与尾块由 [130,65,65] 的流水场景覆盖。

流水用例使用 S=1/81/129、R=32/64/96/128，覆盖双缓冲多轮复用、只有一个 head 的尾 tile、部分核没有 K/V 任务的 GQA，以及相邻调用更换输入数据；FP16/BF16、两种分布和输出布局均检查 Q/K 精度及 V 的逐位保持。

常驻 RoPE 复用用例使用 B=2、S=1、头数 [4096,1,1]，检查同核多个 head tile、Q/K 切换以及共享和独立 batch cache。单次收窄回归用例令输入和权重为 1、eps=2、cos=1.375、sin=0，要求 Q/K 前缀等于 BF16(1.375/√3)=0.79296875，直接检查 FP32 norm 到 RoPE 的精度边界。

负零用例先确认输入 V 的 BF16 位模式为 `0x8000`，再在 CPU 上将输出按 int16 视图逐位比较，同时检查 shape、dtype 和连续性。精度检查保留最大误差坐标、实际值、参考值和失败指标，便于复现对应测试点。

Q/K 的约定数值路径全程 FP32 计算，仅最终输出舍入到输入 dtype，分别按下表取整套混合容差参数。内部 FP32 计算及指标统计使用的 dtype 不改变验收档位；V 只做复制和布局变换，始终要求位模式精确一致。

| Q/K 实际输出 dtype | 验收精度档位 | atol | rtol | 最低匹配率 | 最大误差兜底值 |
| --- | --- | --- | --- | --- | --- |
| FP16 | FLOAT16 | $2^{-10}$ | $2^{-10}$ | 0.99 | 0.1 |
| BF16 | BFLOAT16 | $2^{-7}$ | $2^{-7}$ | 0.99 | 0.1 |

最大绝对误差上限为 $\max(0.1,32\times\mathrm{ULP})$。ULP 按最大误差点的 golden 值及实际输出 dtype 计算：FP16 输出用 FP16 间距，BF16 输出用 BF16 间距。cosine 至少 0.99，模长比在 [0.9,1.1]。小 shape 的三项指标均为硬门禁；生产精度仅 cosine、模长比为硬门禁，混合容差为软约束。小 shape 的指标在 CPU FP64 计算；生产指标留在 NPU，以 FP32 计算，仅打印标量结果。

逐输出日志记录 `tolerance_dtype`、选档依据、实际 `output_dtype`、容差与兜底值、匹配率、最大误差点的坐标和 golden、ULP、32 倍 ULP、最终上限，以及各项通过状态；V 单独打印精确比较结果。

## 精度运行

先 source CANN 环境，确保 torch_npu 和当前 MindIE-SD 算子产物已安装。所有 Python 入口先选卡，再导入 MindIE-SD。小 shape 固定物理 0 卡；生产脚本未指定卡时调用仓内 `tests/tools/select_npu_device.py`，也可指定物理卡号。

```bash
unset ASCEND_RT_VISIBLE_DEVICES
pytest tests/ops/eagle_split_norm_rope_concat/test_eagle_split_norm_rope_concat.py -s

bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id 3
```

生产精度入口默认 `--mode all`：先按分配表执行 8 条单次精度测试，再对这 8 组固定输入分别压测 1000 轮。使用 `--shape B S Htotal D`、`--heads Q K V`、`--rope-type all|half|interleave`、`--cache-width all|full|half` 、`--output-layout all|bsnd|bnsd` 和 `--rotary-dim all|96|128` 直接选择，无需查场景 ID。shape 默认 `1 5672 168 128`，heads 默认 `56 56 56`；四个语义选择项默认 all，指定多项时取交集。每组压测在循环外固定输入，以独立 CPU 副本保存首次输出；1000 轮不包含基线调用。每轮同步后检查全部 Q/K/V 的 shape、dtype、设备、连续性及原始 BF16 位模式，包含正负零；首次输出必须有限，输入摘要在基线前后、结束及输出异常时复核。循环内释放当轮结果，每 100 轮和最后一轮打印进度，不重新计算 golden。

```bash
# 只跑单次精度（8 配置各一种分布，共 8 条）
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id 3 --mode accuracy

# 只测相邻旋转、BNSD 全宽 cache 场景（分配表为 uniform）
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id 3 --mode accuracy --rope-type interleave --cache-width full --output-layout bnsd

# 自定义 BSND shape 与 GQA 头数，检查 INTERLEAVE + 半宽 cache + BNSD
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh \
  --device-id 3 --mode accuracy --shape 2 1024 12 128 --heads 8 2 2 \
  --rope-type interleave --cache-width half --output-layout bnsd

# 无设备访问地预览选择结果；只选半宽 cache 时得到 4 个配置
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --cache-width half --dry-run

# 只跑 R=128 精度，覆盖 4 组代表组合
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id 3 --mode accuracy --rotary-dim 128

# 只跑压测，覆盖默认轮数
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_prod_accuracy.sh --device-id 3 --mode stress --stress-runs 2000
```

精度 Python 入口接受相同的 shape、heads 和四个语义选择参数；`--dry-run` 由 shell 提供，不启动 Python/NPU、不创建输出目录。自定义 shape 用于真实输入和输出契约，要求各维为正 int32、输入总元素数不超过 2147483647、D=128、Q+K+V=Htotal、K=V、Q 能被 K 整除。当前大 shape 精度固定 BF16、norm_type=3、eps=1e-5；R 按代表场景取 96 或 128，cos/sin 为 `[B,S,1,R]`（full）或 `[B,S,1,R/2]`（half）。R=32/64 及 FP16 由小 shape 覆盖；小 shape 已包含 R=128 的两种 RoPE 半宽/全宽逐位等价与默认 R 回归，仍共 100 条。显式 shape 不自动改变 heads，二者不匹配直接报错。未指定语义选择时，自定义 shape 也运行 8 个配置。选择项只过滤表内代表配置，未匹配时报错，不自动补齐未列组合。`--stress-runs` 必须为正整数；`--help` 查看参数。精度与压测默认在脚本所属仓库创建 `log/eagle_split_norm_rope_concat/accuracy_YYYYMMDD_HHMMSS/`，实时输出同时保存为该批次的 `accuracy.log`；`--output DIR` 可指定空目录。同秒批次重名时拒绝覆盖，可稍后重试或指定新目录，不依赖启动目录。压测遵循本算子的逐位确定性要求，任意一位变化即失败，同时报告轮次、分布、输出、差异数量、最大绝对/相对误差和差异位置。异常输出与基线保存到本次批次的 `stress-*/mismatch.pt`。直接运行 Python 精度入口时，`--output` 指定失败张量目录，未指定则使用同样的时间戳默认路径。只跑压测通过仅表示重复执行稳定，不表示已通过 golden 精度验收。

任一测试失败，shell 立即非零退出。全部所选项目通过后，脚本末尾逐项汇总；默认模式为 8 行精度与 8 行压测结果。以下为选择 HALF、BSND 输出及全宽 cache 时的结果，压测轮数按实际参数显示，单选测试类型仅显示所选项目：

```text
PASS accuracy shape=1 5672 168 128 heads=56 56 56 rope=half output=bsnd cache_width=full R=96 dtype=bf16 distribution=uniform outputs=q,k,v (mixed tolerance diagnostic only)
PASS stress shape=1 5672 168 128 heads=56 56 56 rope=half output=bsnd cache_width=full R=96 dtype=bf16 distribution=uniform runs=1000 outputs=q,k,v bit_exact
```

只检查 BF16、S=129、头数 [130,65,65]、R=32 的流水用例和负零复制时，可运行以下命令（选中的流水场景为 HALF、SND 输出、全宽 cache、均匀分布）：

```bash
pytest tests/ops/eagle_split_norm_rope_concat/test_eagle_split_norm_rope_concat.py -k 'bf16-gqa-r32 or value_signed_zero' -s
```

## 生产性能采集

测试目录只有两个 shell 入口：`run_eagle_split_norm_rope_concat_prod_accuracy.sh` 用于精度，`run_eagle_split_norm_rope_concat_msprof_op.sh` 用于性能。性能入口使用同一物理卡、相同 seed 和输入规格，默认依次采集以下两条路径。每条路径只使用正态分布，覆盖 8 个场景，完成该路径全部场景与轮次后再切换路径。

| 路径 | 计时范围 | 工具与指标 | 输出 |
| --- | --- | --- | --- |
| `fused` | packed QKV 直接调用融合算子 | `msprof op` / Task Duration(us) | Q、K、V |
| `composed` | Q/K 链后增加从 packed QKV 切出 V | `msprof` + Event / Layer Duration(us) | Q、K、V |

融合路径调用 `profile_eagle_split_norm_rope_concat_prod.py`，每进程只发出一次目标算子调用，由 `msprof op --kernel-name=EagleSplitNormRopeConcat --warm-up=20 --launch-count=1 --kill=on` 采集。每个场景默认独立运行 20 轮，逐轮从日志提取唯一有效的 Task Duration，报告全部样本、最小值、最大值和波动比例。`MSPROF_WARMUP` 可覆盖工具预热次数（0～500）；同一场景的 (max−min)/min 超过 10% 时考虑增加预热，在新目录按同卡、同输入重新采集完整 20 轮，仍波动则保留警告。提频不足只是待验证原因。融合路径不创建 Event，也不进入多算子计时脚本。

融合 profile 在 CPU 上按目标 BF16 规格生成 packed QKV、两路归一化权重与 cos/sin，再拷贝到指定 NPU；头数列表原样传递。每个输入仍独立采样正态均值和标准差，并记录 seed、采样参数及实际输入元数据。目标调用前后同步，返回后只检查输出数量、shape、dtype 和连续性；有限值及 golden 比较由精度入口负责。融合与拼接均用同一 CPU 造数函数、seed、输入顺序和配置生成输入后拷贝到 NPU，配对输入可逐位复现。

Q/K 性能拼接链分别调用 torch_npu.npu_slice 从头轴提取 [1,5672,56,128]，将 BF16 输入和权重直接交给 torch_npu.npu_rms_norm。归一化输出为 BF16，辅助 rstd 为 [1,5672,56,1] FP32；前 R 维交给 torch_npu.npu_rotary_mul，使用对应的 half 或 interleave 模式及 BF16 cos/sin。R=96 时切出旋转前缀，并拼接归一化尾部 32 维；R=128 时直接整行旋转，省略前缀切片和尾部拼接。半宽 cache 在 BF16 下展开，HALF 使用两份拼接，INTERLEAVE 使用相邻元素重复。完整 QKV 基线还切出头轴偏移 112 的 V；cache 展开、V 拆分和 BNSD 输出转置与连续化均包含在完整调用计时内。

性能基线使用原生算子之间传递 BF16 张量的执行方式，RMSNorm 输出会发生 BF16 舍入，内部计算精度由原生算子决定。融合内核与独立精度 golden 仍按 FP32 Norm → FP32 RoPE → 最终输出舍入执行。因此性能配对的输入配置、输出 shape 和 dtype 一致，但中间舍入不同；性能拼接链不作为融合算子的精度 golden。

拼接入口设置 jit_compile=False，日志记录运行时版本、原生算子名称、实际输入及中间 Tensor 的 shape、dtype、device、stride 和连续性。结果协议为 eagle_snrc_layer_event_v6，配置包含实际 cache shape、compute_dtype=native_operator_internal、intermediate_dtype=bfloat16 和 norm_output_rounding=True；汇总器拒绝旧协议结果，以免混合不同计算路径的耗时。

完整 QKV 拼接路径每进程固定预热 **5 次**，默认正式调用 **10 次**，使用独立 Event 对包围每次完整调用。全部正式调用提交后才读取 Event 时间，每轮取最小值；数据构造、日志和最后一次输出的形状、dtype、有限值检查均在计时区间外。每条路径每个场景默认独立采集 **20 轮**。进程内复用输入，结果反映热缓存条件；Event 时间包含完整调用所提交的设备任务，不使用各 kernel 的 CSV 耗时求和。

```bash
# 默认全跑：融合用 msprof op，拼接用 msprof + Event
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh --device-id 3

# 只跑融合算子
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh --device-id 3 --path fused

# 只跑融合算子的 R=128、相邻旋转、半宽 cache、BNSD 输出
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh --device-id 3 --path fused --rope-type interleave --cache-width half --output-layout bnsd --rotary-dim 128

# 只跑完整 QKV 小算子拼接基线
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh --device-id 3 --path composed

# 仅拼接路径可改为直接 Event 计时，不调用 msprof
bash tests/ops/eagle_split_norm_rope_concat/run_eagle_split_norm_rope_concat_msprof_op.sh --device-id 3 --path composed --no-msprof
```

不传 `--path` 时运行融合与完整 QKV 拼接；`--path fused` 只选融合，`--path composed` 只选完整 QKV 拼接。路径参数仅接受这两个值，不提供同义别名或显式 `all`，重复传入时报错。性能 shell 使用与精度相同的 RoPE、cache 宽度、输出布局及 R 选择项；`--shape` 仅接受 `1 5672 168 128`，其它性能尺寸明确报错。`--dry-run` 可无设备预览配置数及总测量场景数。默认 8 组 × 2 路径 = 16 条性能场景，各重复 20 轮，共 320 次采集；单路径为 160 次，单场景单路径为 20 次。采集轮数不增加场景数。R=128 筛选为 4 组 × 2 路径 = 8 条性能场景。汇总只读取本次所选路径和场景；Task Duration 与完整调用 Event 分别报告，不混算速度比。`--no-msprof` 与包含融合的路径组合时报错。两个单次 profile Python 也接受语义选择项；显式筛选须匹配唯一场景，未传选择项默认 HALF、BSND、R=96、全宽 cache。

可用 `--iterations` 指定拼接路径每轮 Event 样本数，`--runs` 指定各路径独立轮数，`--seed` 和 `--output` 指定造数种子及空目录。融合工具预热默认 20 次，拼接预热固定为 5 次。省略卡号时只调用一次选卡工具并保存报告，后续全部路径沿用该物理卡。默认在本脚本所属仓库根目录下创建 `log/eagle_split_norm_rope_concat/perf_YYYYMMDD_HHMMSS/` 时间戳目录，与启动命令时所在目录无关。同秒批次重名时拒绝覆盖，可稍后重试或指定新目录。

采集目录按 `<case>/normal/run_NN/<path>/` 组织。融合路径保留 msprof op 原始产物、`msprof.log` 和 `task_duration_us.txt`，在各场景的 `normal/` 目录生成 `fused_task_duration_summary.txt`。融合采集不要求 layer 的 Event JSON 或 `PROF_*` 时间线。

拼接路径的工具模式保留 `profiled_timing.json`、`msprof.log`、`event_samples.csv` 以及唯一 `PROF_*` 下的非空 `op_summary*.csv` 和 `msprof_*.json` 时间线；直接计时模式保留 `timing.json`、`timing.log` 和 `event_samples.csv`。两种模式分别汇总，不混用。根目录 `summary.txt` 仅汇总选中的拼接路径，按场景连续列出 Scenario、Test、Layer Duration(us)、DataDir，并给出各场景最小值；`summary.json` 保存所选拼接路径、全部 Event 样本和来源目录。融合与 `composed` 均返回 Q/K/V，工作量边界对应。单 kernel 的 Task Duration 与完整调用 Event 时间口径不同。

全部所选路径、场景和轮次采集及校验成功后，终端末尾先展示一张连续逐轮明细表：列为 `Scenario`、`Test`、`Metric`、`Duration(us)`、`DataDir`，按实际列宽补齐并以 Tab 对齐（默认 8 字符制表位）。逐轮表和最小值总览均使用终端纯文本，不使用竖线分隔。融合与拼接的全部轮次、耗时和落盘路径集中在表内，紧接 `Final performance overview` 各场景最小值汇总，中间不插入分路径统计或说明。融合 DataDir 是含 `msprof.log` 的轮次目录；拼接为该轮 `PROF_*`，直接模式为轮次目录。预热信息和波动提示放在最小值汇总之后。逐轮表另存 `capture_details.txt`，完整回显保存到根目录 `final_summary.txt`，各路径原有明细文件继续保留。单选模式只包含所选路径与场景；采集或汇总失败时非零退出并显示错误，不生成完整成功总报告。

Performance varies across runs. The minimum is the best observed performance under the current conditions, not a hardware limit or evidence that variability has disappeared.

## 910B4 指令仿真

在已加载 CANN 环境、安装当前算子产物的 MindIE-SD 仓库根目录执行以下命令。half-bsnd 使用 BF16 packed QKV [1,5672,168,128]，头数 [56,56,56]，权重各 [128]，cos/sin 各 [1,5672,1,96]；norm_type=3、rope_type=1、rotary_dim=96、eps=1e-5，三路输出均为 BSND [1,5672,56,128]。HALF 表示 RoPE 的前后半区配对方式，不表示 FP16。

```bash
unset ASCEND_RT_VISIBLE_DEVICES
msprof op simulator \
  --soc-version=Ascend910B4 \
  --kernel-name=EagleSplitNormRopeConcat \
  --core-id="0|1" \
  --timeout=10 \
  --launch-count=1 \
  --output="$PWD/log/eagle_split_norm_rope_concat/simulator_half_bsnd_bf16_$(date +%Y%m%d_%H%M%S)" \
  python3 -u tests/ops/eagle_split_norm_rope_concat/profile_eagle_split_norm_rope_concat_prod.py \
  --device-id 0 --rope-type half --output-layout bsnd --cache-width full --rotary-dim 96 --seed 20261017
```

直接传入独立 Python 文件，保留无缓冲输出；不要把多行源码嵌入 msprof 的 python3 -c 参数。入口依次打印六个阶段：设备初始化、CPU 造数、拷贝、目标调用、调用返回并同步、完成元数据检查。若未到目标调用阶段，先检查输入准备和传参，不能仅增加 timeout。--kernel-name 过滤采集对象，不会跳过程序中其他设备运算。

本命令只采集一次目标调用并解析逻辑核 0、1，timeout 单位为分钟；不执行上板性能的预热和 20 轮汇总。结果位于仓根 log/eagle_split_norm_rope_concat 下的本次 simulator 目录。完成标记或 OPPROF 目录不代表有效采集，需打开 trace.json 等结果确认目标 kernel、所选核和非空指令事件；超时流水只能按已执行部分分析，仿真主机耗时不作为设备性能。
