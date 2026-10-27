# 量化

MindIE SD 提供两类量化能力，分别作用于模型的不同部分：

- **Linear 量化**：对线性层的权重和激活值进行低比特处理（INT8/FP8/W8A16 等），减少模型存储空间和计算开销。
- **FA 量化**：对注意力计算中的 Q/K/V 激活值进行 FP8 块量化，降低注意力计算的显存带宽需求。

以下两节分别介绍两种量化的原理和使用方法。

## Linear量化

### 通用原理

量化是将模型的权重（weight）和激活值（activation）从高精度（如 FP32）映射到低精度（如 INT8、FP8）的过程。低精度计算可以减少显存占用和带宽需求，提升推理吞吐。

量化根据是否需要重训练，分为训练后量化（Post-Training Quantization, PTQ）和量化感知训练（Quantization-Aware Training，QAT）。本章节以 PTQ 量化为主，主要分为以下三种类型：

- **动态量化**：仅离线量化权重，在推理时动态计算激活值的量化因子。
- **静态量化**：权重和激活值都是离线量化。
- **Time-Aware 量化**：根据时间维度动态调整量化策略。

下图展示了 INT8 量化示例，将 FP32 映射到 INT8。其中 `[-max(xf), max(xf)]` 是量化前浮点范围，`[-128, 127]` 是量化后范围。

![INT8 量化示意图](../../figures/int8_image.png)

### 技术特点

本仓库通过 `quantize` 接口统一处理 Linear 量化，以下为算法枚举；可用入口与选型限制见“量化方案推荐”。

**权重量化**（仅量化权重，激活值保持原始精度）：

| 算法 | 权重精度 | 说明 |
|------|----------|------|
| W8A16 | INT8 | 基础权重量化 |
| W4A16 | INT4 | 更高压缩比 |
| W4A16_AWQ | INT4 + AWQ | 激活感知的权重量化 |
| W8A16_GPTQ | INT8 + GPTQ | 基于 GPTQ 后训练的权重量化 |
| W4A16_GPTQ | INT4 + GPTQ | 同上，INT4 版本 |

**权重激活量化**（权重和激活值均量化，计算在低精度下完成）：

| 算法 | 量化粒度 | 说明 |
|------|----------|------|
| W8A8 | 逐层 | 基础 INT8 权重激活量化 |
| W8A8_TIMESTEP | 逐层 + 时间步 | 推理中动态切换量化策略 |
| W8A8_DYNAMIC | 逐层 | 激活值动态量化 |
| W8A8_PER_CHANNEL | 逐通道 | 按通道粒度量化 |
| W8A8_PER_TENSOR | 逐张量 | 按张量粒度量化 |
| W8A8_MXFP8 | 逐层 | MXFP8 格式量化 |
| W4A4_DYNAMIC | 逐 token + 逐通道 | INT4 权重激活量化 |
| W4A4_MXFP4_SVD | 逐层 | 当前入口不支持 |
| W4A4_MXFP4_DUALSCALE | 逐层 | MXFP4 双尺度量化 |
| W4A4_MXFP4_DYNAMIC | 逐 token + 逐通道 | MXFP4 动态量化 |
| W4A4_MXFP4_FOUROVERSIX_DYNAMIC | 逐 token + 逐通道 | MXFP4 量化，激活走 `quant_four_over_six_a5`（4/6 自适应） |

### 量化方案推荐

离线量化使用 [msModelSlim](https://gitcode.com/Ascend/msmodelslim) 生成权重和描述文件，MindIE SD 负责加载产物并执行推理。算法原理和工具参数见[量化算法总览](https://gitcode.com/Ascend/msmodelslim/tree/master/docs/zh/knowledge_base/quantization_algorithms)。优先使用工具仓中目标模型已有的量化配方，再按下表选择本仓推理路径。

| 需求 | MindIE SD 推理路径 | msModelSlim 算法与推荐理由 |
| --- | --- | --- |
| 建立 INT8 基线 | `W8A8` / `W8A8_DYNAMIC` | [MinMax](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/minmax/usage_minmax.md) 作为参数估计起点；激活离群明显时，在工具侧增加 [SmoothQuant](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/smooth_quant/usage_smooth_quant.md)，平滑激活与权重分布后再导出。动态路径在推理时计算激活 scale |
| 没有校准数据，优先保留激活精度 | `W8A16` | [HQQ](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/hqq/usage_hqq.md) 使用 INT8 per-channel 权重、浮点激活，无需校准数据；适合先降低权重存储开销 |
| MXFP8 权重重构误差较大 | `W8A8_MXFP8` | [MSE_Round](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/mse_round/usage_mse_round.md) 比较每个 block 的 ceil/floor shared exponent，按 MSE 选择权重缩放；推理侧仍使用 MXFP8 动态激活量化 |
| MXFP4 压缩与计算 | `W4A4_MXFP4_DYNAMIC` | 优先尝试 [Ceil_X](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/ceil_x/usage_ceil_x.md)，工具默认除数为 7.25，通过 ceil shared exponent 减少大值截断；精度不满足时再评估工具侧搜索或局部回退 |
| MXFP4 分块缩放误差明显 | `W4A4_MXFP4_FOUROVERSIX_DYNAMIC` | [FouroverSix](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/fouroversix/usage_fouroversix.md) 在 Scale-to-4 与 Scale-to-6 中按块选择重构误差更小的方案。工具侧优化权重；本仓该路径另外对激活使用 4/6 自适应算子，仅支持 A5，要求 bf16、尾轴 32 对齐 |
| MXFP4 异常通道影响较大 | `W4A4_MXFP4_DUALSCALE` | [DualScale](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/dual_scale/usage_dual_scale.md) 引入第二级 scale；仅在目标模型和算子支持对应两级尺度布局时使用，不能直接替换普通 MXFP4 权重 |
| 扩散时间步对低比特敏感 | `W8A8_TIMESTEP` 或 MXFP4 时间步策略 | 使用 [Linear Quant](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/linear_quant/usage_linear_quant.md) 生成兼容产物；本仓通过 `TimestepManager` 控制策略。MXFP4 可在敏感步切换 W4A8，权重仍保持 MXFP4，示例见下文 |

工具算法名不等于本仓 `QuantAlgorithm` 名称。例如 `ceil_x`、`mse_round` 是生成权重时的参数估计算法，不应直接写成量化描述符中的层类型。导出后须核对描述符枚举值、权重名称、scale 粒度和布局；模型适配及导出要求见 [Linear Quant](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/linear_quant/usage_linear_quant.md)。

当前权重量化入口仅实现 `W8A16` / `W4A16`，AWQ/GPTQ 枚举没有对应替换分支；`W4A4_MXFP4_SVD` 会报不支持。因此不将工具仓中的 AWQ、GPTQ 或 SVDQuant 直接推荐为本仓可运行路径。

对比时固定模型、输入、随机种子和采样步数，先检查层输出误差和生成质量，再测端到端时延及峰值显存。精度不达标时优先回退敏感层或时间步；位宽更低不代表实际更快。

### 接口和使用

所有 Linear 量化算法通过 `quantize` 接口统一触发。

```python
from mindiesd import quantize
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | `nn.Module` | 是 | - | 已初始化的浮点模型 |
| `quant_des_path` | `str` | 否 | `None` | 量化描述符 JSON 路径；未作为位置参数传入时，需要配置 `QuantConfig.quant_des_path` |
| `quant_config` | `QuantConfig` | 否 | `None` | 统一量化配置，可承载 `quant_des_path`、`dtype`、`use_nz`、时间步策略和 `mxfp4_scale_alg` |

`quantize` 会先解析量化描述符 JSON 为 `QuantConfig`，再与用户传入的 `quant_config` 合并；同一字段同时存在时，以用户传入的配置为准。旧接口中的 `timestep_config`、`timestep_policy`、`dtype`、`use_nz` 仍可继续传入，内部会自动收口到 `QuantConfig`。量化描述符路径既可以继续作为 `quantize` 的第二个参数传入，也可以写入 `QuantConfig(quant_des_path=...)`。

#### 使用示例

基础量化：

```python
model = from_pretrain()
model = quantize(model, "quant_model_description_w8a16_0.json")
model.to("npu")
```

等价配置式写法：

```python
from mindiesd import QuantConfig, quantize

quant_config = QuantConfig(quant_des_path="quant_model_description_w8a16_0.json")
model = quantize(model, quant_config=quant_config)
model.to("npu")
```

时间步量化：

```python
from mindiesd import QuantConfig, TimestepManager, TimestepPolicyConfig

timestep_policy = TimestepPolicyConfig()
timestep_policy.register(range(0, 10), "static", target="w8a8_static_linear")

quant_config = QuantConfig(timestep_config=timestep_policy)
model = quantize(model, "quant_model_description_w8a8_timestep_0.json", quant_config=quant_config)

for i, t in enumerate(timesteps):
    TimestepManager.set_timestep_idx(i)
    ...
```

MXFP4 时间步回退：

```python
from mindiesd import QuantConfig, TimestepManager, TimestepPolicyConfig, quantize

timestep_policy = TimestepPolicyConfig()
timestep_policy.register(range(0, 4), "W4A8", target="w4a4_linear")
timestep_policy.register(range(4, 50), "W4A4", target="w4a4_linear")

quant_config = QuantConfig(
    timestep_config=timestep_policy,
    mxfp4_scale_alg=2,
)

model = quantize(model, "quant_model_description_w4a4_mxfp4_0.json", quant_config=quant_config)

for i, timestep in enumerate(timesteps):
    TimestepManager.set_timestep_idx(i)
    noise_pred = model(latents, timestep, encoder_hidden_states)
```

模型侧需要在每个 denoise step 前设置 `TimestepManager.set_timestep_idx(i)`。这和 Wan2.2 `wan/text2video.py` 中按 step 遍历 `timesteps` 的使用方式一致，但本仓的策略语义不同：这里的 Linear 是 `W4A4` 与 `W4A8` 切换，不是原有动静态量化切换。Linear/MM 回退只切换激活值量化精度，权重仍保持 MXFP4。

#### 4/6 自适应激活量化（fouroversix）

`W4A4_MXFP4_FOUROVERSIX_DYNAMIC` 复用 `W4A4_MXFP4_DYNAMIC` 的权重通路，将激活量化替换为自定义算子 `quant_four_over_six_a5`：每个 32 元素块在 D4/D6 两组 E8M0 缩放之间按量化误差自适应选择。时间步 `W4A8` 回退与 `W4A4_MXFP4_DYNAMIC` 一致。

- 权重量化产物与 `W4A4_MXFP4_DYNAMIC` 完全一致，可直接复用其离线权重；描述符 `model_quant_type` 写 `W4A4_MXFP4_FOUROVERSIX` 时，按命名规则权重文件为 `quant_model_weight_w4a4_mxfp4_fouroversix_{rank}.safetensors`。
- 算子要求 bf16 输入且激活尾轴（`in_features`）32 对齐，仅Ascend 950PR&950DT系列产品支持；不满足对齐要求时层构造期即报错。
- `mxfp4_scale_alg` / `mxfp4_dst_type_max` 仅作用于权重量化；激活固定为 4/6 自适应组合。
- 在线量化路径同样支持：`OnlineQuantConfig(quant_type=W4A4_MXFP4_FOUROVERSIX_DYNAMIC)`。

#### 量化权重文件命名

量化权重和描述符文件由 msmodelslim 工具导出，命名规则如下：

- 权重文件：`quant_model_weight_{quant_algo.lower()}_{rank}.safetensors`
- 描述符文件：`quant_model_description_{quant_algo.lower()}_{rank}.json`

单卡量化时 `rank` 为 0，多卡并行时各 rank 分别对应其编号。

#### 量化描述文件校验

`quantize` 在构造量化配置和打开权重文件前，对描述文件进行整体校验。当前支持 MindIE SD 的扁平 JSON 格式：

```json
{
  "model_quant_type": "W8A8",
  "blocks.0.proj.weight": "W8A8",
  "blocks.0.proj.bias": "FLOAT"
}
```

- 文件必须为 UTF-8 编码的标准 JSON 对象，不能包含重复字段、未转义的控制字符或 `NaN`、`Infinity` 等非标准常量。
- `model_quant_type` 必填，值必须为字符串，且精确匹配 `QuantAlgorithm` 的枚举值，例如 `W8A8` 或 `W4A4_MXFP4`。该字段是文件级元数据，不作为层配置。
- 支持可选的 AscendV1 导出元数据：`version`、`kv_cache_type`、`kv_quant_type`、`fa_quant_type`、`reduce_quant_type` 必须为字符串；`group_size` 必须为非负整数，不能为布尔值；`metadata`、`optional` 必须为 JSON 对象。MindIE-SD 不消费的字符串字段不限制为当前导出器枚举值，以保持对新版本导出器的前向兼容。这些字段经校验后排除在层配置之外，不会启用分组、KV-cache、通信或旋转特性。`metadata`、`optional` 内部对象属于导出器扩展信息，不被解释，也不会用于打开额外文件。
- 其余字段为动态层名或张量名，名称不能全为空白或包含 ASCII 控制字符；值必须为字符串，取值为 `QuantAlgorithm` 的枚举值（忽略大小写）或精确的 `FLOAT`。
- 不符合上述格式的字段会被拒绝，不再被静默忽略。动态名称不是固定白名单：陌生名称只要对应合法算法值，仍作为层条目处理。包括元数据对象在内的所有 JSON 数值必须为有限值。
- 校验失败抛出 `ParametersInvalid`，消息指明问题字段或 JSON 的行列位置；不会继续打开权重文件。

此校验检查描述文件的结构、类型与取值，不验证动态名称是否对应真实模型层，也不验证权重张量的形状、数值或量化精度。原有路径、软链接、文件大小和权限检查仍然生效。

## FA量化

### FA 通用原理

FA（Flash Attention）量化针对注意力计算中的 Q/K/V 激活值进行低比特处理。将 Q/K/V 量化为 FP8 后再送入注意力计算内核，可显著降低显存带宽需求，提升推理吞吐。与权重量化不同，FA 量化处理的是推理过程中动态产生的激活值，需要块级别的动态量化策略来平衡精度和加速效果。

### FA 技术特点

本仓库通过 `FP8_DYNAMIC` 和 `MXFP4_DYNAMIC` 算法提供 FA 量化能力。`FP8_DYNAMIC` 使用 FP8 块量化；`MXFP4_DYNAMIC` 复用自定义 `quant_flash_attn` AI Core 算子，并支持按时间步在 `MXFP4`、`FP8`、`FLOAT` 间切换。`MXFP4_FOUROVERSIX_DYNAMIC` 在 MXFP4 FA 的基础上，将 Q/K 的激活量化替换为 `quant_four_over_six_a5`（4/6 自适应）；该算子仅支持尾轴，而 `quant_flash_attn` 要求 V 沿序列轴分块量化，因此 V 仍走 `npu_dynamic_mx_quant`。

#### 旋转（Rotate）

若量化权重中存在预训练旋转矩阵（`q_rot`、`k_rot`），则对 Q 和 K 施加旋转，将异常值分散到各维度，缓解 FP8 量化对异常值的敏感性。权重中没有这两项时跳过旋转，模块初始化不会失败。

#### 块量化（Block Quant）

将 Q/K/V 按块动态量化为 FP8（`float8_e4m3fn`），通过 `npu_dynamic_block_quant` 完成。Q 的 token 块为 128，K 的 token 块为 256；V 的 token 块由下面的 `FP8FAMode` 决定。

#### FP8 Attention

通过 MindIE SD 自带的 `torch.ops.mindiesd.fused_infer_attention_score_v2`
算子进入本仓迁移的 `FusedInferAttentionScore` 实现，在 FP8 域内完成注意力计算，
输出结果反量化为原始精度。`FP8RotateQuantFA` 用枚举 `FP8FAMode` 一次选定量化块、`value_quant_mode` 和 `inner_precise`，避免这三项各自开关、彼此对不齐。

| 模式 | Q/K/V quant mode | `inner_precise` | V 量化块 |
| ------ | ------ | ------ | ------ |
| `HIGH_PRECISION`（默认） | `7/7/7` | 不显式传入（原高精度路径） | 256×128 |
| `C8V16_TILING512` | `7/7/12` | `4`（C8V16） | 512×64 |

后续新特性以新的枚举成员扩展，不要再拆成互不绑定的量化开关。

### FA 选型与工具算法

本仓 `FP8_DYNAMIC`、`MXFP8_DYNAMIC`、`MXFP4_DYNAMIC` 与 `MXFP4_FOUROVERSIX_DYNAMIC` 的 Q/K/V 格式、块大小和旋转权重须与推理算子匹配。可参考工具仓 [FA3 Quant](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/knowledge_base/quantization_algorithms/fa3_quant/usage_fa3_quant.md) 了解 Attention 激活量化与 `dtype` / `scope` 配置，但不能直接将其默认 INT8 per-head 产物用于本仓 FP8/MXFP FA 路径。

先验证 FP8 路径；只有目标设备支持且生成质量达标时，再考虑 MXFP4 及敏感时间步回退。工具侧旋转或缩放等预处理需与模型适配、导出产物一致。

### 接口说明

FA 量化通过 `quantize` 接口统一触发，无需单独调用 FA 量化接口。

```python
from mindiesd import quantize
```

#### FA 使用示例

```python
from mindiesd import quantize

# 加载原始浮点模型
model = from_pretrain()

# 执行量化转换（自动识别 Attention 层并注入 FA 量化）
model = quantize(model, "导出的量化配置文件路径")

# 模型移至 NPU 后执行推理
model.to("npu")
```

`quantize` 内部遍历模型各层，对匹配的 Attention 层自动调用 `add_fa_quant`，注入 `FP8RotateQuantFA` 模块，替换前向计算为旋转→块量化→FP8 Attention 的流程。默认使用 `FP8FAMode.HIGH_PRECISION`。切换到 C8V16 且 V 按 512 token 分块时：

```python
from mindiesd import FP8FAMode, QuantConfig, quantize

model = quantize(
    model,
    "导出的量化配置文件路径",
    quant_config=QuantConfig(fp8_fa_mode=FP8FAMode.C8V16_TILING512),
)
```

FA 量化层通过 `FP8RotateQuantFA` 模块实现，见本节的旋转→块量化→FP8 Attention 流程说明。`C8V16_TILING512` 依赖 FIA 的 C8V16 + V512 路径（`inner_precise=4`、`value_quant_mode=12`）。

MXFP4 FA 使用示例：

```python
from mindiesd import QuantConfig, TimestepManager, TimestepPolicyConfig, quantize

timestep_policy = TimestepPolicyConfig()
timestep_policy.register(range(0, 2), "FLOAT", target="fa")
timestep_policy.register(range(2, 8), "FP8", target="fa")
timestep_policy.register(range(8, 50), "MXFP4", target="fa")

quant_config = QuantConfig(
    timestep_config=timestep_policy,
    mxfp4_scale_alg=2,
)

model = quantize(model, "quant_model_description_mxfp4_dynamic_0.json", quant_config=quant_config)

for i, timestep in enumerate(timesteps):
    TimestepManager.set_timestep_idx(i)
    noise_pred = model(latents, timestep, encoder_hidden_states)
```

FA 没有离线权重约束，因此可以在不同时间步选择任意算法；Linear/MM 则只能在同一 MXFP4 权重下切换激活精度。

`QuantConfig` 中的 `mxfp4_scale_alg` 会透传到动态 MX 量化路径，用于对齐 CANN `aclnnDynamicQuantV2` 的 C7 推理参数；未设置时保持旧接口默认行为。相关接口说明可参考 CANN `aclnnDynamicQuantV2` 文档：<https://gitcode.com/cann/ops-nn/blob/master/quant/dynamic_quant_v2/docs/aclnnDynamicQuantV2.md>。模型侧 timestep 设置方式可参考 Wan2.2 文本生成视频推理循环：<https://modelers.cn/models/MindIE/Wan2.2/blob/main/wan/text2video.py>。

#### 注意事项

- 硬件要求：仅 Atlas 800I A2推理服务器支持此特性。
- Q/K/V 输入布局支持 `BNSD` 和 `BSND`。
- K/V 头数可以少于 Q（GQA, Grouped Query Attention, 分组查询注意力）。`FP8RotateQuantFA` 按 K 的头数向 FIA 传入 `num_key_value_heads`。
- 旋转矩阵（`q_rot`、`k_rot`）可选。若使用，需通过大模型压缩工具 msmodelslim 预先导出，详情请参见 msmodelslim 工具说明；权重中没有这两项时，`FP8RotateQuantFA` 跳过旋转。
