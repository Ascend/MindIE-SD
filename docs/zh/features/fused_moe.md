# Fused MoE

## 功能概述

`fused_moe` 是 MindIE-SD 提供的 MoE 对外入口，用于在 NPU 上完成 MoE 前向推理中的专家选择、Token 分发、专家计算和结果合并。该接口面向开源框架集成场景，调用方传入激活、路由 logits、专家权重和通信配置后，即可通过统一入口完成 routed experts 的前向计算。

MoE 模型中，每个 Token 会根据 router 输出选择少量 experts 参与计算。相比 dense MLP，MoE 可以在扩大模型容量的同时控制单次推理的实际计算量，但也引入了 Token 到 expert 的路由、重排、跨卡通信和结果恢复等额外流程。`fused_moe` 将这些流程封装在统一接口中，减少框架侧重复适配成本，并便于在不同并行策略下复用同一套 MoE 计算入口。

当前 `fused_moe` 包含两条路径：融合算子路径和非融合算子路径。融合算子路径当前版本暂不支持，`use_fused_op=True` 时会回退到非融合算子路径；默认 `use_fused_op=False`，直接使用非融合算子路径。

## 接口说明

```python
from mindiesd import fused_moe
```

### 函数签名

```python
fused_moe(
    hidden_states,
    router_logits,
    num_experts,
    top_k,
    w13_weight,
    w2_weight,
    w13_bias=None,
    w2_bias=None,
    quant_config=None,
    w13_weight_scale=None,
    w2_weight_scale=None,
    tp_group=None,
    ep_group=None,
    dispatcher_type=None,
    tokens_full=True,
    k_group=1,
    group_count=1,
    group_select_mode=0,
    routing_method="softmax",
    renormalize=False,
    routed_scaling_factor=1.0,
    custom_routing_function=None,
    reduce_results=True,
    use_fused_op=False,
) -> torch.Tensor
```

### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `hidden_states` | `torch.Tensor` | 是 | - | 输入激活，形状为 `[..., hidden_size]`，维度不少于 2。 |
| `router_logits` | `torch.Tensor` | 是 | - | 路由 logits，形状为 `[..., num_experts]`，维度不少于 2，前置维度需与 `hidden_states` 一致。 |
| `num_experts` | `int` | 是 | - | 全局 expert 数量，必须与 `router_logits` 最后一维一致。使用 EP 时需能被 EP group size 整除。 |
| `top_k` | `int` | 是 | - | 每个 Token 选择的 expert 数量，取值范围为 `[1, num_experts]`。 |
| `w13_weight` | `torch.Tensor` | 是 | - | 融合后的 gate/up 投影权重，形状为 `[local_experts, hidden_size, 2 * intermediate_size]`，必须为 3D Tensor。 |
| `w2_weight` | `torch.Tensor` | 是 | - | down 投影权重，形状为 `[local_experts, intermediate_size, hidden_size]`，必须与 `w13_weight` 具有相同的 `local_experts`。 |
| `w13_bias` | `torch.Tensor` / `None` | 否 | `None` | gate/up 投影 bias，形状为 `[local_experts, 2 * intermediate_size]`，需与 `w13_weight` 的 expert 和输出维度一致。 |
| `w2_bias` | `torch.Tensor` / `None` | 否 | `None` | down 投影 bias，形状为 `[local_experts, hidden_size]`，需与 `w2_weight` 的 expert 和输出维度一致。 |
| `quant_config` | `QuantConfig` / `None` | 否 | `None` | MindIE-SD 量化配置，用于选择 MoE 前向流程是否启用量化计算。 |
| `w13_weight_scale` | `torch.Tensor` / `None` | 否 | `None` | `w13_weight` 的 quantization scale。 |
| `w2_weight_scale` | `torch.Tensor` / `None` | 否 | `None` | `w2_weight` 的 quantization scale。 |
| `tp_group` | `dist.ProcessGroup` / `None` | 否 | `None` | TP 通信组。未启用 EP 且 TP group size 大于 1 时生效。 |
| `ep_group` | `dist.ProcessGroup` / `None` | 否 | `None` | EP 通信组。EP group size 大于 1 时优先生效。 |
| `dispatcher_type` | `str` / `None` | 否 | `None` | Token 分发策略。可选 `"static"`、`"dynamic"`；`None` 表示根据平台和通信配置自动选择。`"dynamic"` 仅支持 EP 通信场景。 |
| `tokens_full` | `bool` | 否 | `True` | 输入 Token layout 标记。仅支持两种输入 layout：`True` 表示每个 rank 输入全量 Token；`False` 表示每个 rank 输入按通信组均匀切分后的本地 Token shard。 |
| `k_group` | `int` | 否 | `1` | 分组路由时每个 Token 选择的 expert group 数量，取值范围为 `[1, group_count]`。 |
| `group_count` | `int` | 否 | `1` | expert group 总数，`num_experts` 需要能被 `group_count` 整除。 |
| `group_select_mode` | `int` | 否 | `0` | expert group 打分方式。`0` 表示取组内最大分数，`1` 表示取组内 top-2 分数之和。使用 `1` 时每组至少需要 2 个 experts。 |
| `routing_method` | `str` | 否 | `"softmax"` | router logits 的打分方式，可选 `"softmax"` 或 `"sigmoid"`。 |
| `renormalize` | `bool` | 否 | `False` | 是否对 softmax 路由选中的 top-k routing weights 重新归一化。sigmoid 路由按 NPU gating top-k 算子语义输出归一化后的权重。 |
| `routed_scaling_factor` | `float` | 否 | `1.0` | 路由权重缩放系数，在专家选择阶段生效。 |
| `custom_routing_function` | `callable` / `None` | 否 | `None` | 自定义路由函数，调用形式为 `custom_routing_function(hidden_states, gating_output, topk, renormalize)`，需返回 `(topk_weights, topk_ids)`。 |
| `reduce_results` | `bool` | 否 | `True` | static MoE 下是否对完整 Token 输出做通信规约；dynamic MoE 路径不使用该参数。 |
| `use_fused_op` | `bool` | 否 | `False` | 是否优先启用融合算子路径。当前版本暂不支持该路径，设置为 `True` 时会回退到非融合算子路径；默认 `False` 直接使用非融合算子路径。 |

### 返回值

`torch.Tensor`：MoE 前向计算结果，形状与 `hidden_states` 一致。

## 融合算子路径（预留）

当前版本暂不支持融合算子路径。若设置 `use_fused_op=True`，接口会回退到非融合算子路径完成计算。

---

## 非融合算子路径（当前默认）

该路径将 MoE 前向拆成多个阶段完成，支持非量化和量化 MoE 推理，覆盖单卡、TP 和 EP 场景，并提供 static 和 dynamic 两种 Token 分发方式。static dispatcher 适用于单卡、TP 以及部分 EP 场景；dynamic dispatcher 面向 EP 场景下的 all-to-all Token 交换。

### 执行流程

该路径将 MoE 前向过程拆分为以下阶段：

1. **prepare**：整理输入激活和 router logits，并根据输入 layout 准备后续计算所需的数据。
2. **select_experts**：根据 router 输出，为每个 Token 选择对应的 top-k experts，并生成 routing weights。
3. **dispatch**：根据专家选择结果，将 Token 按 expert 路由并重排，生成专家计算所需的输入。
4. **mlp**：执行专家侧 grouped MLP 计算，完成 routed experts 的前馈计算。
5. **combine**：将专家输出按原 Token 顺序合并，恢复 routed MoE 的输出结果。
6. **finalize**：完成输出恢复和通信后的收尾处理。

### Dispatcher 策略

支持两类 Token 分发策略：

- **static dispatcher**：使用静态 Token 分发路径，适用于单卡、TP 场景，以及部分 EP 场景。该路径通过 NPU MoE routing 算子完成 Token 排序、expert token 统计和结果恢复。
- **dynamic dispatcher**：使用动态 Token 分发路径，适用于 EP 场景。该路径会根据 Token 到 expert 的分布执行 all-to-all 通信，并在专家计算前后完成 Token 顺序恢复。

当 `dispatcher_type=None` 时，接口根据通信模式和 NPU 型号自动选择：EP 场景下 A3/A5 硬件使用 dynamic dispatcher，A2 硬件使用 static dispatcher；非 EP 场景（单卡/TP）始终使用 static dispatcher。也可以通过 `dispatcher_type="static"` 或 `dispatcher_type="dynamic"` 显式指定。

### 通信配置

通过 `tp_group` 和 `ep_group` 支持分布式 MoE 推理。当前接口一次仅启用一种 MoE 通信策略：

- 当传入有效 `ep_group` 时，优先使用 EP 通信策略。
- 当未启用 EP 且传入有效 `tp_group` 时，使用 TP 通信策略。
- 当二者均未启用时，按单卡路径执行。

调用方需要保证输入激活和 router logits 的 layout 与 `tokens_full` 配置一致。

### 量化配置

未传入 `quant_config`，或 `quant_config.quant_algo` 为 `None` / `NO_QUANT` 时，按非量化方式执行 MoE 前向流程。当前支持以下量化配置：

- `QuantConfig(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)`：W8A8 dynamic quantization，权重使用 INT8。

### 路由选择

默认路由使用 NPU gating top-k 算子完成专家选择。`routing_method="softmax"` 时对 router logits 做 softmax，`routing_method="sigmoid"` 时做 sigmoid。softmax 路由可通过 `renormalize=True` 对选中的 top-k weights 做重新归一化；sigmoid 路由按 NPU gating top-k 算子语义输出归一化后的权重。

未启用分组路由时保持默认 `k_group=1`、`group_count=1` 即可。启用分组路由时，会先按 `group_select_mode` 选择 expert group，再在选中的 group 中选择每个 Token 的 top-k experts。

分组路由需要满足以下约束：

- `num_experts` 能被 `group_count` 整除。
- `k_group` 取值范围为 `[1, group_count]`。
- `top_k` 不超过被选中 expert group 内的 expert 总数。
- `group_select_mode=1` 时，每个 expert group 至少包含 2 个 experts。

### 使用示例

#### 用例说明

实际接入框架时，`hidden_states`、`router_logits` 和专家权重通常来自模型 forward 和权重加载流程。
分布式示例默认 `tp_group` 或 `ep_group` 已完成初始化。

#### 单卡 MoE

单卡场景不需要传入通信组，使用默认策略即可。

```python
import torch
from mindiesd import fused_moe

num_tokens = 8
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2
dtype = torch.bfloat16
device = "npu"

hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
w13_weight = torch.randn(num_experts, hidden_size, 2 * intermediate_size, device=device, dtype=dtype)
w2_weight = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)
w13_bias = torch.randn(num_experts, 2 * intermediate_size, device=device, dtype=dtype)
w2_bias = torch.randn(num_experts, hidden_size, device=device, dtype=dtype)

out = fused_moe(
    hidden_states=hidden_states,
    router_logits=router_logits,
    num_experts=num_experts,
    top_k=top_k,
    w13_weight=w13_weight,
    w2_weight=w2_weight,
    w13_bias=w13_bias,
    w2_bias=w2_bias,
    renormalize=True,
)
```

#### 分组路由 MoE

当模型需要先选择 expert group，再从选中 group 中选择 top-k experts 时，可以配置分组路由参数。下面示例使用 sigmoid 路由，并通过 `routed_scaling_factor` 对路由权重做缩放。

```python
import torch
from mindiesd import fused_moe

num_tokens = 8
hidden_size = 4096
intermediate_size = 14336
num_experts = 16
top_k = 2
dtype = torch.bfloat16
device = "npu"

hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
w13_weight = torch.randn(num_experts, hidden_size, 2 * intermediate_size, device=device, dtype=dtype)
w2_weight = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)

out = fused_moe(
    hidden_states=hidden_states,
    router_logits=router_logits,
    num_experts=num_experts,
    top_k=top_k,
    w13_weight=w13_weight,
    w2_weight=w2_weight,
    k_group=1,
    group_count=4,
    group_select_mode=1,
    routing_method="sigmoid",
    routed_scaling_factor=0.5,
)
```

#### INT8 dynamic quant MoE

INT8 路径要求 `w13_weight` 和 `w2_weight` 为 `torch.int8`，并传入对应的 quantization scale。MindIE-SD 会在 MLP 计算前检查权重格式；若权重不是 NPU NZ 格式，会自动转换为 NZ 后再调用 INT8 grouped MLP 算子。

```python
import torch
from mindiesd import fused_moe
from mindiesd.quantization.config import QuantConfig
from mindiesd.quantization.mode import QuantAlgorithm

num_tokens = 8
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2
dtype = torch.bfloat16
device = "npu"

hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
w13_weight = torch.randint(
    -128,
    127,
    (num_experts, hidden_size, 2 * intermediate_size),
    device=device,
    dtype=torch.int8,
)
w2_weight = torch.randint(
    -128,
    127,
    (num_experts, intermediate_size, hidden_size),
    device=device,
    dtype=torch.int8,
)
w13_weight_scale = torch.rand(num_experts, 2 * intermediate_size, device=device, dtype=dtype)
w2_weight_scale = torch.rand(num_experts, hidden_size, device=device, dtype=dtype)

out = fused_moe(
    hidden_states=hidden_states,
    router_logits=router_logits,
    num_experts=num_experts,
    top_k=top_k,
    w13_weight=w13_weight,
    w2_weight=w2_weight,
    quant_config=QuantConfig(quant_algo=QuantAlgorithm.W8A8_DYNAMIC),
    w13_weight_scale=w13_weight_scale,
    w2_weight_scale=w2_weight_scale,
    dispatcher_type="static",
    tokens_full=True,
    reduce_results=False,
)
```

#### TP static MoE

TP 场景传入 `tp_group`。当 `tokens_full=True` 时，表示每个 rank 输入全量 Token。

```python
import torch
from mindiesd import fused_moe

num_tokens = 8
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2
dtype = torch.bfloat16
device = "npu"

hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
w13_weight = torch.randn(num_experts, hidden_size, 2 * intermediate_size, device=device, dtype=dtype)
w2_weight = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)

out = fused_moe(
    hidden_states=hidden_states,
    router_logits=router_logits,
    num_experts=num_experts,
    top_k=top_k,
    w13_weight=w13_weight,
    w2_weight=w2_weight,
    tp_group=tp_group,
    dispatcher_type="static",
    tokens_full=True,
    reduce_results=True,
)
```

#### EP static MoE

EP 场景也可以显式指定 static dispatcher。此时 `w13_weight` 和 `w2_weight` 表示当前 rank 持有的本地 expert 权重，`ep_group` 用于完成 MoE 通信。

```python
import torch
from mindiesd import fused_moe

num_tokens = 8
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2
dtype = torch.bfloat16
device = "npu"
ep_world_size = torch.distributed.get_world_size(ep_group)
local_experts = num_experts // ep_world_size

hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
w13_weight = torch.randn(local_experts, hidden_size, 2 * intermediate_size, device=device, dtype=dtype)
w2_weight = torch.randn(local_experts, intermediate_size, hidden_size, device=device, dtype=dtype)

out = fused_moe(
    hidden_states=hidden_states,
    router_logits=router_logits,
    num_experts=num_experts,
    top_k=top_k,
    w13_weight=w13_weight,
    w2_weight=w2_weight,
    ep_group=ep_group,
    dispatcher_type="static",
    tokens_full=True,
    reduce_results=True,
    renormalize=True,
)
```

#### EP dynamic MoE

EP dynamic 路径需要传入 `ep_group`。当每个 rank 输入按通信组均匀切分后的本地
Token shard 时，设置 `tokens_full=False`；此时 `local_hidden_states` 和
`local_router_logits` 表示当前 rank 的本地 Token 输入，`w13_weight` 和
`w2_weight` 仍表示当前 rank 的本地 expert 权重。使用该 layout 时，Token 数量
需要能被通信组大小整除。

```python
import torch
from mindiesd import fused_moe

num_tokens = 8
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2
dtype = torch.bfloat16
device = "npu"
ep_world_size = torch.distributed.get_world_size(ep_group)
local_experts = num_experts // ep_world_size
local_num_tokens = num_tokens // ep_world_size
local_hidden_states = torch.randn(local_num_tokens, hidden_size, device=device, dtype=dtype)
local_router_logits = torch.randn(local_num_tokens, num_experts, device=device, dtype=dtype)
w13_weight = torch.randn(local_experts, hidden_size, 2 * intermediate_size, device=device, dtype=dtype)
w2_weight = torch.randn(local_experts, intermediate_size, hidden_size, device=device, dtype=dtype)

out = fused_moe(
    hidden_states=local_hidden_states,
    router_logits=local_router_logits,
    num_experts=num_experts,
    top_k=top_k,
    w13_weight=w13_weight,
    w2_weight=w2_weight,
    ep_group=ep_group,
    dispatcher_type="dynamic",
    tokens_full=False,
    renormalize=True,
)
```

#### 自定义 routing function

当模型已有自定义路由逻辑时，可通过 `custom_routing_function` 接入。该函数需要返回 `(topk_weights, topk_ids)`。

```python
import torch
from mindiesd import fused_moe

num_tokens = 8
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2
dtype = torch.bfloat16
device = "npu"

hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
w13_weight = torch.randn(num_experts, hidden_size, 2 * intermediate_size, device=device, dtype=dtype)
w2_weight = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)

def custom_routing_function(hidden_states, gating_output, topk, renormalize):
    topk_result = gating_output.softmax(dim=-1).topk(topk, dim=-1)
    topk_weights = topk_result.values
    topk_ids = topk_result.indices
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


out = fused_moe(
    hidden_states=hidden_states,
    router_logits=router_logits,
    num_experts=num_experts,
    top_k=top_k,
    w13_weight=w13_weight,
    w2_weight=w2_weight,
    custom_routing_function=custom_routing_function,
    renormalize=True,
)
```

### 注意事项

- 当前接口仅支持前向推理，不支持反向梯度计算。
- 当前支持非量化 MoE 和 INT8 dynamic quant MoE 路径。
- `w13_weight` 和 `w2_weight` 需要使用 grouped matmul 所需的权重布局。
- `router_logits` 的最后一维必须等于 `num_experts`。
- `dynamic` dispatcher 需要在 EP 通信场景下使用。
- 使用分布式通信时，调用方需要提前完成通信组初始化，并保证各 rank 上的输入 layout 一致。
