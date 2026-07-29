# Fused MoE

## Overview

`fused_moe` is the MindIE-SD entry point for MoE, responsible for expert selection, token dispatch, expert computation, and result combining in MoE forward inference on NPU. This API targets open-source framework integration scenarios: callers provide activations, router logits, expert weights, and communication configurations, and the unified entry handles the entire routed experts forward computation.

In MoE models, each token selects a small number of experts based on router output. Compared to dense MLP, MoE can expand model capacity while controlling the actual computation per inference step, but introduces additional flows including token-to-expert routing, reordering, cross-card communication, and result recovery. `fused_moe` encapsulates these flows in a unified API, reducing repetitive adaptation costs on the framework side and enabling reuse of a single MoE computation entry across different parallelism strategies.

Currently `fused_moe` includes two paths: a fused operator path and a non-fused operator path. The fused operator path is not yet supported in the current version; setting `use_fused_op=True` falls back to the non-fused operator path. The default `use_fused_op=False` directly uses the non-fused operator path.

## API Reference

```python
from mindiesd import fused_moe
```

### Function Signature

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

### Parameters

| Parameter | Type | Required | Default | Description |
| ------ | ------ | ------ | -------- | ------ |
| `hidden_states` | `torch.Tensor` | Yes | - | Input activations, shape `[..., hidden_size]`, at least 2 dimensions. |
| `router_logits` | `torch.Tensor` | Yes | - | Router logits, shape `[..., num_experts]`, at least 2 dimensions; leading dimensions must match `hidden_states`. |
| `num_experts` | `int` | Yes | - | Total number of experts, must match the last dimension of `router_logits`. Must be divisible by EP group size when using EP. |
| `top_k` | `int` | Yes | - | Number of experts selected per token, range `[1, num_experts]`. |
| `w13_weight` | `torch.Tensor` | Yes | - | Fused gate/up projection weight, shape `[local_experts, hidden_size, 2 * intermediate_size]`, must be a 3D tensor. |
| `w2_weight` | `torch.Tensor` | Yes | - | Down projection weight, shape `[local_experts, intermediate_size, hidden_size]`, must have the same `local_experts` as `w13_weight`. |
| `w13_bias` | `torch.Tensor` / `None` | No | `None` | Gate/up projection bias, shape `[local_experts, 2 * intermediate_size]`, must match `w13_weight` expert and output dimensions. |
| `w2_bias` | `torch.Tensor` / `None` | No | `None` | Down projection bias, shape `[local_experts, hidden_size]`, must match `w2_weight` expert and output dimensions. |
| `quant_config` | `QuantConfig` / `None` | No | `None` | MindIE-SD quantization config for enabling quantization in the MoE forward flow. |
| `w13_weight_scale` | `torch.Tensor` / `None` | No | `None` | Quantization scale for `w13_weight`. |
| `w2_weight_scale` | `torch.Tensor` / `None` | No | `None` | Quantization scale for `w2_weight`. |
| `tp_group` | `dist.ProcessGroup` / `None` | No | `None` | TP communication group. Takes effect when EP is not enabled and TP group size > 1. |
| `ep_group` | `dist.ProcessGroup` / `None` | No | `None` | EP communication group. Takes priority when EP group size > 1. |
| `dispatcher_type` | `str` / `None` | No | `None` | Token dispatch strategy. Options: `"static"`, `"dynamic"`; `None` auto-selects based on platform and communication config. `"dynamic"` is only supported in EP scenarios. |
| `tokens_full` | `bool` | No | `True` | Input token layout marker. Only two layouts supported: `True` means each rank inputs full tokens; `False` means each rank inputs evenly split local token shards per communication group. |
| `k_group` | `int` | No | `1` | Number of expert groups selected per token in grouped routing, range `[1, group_count]`. |
| `group_count` | `int` | No | `1` | Total number of expert groups; `num_experts` must be divisible by `group_count`. |
| `group_select_mode` | `int` | No | `0` | Expert group scoring method. `0` uses max score within group, `1` uses sum of top-2 scores within group. Each group must have at least 2 experts when using `1`. |
| `routing_method` | `str` | No | `"softmax"` | Router logit scoring method, options: `"softmax"` or `"sigmoid"`. |
| `renormalize` | `bool` | No | `False` | Whether to re-normalize top-k routing weights selected by softmax routing. Sigmoid routing outputs normalized weights per NPU gating top-k operator semantics. |
| `routed_scaling_factor` | `float` | No | `1.0` | Routing weight scaling factor, applied during expert selection. |
| `custom_routing_function` | `callable` / `None` | No | `None` | Custom routing function, called as `custom_routing_function(hidden_states, gating_output, topk, renormalize)`, must return `(topk_weights, topk_ids)`. |
| `reduce_results` | `bool` | No | `True` | Whether to perform communication reduction on full token output in static MoE; unused in dynamic MoE path. |
| `use_fused_op` | `bool` | No | `False` | Whether to prefer the fused operator path. Currently unsupported; setting `True` falls back to the non-fused path. Default `False` directly uses the non-fused path. |

### Return Value

`torch.Tensor`: MoE forward computation result, same shape as `hidden_states`.

## Fused Operator Path (Reserved)

The fused operator path is not yet supported in the current version. Setting `use_fused_op=True` falls back to the non-fused operator path.

---

## Non-Fused Operator Path (Current Default)

This path splits the MoE forward pass into multiple stages, supporting both non-quantized and quantized MoE inference, covering single-card, TP, and EP scenarios, and providing static and dynamic token dispatch methods. The static dispatcher is suitable for single-card, TP, and some EP scenarios; the dynamic dispatcher targets all-to-all token exchange in EP scenarios.

### Execution Flow

This path divides the MoE forward process into the following stages:

1. **prepare**: Organize input activations and router logits, and prepare data needed for subsequent computation based on input layout.
2. **select_experts**: Select top-k experts for each token based on router output and generate routing weights.
3. **dispatch**: Route and reorder tokens by expert according to expert selection results, generating inputs for expert computation.
4. **mlp**: Execute expert-side grouped MLP computation to complete routed experts' feed-forward computation.
5. **combine**: Merge expert outputs back to the original token order to recover the routed MoE output.
6. **finalize**: Complete post-processing after output recovery and communication.

### Dispatcher Strategy

Two token dispatch strategies are supported:

- **static dispatcher**: Uses a static token dispatch path, suitable for single-card, TP, and some EP scenarios. This path completes token sorting, expert token statistics, and result recovery via NPU MoE routing operators.
- **dynamic dispatcher**: Uses a dynamic token dispatch path, suitable for EP scenarios. This path performs all-to-all communication based on token-to-expert distribution and restores token order before and after expert computation.

When `dispatcher_type=None`, the API auto-selects based on communication mode and NPU model: Atlas 800I A3 SuperPoD Server / Ascend 950PR / Ascend 950DT  uses dynamic dispatcher in EP scenarios, Atlas 800I A2 inference server uses static dispatcher; non-EP scenarios (single-card/TP) always use static dispatcher. You can also explicitly specify via `dispatcher_type="static"` or `dispatcher_type="dynamic"`.

### Communication Configuration

Distributed MoE inference is supported through `tp_group` and `ep_group`. The current API enables only one MoE communication strategy at a time:

- When a valid `ep_group` is provided, the EP communication strategy takes priority.
- When EP is not enabled and a valid `tp_group` is provided, the TP communication strategy is used.
- When neither is enabled, the single-card path is used.

Callers must ensure that input activation and router logit layouts are consistent with the `tokens_full` configuration.

### Quantization Configuration

When `quant_config` is not provided, or `quant_config.quant_algo` is `None` / `NO_QUANT`, the MoE forward flow runs in non-quantized mode. The following quantization configuration is currently supported:

- `QuantConfig(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)`: W8A8 dynamic quantization, weights using INT8.

### Routing Selection

Default routing uses NPU gating top-k operators for expert selection. `routing_method="softmax"` applies softmax to router logits; `routing_method="sigmoid"` applies sigmoid. Softmax routing can re-normalize selected top-k weights via `renormalize=True`; sigmoid routing outputs normalized weights per NPU gating top-k operator semantics.

When grouped routing is not enabled, keep the defaults `k_group=1`, `group_count=1`. When grouped routing is enabled, expert groups are first selected by `group_select_mode`, then top-k experts are selected per token within the selected groups.

Grouped routing requires the following constraints:

- `num_experts` must be divisible by `group_count`.
- `k_group` range is `[1, group_count]`.
- `top_k` must not exceed the total number of experts within the selected expert groups.
- When `group_select_mode=1`, each expert group must contain at least 2 experts.

### Usage Examples

#### Usage Notes

In actual framework integration, `hidden_states`, `router_logits`, and expert weights typically come from model forward and weight loading flows.
Distributed examples assume `tp_group` or `ep_group` have already been initialized.

#### Single-Card MoE

Single-card scenarios do not require communication groups; default strategy is used.

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

#### Grouped Routing MoE

When the model needs to first select expert groups and then select top-k experts from the selected groups, configure the grouped routing parameters. The example below uses sigmoid routing and scales routing weights via `routed_scaling_factor`.

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

#### INT8 Dynamic Quant MoE

The INT8 path requires `w13_weight` and `w2_weight` to be `torch.int8`, along with corresponding quantization scales. MindIE-SD checks weight format before MLP computation; if weights are not in NPU NZ format, they are automatically converted to NZ before calling INT8 grouped MLP operators.

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

#### TP Static MoE

TP scenarios require `tp_group`. When `tokens_full=True`, each rank inputs full tokens.

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

#### EP Static MoE

EP scenarios can also explicitly specify the static dispatcher. Here, `w13_weight` and `w2_weight` represent the local expert weights held by the current rank, and `ep_group` is used for MoE communication.

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

#### EP Dynamic MoE

The EP dynamic path requires `ep_group`. When each rank inputs locally sharded tokens evenly split by the communication group, set `tokens_full=False`. Here, `local_hidden_states` and `local_router_logits` represent the current rank's local token input, while `w13_weight` and `w2_weight` still represent the current rank's local expert weights. With this layout, the token count must be divisible by the communication group size.

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

#### Custom Routing Function

When the model already has custom routing logic, it can be integrated via `custom_routing_function`. This function must return `(topk_weights, topk_ids)`.

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

### Notes

- The current API only supports forward inference; backward gradient computation is not supported.
- Currently supports non-quantized MoE and INT8 dynamic quant MoE paths.
- `w13_weight` and `w2_weight` must use the weight layout required by grouped matmul.
- The last dimension of `router_logits` must equal `num_experts`.
- The `dynamic` dispatcher must be used in EP communication scenarios.
- When using distributed communication, callers must initialize communication groups in advance and ensure consistent input layout across all ranks.
