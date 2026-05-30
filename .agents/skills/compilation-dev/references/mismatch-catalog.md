# Pattern Mismatch 类型目录

基于 Wan2.2 / FLUX.1-dev 实际调试中发现并修复的 7 类 mismatch。

---

## 类型 1: 算子分解差异

**症状**:
Pattern 用高级 API（`torch.nn.functional.rms_norm`），
但模型 graph 中该算子已被 Dynamo 分解为原子 op 序列。

**根本原因**:
`_custom_decomposition.py` 的 `select_pattern_decomp_table()` 移除了 `aten.rms_norm` 的分解规则，
导致 pattern trace 中保留单个 `aten.rms_norm` 节点；
但 Dynamo 对模型的 trace 使用默认 decomposition table，`aten.rms_norm` 被分解。

**模型 graph 示例** (Wan2.2 RMSNorm):

```text
pow = aten.pow.Tensor_Scalar(view_15, 2)       # f32[1,75600,5120]
mean = aten.mean.dim(pow, [2], True)            # f32[1,75600,1]
add = aten.add.Scalar(mean, 9.999e-7)           # f32[1,75600,1]
rsqrt = aten.rsqrt.default(add)                  # f32[1,75600,1]
mul = aten.mul.Tensor(view_15, rsqrt)            # f32[1,75600,5120]
mul_w = aten.mul.Tensor(mul, weight)             # f32[1,75600,5120]
```

**错误 Pattern**:

```python
# 追踪为单个 aten.rms_norm 节点 → 不会匹配分解后的 7 节点序列
torch.nn.functional.rms_norm(x, [shape], weight, eps)
```

**正确 Pattern**:

```python
def func(x, weight):
    variance = x.pow(2).mean(-1, keepdim=True)
    result = x * torch.rsqrt(variance + epsilon)
    return result * weight
```

**教训**: 始终检查模型 graph 中算子是否被分解，pattern 必须匹配分解后的形式。
`select_pattern_decomp_table()` 的排除行为仅影响 pattern trace，不影响模型 trace。

---

## 类型 2: 参数顺序颠倒

**症状**:
`aten.add.Tensor(a, b)` 的参数 `(a, b)` 在模型 graph 和 pattern trace 中顺序不同。

**根本原因**:
Python 表达式 `1 + scale` 和 `scale + 1` 数学上等价，但 FX 追踪的 `aten.add.Tensor(target, args)`
中 args 顺序取决于表达式写法。pattern matcher 按 exact match 比较参数值。

**模型 graph**:

```python
add_1 = aten.add.Tensor(getitem_7, 1)  # scale + 1
```

**错误 Pattern**:

```python
1 + scale  # Python traces to aten.add.Tensor(1, scale)
```

**正确 Pattern**:

```python
scale + 1  # traces to aten.add.Tensor(scale, 1) → 匹配!
```

**修复**: 在 pattern 中使用与模型源码一致的表达式顺序。

---

## 类型 3: Dtype 路径不一致

**症状**:
模型 graph 中算子输入已是某 dtype（如 fp32），但 pattern 定义了额外的 dtype cast 操作，
导致模型 graph 中不存在的节点出现在 pattern 中。

**根本原因**:
模型在传入该算子前已经做了 dtype 转换（如 Wan 的 `hidden_states.float()` 在 norm 之前调用），
cast 节点出现在 graph 上游而非该算子的子图中。
pattern 从中途入口开始，不应包含上游的 cast。

**模型 graph** (Wan2.2):

```text
view_15  ← 输入已是 f32，无 cast 节点
pow = aten.pow.Tensor_Scalar(view_15, 2)  ← 直接从 f32 开始
```

**错误 Pattern**:

```python
x.to(torch.float32).pow(2)  # 引入 aten._to_copy 节点 → 模型 graph 中无此节点
```

**正确 Pattern**:

```python
x.pow(2)  # x 已是 fp32，无需 cast
```

**修复**: 移除 pattern 中与模型 graph 不匹配的冗余 dtype cast。
同时更新 `inputs()` 的 dtype 为 fp32（若模型 graph 中的输入确实是 fp32）。

---

## 类型 4: 缺少中间节点

**症状**:
模型 graph 中算子子图包含 20+ 节点，但 pattern 函数只定义了 10 个节点，
因为跳过了模型 graph 中内联的中间操作。

**根本原因**:
模型代码中子图的"真实入口"比我们预期的更靠前。
pattern 的 `inputs()` 应定义为该子图在模型 graph 中的入口 tensor，
而不能假设某些操作已被上游完成。

**模型 graph** (Wan2.2 RoPE):

```text
slice_9 = aten.slice.Tensor(freqs_cos, 3, 0, MAX, 2)   # freqs_cos[...,0::2]
slice_10 = aten.slice.Tensor(freqs_sin, 3, 1, MAX, 2)   # freqs_sin[...,1::2]
# ... 然后才使用 cos/sin 做 rotation
```

**错误 Pattern**: cos/sin 作为直接输入（假设已切片）
**正确 Pattern**: `freqs_cos` / `freqs_sin` 作为输入，pattern 内部包含 `[..., 0::2]` slice 操作

**修复**: `inputs()` 从模型 graph 中该子图的真实入口开始定义，
`pattern()` 内部包含所有到 target op 的中间操作。

---

## 类型 5: Dim 字面量差异

**症状**:
`mean(x, [-1], keepdim=True)` 与 `mean(x, [2], keepdim=True)` 不匹配。

**根本原因**:
3D 输入时 `-1 == 2`，但 pattern matcher 比较的是 FX node 的
字面量参数值，而非解析后的维度索引。

**修复**: pattern 中使用与模型 graph 一致的具体 dim 索引值。

**适用条件**: 此修复在模型输入 rank 固定时可靠。若模型 shape 可变，
需确保 pattern 的所有调用场景都使用相同 rank 的输入。

---

## 类型 6: Fake vs Real Tensor 标量转换差异

**症状**:
Pattern 和 model 都使用相同的高级 API（如 `torch.rms_norm(eps=1e-6)`），
两边都通过电感分解表分解为相同的原子 op 序列，但 pattern 仍不匹配。
debug graph 中检查 `add.Scalar(mean, eps)` 的 eps 字面量发现两图值不同。

**根本原因**:
`make_fx` 使用 fake tensor 追踪 pattern 函数：标量参数不经实际数值运算，直接存入 FX node 作为 Python float 字面量。
Dynamo 使用 real/fake tensor 追踪模型：当标量与不同精度 tensor 做运算时（如 f32 tensor + f64 scalar），
PyTorch 将 scalar 隐式 cast 到 tensor dtype，结果以转换后的值存入 FX node。
两个值虽然极为接近（差 `~2.5e-15`），但 Python `==` 比较为 `False`，pattern matcher 判定为不同节点。

**模型 graph** (Wan2.2 RMSNorm):

```text
pow_1 = aten.pow.Tensor_Scalar(view_15, 2)
mean  = aten.mean.dim(pow_1, [2], True)
add   = aten.add.Scalar(mean, 9.999999974752427e-07)   ← float32(1e-6)
rsqrt = aten.rsqrt.default(add)
```

**Pattern trace** (make_fx):

```text
pow = aten.pow.Tensor_Scalar(hidden_states, 2)
mean = aten.mean.dim(pow, [2], True)
add = aten.add.Scalar(mean, 1e-6)                       ← 原始 float64 值
rsqrt = aten.rsqrt.default(add)
```

**错误 Pattern**: 使用原始 Python float `epsilon=1e-6` → make_fx 不模拟 f32 cast → 字面量偏离。
**正确 Pattern**: 远端 `make_fx` 输出 pattern 的实际 FX graph，与 model graph 做逐节点 diff，
定位偏差的字面量，直接将 model graph 中的精确值硬编码到 pattern 中。

**诊断工具**（远端容器内执行）:

```python
from torch.fx.experimental.proxy_tensor import make_fx
from mindiesd.compilation._custom_decomposition import select_pattern_decomp_table
from mindiesd.compilation.patterns.xxx_pattern import XxxPatternGroup

p = XxxPatternGroup[0]
cpu_in = [torch.empty(*inp.shape, dtype=inp.dtype) for inp in p.inputs()]
gm = make_fx(p.pattern, decomposition_table=select_pattern_decomp_table())(*cpu_in)

for n in gm.graph.nodes:
    if n.op == 'call_function':
        print(f'{n.target}  args={n.args}')
```

将此输出与 debug log 中 model graph 的目标节点序列逐行对齐，找到第一个标量值不匹配的节点。

**修复**: 将 model graph 中的精确标量值硬编码到 pattern 闭包中。
**风险**: 硬编码值可能随 torch 版本升级而变化，需在升级后重验证。

---

## 类型 7: placeholder vs get_attr 参数来源不一致

**症状**:

- `PatternMatchPass` 日志显示 pattern 注册成功、match count > 0
- 单元测试通过（`cosine_similarity > 2^-7`）
- 但全模型 profiling 的 `kernel_details.csv` 中**无融合 kernel 出现**，原始分解 kernel 未消失
- 全模型 `PatternMatchPass replace N` 统计中的 N 可能包含其他 pattern 的匹配（如 GELU），
  并**不代表本 pattern 被命中**

**根本原因**:
`torch._inductor.pattern_matcher.register_replacement` 要求 pattern 的所有参数在 traced graph
中为 `placeholder` 节点（即函数输入参数）。当目标算子在全模型中使用 `nn.Module` 风格的参数时
（如 `nn.RMSNorm(self.weight)` 的 `self.weight`），该参数在 FX graph 中表现为 `get_attr` 节点，
而非 `placeholder`。

pattern matcher 按 node 类型做精确匹配：`placeholder` ≠ `get_attr` → 匹配失败，
且**无错误日志或警告**，match count 不增加，静默跳过。

**对比**: GELU 的 `torch.nn.GELU(approximate="tanh")` 无 learnable parameters → graph 中无
`get_attr` 节点 → `register_replacement` 正常工作。

**模型 graph 示例** (Wan2.2 RMSNorm):

```text
# weight 来自 module attribute (get_attr)
mul_5 = aten.mul.Tensor(mul_4, arg24_1)   # arg24_1 = get_attr(self.norm_q.weight)
```

**错误 Pattern**: weight 作为函数输入（placeholder）

```python
@staticmethod
def pattern(x, weight):       # weight → FX placeholder
    ...
    return x * rsqrt(...) * weight
```

**正确 Pattern** (自定义 Graph Pass):

无法通过 `register_replacement` 匹配。需在 `PatternMatchPass` 中实现自定义 graph traversal：

```python
def _rewrite_rmsnorm_to_fused(self, graph):
    """Graph-level pass: direct node walk for get_attr weight patterns."""
    for node in list(graph.graph.nodes):
        # Walk the chain: mul_final → mul_mid → rsqrt → add → mean → pow
        # Verify x node is shared between pow and mul_mid
        # weight_node directly from get_attr — no placeholder matching needed
        ...
        with graph.graph.inserting_before(mul_final):
            rms = graph.graph.call_function(torch.ops.npu.npu_rms_norm.default, ...)
        mul_final.replace_all_uses_with(getitem(rms, 0))
```

**执行位置**: 必须在 `graph_rewrite_after_freezing` 中调用（freeze 阶段不识别 NPU custom ops）。

**判据**:

- 模型 graph dump 中 target 参数来源为 `get_attr(name.weight)` → 类型 7
- 模型 graph 中无 `get_attr` 节点 → 普通 gateway pattern 可处理

**修复**: 放弃 `register_replacement` 路径，使用自定义 graph traversal + 手动 node 替换。

---

## 维护与更新

当发现新的 mismatch 类型或有更优的修复策略时更新此文件。
每条新增类型需包含：症状、根本原因、模型 graph 示例、错误/正确 pattern 对比、修复方式。
