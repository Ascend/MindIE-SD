# 自定义 Graph Pass 实现指南

当 `register_replacement` 因 mismatch 类型 6（placeholder vs get_attr）失效时，
使用自定义 FX graph traversal pass 替代。

---

## 适用条件

- 目标算子在全模型中使用 `nn.Module` 参数（weight/bias 为 `get_attr` 节点）
- `register_replacement` 注册的 pattern 单元测试通过但全模型 kernel diff 确认未命中
- 典型案例：`nn.RMSNorm(self.weight)` → weight 为 `get_attr`，pattern 的 weight 为 `placeholder` → 匹配失败

---

## 实现步骤

1. 在 `PatternMatchPass`（`pattern_match_pass.py`）中新增方法（如 `_rewrite_rmsnorm_to_fused`）
2. 遍历 `graph.graph.nodes`，从终端 node（如 `aten.mul.Tensor`）出发回溯验证完整 pattern chain
3. 确认 chain 中 x 节点在 pow 和 mul_mid 中引用一致，使用 `graph.graph.inserting_before` 在终端节点前插入替换节点
4. 替换节点使用 `graph.graph.call_function(torch.ops.npu.xxx.default, args=...)` 创建
5. 用 `node.replace_all_uses_with(new_node)` 重定向所有引用，然后 `graph.graph.erase_node` 清理旧 chain
6. **该 pass 必须在 `graph_rewrite_after_freezing` 中调用**（非 `graph_rewrite_before_freezing`），
   否则 `torch._inductor.freezing.freeze()` 的 `node_copy` 会因不识别 NPU custom op 而 Crash

---

## 代码模板（RMSNorm 示例）

```python
import operator

def _rewrite_rmsnorm_to_fused(self, graph: torch.fx.GraphModule) -> int:
    if not torch.npu.is_available():
        return 0
    import torch_npu

    def _n(n):
        return n if isinstance(n, torch.fx.Node) else None

    replaced = 0
    for node in list(graph.graph.nodes):
        node_obj = _n(node)
        if not node_obj or node_obj.target not in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.default):
            continue

        # Determine mul_mid (normed x rsqrt) and weight_node (get_attr)
        arg0, arg1 = _n(node_obj.args[0]), _n(node_obj.args[1])
        if not arg0 or not arg1:
            continue
        if arg0.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.default):
            mul_mid, weight_node = arg0, arg1
        elif arg1.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.default):
            mul_mid, weight_node = arg1, arg0
        else:
            continue

        # Find rsqrt and x in mul_mid = x * rsqrt
        ma, mb = _n(mul_mid.args[0]), _n(mul_mid.args[1])
        if not ma or not mb:
            continue
        if ma.target in (torch.ops.aten.rsqrt.default, torch.ops.aten.rsqrt):
            rsqrt_node, x_node = ma, mb
        elif mb.target in (torch.ops.aten.rsqrt.default, torch.ops.aten.rsqrt):
            rsqrt_node, x_node = mb, ma
        else:
            continue

        # Trace back: rsqrt add mean pow, verify shared x
        add_node = _n(rsqrt_node.args[0])
        if not add_node or add_node.target not in (torch.ops.aten.add.Scalar, torch.ops.aten.add.Tensor):
            continue
        mean_node = _n(add_node.args[0])
        if not mean_node or mean_node.target != torch.ops.aten.mean.dim:
            continue
        pow_node = _n(mean_node.args[0])
        if not pow_node or pow_node.target != torch.ops.aten.pow.Tensor_Scalar:
            continue
        if _n(pow_node.args[0]) is not x_node:
            continue

        # Extract epsilon
        eps = 1e-6
        if len(add_node.args) >= 2:
            try:
                eps = float(add_node.args[1])
            except (TypeError, ValueError):

        with graph.graph.inserting_before(node_obj):
            result = graph.graph.call_function(
                torch.ops.aten.mul.Tensor, (weight_node, mul_mid)
            )

        node_obj.replace_all_uses_with(result)
        graph.graph.erase_node(node_obj)

        for dead in (node_obj, mul_mid, rsqrt_node, add_node, mean_node, pow_node):
            if len(dead.users) == 0:
                graph.graph.erase_node(dead)
        replaced += 1

    return replaced
```

---

## 在后端中调用

```python
# mindiesd/compilation/mindie_sd_backend.py
def graph_rewrite_after_freezing(fx_graph, inputs):
    self.__class__.apply_redundant_node_elimination_pass(fx_graph, inputs)
    patterns._rewrite_rmsnorm_to_fused(fx_graph)  # ← after freeze
    self.__class__.apply_decompose_auto_functionalized_pass(fx_graph)
    return fx_graph
```

---

## 验证

全模型 profiling + kernel diff：

- **eager trace 中消失的 kernel**: `PowTensorScalar`、`ReduceMean`、`Rsqrt`
- **compile trace 中新增的 kernel**: `RmsNorm`（即 `npu_rms_norm`）
- **判断标准**: 原始分解 kernel 耗时 ≈ 新增融合 kernel 耗时

---

## 维护与更新

当发现新的 get_attr 场景或更优的节点遍历策略时更新本文件。
