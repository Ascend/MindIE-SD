# Copy 算子消减全流程（default/Inductor 路径）

Pattern 匹配成功后，检查 compile 是否引入额外 Copy 算子并制定消减方案。
本文件只描述**本仓已实现**的 default（aot_autograd + Inductor）路径；
批量下发（aclgraph）见 `aclgraph-dev` skill。

---

## 1. Copy 引入机制

`default` 后端使用 `aot_autograd` 作为编译包装器，其 functionalization 将所有
in-place view/reshape 转为 `_to_copy` 节点 → Inductor codegen → InplaceCopy NPU kernel。

> 历史上记录过的 torchair_ge / npugraph_ex "四后端对比"**在本仓未实现**
> （`mindiesd/compilation/compiliation_config.py` 无 `backend_mode` 及对应常量），
> 不作为可执行方案。

---

## 2. Copy 检测方法

在 `kernel_details.csv` 中搜索 copy 类算子：

```bash
grep "InplaceCopy\|ViewCopy\|TensorMove\|StridedSlice" kernel_details.csv | wc -l
```

关注以下 Copy 算子类型：

| 算子 | 含义 | 典型来源 |
|------|------|---------|
| `InplaceCopy_ViewCopyAiCore` | 大尺寸 tensor reshape copy | VAE decoder 的 3D→2D 变换 |
| `InplaceCopy_TensorMoveAiCore` | functionalization 引入的 tensor move | aot_autograd `_to_copy` |
| `InplaceCopy_StridedSliceAiCore` | strided slice copy | 3D attention 的 QKV 重组 |
| `InplaceCopy_TransposeAiCore` | transpose copy | Attention score 计算 |
| `InplaceCopy_CastAiCore` | dtype cast copy | bf16↔f32 转换 |

### Kernel diff 对比法

同时采集 eager 和 compile profiling，按 kernel 名称聚合耗时后 diff：

1. **同名 kernel 耗时差**排序，定位膨胀源（如 ViewCopy 569→1137ms）
2. **eager_only kernel**（被融合的原始算子）与 **compile_only kernel**（新增融合算子）对账

---

## 3. Copy 膨胀根因分析

Copy 膨胀的程度取决于模型结构：

| 因素 | 低风险 (Copy 无膨胀) | 高风险 (Copy 膨胀) |
|------|:---:|:---:|
| **Attention 维度** | 2D (FLUX.1-dev) | 3D (Wan2.2) |
| **Norm 层类型** | 标准 LayerNorm/RMSNorm | FP32LayerNorm (→ native_layer_norm) |
| **Pattern 命中率** | 4/4 全部命中 | 仅 GELU 命中 |
| **VAE 结构** | 简单 2D Conv | 复杂 3D Conv + StridedSlice |

---

## 4. 消减方案

### 方案 A: 修复 Pattern 匹配（default 路径，推荐）

条件: 模型使用标准 Norm 层 (LayerNorm/RMSNorm)，非 FP32LayerNorm。
提高 pattern 命中率可减少 functionalization 引入的 Copy（FLUX.1-dev 实测 4 pattern
全部命中时 Copy 不增反减）。

### 方案 B: 混合模式

对于 VAE 部分使用 eager 模式（`--skip-vae`），仅 transformer 走 compile。

### 方案 C: aclgraph 批量下发

静态 shape / 大 batch 场景改用 `aclgraph` 批量下发（`CompilationConfig.aclgraph_only` /
`aclgraph_with_compile`），replay 省去 host launch；机制与调优见 `aclgraph-dev` skill。

---

## 5. 复验验证

```bash
# eager vs compile 双模式采集
python wan_infer.py --device_id 0 --profile              # No-Compile
python wan_infer.py --device_id 0 --profile --compile    # default

# Copy 对比
grep -c "InplaceCopy\|ViewCopy\|TensorMove\|StridedSlice" */kernel_details.csv
```

验证标准:

- compile 的 Copy count ≤ No-Compile 或膨胀可解释（新增融合 kernel 带来收益）
- 无异常新增 TensorMove/StridedSlice 算子
- 计时遵循 `benchmark-guide.md`（L2-flush 放计时区外、warm/cold 双档）

---

## 维护与更新

当发现新的 Copy 膨胀场景或 aot_autograd/Inductor 行为变化时更新本文件。
