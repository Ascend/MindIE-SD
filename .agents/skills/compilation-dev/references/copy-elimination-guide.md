# Copy 算子消减全流程

Pattern 匹配成功后，检查 compile 是否引入额外 Copy 算子并制定消减方案。

---

## 1. Copy 引入机制

`default` 后端使用 `aot_autograd` 作为编译包装器，其 functionalization 将所有 in-place view/reshape 转为 `_to_copy` 节点 → Inductor codegen → InplaceCopy NPU kernel。

**torchair_ge** 通过 `torch_npu.dynamo.torchair.get_npu_backend()` 获取 GE 模式后端，绕过 aot_autograd，消除此链条。完整链路对比见 `references/backend-comparison.md`。

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

**已验证结论** (8-mode 全量数据见 `references/backend-comparison.md`):

- **torchair_ge** 是唯一能绕过 AOT Autograd 消除 Copy 膨胀的后端（Wan2.2 ViewCopy 16→8）
- **npugraph_ex** 在 torch 2.9.0 上与 default 等价（仍走 aot_autograd，Copy 存在）
- **default** 在 FLUX.1-dev 上最优（4 pattern 全部命中，-4.0% 加速）

---

## 4. 消减方案

**方案 A: 切换 torchair_ge (推荐，适用于 3D attention 模型)**

原理: torchair GE 图模式直接下沉 ACL graph，无 functionalization → 无 Copy 膨胀。

**效果** (Wan2.2, torch 2.9.0 已验证):

- ViewCopy: 16→8 (-569ms)
- TensorMove: 16→0 (-40ms)
- StridedSlice: 8→0 (-25ms)
- Timed 推理: 7632ms→7023ms (-8%, 与 No-Compile 持平)

**方案 B: 修复 Pattern 匹配 (default 路径)**

条件: 模型使用标准 Norm 层 (LayerNorm/RMSNorm)，非 FP32LayerNorm。
FLUX.1-dev 已生效：4 pattern 全部命中 → Copy 不增反减 (-81%)。

**方案 C: 混合模式**

对于 VAE 部分使用 eager 模式（`--skip-vae`），仅 transformer 走 compile。

**方案 D: 试验 npugraph_ex（不推荐）**

原生 `backend="npugraph_ex"` 在 torch 2.9.0 上与 default 等价。仅用于对比验证或未来 torch 版本升级后重新评估。

---

## 5. 复验验证

```bash
# 四模式采集
python wan_infer.py --device_id 0 --profile              # No-Compile
python wan_infer.py --device_id 0 --profile --compile    # default
python wan_infer.py --device_id 0 --profile --npugraph   # torchair_ge

# Copy 对比
grep -c "InplaceCopy\|ViewCopy\|TensorMove\|StridedSlice" */kernel_details.csv
```

验证标准:

- torchair_ge (--npugraph) 的 Copy count ≤ No-Compile
- torchair_ge 的 Copy duration ≤ No-Compile * 1.05
- 无新增 TensorMove/StridedSlice 算子

---

## 维护与更新

当发现新的 Copy 膨胀场景或后端支持矩阵变化时更新本文件。
