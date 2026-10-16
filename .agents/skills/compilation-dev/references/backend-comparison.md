# 四后端架构对比

MindIE-SD `torch.compile` 支持四种编译后端，通过 `CompilationConfig.backend_mode` 单点路由。

## 架构

```text
CompilationConfig.backend_mode
├── "default"         → _compile_default()      aot_autograd + PatternMatcherPass
├── "torchair_ge"     → _compile_torchair_ge()   torchair GE → ACL graph
├── "npugraph_ex"     → _compile_npugraph_ex()   backend="npugraph_ex"
└── "aclgraph"        → _compile_aclgraph()      NPUGraph static capture
```

## 机制对比

| 维度 | default | torchair_ge | npugraph_ex | aclgraph |
|------|:---:|:---:|:---:|:---:|
| **编译链路** | Dynamo → aot_autograd → Inductor | Dynamo → torchair GE → ACL graph | Dynamo → npugraph_ex → ACL graph | Dynamo → NPUGraph capture |
| **API** | `backend=MindieSDBackend()` | `get_npu_backend()` | `backend="npugraph_ex"` | `create_aclgraph_backend()` |
| **aot_autograd** | ✅ 使用 | ❌ 绕过 | ⚠️ torch 2.9 仍走 aot | ❌ 绕过 |
| **functionalization** | ✅ 产生 `_to_copy` | ❌ 无 | ⚠️ 有（Copy 未消除） | ❌ 无 |
| **PatternMatcherPass** | ✅ 生效 | ❌ 无（需 register_replacement） | ❌ 无 | ❌ 无 |
| **Copy 膨胀** | ⚠️ 模型相关 | ✅ 消除 | ❌ 存在 | ✅ 消除 |
| **适用场景** | Pattern 全覆盖模型 | 3D attention + 自定义 Norm | 实验/调试 | 静态 shape 大 batch |

## 编译链路

```text
default:     Dynamo → aot_autograd → pattern rewrites → freezing → Inductor codegen → NPU
torchair_ge: Dynamo → torchair GE → ACL graph → NPU (跳过 aot_autograd)
npugraph_ex: Dynamo → aot_autograd → ACL graph → NPU (torch 2.9 与 default 同)
aclgraph:    Dynamo → aot_autograd(可选) → NPUGraph capture → replay
```

## Wan2.2 实测 (torch 2.9.0, Ascend 910B)

| 指标 | No-Compile | default | torchair_ge | npugraph_ex |
|------|:---:|:---:|:---:|:---:|
| **Timed 推理** | 7007ms | 7632ms | **7023ms** | 7649ms |
| **vs NC** | — | +8.7% | **+0.2%** | +9.2% |
| 总算子数 | 448 | 480 | **448** | 480 |
| **Copy 总耗时** | 579ms | 1213ms | **579ms** | 1213ms |
| ViewCopy | 8/568ms | 16/1137ms | **8/568ms** | 16/1137ms |
| TensorMove | — | 16/40ms | **—** | 16/40ms |
| StridedSlice | — | 8/25ms | **—** | 8/25ms |

## FLUX.1-dev 实测 (torch 2.9.0, Ascend 910B)

| 指标 | No-Compile | default | torchair_ge | npugraph_ex |
|------|:---:|:---:|:---:|:---:|
| **Timed 推理** | 914ms | **878ms** | 908ms | 878ms |
| **vs NC** | — | **-4.0%** | -0.7% | **-4.0%** |
| 总算子数 | 2361 | **1727** | 2361 | **1727** |
| Copy 总耗时 | 4ms | 4ms | 4ms | 4ms |
| LayerNorm | 6.5ms | **0.2ms** (-97%) | 6.5ms | **0.2ms** (-97%) |
| Mul | 20.3ms | **9.6ms** (-53%) | 19.4ms | **9.4ms** (-54%) |
| RoPE | — | **10.1ms (融合)** | — | **10.1ms (融合)** |

## Copy 消减原理

**问题**: `aot_autograd` 作为 default 后端的编译包装器，会将所有 in-place view/reshape 操作转化为独立的 `_to_copy` 节点。这些节点在 Inductor codegen 阶段降级为 NPU InplaceCopy kernel（ViewCopy / TensorMove / StridedSlice）。

**链条**:

```text
aot_autograd → functionalization → _to_copy → Inductor codegen → InplaceCopy kernel
```

**torchair_ge 解决方案**: 通过 `get_npu_backend()` 获取的 GE 模式后端，将 FX graph 直接下沉到 ACL graph，完全绕过 `aot_autograd` → 无 functionalization → 无 `_to_copy` → **无 Copy 膨胀**。

**npugraph_ex 现状**: 在 torch 2.9.0 上，`backend="npugraph_ex"` 行为与 default 一致，仍经过 aot_autograd，不能独立消除 Copy 膨胀。

## 后端选择决策树

```text
模型使用 FP32LayerNorm？ → YES → torchair_ge
  ↓ NO
3D attention + ViewCopy 路径？ → YES → torchair_ge
  ↓ NO
标准 LayerNorm/RMSNorm？ → YES → default (Pattern 融合最优)
  ↓
大 batch 静态 shape？ → YES → aclgraph (NPUGraph replay)
  ↓
实验/对比 → npugraph_ex (与 default 等价)
```

## 配置使用

```python
from mindiesd.compilation import CompilationConfig

# torchair_ge (消除 Copy 膨胀)
CompilationConfig.backend_mode = CompilationConfig.BACKEND_TORCHAIR_GE
CompilationConfig.torchair_ge.inplace_pass = False

# default (Pattern 融合)
CompilationConfig.backend_mode = CompilationConfig.BACKEND_DEFAULT
CompilationConfig.fusion_patterns.enable_rms_norm = True

# npugraph_ex (原生后端)
CompilationConfig.backend_mode = CompilationConfig.BACKEND_NPUGRAPH_EX

# aclgraph (静态 capture)
CompilationConfig.backend_mode = CompilationConfig.BACKEND_ACLGRAPH
CompilationConfig.aclgraph.compile_first = True
```

## 相关 skill

- `compilation-dev/SKILL.md` Phase 7: Copy 消减全流程
- `performance-analysis/SKILL.md`: 5 层递进分析管道
- `performance-analysis/references/heuristics.md`: 编译路径选择决策表
