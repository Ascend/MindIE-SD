# 编译特性

MindIE SD 基于 PyTorch 的 `torch.compile` 编译器提供自定义后端 `MindieSDBackend()`，在昇腾芯片上提供两套互补的加速能力：

- **Pattern 融合**：利用 Pattern Matcher 将常见算子组合自动替换为昇腾融合算子，减少 kernel 启动开销
- **ACLGraph 加速**：通过 `torch.npu.NPUGraph` 将计算图捕获为静态执行图，replay 时跳过动态图调度

两项能力通过 `CompilationConfig` 统一控制。

>[!NOTE]说明
>当使能该特性后，模型运行初期存在一定的编译耗时（默认最多进行8次尝试），但是在后续运行中，一般不会再次编译。在实际benchmark测试过程中，需要将预热阶段的耗时去除。

## 基本用法

两种加速能力共享相同的入口：对模型或其子模块调用 `torch.compile` 并指定 `MindieSDBackend()`。

对 transformer 整体 compile：

```python
pipe = FluxPipeline.from_pretrained(...)
transformer = torch.compile(pipe.transformer, backend=MindieSDBackend())
setattr(pipe, "transformer", transformer)
```

对单个 Module 使用装饰器：

```python
@torch.compile(backend=MindieSDBackend())
class FluxSingleTransformerBlock(nn.Module):
```

对 forward 函数使用装饰器：

```python
class FluxSingleTransformerBlock(nn.Module):
    @torch.compile(backend=MindieSDBackend())
    def forward(...):
```

## Pattern 融合

`MindieSDBackend()` 内置了多组算子融合 Pattern，编译时自动匹配并替换为昇腾优化算子。各 Pattern 的开关可通过 `CompilationConfig.fusion_patterns` 单独控制：

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.fusion_patterns.enable_rms_norm = False   # 关闭 RMSNorm 融合
CompilationConfig.fusion_patterns.enable_rope = False       # 关闭 RoPE 融合
```

### 支持度

|     模型     | RMSNorm | Rope | fastGelu | adaLN |
|:----------:|:------: |:---: |:---: |:-----:|
| flux.1-dev | ✅      | ✅   | ✅️ |  ✅️   |

## ACLGraph 加速

在 Pattern 融合的基础上，可进一步启用 ACLGraph 将优化后的图捕获为静态执行计划。

### 配置方式

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `aclgraph_only` | `False` | 仅 ACLGraph，跳过 Pattern 融合 |
| `aclgraph_with_compile` | `False` | 先 Pattern 融合，再捕获为 ACLGraph |

两者互斥，同时开启时 `aclgraph_with_compile` 优先级更高。

### 使用示例

在上方 [基本用法](#基本用法) 的基础上，调用前配置 `CompilationConfig` 即可启用：

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.aclgraph_with_compile = True
# 之后调用 torch.compile(..., backend=MindieSDBackend()) 即自动启用
```

### 变长输入处理

语音等场景下输入长度不固定，可通过外部 padding 适配：

```python
max_len = 512

model = torch.compile(transformer, backend=MindieSDBackend())
_ = model(torch.randn(max_len, dim, device="npu"))  # 触发捕获

for audio_chunk in chunks:
    actual_len = audio_chunk.shape[0]
    padded = torch.nn.functional.pad(audio_chunk, (0, 0, 0, max_len - actual_len))
    output = model(padded)[:actual_len]
```

### 限制与注意事项

- **环境依赖**：仅昇腾 NPU 环境支持
- **输入 shape**：运行时输入 shape 须与捕获时一致，变更会触发重新捕获
- **动态特性**：不支持动态 shape、动态 control flow 或 conditional branching
- **首次耗时**：首次触发图捕获存在一次性耗时开销
- **graph.update**：不提供 `graph.update` 接口（该接口用于 LLM 场景动态注入 attention metadata，SD 场景不需要）
- **配置时机**：`CompilationConfig` 需在 `torch.compile()` 调用前完成配置

### 问题定位技巧

- 相关的定位手段与PyTorch的compile一致，[mindie_sd_backend.py](../../../mindiesd/compilation/mindie_sd_backend.py)中定义了日志模块，开启后，可以观察到pattern使能前后的图变化情况。配合torch.compile缩小范围，可以识别pattern失效的原因。
- 通过控制compile的范围，可以有效控制问题定位的范围。
- 其他定位手段可以参考[PyTorch](https://docs.pytorch.org/docs/main/generated/torch.compile.html)官网。
