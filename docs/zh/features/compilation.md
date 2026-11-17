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

CompilationConfig.fusion_patterns.enable_rms_norm = False      # 关闭 RMSNorm 融合
CompilationConfig.fusion_patterns.enable_rope = False          # 关闭 RoPE 融合
CompilationConfig.fusion_patterns.enable_adalayernorm = False  # 关闭 adaLN 融合
CompilationConfig.fusion_patterns.enable_fast_gelu = False     # 关闭 fastGelu 融合
CompilationConfig.fusion_patterns.enable_mul_add = False       # 关闭 Mul+Add 融合
```

各融合算子的详细 API 说明请参见[核心加速API中的融合算子章节](core_layers.md#融合算子)。

### 接口说明

#### MindieSDBackend

```python
from mindiesd.compilation import MindieSDBackend
```

作为 `torch.compile` 的 `backend` 参数传入，自动启用 Pattern 融合和 ACLGraph 加速。

#### CompilationConfig

```python
from mindiesd.compilation import CompilationConfig
```

Pattern 融合的开关通过 `fusion_patterns` 控制：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `fusion_patterns.enable_rms_norm` | `True` | RMSNorm 融合 |
| `fusion_patterns.enable_rope` | `True` | RoPE 融合 |
| `fusion_patterns.enable_adalayernorm` | `True` | AdaLayerNorm 融合 |
| `fusion_patterns.enable_fast_gelu` | `True` | fastGELU 融合 |
| `fusion_patterns.enable_mul_add` | `True` | Mul+Add 融合 |
| `fusion_patterns.enable_gelu_quant_fusion` | `False` | gelu → 动态 MX 量化链融合为 `npu_gelu_quant`；默认关闭，需 NPU 数值回归后再开启 |
| `fusion_patterns.enable_rope_cast_fusion` | `False` | 消除 `npu_rotary_mul` 两端的 fp32 cast；默认关闭，需质量回归后再开启 |
| `fusion_patterns.gelu_quant_approximate` | `"tanh"` | gelu-quant 融合在链头为 `npu_fast_gelu`（无可透传的近似参数）时采用的近似模式；链头为 `aten.gelu` 时透传其自身设置 |

## 后端选择（backend_mode）

`CompilationConfig.backend_mode` 是后端选择的单点入口。为 `None`（默认）时按旧的布尔字段推导，保持向后兼容；优先级为：显式 `backend_mode` > `aclgraph_with_compile` > `aclgraph_only` > `default`。

| 取值 | 说明 |
|------|------|
| `"default"` | 仅 Pattern 融合（AOT Autograd），不做图捕获 |
| `"aclgraph_only"` | 仅 ACLGraph 捕获；`aclgraph_enable_patterns=True` 时先施加轻量算子展开与 Pattern 融合 |
| `"aclgraph_with_compile"` | 先完整编译（Pattern 融合 + AOT Autograd），再捕获为 ACLGraph |
| `"npugraph_ex"` | 走 torch_npu 的 `npugraph_ex.compile_fx` 入口 |

`BackendMode` 枚举成员名（如 `"NPUGRAPH_EX"`）与取值字符串等价。无法识别的取值会告警并被忽略，回退到按布尔字段推导；目标后端不可用时（NPU 不可用、`npugraph_ex` 不可用）自动降级到 `default` 并告警。

### npugraph_ex 后端

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.backend_mode = "npugraph_ex"
CompilationConfig.aclgraph_enable_patterns = True  # 默认为 False；开启后捕获前先做 Pattern 融合
CompilationConfig.npugraph_ex_options = None       # 可选：透传给 npugraph_ex.compile_fx 的 options 字典
                                                   # 未显式配置时默认补 reuse_graph_pool_in_same_fx=False
                                                   # 与 clone_output=True（见下文两节）
```

`npugraph_ex` 入口自身完成 AOT 分解与图执行，MindIE SD 只负责在捕获前施加 Pattern 融合，因此不需要 `apply_lightweight_decomp` 的 `make_fx` 预展开。

#### 多图场景：关闭跨图内存池复用

torch_npu 侧 `reuse_graph_pool_in_same_fx` 默认为开启，即同一 FX 中的多张 ACLGraph 共享一个内存池。当一个进程内先后出现多张图——例如一次任务里依次跑两个不同文本长度的 prompt，或输入 shape 变化导致图被重新捕获——后一张图会复用前一张图的缓冲区，使**跨图断点传递的张量**（典型是由 `Tensor.tolist()` 断图后仍需在 eager 侧消费的 shape 张量，以及模型返回张量）的 shape 或数据被覆写成指针量级的垃圾值。

MindIE SD 在 `npugraph_ex_options` **未显式配置该项时默认置为 `False`**，并在首次生效时打印一条提示；若用户显式传 `True`，则尊重该取值，同时打印一次风险告警（含上述报错特征与本节的规避方式）。

典型现象（偶发，同一配置时过时不过，单次运行不具备判别力）：

- `RuntimeError: Dynamo failed to run FX node with fake tensors: call_method view(...)`，报错中的符号取值在 `1e12` 量级；
- `RuntimeError: The size of tensor a (5736774568800) must match the size of tensor b (0) ...`；
- `RuntimeError: shape '[5833846557024, 34, 25, ...]' is invalid for input of size ...`。

由于默认值已关闭跨图复用，通常无需额外配置；确实需要共享内存池时再显式打开：

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.backend_mode = "npugraph_ex"
# 默认即为 False（每张图独立内存池）；显式置 True 会打印风险告警
CompilationConfig.npugraph_ex_options = {"reuse_graph_pool_in_same_fx": True}
```

代价是每张图各自持有一份内存池，峰值显存随图数量上升，显存紧张时需要结合图数量评估。

注意 `clone_output=True` 不能替代该选项：它解决的是下一节的输出缓冲跨调用复用问题，解决不了输出元数据本身被跨图覆写的问题。

#### 单步内多次调用同一张图：开启输出克隆

torch_npu 侧 `clone_output` 默认为 `False`，replay 直接返回静态输出缓冲。同一步内对同一张图连续调用两次时（CFG 的 cond/uncond 是最常见形态，双流、teacher forcing 同理），第二次 replay 会覆盖第一次返回的张量，调用方拿到的两个结果完全相同：

```python
noise_pred_cond   = model(x, context=text)   # 写进静态输出缓冲 B
noise_pred_uncond = model(x, context=null)   # 同一张图，再次写进 B
noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
#          = noise_pred_uncond            # cond 已被覆盖，引导项恒为 0
```

引导项归零，等效 `guide_scale=1`。**不会报错**，但生成结果的动态与文本响应明显变差：背景几乎静止、主体没有动作，而画面主体仍与参考图一致（图像条件始终在，所以看起来"功能正常"）。

MindIE SD 在未显式配置该项时默认置为 `True`，首次生效时打印一条提示；显式置 `False` 会打印一次风险告警。代价是每次 replay 多一次输出拷贝，并额外占用一份输出大小的显存。`aclgraph_only` / `aclgraph_with_compile` 路径不受此问题影响，因为它们由 `safe_output_mode`（默认 `True`）在每次 replay 后克隆输出。

与上一节的 `reuse_graph_pool_in_same_fx` 是两个独立问题：池复用导致偶发崩溃与垃圾 shape，输出缓冲复用导致稳定但不报错的数值错误，二者不能互相替代。

## ACLGraph 加速

在 Pattern 融合的基础上，可进一步启用 ACLGraph 将优化后的图捕获为静态执行计划。

### 配置方式

`aclgraph_only` 与 `aclgraph_with_compile` 互斥，同时开启时 `aclgraph_with_compile` 优先级更高。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `aclgraph_only` | `False` | 仅 ACLGraph，跳过 Pattern 融合 |
| `aclgraph_with_compile` | `False` | 先 Pattern 融合，再捕获为 ACLGraph |
| `aclgraph_breakpoints` | `None` | 分段捕获断点节点名列表，为空时整图单段捕获（见 [Breakable Graph 分段捕获](#breakable-graph-分段捕获)） |
| `aclgraph_enable_patterns` | `False` | `aclgraph_only` / `npugraph_ex` 下是否在捕获前做轻量算子展开与 Pattern 融合 |
| `enable_freezing` | `True` | 编译前是否执行常量折叠 |
| `safe_output_mode` | `True` | ACLGraph replay 时是否 clone 输出 |
| `graph_log_url` | `None` | 调试用 graph transform 日志 URL |

### 使用示例

在上方 [基本用法](#基本用法) 的基础上，调用前配置 `CompilationConfig` 即可启用：

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.aclgraph_with_compile = True
# 之后调用 torch.compile(..., backend=MindieSDBackend()) 即自动启用
```

### Breakable Graph 分段捕获

整图捕获因局部算子或控制流而失败时，可用 `aclgraph_breakpoints` 指定断点节点名，把图切分为若干顺序子图逐段捕获。段与段之间通过静态缓冲衔接：前段输出原地写入缓冲，后段直接读取，无需额外拷贝。

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.backend_mode = "aclgraph_only"
CompilationConfig.aclgraph_breakpoints = ["transformer_blocks_0", "transformer_blocks_12"]
```

- 断点是 FX 图的**节点名**（`fx.Node.name`），不是模块名。可从 `graph_log_url` 记录的图或 `logger.debug(graph.print_readable())` 的输出中获取。每个命中的断点节点作为新一段的起点。
- 断点为空或全部未命中时不做切分，退化为整图单段捕获。
- 返回结构按原始 `output` 的容器类型与顺序组装；由非末段产生的图输出同样会被返回，不会丢失。
- 当前仅在 `aclgraph_only` 下生效；`aclgraph_with_compile` 会告警并退回整图捕获。

**限制**：分段捕获按首次捕获时的输入形状固定。运行时 shape/dtype 与捕获时不一致会显式抛出 `RuntimeError`（不会静默错算），请在模型侧保持一致（必要时要 padding 到固定长度）。捕获期被固化进 kernel 参数的标量输入（`int`/`bool`/`float`/`str`）同理：回放时按值校验，取值变化同样直接报错。

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

### 环境变量

以下环境变量用于兼容性与数值口径控制。**其中前两项在 `mindiesd.compilation` 模块导入时读取**，需在导入前设置。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MINDIE_SD_ATTENTION_DYNAMO_DISABLE` | `1` | 对 `attention_forward` 关闭 Dynamo 追踪。关闭追踪是为绕开旧环境 autocast 相关的 trace 崩溃，避免打断整图编译。设为 `0` 可让 attention 回到图内，用于在 NPU 上复验该崩溃是否已消失并做 A/B 对比 |
| `MINDIE_SD_DYNAMO_COMPAT_PATCHES` | `1` | 进程级 Dynamo 兼容补丁：NPU 下的 triton/TMA 设备探测、`torch_npu` 对 `os.getenv` 的包装（会导致 trace 期碎图）。设为 `0` 整体关闭，用于与 torch_npu 官方补丁共存或排除本补丁影响 |
| `MINDIE_SD_ADALN_DTYPE_ALIGN` | `1` | AdaLayerNorm 融合前把调制量对齐到 `x.dtype`。融合算子要求所有输入 dtype 一致，不能依赖 eager 的类型提升。设为 `0` 关闭对齐，此时需调用方自行保证 dtype 一致 |

### 问题定位技巧

- 相关的定位手段与 PyTorch 的 compile 一致，[mindie_sd_backend.py](../../../mindiesd/compilation/mindie_sd_backend.py)中定义了日志模块，开启后，可以观察到pattern使能前后的图变化情况。配合torch.compile缩小范围，可以识别pattern失效的原因。
- 通过控制compile的范围，可以有效控制问题定位的范围。
- 其他定位手段可以参考[PyTorch](https://docs.pytorch.org/docs/main/generated/torch.compile.html)官网。

## 参考示例

- **[FLUX.1-dev 模型推理优化指南](../../../examples/cache-dit/FLUX.1-dev.md)**：在 FLUX.1-dev 上组合使用 `MindieSDBackend()` 编译优化与 Cache-DiT DBCache 的单卡端到端案例。可作为落地参考的完整示例，主要内容包括：

  - 优化前后对比：同一推理代码在未使能 / 使能编译优化两种运行下的耗时对比方法与预期效果
  - 落地要点：`pip install cache-dit` 安装依赖；DB cache 依赖 `PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'` 环境变量，否则可能内存溢出
  - 效果验证：通过 `MINDIE_LOG_LEVEL=debug` 观察 `PatternMatchPass replace` 匹配数量，确认 RMSNorm、RoPE 等融合算子被替换后再统计耗时
  - 完整可运行脚本：整合编译优化与 transformer 双流/单流 block 的缓存参数配置
