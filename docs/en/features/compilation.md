# Compilation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:14:22.082Z pushedAt=2026-06-08T02:57:38.985Z -->

MindIE SD provides a custom backend `MindieSDBackend()` based on PyTorch's `torch.compile` compiler, delivering two complementary acceleration capabilities on Ascend chips:

- **Pattern Fusion**: Uses Pattern Matcher to automatically replace common operator combinations with Ascend fused operators, reducing kernel launch overhead.

- **ACLGraph Acceleration**: Captures the computation graph as a static execution graph via `torch.npu.NPUGraph`, skipping dynamic graph scheduling during replay.

Both capabilities are uniformly controlled via `CompilationConfig`.

>[!NOTE]Note
>When this feature is enabled, there is a certain compilation overhead during the initial model run (up to 8 attempts by default), but it generally will not recompile in subsequent runs. During actual benchmark testing, the warm-up phase overhead needs to be excluded.

## Basic Usage

Both acceleration capabilities share the same entry point: calling `torch.compile` on the model or its submodules and specifying `MindieSDBackend()`.

Compile the entire transformer:

```python
pipe = FluxPipeline.from_pretrained(...)
transformer = torch.compile(pipe.transformer, backend=MindieSDBackend())
setattr(pipe, "transformer", transformer)
```

Use a decorator for a single module:

```python
@torch.compile(backend=MindieSDBackend())
class FluxSingleTransformerBlock(nn.Module):
```

Use a decorator on the `forward` function:

```python
class FluxSingleTransformerBlock(nn.Module):
    @torch.compile(backend=MindieSDBackend())
    def forward(...):
```

## Pattern Fusion

`MindieSDBackend()` has multiple sets of operator fusion patterns built in. During compilation, these patterns are automatically matched and replaced with Ascend-optimized operators. The switch for each pattern can be individually controlled via `CompilationConfig.fusion_patterns`:

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.fusion_patterns.enable_rms_norm = False      # Disable RMSNorm fusion
CompilationConfig.fusion_patterns.enable_rope = False          # Disable RoPE fusion
CompilationConfig.fusion_patterns.enable_adalayernorm = False  # Disable adaLN Fusion
CompilationConfig.fusion_patterns.enable_fast_gelu = False     # Disable fastGelu Fusion
CompilationConfig.fusion_patterns.enable_mul_add = False       # Disable Mul+Add Fusion
 ```

For detailed API descriptions of each fusion operator, see [fused operators in core_layers.md](core_layers.md#fused-operator).

### API Description

#### `MindieSDBackend`

```python
from mindiesd.compilation import MindieSDBackend
```

Passed as the `backend` parameter of `torch.compile`, it automatically enables Pattern Fusion and ACLGraph Acceleration.

#### `CompilationConfig`

```python
from mindiesd.compilation import CompilationConfig
```

The switch for Pattern Fusion is controlled via `fusion_patterns`:

| Configuration Item | Default Value | Description |
|--------|--------|------|
| `fusion_patterns.enable_rms_norm` | `True` | RMSNorm fusion |
| `fusion_patterns.enable_rope` | `True` | RoPE fusion |
| `fusion_patterns.enable_adalayernorm` | `True` | AdaLayerNorm fusion |
| `fusion_patterns.enable_fast_gelu` | `True` | fastGELU fusion |
| `fusion_patterns.enable_mul_add` | `True` | Mul+Add fusion |

## ACLGraph Acceleration

Based on Pattern Fusion, you can further enable ACLGraph to capture the optimized graph as a static execution plan.

### Configuration Method

`aclgraph_only` and `aclgraph_with_compile` are mutually exclusive. When both are enabled, `aclgraph_with_compile` takes higher priority.

| Configuration Item | Default Value | Description |
|--------|--------|------|
| `aclgraph_only` | `False` | ACLGraph only, skipping Pattern Fusion |
| `aclgraph_with_compile` | `False` | Pattern Fusion first, then ACLGraph |
| `enable_freezing` | `True` | Whether to perform constant folding before compilation |
| `safe_output_mode` | `True` | Whether to clone outputs during ACLGraph replay |
| `graph_log_url` | `None` | Debug URL for graph transform logs |

### Usage Example

Based on the [Basic Usage](#basic-usage) above, configure `CompilationConfig` before calling to enable the feature:

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.aclgraph_with_compile = True
# Afterwards, calling torch.compile(..., backend=MindieSDBackend()) automatically enables it
```

### Variable-Length Input Handling

For scenarios like voice input where the input length is variable, external padding can be used for adaptation.

```python
max_len = 512

model = torch.compile(transformer, backend=MindieSDBackend())
_ = model(torch.randn(max_len, dim, device="npu"))  # Trigger capture

for audio_chunk in chunks:
    actual_len = audio_chunk.shape[0]
    padded = torch.nn.functional.pad(audio_chunk, (0, 0, 0, max_len - actual_len))
    output = model(padded)[:actual_len]
```

### Constraints

- **Environment**: Only supported on Ascend NPU environments

- **Input shape**: The input shape at runtime must be consistent with that at capture time; changes will trigger re-capture.

- **Dynamic features**: Dynamic shapes, dynamic control flow, or conditional branching are not supported.

- **First-time latency**: The first-time graph capture incurs a one-time latency overhead.

- **graph.update**: The `graph.update` interface is not provided (this interface is used for dynamically injecting attention metadata in LLM scenarios and is not required for SD scenarios).

- **Configuration timing**: `CompilationConfig` must be configured before calling `torch.compile()`.

### Troubleshooting Tips

- The positioning approach aligns with PyTorch's `compile`. The logging module defined in [mindie_sd_backend.py](../../../mindiesd/compilation/mindie_sd_backend.py) allows observation of graph changes before and after pattern application when enabled. Use it alongside `torch.compile` to narrow down the scope and identify why a pattern fails to take effect.

- By scoping the compilation, you can effectively narrow down the issue location.

- For other localization methods, see the [PyTorch](https://docs.pytorch.org/docs/main/generated/torch.compile.html) official website.
