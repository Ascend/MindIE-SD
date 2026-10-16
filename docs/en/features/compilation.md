# Compilation Features

MindIE SD provides a custom backend `MindieSDBackend()` based on PyTorch's `torch.compile` compiler, delivering two complementary acceleration capabilities on Ascend chips:

- **Pattern Fusion**: Uses Pattern Matcher to automatically replace common operator combinations with Ascend fusion operators, reducing kernel launch overhead
- **ACLGraph Acceleration**: Captures the computation graph as a static execution graph via `torch.npu.NPUGraph`, skipping dynamic graph scheduling during replay

Both capabilities are uniformly controlled through `CompilationConfig`.

>[!NOTE] Note
>When this feature is enabled, there is a certain compilation overhead during initial model execution (up to 8 attempts by default), but generally no recompilation occurs in subsequent runs. During actual benchmark testing, the warm-up phase overhead should be excluded.

## Basic Usage

Both acceleration capabilities share the same entry point: call `torch.compile` on the model or its submodule and specify `MindieSDBackend()`.

Compile the entire transformer:

```python
pipe = FluxPipeline.from_pretrained(...)
transformer = torch.compile(pipe.transformer, backend=MindieSDBackend())
setattr(pipe, "transformer", transformer)
```

Use decorator on a single Module:

```python
@torch.compile(backend=MindieSDBackend())
class FluxSingleTransformerBlock(nn.Module):
```

Use decorator on a forward function:

```python
class FluxSingleTransformerBlock(nn.Module):
    @torch.compile(backend=MindieSDBackend())
    def forward(...):
```

## Pattern Fusion

`MindieSDBackend()` has multiple built-in operator fusion patterns. At compile time, they are automatically matched and replaced with Ascend-optimized operators. Each pattern can be individually toggled via `CompilationConfig.fusion_patterns`:

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.fusion_patterns.enable_rms_norm = False      # Disable RMSNorm fusion
CompilationConfig.fusion_patterns.enable_rope = False          # Disable RoPE fusion
CompilationConfig.fusion_patterns.enable_adalayernorm = False  # Disable adaLN fusion
CompilationConfig.fusion_patterns.enable_fast_gelu = False     # Disable fastGelu fusion
CompilationConfig.fusion_patterns.enable_mul_add = False       # Disable Mul+Add fusion
```

For detailed API documentation of each fusion operator, see [the fusion operator section in core_layers.md](core_layers.md#fused-operators).

### API Reference

#### MindieSDBackend

```python
from mindiesd.compilation import MindieSDBackend
```

Passed as the `backend` parameter to `torch.compile`, automatically enables Pattern Fusion and ACLGraph acceleration.

#### CompilationConfig

```python
from mindiesd.compilation import CompilationConfig
```

Pattern fusion toggles are controlled via `fusion_patterns`:

| Option | Default | Description |
|--------|--------|------|
| `fusion_patterns.enable_rms_norm` | `True` | RMSNorm fusion |
| `fusion_patterns.enable_rope` | `True` | RoPE fusion |
| `fusion_patterns.enable_adalayernorm` | `True` | AdaLayerNorm fusion |
| `fusion_patterns.enable_fast_gelu` | `True` | fastGELU fusion |
| `fusion_patterns.enable_mul_add` | `True` | Mul+Add fusion |

## ACLGraph Acceleration

On top of Pattern Fusion, you can further enable ACLGraph to capture the optimized graph as a static execution plan.

### Configuration

`aclgraph_only` and `aclgraph_with_compile` are mutually exclusive; when both are enabled, `aclgraph_with_compile` takes higher priority.

| Option | Default | Description |
|--------|--------|------|
| `aclgraph_only` | `False` | ACLGraph only, skip Pattern Fusion |
| `aclgraph_with_compile` | `False` | Pattern Fusion first, then capture as ACLGraph |
| `enable_freezing` | `True` | Whether to perform constant folding before compilation |
| `safe_output_mode` | `True` | Whether to clone output during ACLGraph replay |
| `graph_log_url` | `None` | Graph transform log URL for debugging |

### Usage Example

Based on the [Basic Usage](#basic-usage) above, configure `CompilationConfig` before calling:

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.aclgraph_with_compile = True
# Subsequent calls to torch.compile(..., backend=MindieSDBackend()) will automatically enable it
```

### Variable-Length Input Handling

In scenarios such as audio, input lengths are not fixed and can be adapted via external padding:

```python
max_len = 512

model = torch.compile(transformer, backend=MindieSDBackend())
_ = model(torch.randn(max_len, dim, device="npu"))  # Trigger capture

for audio_chunk in chunks:
    actual_len = audio_chunk.shape[0]
    padded = torch.nn.functional.pad(audio_chunk, (0, 0, 0, max_len - actual_len))
    output = model(padded)[:actual_len]
```

### Limitations and Notes

- **Environment dependency**: Only supported on Ascend NPU environments
- **Input shape**: Runtime input shape must match the shape at capture time; changes will trigger re-capture
- **Dynamic features**: Dynamic shapes, dynamic control flow, or conditional branching are not supported
- **First-time overhead**: The first graph capture incurs a one-time overhead
- **graph.update**: The `graph.update` interface is not provided (this interface is used for dynamically injecting attention metadata in LLM scenarios and is not needed for SD scenarios)
- **Configuration timing**: `CompilationConfig` must be configured before calling `torch.compile()`

### Troubleshooting Tips

- The troubleshooting methods are consistent with PyTorch's compile. A logging module is defined in [mindie_sd_backend.py](../../../mindiesd/compilation/mindie_sd_backend.py). After enabling it, you can observe graph changes before and after pattern activation. Combined with narrowing the scope via torch.compile, you can identify the cause of pattern failures.
- Controlling the compile scope can effectively control the troubleshooting scope.
- For other troubleshooting methods, refer to the [PyTorch official documentation](https://docs.pytorch.org/docs/main/generated/torch.compile.html).

## Reference Examples

- **[FLUX.1-dev Inference Optimization Guide](../../../examples/cache-dit/FLUX.1-dev.md)**: An end-to-end single-card example that combines `MindieSDBackend()` compilation optimization with Cache-DiT DBCache on FLUX.1-dev. A complete, runnable reference covering:

  - Before/after comparison: how to compare runtime between the baseline and compilation-optimized runs of the same inference code, and the expected effect
  - Practical prerequisites: install dependencies with `pip install cache-dit`; DBCache requires the `PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'` environment variable, otherwise memory overflow may occur
  - Effect verification: watch the `PatternMatchPass replace` count in the debug log (`MINDIE_LOG_LEVEL=debug`) to confirm fused operators such as RMSNorm and RoPE are replaced, then measure runtime
  - A complete runnable script: integrates `MindieSDBackend()` compilation optimization with cache parameter configuration for both dual-stream and single-stream transformer blocks
