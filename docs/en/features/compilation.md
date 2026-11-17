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

For detailed API documentation of each fusion operator, see [the fusion operator section in Core Acceleration API](core_layers.md#fused-operators).

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
| `fusion_patterns.enable_gelu_quant_fusion` | `False` | Fuses the gelu → dynamic MX quant chain into `npu_gelu_quant`; disabled by default until validated numerically on NPU |
| `fusion_patterns.enable_rope_cast_fusion` | `False` | Removes the fp32 casts around both ends of `npu_rotary_mul`; disabled by default until quality regression passes |
| `fusion_patterns.gelu_quant_approximate` | `"tanh"` | Approximation mode used by gelu-quant fusion when the chain head is `npu_fast_gelu` (which has no approximation argument to forward); when the head is `aten.gelu`, its own setting is forwarded |

## Backend Selection (backend_mode)

`CompilationConfig.backend_mode` is the single entry point for backend selection. When it is `None` (default), the backend is derived from the legacy boolean fields for backward compatibility. Priority: explicit `backend_mode` > `aclgraph_with_compile` > `aclgraph_only` > `default`.

| Value | Description |
|-------|-------------|
| `"default"` | Pattern fusion only (AOT Autograd), no graph capture |
| `"aclgraph_only"` | ACLGraph capture only; with `aclgraph_enable_patterns=True`, lightweight operator decomposition and Pattern fusion are applied first |
| `"aclgraph_with_compile"` | Full compilation (Pattern fusion + AOT Autograd) first, then capture as ACLGraph |
| `"npugraph_ex"` | Uses torch_npu's `npugraph_ex.compile_fx` entry point |

Enum member names (such as `"NPUGRAPH_EX"`) are equivalent to the value strings. Unrecognized values are ignored with a warning and fall back to the boolean-field derivation. When the target backend is unavailable (no NPU, or `npugraph_ex` missing), the backend degrades to `default` with a warning.

### npugraph_ex Backend

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.backend_mode = "npugraph_ex"
CompilationConfig.aclgraph_enable_patterns = True  # Default False; when enabled, Pattern fusion runs before capture
CompilationConfig.npugraph_ex_options = None       # Optional: options dict forwarded to npugraph_ex.compile_fx
                                                   # Defaults reuse_graph_pool_in_same_fx to False and
                                                   # clone_output to True when unset (see the two sections below)
```

The `npugraph_ex` entry point performs AOT decomposition and graph execution itself; MindIE SD only applies Pattern fusion before capture, so the `make_fx` pre-decomposition of `apply_lightweight_decomp` is not needed.

#### Multi-graph workloads: disable cross-graph memory-pool reuse

torch_npu enables `reuse_graph_pool_in_same_fx` by default, which lets multiple ACLGraphs in the same FX share one memory pool. When more than one graph appears in a process -- for example two prompts with different text lengths in a single task, or a graph re-captured because the input shape changed -- a later graph reuses the buffers of an earlier one. Tensors that cross a graph break (typically shape tensors produced before a `Tensor.tolist()` break and consumed in eager code, as well as the model's returned tensors) then have their shape or data overwritten with pointer-magnitude garbage.

MindIE SD **defaults this option to `False` when `npugraph_ex_options` does not set it**, and logs a hint the first time it takes effect. If you set it to `True` explicitly, your value is honored and a one-time warning is logged with the error signatures above and the workaround below.

Typical symptoms (intermittent; the same configuration may pass on one run and fail on the next, so a single run proves nothing):

- `RuntimeError: Dynamo failed to run FX node with fake tensors: call_method view(...)` with symbol values around `1e12`;
- `RuntimeError: The size of tensor a (5736774568800) must match the size of tensor b (0) ...`;
- `RuntimeError: shape '[5833846557024, 34, 25, ...]' is invalid for input of size ...`.

Because the default already disables cross-graph reuse, no extra configuration is normally needed; enable sharing explicitly only when you really want it:

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.backend_mode = "npugraph_ex"
# Default is already False (one pool per graph); setting True logs a risk warning
CompilationConfig.npugraph_ex_options = {"reuse_graph_pool_in_same_fx": True}
```

The cost is one memory pool per graph, so peak memory grows with the number of graphs; evaluate it when memory is tight.

Note that `clone_output=True` is not a substitute: it solves the output-buffer reuse problem in the next section, not output metadata being overwritten across graphs.

#### Multiple calls to the same graph within one step: enable output cloning

torch_npu defaults `clone_output` to `False`, so a replay returns the static output buffer directly. When the same graph is called twice within one step (classifier-free guidance cond/uncond is the most common case; dual-stream and teacher forcing behave the same way), the second replay overwrites the tensor returned by the first, and the caller receives two identical results:

```python
noise_pred_cond   = model(x, context=text)   # writes static output buffer B
noise_pred_uncond = model(x, context=null)   # same graph, writes B again
noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
#          = noise_pred_uncond            # cond was overwritten, guidance term is always 0
```

The guidance term vanishes, which is equivalent to `guide_scale=1`. **No error is raised**, but the generated result loses most of its motion and text response: the background is nearly static and the subject does not move, while the subject still matches the reference image (the image conditioning is still present, so it looks functionally fine).

MindIE SD **defaults this option to `True` when `npugraph_ex_options` does not set it**, and logs a hint the first time it takes effect; an explicit `False` logs a one-time warning. The cost is one extra output copy per replay and one extra output-sized buffer. The `aclgraph_only` / `aclgraph_with_compile` paths are unaffected because `safe_output_mode` (default `True`) clones outputs after each replay.

This is independent of `reuse_graph_pool_in_same_fx` in the previous section: pool reuse causes intermittent crashes and garbage shapes, output-buffer reuse causes a stable but silent numerical error. Neither substitutes for the other.

## ACLGraph Acceleration

On top of Pattern Fusion, you can further enable ACLGraph to capture the optimized graph as a static execution plan.

### Configuration

`aclgraph_only` and `aclgraph_with_compile` are mutually exclusive; when both are enabled, `aclgraph_with_compile` takes higher priority.

| Option | Default | Description |
|--------|--------|------|
| `aclgraph_only` | `False` | ACLGraph only, skip Pattern Fusion |
| `aclgraph_with_compile` | `False` | Pattern Fusion first, then capture as ACLGraph |
| `aclgraph_breakpoints` | `None` | Node names used as segmentation breakpoints; empty means a single-segment whole-graph capture (see [Breakable Graph Segmentation](#breakable-graph-segmentation)) |
| `aclgraph_enable_patterns` | `False` | Whether to apply lightweight operator decomposition and Pattern fusion before capture in `aclgraph_only` / `npugraph_ex` |
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

### Breakable Graph Segmentation

When whole-graph capture fails because of individual operators or control flow, `aclgraph_breakpoints` can name breakpoint nodes to split the graph into several sequential subgraphs captured one segment at a time. Segments are chained through static buffers: each segment writes its outputs in place, and the next segment reads them directly, so no extra copy is required.

```python
from mindiesd.compilation import CompilationConfig

CompilationConfig.backend_mode = "aclgraph_only"
CompilationConfig.aclgraph_breakpoints = ["transformer_blocks_0", "transformer_blocks_12"]
```

- A breakpoint is an **FX node name** (`fx.Node.name`), not a module name. You can read them from the graph recorded via `graph_log_url` or from `logger.debug(graph.print_readable())`. Each matched node starts a new segment.
- If the list is empty or nothing matches, no split happens and capture degrades to a single whole-graph segment.
- The return value is assembled in the original `output` structure and order; graph outputs produced by non-final segments are returned too, never dropped.
- Segmentation currently applies to `aclgraph_only` only; `aclgraph_with_compile` logs a warning and falls back to whole-graph capture.

**Limitation**: segmented capture is fixed to the input shapes seen at first capture. A shape/dtype change at runtime raises an explicit `RuntimeError` (it never silently computes the wrong result), so keep inputs identical to capture time (pad to a fixed length if necessary). Scalar inputs (`int`/`bool`/`float`/`str`) baked into kernel arguments at capture time are checked by value on replay and raise the same way when the value changes.

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

### Environment Variables

These environment variables control compatibility behavior and numeric semantics. **The first two are read when `mindiesd.compilation` is imported**, so they must be set before the import.

| Variable | Default | Description |
|----------|---------|-------------|
| `MINDIE_SD_ATTENTION_DYNAMO_DISABLE` | `1` | Disables Dynamo tracing for `attention_forward`. Tracing is disabled to work around an autocast-related trace crash in older environments that breaks whole-graph compilation. Set to `0` to bring attention back into the graph, for re-validating the crash on NPU and for A/B comparison |
| `MINDIE_SD_DYNAMO_COMPAT_PATCHES` | `1` | Process-wide Dynamo compatibility patches: triton/TMA device probes on NPU, and the `torch_npu` wrapper around `os.getenv` (which causes graph breaks during trace). Set to `0` to disable them entirely, either to coexist with the official torch_npu patches or to rule out their effect |
| `MINDIE_SD_ADALN_DTYPE_ALIGN` | `1` | Aligns AdaLayerNorm modulation tensors to `x.dtype` before fusion. The fused operator requires identical input dtypes and cannot rely on eager type promotion. Set to `0` to opt out, in which case the caller must ensure matching dtypes |

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
