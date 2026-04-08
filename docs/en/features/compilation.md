# Compilation Features

## Overview

Based on multimodal model structure patterns and Ascend optimization practice, MindIE SD uses the PyTorch `torch.compile` compiler and its pattern-matcher capability to provide a custom `MindieSDBackend()`. This backend supports automatic enablement of fused operators on Ascend chips. The built-in fusion patterns can be controlled through the `CompilationConfig` switch set.

> [!NOTE]
> After this feature is enabled, there is some compilation overhead during the initial model run. By default, up to eight attempts are made. In later steady-state runs the model is usually not compiled again, so benchmark measurements should exclude the warm-up stage.

## Usage

Enable compilation for the whole transformer module in the entry script:

```python
pipe = FluxPipeline.from_pretrained(...)
transformer = torch.compile(pipe.transformer, backend=MindieSDBackend())
setattr(pipe, "transformer", transformer)
```

You can also apply it to an individual module:

```python
@torch.compile(backend=MindieSDBackend())
class FluxSingleTransformerBlock(nn.Module):
```

Or apply it to the `forward` method directly:

```python
class FluxSingleTransformerBlock(nn.Module):
    @torch.compile(backend=MindieSDBackend())
    def forward(...):
```

## Support matrix

**Operator fusion coverage**

| Model | RMSNorm | Rope | fastGelu | adaLN | FA |
|:----------:|:------:|:---:|:---:|:-----:|:--:|
| flux.1-dev | ✅ | ✅ | ✅ | ✅ | ❌ |

## Troubleshooting tips

- Troubleshooting follows the same workflow as standard PyTorch `compile`. The logging helpers defined in [mindie_sd_backend.py](../../../mindiesd/compilation/mindie_sd_backend.py) can be enabled to observe the graph before and after pattern enablement. Together with narrowing the `torch.compile` scope, this helps identify why a pattern fails to match.
- You can narrow the troubleshooting scope effectively by controlling how much of the model is compiled.
- Additional troubleshooting guidance can be found in the [PyTorch `torch.compile` documentation](https://docs.pytorch.org/docs/main/generated/torch.compile.html).
