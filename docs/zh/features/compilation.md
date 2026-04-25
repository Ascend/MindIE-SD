# 编译特性

基于多模态模型的结构特征和昇腾性能优化实践，MindIE SD利用PyTorch的torch.compile编译器及其Pattern Matcher能力，自定义实现了`MindieSDBackend()`，支持昇腾芯片上融合算子的自动使能。现内置的融合Pattern本身可以通过开关（`CompilationConfig`）进行控制。

>[!NOTE]说明
>当使能该特性后，模型运行初期存在一定的编译耗时（默认最多进行8次尝试），但是在后续运行中，一般不会再次编译。在实际benchmark测试过程中，需要将预热阶段的耗时去除。

## Pattern支持度

|     模型     | RMSNorm | Rope | fastGelu | adaLN |
|:----------:|:------: |:---: |:---: |:-----:|
| flux.1-dev | ✅      | ✅   | ✅️ |  ✅️   |

## 使用方法

在入口脚本中，将transformer模块整体进行compile，可以通过如下方式使能：

```python
pipe = FluxPipeline.from_pretrained(...)
transformer = torch.compile(pipe.transformer, backend=MindieSDBackend())
setattr(pipe, "transformer", transformer)
```

也可以针对单个Module针对性使用：

```python
@torch.compile(backend=MindieSDBackend())
class FluxSingleTransformerBlock(nn.Module):
```

或者对forward函数使用：

```python
class FluxSingleTransformerBlock(nn.Module):
    @torch.compile(backend=MindieSDBackend())
    def forward(...):
```

### 问题定位技巧

- 相关的定位手段与PyTorch的compile一致，[mindie_sd_backend.py](../../../mindiesd/compilation/mindie_sd_backend.py)中定义了日志模块，开启后，可以观察到pattern使能前后的图变化情况。配合torch.compile缩小范围，可以识别pattern失效的原因。
- 通过控制compile的范围，可以有效控制问题定位的范围。
- 其他定位手段可以参考[PyTorch](https://docs.pytorch.org/docs/main/generated/torch.compile.html)官网。
