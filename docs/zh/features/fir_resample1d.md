# FIR Resample1D

MindIE SD 提供模型无关的受限 FIR 一维重采样接口。该接口提供 PyTorch 参考路径，
并在支持 Triton-Ascend 的昇腾 NPU 上使用优化内核。

```python
from mindiesd import fir_resample1d

output = fir_resample1d(x, fir_filter, up=2, down=1, padding="replicate")
```

首版接口有意采用严格支持范围：

- `x`：连续 FP32 `[B, C, T]`；
- `fir_filter`：所有批次和通道共享的连续 FP32 `[12]`；
- 仅支持 `up=2`、`down=1`；
- 仅支持 `padding="replicate"`；
- 输出形状为 `[B, C, 2*T]`，且 `2*T` 不超过 `2^31 - 12`；
- 不自动归一化或缩放滤波器系数；
- 仅用于推理。

令偶数位置保存输入样本、奇数位置为零的虚拟高采样率信号为 `u`。对于输出位置
`n`，接口计算：

```text
output[n] = sum(fir_filter[k] * u[n + k - 5]), k = 0..11
```

越过输入首尾的偶数位置使用边界样本，奇数位置仍为零。该定义明确固定了输出长度、
相位、滤波器顺序和边界行为。

`is_fir_resample1d_supported` 用于判断给定张量是否能够使用优化 NPU 后端。返回
`False` 时，公共接口仍可使用 PyTorch 参考路径执行受支持的配置。
