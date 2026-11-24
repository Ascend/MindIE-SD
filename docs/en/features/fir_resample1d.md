# FIR Resample1D

MindIE SD provides a model-independent, restricted 1D FIR resampling API. It
includes a PyTorch reference path and uses an optimized kernel on Ascend NPUs
with Triton-Ascend support.

```python
from mindiesd import fir_resample1d

output = fir_resample1d(x, fir_filter, up=2, down=1, padding="replicate")
```

Version 1 intentionally has a strict support envelope:

- `x`: contiguous FP32 `[B, C, T]`;
- `fir_filter`: contiguous FP32 `[12]`, shared by all batches and channels;
- only `up=2` and `down=1`;
- only `padding="replicate"`;
- output shape `[B, C, 2*T]`, with `2*T` no greater than `2^31 - 12`;
- filter coefficients are not normalized or rescaled;
- inference only.

Let `u` be a virtual high-rate signal with input samples at even positions and
zeros at odd positions. At output position `n`, the API computes:

```text
output[n] = sum(fir_filter[k] * u[n + k - 5]), k = 0..11
```

Even positions outside the input range use the nearest boundary sample. Odd
positions remain zero. This definition fixes output length, phase, tap order,
and boundary behavior.

Use `is_fir_resample1d_supported` to determine whether the optimized NPU
backend supports given tensors. When it returns `False`, the public API can
still execute supported configurations through its PyTorch reference path.
