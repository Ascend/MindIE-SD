#pragma once
#include <ATen/ATen.h>

at::Tensor eagle_quant_batch_matmul_swiglu_impl_npu(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &scale,
    const c10::optional<at::Tensor> &offset, const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &pertoken_scale, c10::optional<at::ScalarType> output_dtype, bool transpose_x1,
    bool transpose_x2, int64_t group_size);
