#include <algorithm>
#include <string_view>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "eagle_quant_batch_matmul_swiglu.h"

namespace {
constexpr std::string_view OP_NAME = "aclnnEagleQuantBatchMatmulSwiglu";
constexpr std::string_view NZ_OP_NAME = "aclnnEagleQuantBatchMatmulSwigluWeightNz";
struct QbmmWeightTensor {
    const at::Tensor &tensor;
};

// ADL selects this converter inside EXEC_NPU_CMD; its AclTensor is released by the bridge.
AclTensor *ConvertType(const QbmmWeightTensor &weight) {
    const auto &tensor = weight.tensor;
    if (at_npu::native::get_npu_format(tensor) != ACL_FORMAT_FRACTAL_NZ) {
        return ::ConvertType(tensor);
    }
    const auto storageSizes = at_npu::native::get_npu_storage_sizes(tensor);
    auto *result = aclCreateTensor(tensor.sizes().data(), tensor.sizes().size(),
        ConvertToAclDataType(tensor.scalar_type()), tensor.strides().data(), tensor.storage_offset(),
        ACL_FORMAT_FRACTAL_NZ, storageSizes.data(), storageSizes.size(), const_cast<void *>(tensor.storage().data()));
    TORCH_CHECK(result != nullptr, "failed to create NZ weight descriptor");
    return result;
}

void CheckTensor(const at::Tensor &tensor, const at::Tensor &x1, const char *name, bool allowNz = false) {
    TORCH_CHECK(tensor.device() == x1.device(), name, " must be on the same NPU as x1");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    const auto format = at_npu::native::get_npu_format(tensor);
    TORCH_CHECK(format == ACL_FORMAT_ND || (allowNz && format == ACL_FORMAT_FRACTAL_NZ), name,
        allowNz ? " must use ND or FRACTAL_NZ storage" : " must use ND storage");
}
}

at::Tensor eagle_quant_batch_matmul_swiglu_impl_npu(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &scale,
    const c10::optional<at::Tensor> &offset, const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &pertoken_scale, c10::optional<at::ScalarType> requested_dtype, bool transpose_x1,
    bool transpose_x2, int64_t group_size) {
    const auto output_dtype = requested_dtype.value_or(at::kBFloat16);
    TORCH_CHECK(
        x1.dim() >= 2 && x1.dim() <= 6 && x2.dim() >= 2 && x2.dim() <= 6, "x1 and x2 must have 2 to 6 dimensions");
    TORCH_CHECK(
        x1.scalar_type() == at::kChar && x2.scalar_type() == at::kChar, "the 910B plugin accepts INT8 x1 and x2");
    TORCH_CHECK(output_dtype == at::kBFloat16 || output_dtype == at::kHalf, "output_dtype must be BF16 or FP16");
    TORCH_CHECK(!bias.has_value(), "bias is not supported; pass None");
    TORCH_CHECK(pertoken_scale.has_value(), "pertoken_scale is required");
    TORCH_CHECK(scale.scalar_type() == at::kFloat && pertoken_scale->scalar_type() == at::kFloat,
        "channel and token scales must be FP32");
    TORCH_CHECK(group_size == 0, "910B requires group_size=0");
    TORCH_CHECK(!offset.has_value(), "offset must be None: this bridge does not perform ACLNN scale packing");
    CheckTensor(x1, x1, "x1");
    CheckTensor(x2, x1, "x2", true);
    CheckTensor(scale, x1, "scale");
    if (bias.has_value()) {
        CheckTensor(*bias, x1, "bias");
    }
    if (pertoken_scale.has_value()) {
        CheckTensor(*pertoken_scale, x1, "pertoken_scale");
    }
    const auto m = x1.size(transpose_x1 ? -1 : -2);
    const auto k = x1.size(transpose_x1 ? -2 : -1);
    const auto n = x2.size(transpose_x2 ? -2 : -1);
    TORCH_CHECK(m > 0 && k > 0, "M and K must be positive");
    TORCH_CHECK(n > 0 && n % 2 == 0, "SwiGLU requires positive even N");
    if (at_npu::native::get_npu_format(x2) == ACL_FORMAT_FRACTAL_NZ) {
        TORCH_CHECK((n / 2) % (transpose_x2 ? 16 : 32) == 0,
            "NZ weights require N/2 aligned to 16 when transposed, otherwise 32");
    }
    TORCH_CHECK(scale.dim() == 1 && scale.numel() == n, "scale must have shape [N]");
    TORCH_CHECK(pertoken_scale->dim() == 1 && pertoken_scale->numel() == m, "pertoken_scale must have shape [M]");
    TORCH_CHECK(k == x2.size(transpose_x2 ? -1 : -2), "x1 and x2 K dimensions differ");
    const int64_t rank = std::max(x1.dim(), x2.dim());
    std::vector<int64_t> shape(rank, 1);
    for (int64_t i = 0; i < rank - 2; ++i) {
        const auto ai = i - (rank - x1.dim());
        const auto bi = i - (rank - x2.dim());
        const auto a = ai < 0 ? 1 : x1.size(ai);
        const auto b = bi < 0 ? 1 : x2.size(bi);
        TORCH_CHECK(a == b || a == 1 || b == 1, "batch dimensions cannot broadcast");
        shape[i] = a == 1 ? b : a;
    }
    shape[rank - 2] = m;
    int64_t batch = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        batch *= shape[i];
    }
    auto batchA = x1.numel() / (m * k);
    auto batchB = x2.numel() / (k * n);
    TORCH_CHECK((batchA == 1 || batchA == batch) && (batchB == 1 || batchB == batch),
        "only equal batches or whole-matrix broadcasting are supported");
    shape[rank - 1] = n / 2;
    at::Tensor out = at_npu::native::empty_with_format(shape, x1.options().dtype(output_dtype), ACL_FORMAT_ND);
    int64_t dtype = static_cast<int64_t>(ConvertToAclDataType(output_dtype));
    // Generated ACLNN follows the OpDef input order, then attributes, then output.
    const QbmmWeightTensor weight{x2};
    if (at_npu::native::get_npu_format(x2) == ACL_FORMAT_FRACTAL_NZ) {
        EXEC_NPU_CMD<NZ_OP_NAME>(
            x1, weight, scale, offset, bias, pertoken_scale, dtype, transpose_x1, transpose_x2, group_size, out);
    } else {
        EXEC_NPU_CMD<OP_NAME>(
            x1, weight, scale, offset, bias, pertoken_scale, dtype, transpose_x1, transpose_x2, group_size, out);
    }
    return out;
}
