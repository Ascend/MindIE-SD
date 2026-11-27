/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <c10/core/DeviceGuard.h>
#include "pytorch_npu_helper.h"
#include "eagle_hadamard_quant.h"
#include "../ops/eagle_hadamard_quant/op_host/eagle_hadamard_quant_params.h"

namespace {
constexpr std::string_view EAGLE_HADAMARD_QUANT_NAME = "aclnnEagleHadamardQuant";
// Resolve the optional enum at compile time while keeping the schema available on older Torch.
template <typename Scalar> auto EagleScaleType(int) -> decltype(Scalar::Float8_e8m0fnu) {
    return Scalar::Float8_e8m0fnu;
}

template <typename Scalar> Scalar EagleScaleType(...) {
    TORCH_CHECK(false, "eagle_hadamard_quant requires a PyTorch build with native torch.float8_e8m0fnu support");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> eagle_hadamard_quant_impl_npu(
    const at::Tensor &x, const at::Tensor &sign_h, int64_t block_size, const std::string &dst_type, bool is_sign_h_nz) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be an NPU tensor");
    TORCH_CHECK(x.device() == sign_h.device(), "x and sign_h must be on the same device");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && sign_h.scalar_type() == at::kBFloat16, "x and sign_h must be BF16");
    TORCH_CHECK(x.dim() == 2, "x must be [M,N]");
    const int64_t m = x.size(0), n = x.size(1);
    const auto p = EagleHadamardQuant::CheckParams(
        m, n, sign_h.sizes().data(), sign_h.dim(), block_size, dst_type.c_str(), is_sign_h_nz);
    TORCH_CHECK(p.error == nullptr, p.error);
    const auto yType = p.format == 0 ? at::kByte : at::kFloat8_e4m3fn;
    auto y = at::empty({m, p.outputColumns}, x.options().dtype(yType));
    auto scale = at::empty({m, p.scalePairs, 2}, x.options().dtype(EagleScaleType<at::ScalarType>(0)));
    if (m == 0) {
        return {y, scale};
    }
    const c10::DeviceGuard guard(x.device());
    const char *soc = aclrtGetSocName();
    TORCH_CHECK(
        soc != nullptr && std::string_view(soc).find("Ascend950") == 0, "eagle_hadamard_quant requires DAV-3510");
    auto xNd = at_npu::native::npu_format_cast(x, ACL_FORMAT_ND).contiguous();
    auto signHNd = at_npu::native::npu_format_cast(sign_h, ACL_FORMAT_ND).contiguous();
    // The shared helper's scalar table does not include Torch's E8M0 enum yet.
    const TensorWrapper scaleWrapper{scale, ACL_DTYPE_FLOAT8_E8M0};
    const char *dstType = dst_type.c_str();
    EXEC_NPU_CMD<EAGLE_HADAMARD_QUANT_NAME>(xNd, signHNd, block_size, dstType, is_sign_h_nz, y, scaleWrapper);
    return {y, scale};
}

std::tuple<at::Tensor, at::Tensor> eagle_hadamard_quant_impl_meta(
    const at::Tensor &x, const at::Tensor &sign_h, int64_t block_size, const std::string &dst_type, bool is_sign_h_nz) {
    TORCH_CHECK(x.device() == sign_h.device(), "x and sign_h must be on the same device");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && sign_h.scalar_type() == at::kBFloat16, "x and sign_h must be BF16");
    TORCH_CHECK(x.dim() == 2 && sign_h.dim() <= 5, "x must be [M,N] and sign_h must have at most 5 dimensions");
    std::array<int64_t, 5> dims{};
    for (int64_t i = 0; i < sign_h.dim(); ++i) {
        dims[i] = sign_h.sym_size(i).maybe_as_int().value_or(-1);
    }
    // Preserve symbolic batch; execution-time uint32 limits stay in the NPU/tiling validator.
    const auto n = x.sym_size(1);
    const auto p = EagleHadamardQuant::CheckParams(x.sym_size(0).maybe_as_int().value_or(-1),
        n.maybe_as_int().value_or(-1), dims.data(), sign_h.dim(), block_size, dst_type.c_str(), is_sign_h_nz, true);
    TORCH_CHECK(p.error == nullptr, p.error);
    TORCH_SYM_CHECK(n.sym_gt(0), "hd must be positive");
    TORCH_SYM_CHECK((n % block_size).sym_eq(0), "hd must be a multiple of block_size");
    if (is_sign_h_nz) {
        for (int64_t i = 1; i < 5; ++i) {
            TORCH_SYM_CHECK(sign_h.sym_size(i).sym_eq(i < 3 ? block_size / 16 : 16),
                "NZ sign_h must have shape [B, block_size/16, block_size/16, 16, 16]");
        }
    } else {
        TORCH_SYM_CHECK(sign_h.sym_size(-2).sym_eq(block_size), "block_size must match sign_h");
        TORCH_SYM_CHECK(sign_h.sym_size(-1).sym_eq(block_size), "block_size must match sign_h");
    }
    if (sign_h.dim() != 2) {
        const auto blocks = sign_h.sym_size(0);
        TORCH_SYM_CHECK(blocks.sym_gt(0), "sign_h must contain at least one rotation");
        const auto valid = blocks.sym_eq(n / block_size);
        TORCH_SYM_CHECK(is_sign_h_nz ? valid | blocks.sym_eq(1) : valid,
            "sign_h must contain hd/block_size rotations or one shared rotation (2D ND or NZ)");
    }
    const auto yType = p.format == 0 ? at::kByte : at::kFloat8_e4m3fn;
    auto y = at::empty_symint({x.sym_size(0), p.format == 0 ? n / 2 : n}, x.options().dtype(yType));
    auto scale =
        at::empty_symint({x.sym_size(0), (n + 63) / 64, 2}, x.options().dtype(EagleScaleType<at::ScalarType>(0)));
    return {y, scale};
}
