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

#include <cmath>
#include <limits>
#include <string_view>
#include <vector>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "eagle_split_norm_rope_concat.h"

namespace {
constexpr std::string_view EAGLE_SPLIT_NORM_ROPE_CONCAT_NAME = "aclnnEagleSplitNormRopeConcat";

void CheckFloatingInput(const at::Tensor &tensor, const at::Tensor &qkv, const char *name) {
    TORCH_CHECK(tensor.device() == qkv.device(), name, " must be on the same NPU as qkv");
    TORCH_CHECK(tensor.scalar_type() == qkv.scalar_type(), name, " must have the same dtype as qkv");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() > 0 && tensor.numel() <= std::numeric_limits<int32_t>::max(), name,
        " numel must be in [1, INT32_MAX]");
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> eagle_split_norm_rope_concat_impl_npu(const at::Tensor &qkv,
    const c10::optional<at::Tensor> &norm_query_weight, const c10::optional<at::Tensor> &norm_key_weight,
    const at::Tensor &rope_cos, const at::Tensor &rope_sin, at::IntArrayRef head_counts, int64_t norm_type,
    int64_t rope_type, c10::optional<int64_t> rotary_dim, double eps, bool output_bnsd) {
    TORCH_CHECK(qkv.device().type() == c10::DeviceType::PrivateUse1, "qkv must be an NPU tensor");
    TORCH_CHECK(qkv.dim() == 3 || qkv.dim() == 4, "qkv must be [S,Htotal,D] or [B,S,Htotal,D]");
    TORCH_CHECK(
        qkv.scalar_type() == at::kHalf || qkv.scalar_type() == at::kBFloat16, "qkv must be float16 or bfloat16");
    CheckFloatingInput(qkv, qkv, "qkv");
    const int64_t batch = qkv.dim() == 4 ? qkv.size(0) : 1;
    const int64_t sequence = qkv.size(-3);
    const int64_t totalHeads = qkv.size(-2);
    const int64_t dim = qkv.size(-1);
    const int64_t rotaryDim = rotary_dim.value_or(dim);
    TORCH_CHECK(dim == 128, "head_dim must be 128; other dimensions are not supported");
    TORCH_CHECK(rotaryDim >= 32 && rotaryDim <= dim && rotaryDim % 32 == 0,
        "rotary_dim must be a multiple of 32 in [32,head_dim]");
    TORCH_CHECK(norm_type == 2 || norm_type == 3,
        "norm_type must be 2 (RMS_NORM) or 3 (RMS_NORM_AFFINE); other modes are not supported");
    TORCH_CHECK(rope_type == 0 || rope_type == 1, "rope_type must be 0 (interleave) or 1 (half)");
    const float floatEps = static_cast<float>(eps);
    TORCH_CHECK(std::isfinite(eps) && std::isfinite(floatEps) && floatEps > 0,
        "eps must be finite and positive after conversion to float32");
    TORCH_CHECK(head_counts.size() == 3, "head_counts must contain exactly three integers [Hq,Hk,Hv]");
    for (const auto head : head_counts) {
        TORCH_CHECK(head > 0 && head <= totalHeads, "each head count must be in [1,Htotal]");
    }
    TORCH_CHECK(head_counts[0] + head_counts[1] + head_counts[2] == totalHeads, "head counts must sum to Htotal");
    TORCH_CHECK(head_counts[1] == head_counts[2] && head_counts[0] % head_counts[1] == 0,
        "require Hk=Hv and Hq divisible by Hk");
    const auto checkWeight = [&](const c10::optional<at::Tensor> &weight, const char *name) {
        const bool present = weight.has_value() && weight->defined();
        TORCH_CHECK(norm_type != 3 || present, name, " is required for norm_type=3");
        if (present) {
            CheckFloatingInput(*weight, qkv, name);
            TORCH_CHECK(weight->dim() == 1 && weight->size(0) == dim, name, " must have shape [D]");
        }
    };
    checkWeight(norm_query_weight, "norm_query_weight");
    checkWeight(norm_key_weight, "norm_key_weight");
    CheckFloatingInput(rope_cos, qkv, "rope_cos");
    CheckFloatingInput(rope_sin, qkv, "rope_sin");
    TORCH_CHECK(rope_cos.sizes() == rope_sin.sizes(), "rope_cos and rope_sin must have identical shapes");
    TORCH_CHECK(rope_cos.dim() >= 2 && rope_cos.dim() <= 4, "RoPE cache must have shape [S,W], [Bc,S,W] or [Bc,S,1,W]");
    TORCH_CHECK(rope_cos.size(-1) == rotaryDim || rope_cos.size(-1) == rotaryDim / 2,
        "RoPE cache last dimension must equal rotary_dim or rotary_dim/2 (head_dim if omitted)");
    const int64_t cacheBatch = rope_cos.dim() == 2 ? 1 : rope_cos.size(0);
    const int64_t cacheSequence = rope_cos.dim() == 4 ? rope_cos.size(1) : rope_cos.size(-2);
    TORCH_CHECK(cacheBatch == 1 || cacheBatch == batch, "RoPE cache batch must be 1 or B");
    TORCH_CHECK(cacheSequence == sequence, "RoPE cache sequence dimension must equal S");
    TORCH_CHECK(rope_cos.dim() != 4 || rope_cos.size(2) == 1, "RoPE cache head dimension must be 1");

    const c10::DeviceGuard deviceGuard(qkv.device());
    const auto allocate = [&](int64_t headCount) {
        std::vector<int64_t> shape;
        if (output_bnsd) {
            shape = {batch, headCount, sequence, dim};
        } else if (qkv.dim() == 4) {
            shape = {batch, sequence, headCount, dim};
        } else {
            shape = {sequence, headCount, dim};
        }
        return at_npu::native::empty_with_format(shape, qkv.options(), ACL_FORMAT_ND);
    };
    auto query = allocate(head_counts[0]);
    auto key = allocate(head_counts[1]);
    auto value = allocate(head_counts[2]);
    // ValueDepend(REQUIRED) generates an aclIntArray argument for this metadata input.
    EXEC_NPU_CMD<EAGLE_SPLIT_NORM_ROPE_CONCAT_NAME>(qkv, head_counts, norm_query_weight, norm_key_weight, rope_cos,
        rope_sin, norm_type, rope_type, rotaryDim, eps, output_bnsd, query, key, value);
    return std::make_tuple(query, key, value);
}
