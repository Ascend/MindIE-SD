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

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "eagle_ffn_linear.h"

using namespace at;

constexpr std::string_view EAGLE_FFN_OP_NAME = "aclnnEagleFfnV2";

namespace {

enum class FfnLayout { LINEAR, CANONICAL, INVALID };

FfnLayout ffn_detect_layout(int64_t w1d0, int64_t w1d1, int64_t w2d0, int64_t w2d1, int64_t x_k, bool swiglu)
{
    const int64_t hidden_w = swiglu ? w1d0 / 2 : w1d0;
    const int64_t hidden_w2 = swiglu ? w1d1 / 2 : w1d1;
    if (w1d1 == x_k && w1d0 != x_k) {
        return FfnLayout::LINEAR;
    }
    if (w1d0 == x_k && w1d1 != x_k) {
        return FfnLayout::CANONICAL;
    }
    if (w1d0 == x_k && w1d1 == x_k) {
        return (w2d0 == hidden_w2 && w2d1 != hidden_w) ? FfnLayout::CANONICAL : FfnLayout::LINEAR;
    }
    return FfnLayout::INVALID;
}

std::string to_lower(std::string_view s)
{
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

} // namespace

at::Tensor eagle_ffn_linear_mindie_sd_impl_npu(const at::Tensor &x, const at::Tensor &weight1, const at::Tensor &weight2,
                                         const c10::optional<at::Tensor> &bias1, const c10::optional<at::Tensor> &bias2,
                                         c10::string_view activation, int64_t inner_precise)
{
    TORCH_CHECK(x.dim() >= 2, "eagle_ffn_linear: x must have at least 2 dimensions, got ", x.dim(), "D");
    TORCH_CHECK(weight1.dim() == 2 && weight2.dim() == 2, "eagle_ffn_linear: weight1/weight2 must be 2D");

    const std::string act = to_lower(std::string_view(activation.data(), activation.size()));
    TORCH_CHECK(act == "gelu" || act == "silu" || act == "swiglu",
                "eagle_ffn_linear: activation must be gelu/silu/swiglu, got ", act);

    at::Tensor x_c = x.contiguous();
    at::Tensor w1_c = weight1.contiguous();
    at::Tensor w2_c = weight2.contiguous();

    const int64_t x_k = x_c.size(x_c.dim() - 1);
    const int64_t w1d0 = w1_c.size(0);
    const int64_t w1d1 = w1_c.size(1);
    const int64_t w2d0 = w2_c.size(0);
    const int64_t w2d1 = w2_c.size(1);
    const bool swiglu = (act == "swiglu");
    const FfnLayout layout = ffn_detect_layout(w1d0, w1d1, w2d0, w2d1, x_k, swiglu);
    TORCH_CHECK(layout != FfnLayout::INVALID,
                "eagle_ffn_linear: weight1 shape [", w1d0, ", ", w1d1, "] does not match x K=", x_k,
                " (expect [K,N] canonical or [N,K] linear)");
    const bool is_linear = (layout == FfnLayout::LINEAR);
    TORCH_CHECK(!swiglu || is_linear, "eagle_ffn_linear: swiglu only supports linear layout weight1 [2H,K]");

    c10::optional<at::Tensor> b1;
    c10::optional<at::Tensor> b2;
    b1 = (bias1.has_value() && bias1.value().defined()) ? c10::optional<at::Tensor>(bias1.value().contiguous())
                                                        : c10::nullopt;
    b2 = (bias2.has_value() && bias2.value().defined()) ? c10::optional<at::Tensor>(bias2.value().contiguous())
                                                        : c10::nullopt;
    TORCH_CHECK(b1.has_value() == b2.has_value(), "eagle_ffn_linear: bias1/bias2 must be both present or both absent");

    // 输出尺寸：保留 x 前导维，末维 = N（linear 取 w2.d0，canonical 取 w2.d1）
    auto out_size = x_c.sizes().vec();
    out_size[x_c.dim() - 1] = is_linear ? w2d0 : w2d1;
    at::Tensor out = at::empty(out_size, x_c.options());

    // EXEC_NPU_CMD 要求实参为左值：空可选参数与字符串统一绑定具名局部量
    const bool tokens_index_flag = false;
    const c10::optional<at::Tensor> no_tensor;
    const char *activation_cstr = act.c_str();
    EXEC_NPU_CMD<EAGLE_FFN_OP_NAME>(x_c, w1_c, w2_c,
                              no_tensor,  // expert_tokens（单专家）
                              b1, b2,
                              no_tensor, no_tensor, no_tensor, no_tensor,
                              no_tensor, no_tensor, no_tensor, no_tensor,
                              activation_cstr, inner_precise, tokens_index_flag, out);
    return out;
}
