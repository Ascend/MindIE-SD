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
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <string_view>
#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "quant_four_over_six_a5.h"

using namespace at;

constexpr std::string_view QUANT_FOUR_OVER_SIX_A5_OP_NAME = "aclnnQuantFourOverSixA5V2";

// ge::DataType 枚举值：40 = DT_FLOAT4_E2M1
// T9b: 仅保留 46 自适应路径（dst_type_max=4.0, bf16 -> fp4x2_e2m1, scale_alg=2, blocksize=32,
// 尾轴）。dst_type 41/36/35 等非 46 组合已随清理删除（def 亦仅注册 bf16 x FLOAT4_E2M1 组合），
// 此处强制 dst_type==40 即天然排除，无需逐项拒绝。
constexpr int64_t DST_TYPE_FLOAT4_E2M1 = 40;
constexpr int64_t SCALE_ALG_ADAPTIVE_46 = 2;
constexpr double DST_TYPE_MAX_ADAPTIVE_46 = 4.0;

// T9c/B2: 46 路径（tilingKey=12）kernel 支持 rint/round/floor 三种 round_mode（tiling_arch35.cpp
// GetRoundMode 白名单一致）；其余取值拒绝。
constexpr const char *ROUND_MODE_WHITELIST[] = {"rint", "round", "floor"};

// y 的 torch 层 dtype：float4（dst_type=40, FLOAT4_E2M1）在 torch 层以 uint8 呈现
// （4bit 打包），经 MakeTensorWrapper 指定 CANN_DTYPE_FLOAT4_E2M1 使 aclTensor 语义为 float4。
static void CheckQuantFourOverSixA5Attrs(int64_t dstType, int64_t blocksize, int64_t scaleAlg,
    double dstTypeMax, const std::string &roundMode)
{
    // 46 自适应组合校验（与 host tiling 收紧一致）
    TORCH_CHECK(dstType == DST_TYPE_FLOAT4_E2M1,
        "quant_four_over_six_a5: only dst_type=40 (FLOAT4_E2M1) is supported "
        "(4/6 adaptive is the only remaining path), got ", dstType);
    TORCH_CHECK(scaleAlg == SCALE_ALG_ADAPTIVE_46,
        "quant_four_over_six_a5: scale_alg must be 2 (4/6 adaptive), got ", scaleAlg);
    TORCH_CHECK(dstTypeMax == DST_TYPE_MAX_ADAPTIVE_46,
        "quant_four_over_six_a5: dst_type_max must be 4.0 (4/6 adaptive), got ", dstTypeMax);
    // T9c/B1: 46 自适应仅 blocksize=32（T9a 已证：尾轴模板仅支持 blockSize==32）
    TORCH_CHECK(blocksize == 32,
        "quant_four_over_six_a5: blocksize must be 32 (4/6 adaptive tail-axis template only "
        "supports blocksize=32), got ", blocksize);
    // T9c/B2: round_mode 白名单 {rint, round, floor}（kernel tilingKey=12 支持三种）
    bool roundModeValid = false;
    for (const char *mode : ROUND_MODE_WHITELIST) {
        if (roundMode == mode) {
            roundModeValid = true;
            break;
        }
    }
    TORCH_CHECK(roundModeValid,
        "quant_four_over_six_a5: round_mode must be one of {rint, round, floor}, got '", roundMode, "'");
}

std::tuple<at::Tensor, at::Tensor> quant_four_over_six_a5_mindie_sd_impl_npu(const at::Tensor &x,
    int64_t axis, std::string round_mode, int64_t dst_type, int64_t blocksize, int64_t scale_alg,
    double dst_type_max)
{
    TORCH_CHECK(x.dim() >= 1 && x.dim() <= 7, "quant_four_over_six_a5: x dim must be within [1, 7], got ",
        x.dim());
    // R1: 46 自适应仅支持 bf16 输入（fp16 在 kernel 中无 46 实现，强制拦截防静默无输出）
    TORCH_CHECK(x.scalar_type() == at::kBFloat16,
        "quant_four_over_six_a5: x must be bf16 (4/6 adaptive is the only remaining path), got ",
        x.scalar_type());
    CheckQuantFourOverSixA5Attrs(dst_type, blocksize, scale_alg, dst_type_max, round_mode);

    int64_t dim = (axis >= 0) ? axis : axis + x.dim();
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "quant_four_over_six_a5: invalid axis ", axis,
        " for a ", x.dim(), "D tensor");
    // 46 自适应仅支持尾轴（kernel 仅在尾轴模板实现）
    TORCH_CHECK(dim == x.dim() - 1,
        "quant_four_over_six_a5: 4/6 adaptive (dst_type_max=4) only supports the tail axis, got axis=",
        axis);
    // 硬性条件：量化轴（尾轴）长度必须 32 对齐（kernel block 均满块处理，无 tail block）
    TORCH_CHECK(x.size(dim) % 32 == 0,
        "quant_four_over_six_a5: axis dim must be 32-aligned (46 adaptive is block-aligned only), got axis dim=",
        x.size(dim));


    // mxscale 形状（与 infershape 公式一致）：x 形状中 axis 维改为 ceil(ceil(dim/bs)/2)，并 append 2
    std::vector<int64_t> scaleShape(x.sizes().begin(), x.sizes().end());
    scaleShape[dim] = ((x.size(dim) + blocksize - 1) / blocksize + 1) / 2;
    scaleShape.push_back(2);

    if (x.numel() == 0) {
        // 空输入分支：fp4 输出打包减半（uint8）
        auto emptyYShape = x.sizes().vec();
        emptyYShape.back() /= 2;
        at::Tensor emptyY = at_npu::native::empty_with_format(
            emptyYShape, x.options().dtype(at::kByte), at_npu::native::get_npu_format(x));
        at::Tensor emptyScale = at_npu::native::empty_with_format(
            scaleShape, x.options().dtype(at::kByte), at_npu::native::get_npu_format(x));
        return std::make_tuple(emptyY, emptyScale);
    }

    // y：float4（40）用 uint8 tensor + MakeTensorWrapper 覆盖为 acl float4 语义；
    //     CollectB4ShapeInfo 会把 torch 层 uint8 shape 的最后一维 ×2 作为 acl
    //     float4 shape（4bit 每字节 2 元素打包），故 torch 层 shape = x.shape
    //     最后一维减半。
    //     mxscale（E8M0）为 1 字节/元素，torch 层 uint8 shape 即目标 shape，
    //     经 MakeTensorWrapper 覆盖为 acl E8M0。
    TORCH_CHECK(x.size(x.dim() - 1) % 2 == 0,
        "quant_four_over_six_a5: last dim of x must be even for float4 packing, got ", x.size(x.dim() - 1));
    auto yShape = x.sizes().vec();
    yShape.back() /= 2;
    at::Tensor y = at_npu::native::empty_with_format(
        yShape, x.options().dtype(at::kByte), at_npu::native::get_npu_format(x));
    at::Tensor mxscale = at_npu::native::empty_with_format(
        scaleShape, x.options().dtype(at::kByte), at_npu::native::get_npu_format(x));

    auto yWrapper = MakeTensorWrapper(y, c10::optional<int64_t>(CANN_DTYPE_FLOAT4_E2M1));
    auto mxscaleWrapper = MakeTensorWrapper(mxscale, c10::optional<int64_t>(CANN_DTYPE_FLOAT8_E8M0));

    const char *roundModePtr = round_mode.c_str();

    EXEC_NPU_CMD<QUANT_FOUR_OVER_SIX_A5_OP_NAME>(
        x, axis, roundModePtr, dst_type, blocksize, scale_alg, dst_type_max, yWrapper, mxscaleWrapper);

    return std::make_tuple(y, mxscale);
}
