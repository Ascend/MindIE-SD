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


/*!
 * \file quant_four_over_six_a5_infershape.cpp
 * \brief
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"
#include <cmath>

// OPBASE 移除（同仓惯例）：源算子依赖 Ops::Base 框架库（MindIE 未随迁），
// 此处以 gert::Shape 原生语义内联等价实现（参考 eagle_quant_block_sparse_attention）。
namespace {
constexpr int64_t UNKNOWN_DIM_VALUE_Q = -2;
inline bool IsUnknownRankShapeQ(const gert::Shape &s)
{
    return s.GetDimNum() == 1 && s.GetDim(0) == UNKNOWN_DIM_VALUE_Q;
}
inline void SetUnknownRankShapeQ(gert::Shape &s)
{
    s.SetDimNum(0);
    s.AppendDim(UNKNOWN_DIM_VALUE_Q);
}
template <typename T> inline bool IsFloatEqualQ(T a, T b)
{
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    if (std::isinf(a) || std::isinf(b)) {
        return std::signbit(a) == std::signbit(b);
    }
    return a == b;
}
template <typename T> inline T CeilDivQ(T a, T b) { return (a + b - 1) / b; }
}  // namespace

using namespace ge;
namespace ops {

// 注：Shape2String 模板（namespace ops）由 MindIE-SD csrc/ops/utils/inc/log/inner/dfx_base.h 提供
// （经 log/log.h 引入，实现与源算子本地定义一致），此处不再重复定义，避免 redefinition。

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr size_t INDEX_ATTR_AXIS = 0;
constexpr size_t INDEX_ATTR_ROUND_MODE = 1;
constexpr size_t INDEX_ATTR_DST_TYPE = 2;
constexpr size_t INDEX_ATTR_BLOCK_SIZE = 3;
constexpr size_t INDEX_ATTR_SCALE_ALG = 4;
constexpr size_t INDEX_ATTR_DST_TYPE_MAX = 5;
constexpr int64_t ALIGN_NUM = 2;
constexpr size_t MAX_DIM_NUM = 7;
// T9c: 46 组合守卫常量（与 plugin/tiling 侧一致）
constexpr int32_t BLOCK_SIZE_46 = 32;
constexpr int32_t SCALE_ALG_46 = 2;
constexpr float DST_TYPE_MAX_46 = 4.0f;
// T9b: 仅保留 46 自适应路径——y 仅 FLOAT4_E2M1（41/36/35 已随清理删除）
static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {
    ge::DT_FLOAT4_E2M1};

graphStatus InferShapeForQuantFourOverSixA5(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForQuantFourOverSixA5");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* scaleShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    OP_CHECK_IF(
        xShape->GetDimNum() < 1 || xShape->GetDimNum() > MAX_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
            std::to_string(xShape->GetDimNum()), "The shape dim of x must be within the range [1, 7]"),
        return ge::GRAPH_FAILED);

    if (IsUnknownRankShapeQ(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, scale shape to (-2, )");
        SetUnknownRankShapeQ(*yShape);
        SetUnknownRankShapeQ(*scaleShape);
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *xShape;

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int32_t* axis = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    const int32_t* blockSize = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSize);

    // ===== T9c: 46 组合守卫（aclnn/GE 直调路径补齐，语义与 plugin/tiling 侧一致）=====
    const char* roundMode = attrsPtr->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, roundMode);
    const int32_t* scaleAlg = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_SCALE_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleAlg);
    const float* dstTypeMax = attrsPtr->GetAttrPointer<float>(INDEX_ATTR_DST_TYPE_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstTypeMax);

    // B4: axis 范围（归一后须落在 [0, rank) 内，防越界访问）
    int64_t dim = (*axis >= 0) ? static_cast<int64_t>(*axis)
                               : static_cast<int64_t>(*axis) + static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(
        dim < 0 || dim >= static_cast<int64_t>(xShape->GetDimNum()),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis",
            std::to_string(*axis),
            "The value of axis must be within the range [-" + std::to_string(xShape->GetDimNum()) + ", " +
                std::to_string(xShape->GetDimNum() - 1) + "]"),
        return ge::GRAPH_FAILED);
    // B3: 46 自适应仅尾轴（kernel 仅在尾轴模板实现）
    OP_CHECK_IF(
        dim != static_cast<int64_t>(xShape->GetDimNum()) - 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis",
            std::to_string(*axis),
            "dst_type_max=4 (4/6 adaptive) is only supported on the tail axis"),
        return ge::GRAPH_FAILED);
    // B5: 硬性条件——量化轴（尾轴）长度必须 32 对齐（kernel 无 tail block 处理）
    if (xShape->GetDim(dim) != UNKNOWN_DIM_VALUE_) {
        OP_CHECK_IF(
            xShape->GetDim(dim) % BLOCK_SIZE_46 != 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis",
                std::to_string(*axis),
                "The axis dim must be 32-aligned (4/6 adaptive is block-aligned only)"),
            return ge::GRAPH_FAILED);
    }
    // B1: 46 自适应仅 blocksize=32
    OP_CHECK_IF(
        *blockSize != BLOCK_SIZE_46,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blocksize",
            std::to_string(*blockSize),
            "dst_type_max=4 (4/6 adaptive) is only supported with blocksize=32"),
        return ge::GRAPH_FAILED);
    // B3: scale_alg 必须为 2
    OP_CHECK_IF(
        *scaleAlg != SCALE_ALG_46,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale_alg",
            std::to_string(*scaleAlg),
            "dst_type_max=4 (4/6 adaptive) only supports scale_alg=2"),
        return ge::GRAPH_FAILED);
    // B3: dst_type_max 必须为 4.0（46 自适应触发值；0/6/7 等非自适应路径已随清理删除）
    OP_CHECK_IF(
        !IsFloatEqualQ(*dstTypeMax, DST_TYPE_MAX_46),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_type_max",
            std::to_string(*dstTypeMax),
            "dst_type_max must be 4.0 (4/6 adaptive is the only remaining path)"),
        return ge::GRAPH_FAILED);
    // B2: round_mode 白名单 {rint, round, floor}（kernel tilingKey=12 支持三种）
    std::string roundModeStr = roundMode;
    OP_CHECK_IF(
        roundModeStr != "rint" && roundModeStr != "round" && roundModeStr != "floor",
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "round_mode", roundModeStr,
            "The value of round_mode must be one of [rint, round, floor]"),
        return ge::GRAPH_FAILED);
    // ===== T9c 守卫结束 =====

    int64_t dimSize = 0;
    if (xShape->GetDim(dim) == UNKNOWN_DIM_VALUE_) {
        dimSize = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize = CeilDivQ(xShape->GetDim(dim), static_cast<int64_t>(*blockSize));
        dimSize = (dimSize + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *scaleShape = *xShape;
    scaleShape->SetDim(dim, dimSize);
    scaleShape->AppendDim(ALIGN_NUM);

    OP_LOGD(
        context->GetNodeName(), "x shape is : %s, mxscale shape is %s.", Shape2String(*xShape).c_str(),
        Shape2String(*scaleShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForQuantFourOverSixA5");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForQuantFourOverSixA5(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForQuantFourOverSixA5");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int32_t* dstDtype = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "dst_type",
            ge::TypeUtils::DataTypeToSerialString(outDtype), "DT_FLOAT4_E2M1"),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, outDtype);
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForQuantFourOverSixA5");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantFourOverSixA5)
    .InferShape(InferShapeForQuantFourOverSixA5)
    .InferDataType(InferDataTypeForQuantFourOverSixA5);
} // namespace ops
