/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file eagle_quant_batch_matmul_swiglu_infershape.cpp
 * \brief
 */
#include "deps/matmul/common/op_host/matmul_common_infershape.h"
#include "log/log.h"
#include "deps/matmul/common/op_host/log_format_util.h"
#include "runtime/infer_datatype_context.h"

namespace {
constexpr size_t EAGLE_QUANT_BATCH_MATMUL_SWIGLU_MAX_SHAPE_SIZE = 6;
constexpr size_t EAGLE_QUANT_BATCH_MATMUL_SWIGLU_MIN_SHAPE_SIZE = 2;
static ge::graphStatus InferShapeForEagleQuantBatchMatmulSwiglu(gert::InferShapeContext *context) {
    if (context == nullptr) {
        OP_LOGE("EagleQuantBatchMatmulSwiglu", "InferShapeContext is null.");
        return ge::GRAPH_FAILED;
    }
    auto shape_x1 = context->GetInputShape(0);
    auto shape_x2 = context->GetInputShape(1);
    auto scale = context->GetInputShape(2);
    if (shape_x1 == nullptr || shape_x2 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x1, x2", "null", "x1 and x2 can not be null");
        return ge::GRAPH_FAILED;
    }
    auto dim_a = shape_x1->GetDimNum();
    auto dim_b = shape_x2->GetDimNum();
    if (scale == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "scale", "null", "scale can not be null");
        return ge::GRAPH_FAILED;
    }
    bool any_unknow_rank =
        Ops::EagleQbmm::CheckIsUnknownDimNum(*shape_x1) || Ops::EagleQbmm::CheckIsUnknownDimNum(*shape_x2);
    if (!any_unknow_rank &&
        (dim_a < EAGLE_QUANT_BATCH_MATMUL_SWIGLU_MIN_SHAPE_SIZE ||
            dim_a > EAGLE_QUANT_BATCH_MATMUL_SWIGLU_MAX_SHAPE_SIZE ||
            dim_b < EAGLE_QUANT_BATCH_MATMUL_SWIGLU_MIN_SHAPE_SIZE ||
            dim_b > EAGLE_QUANT_BATCH_MATMUL_SWIGLU_MAX_SHAPE_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x1, x2",
            Ops::EagleQbmm::FormatString("%zuD, %zuD", dim_a, dim_b).c_str(),
            "the shape dims of x1 and x2 must be in the range of 2 to 6");
        return ge::GRAPH_FAILED;
    }
    // first transpose attr is transpose_x1, its index is 1 and bias input tensor index is 4
    auto status = Ops::EagleQbmm::InferShapeForBatchMatMul(context, 1, 4UL);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    auto output = context->GetOutputShape(0);
    if (!Ops::EagleQbmm::CheckIsUnknownDimNum(*output)) {
        auto last = output->GetDimNum() - 1;
        auto n = output->GetDim(last);
        if (n >= 0) {
            if (n == 0 || n % 2 != 0) {
                OP_LOGE(context->GetNodeName(), "SwiGLU requires positive even N.");
                return ge::GRAPH_FAILED;
            }
            output->SetDim(last, n / 2);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 由于编译期InferShapeRange在InferShape之后，因此该函数无需重复InferShape的dtype校验
static ge::graphStatus InferShapeRangeForEagleQuantBatchMatmulSwiglu(gert::InferShapeRangeContext *context) {
    // first transpose attr is transpose_x1, its index is 1 and bias input tensor index is 4
    auto status = Ops::EagleQbmm::InferShapeRangeForBatchMatMul(context, 1, 4);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    auto range = context->GetOutputShapeRange(0);
    for (auto shape : {range->GetMin(), range->GetMax()}) {
        if (shape != nullptr && shape->GetDimNum() >= 2) {
            auto last = shape->GetDimNum() - 1;
            auto n = shape->GetDim(last);
            if (n >= 0) {
                shape->SetDim(last, n / 2);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace
namespace Ops::EagleQbmm::MatMul {
IMPL_OP_INFERSHAPE(EagleQuantBatchMatmulSwiglu)
    .InferShape(InferShapeForEagleQuantBatchMatmulSwiglu)
    .InferShapeRange(InferShapeRangeForEagleQuantBatchMatmulSwiglu)
    .PrivateAttr("enable_uncache", 0L);
}
