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

#include <register/op_impl_registry.h>
#include <graph/utils/type_utils.h>
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeIndexedAdaLayerNorm(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *scaleShape = context->GetInputShape(1);
    const gert::Shape *shiftShape = context->GetInputShape(2);
    const gert::Shape *indexShape = context->GetInputShape(3);
    gert::Shape *outShape = context->GetOutputShape(0);
    if (xShape == nullptr || scaleShape == nullptr || shiftShape == nullptr || indexShape == nullptr ||
        outShape == nullptr) {
        OP_LOGE("IndexedAdaLayerNorm", "Input or output shape is null.");
        return ge::GRAPH_FAILED;
    }

    if (xShape->GetDimNum() != 3) {
        OP_LOGE("IndexedAdaLayerNorm", "x must be 3D [B, T, H], got %zu.", xShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (scaleShape->GetDimNum() != 3 || shiftShape->GetDimNum() != 3) {
        OP_LOGE("IndexedAdaLayerNorm", "scale_table/shift_table must be 3D [B, K, H].");
        return ge::GRAPH_FAILED;
    }
    if (indexShape->GetDimNum() != 1) {
        OP_LOGE("IndexedAdaLayerNorm", "mod_index must be 1D [T], got %zu.", indexShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = xShape->GetDim(0);
    const int64_t seqLen = xShape->GetDim(1);
    const int64_t hidden = xShape->GetDim(2);
    if (scaleShape->GetDim(0) != batch || shiftShape->GetDim(0) != batch) {
        OP_LOGE("IndexedAdaLayerNorm", "Batch of scale/shift tables must match x.");
        return ge::GRAPH_FAILED;
    }
    if (scaleShape->GetDim(2) != hidden || shiftShape->GetDim(2) != hidden) {
        OP_LOGE("IndexedAdaLayerNorm", "Hidden size of scale/shift tables must match x.");
        return ge::GRAPH_FAILED;
    }
    if (scaleShape->GetDim(1) <= 0 || shiftShape->GetDim(1) != scaleShape->GetDim(1)) {
        OP_LOGE("IndexedAdaLayerNorm", "scale/shift tables must have the same positive K.");
        return ge::GRAPH_FAILED;
    }
    if (indexShape->GetDim(0) != seqLen) {
        OP_LOGE("IndexedAdaLayerNorm", "mod_index length must equal T.");
        return ge::GRAPH_FAILED;
    }

    *outShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexedAdaLayerNorm(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexedAdaLayerNorm)
    .InferShape(InferShapeIndexedAdaLayerNorm)
    .InferDataType(InferDataTypeIndexedAdaLayerNorm);

} // namespace ops
