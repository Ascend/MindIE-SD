/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

static ge::graphStatus InferShapeMulAdd(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *aShape = context->GetInputShape(0);
    const gert::Shape *bShape = context->GetInputShape(1);
    const gert::Shape *cShape = context->GetInputShape(2);
    gert::Shape *outShape = context->GetOutputShape(0);

    if (aShape == nullptr || bShape == nullptr || cShape == nullptr || outShape == nullptr) {
        OP_LOGE("MulAdd", "Input or output shape of MulAdd is null.");
        return ge::GRAPH_FAILED;
    }

    if (aShape->GetDimNum() != 3 || bShape->GetDimNum() != 3 || cShape->GetDimNum() != 3) {
        OP_LOGE("MulAdd", "Inputs of MulAdd must be 3-dimensional, got a:%zu, b:%zu, c:%zu.",
                aShape->GetDimNum(), bShape->GetDimNum(), cShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (aShape->GetDim(0) != bShape->GetDim(0) || aShape->GetDim(0) != cShape->GetDim(0)) {
        OP_LOGE("MulAdd", "Batch size of MulAdd inputs must be the same, got a:%ld, b:%ld, c:%ld.",
                aShape->GetDim(0), bShape->GetDim(0), cShape->GetDim(0));
        return ge::GRAPH_FAILED;
    }
    if (aShape->GetDim(1) != bShape->GetDim(1)) {
        OP_LOGE("MulAdd", "Sequence length of MulAdd inputs a and b must be the same, got a:%ld, b:%ld.",
                aShape->GetDim(1), bShape->GetDim(1));
        return ge::GRAPH_FAILED;
    }
    if (aShape->GetDim(2) != bShape->GetDim(2) || aShape->GetDim(2) != cShape->GetDim(2)) {
        OP_LOGE("MulAdd", "Hidden size of MulAdd inputs must be the same, got a:%ld, b:%ld, c:%ld.",
                aShape->GetDim(2), bShape->GetDim(2), cShape->GetDim(2));
        return ge::GRAPH_FAILED;
    }
    if (cShape->GetDim(1) != 1) {
        OP_LOGE("MulAdd", "The second dimension of MulAdd input c must be 1, got %ld.", cShape->GetDim(1));
        return ge::GRAPH_FAILED;
    }

    *outShape = *aShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMulAdd(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MulAdd)
    .InferShape(InferShapeMulAdd)
    .InferDataType(InferDataTypeMulAdd);

} // namespace ops
