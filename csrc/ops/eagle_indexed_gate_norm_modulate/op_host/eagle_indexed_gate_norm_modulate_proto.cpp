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
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = *context->GetInputShape(0);
    if (shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = shape;
    *context->GetOutputShape(1) = shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *dtype = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    if (dtype == nullptr || (*dtype != 0 && *dtype != 1)) {
        return ge::GRAPH_FAILED;
    }
    const auto outputType = *dtype == 0 ? ge::DT_FLOAT16 : ge::DT_BF16;
    context->SetOutputDataType(0, outputType);
    context->SetOutputDataType(1, outputType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EagleIndexedGateNormModulate).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
