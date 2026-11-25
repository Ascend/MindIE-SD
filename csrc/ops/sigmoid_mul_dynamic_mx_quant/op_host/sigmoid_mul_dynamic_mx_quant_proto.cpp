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
    const auto *x = context->GetInputShape(0);
    const auto *gate = context->GetInputShape(1);
    auto *y = context->GetOutputShape(0);
    auto *scale = context->GetOutputShape(1);
    if (!x || !gate || !y || !scale || x->GetDimNum() != 2 || gate->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    *y = *x;
    for (size_t i = 0; i < 2; ++i) {
        const auto a = x->GetDim(i);
        const auto b = gate->GetDim(i);
        if (a >= 0 && b >= 0 && a != b) {
            return ge::GRAPH_FAILED;
        }
        y->SetDim(i, a >= 0 ? a : b);
    }
    const int64_t cols = y->GetDim(1);
    scale->SetDimNum(3);
    scale->SetDim(0, y->GetDim(0));
    scale->SetDim(1, cols < 0 ? -1 : cols / 64 + (cols % 64 != 0));
    scale->SetDim(2, 2);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidMulDynamicMxQuant).InferShape(InferShape).InferDataType(InferType);
} // namespace ops
