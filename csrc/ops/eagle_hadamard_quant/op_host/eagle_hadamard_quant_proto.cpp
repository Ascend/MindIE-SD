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
#include "eagle_hadamard_quant_params.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *ctx) {
    if (ctx == nullptr || ctx->GetAttrs() == nullptr || ctx->GetRequiredInputShape(0) == nullptr ||
        ctx->GetRequiredInputShape(1) == nullptr || ctx->GetOutputShape(0) == nullptr ||
        ctx->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *blockSize = ctx->GetAttrs()->GetInt(0);
    const auto *dstType = ctx->GetAttrs()->GetStr(1);
    const auto *isNz = ctx->GetAttrs()->GetBool(2);
    if (blockSize == nullptr || dstType == nullptr || isNz == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &x = *ctx->GetRequiredInputShape(0);
    const auto p = EagleHadamardQuant::CheckShape(x, *ctx->GetRequiredInputShape(1), *blockSize, dstType, *isNz, true);
    if (p.error != nullptr) {
        return ge::GRAPH_FAILED;
    }
    *ctx->GetOutputShape(0) = {x.GetDim(0), p.outputColumns};
    *ctx->GetOutputShape(1) = {x.GetDim(0), p.scalePairs, 2};
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
    if (ctx == nullptr || ctx->GetAttrs() == nullptr || ctx->GetRequiredInputDataType(0) != ge::DT_BF16 ||
        ctx->GetRequiredInputDataType(1) != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    const auto *dstType = ctx->GetAttrs()->GetStr(1);
    if (dstType == nullptr || (std::strcmp(dstType, "mxfp4") != 0 && std::strcmp(dstType, "mxfp8") != 0)) {
        return ge::GRAPH_FAILED;
    }
    ctx->SetOutputDataType(0, std::strcmp(dstType, "mxfp4") == 0 ? ge::DT_UINT8 : ge::DT_FLOAT8_E4M3FN);
    ctx->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EagleHadamardQuant).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
