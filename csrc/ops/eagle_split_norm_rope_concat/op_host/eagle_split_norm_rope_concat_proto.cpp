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
#include "eagle_split_norm_rope_concat_shape.h"

namespace ops {
static ge::graphStatus InferEagleSplitNormRopeConcat(gert::InferShapeContext *ctx) {
    if (ctx == nullptr || ctx->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 6; ++i) {
        if (i == 2 || i == 3) {
            continue;
        }
        if (ctx->GetRequiredInputShape(i) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *attrs = ctx->GetAttrs();
    const auto *norm = attrs->GetInt(0);
    const auto *rope = attrs->GetInt(1);
    const auto *rotary = attrs->GetInt(2);
    const auto *eps = attrs->GetFloat(3);
    const auto *bnsd = attrs->GetBool(4);
    const auto *heads = ctx->GetInputTensor(1);
    if (norm == nullptr || rotary == nullptr || rope == nullptr || eps == nullptr || bnsd == nullptr ||
        heads == nullptr || (*norm != 2 && *norm != 3)) {
        return ge::GRAPH_FAILED;
    }
    eagle_snrc::ShapeInfo info{};
    if (!eagle_snrc::ParseShape(*ctx->GetRequiredInputShape(0), *ctx->GetRequiredInputShape(1),
            heads->GetData<int64_t>(), *rotary, *rope, *eps, info) ||
        !eagle_snrc::CheckRope(*ctx->GetRequiredInputShape(4), *ctx->GetRequiredInputShape(5), info.rotaryDim, info)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 2; i < 4; ++i) {
        const auto *weight = ctx->GetOptionalInputShape(i);
        if (weight == nullptr) {
            if (*norm == 3) {
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        if (weight->GetDimNum() != 1 || weight->GetDim(0) != info.dim) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t i = 0; i < 3; ++i) {
        auto *out = ctx->GetOutputShape(i);
        if (out == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (*bnsd) {
            *out = {info.batch, info.heads[i], info.seq, info.dim};
        } else if (ctx->GetRequiredInputShape(0)->GetDimNum() == 3) {
            *out = {info.seq, info.heads[i], info.dim};
        } else {
            *out = {info.batch, info.seq, info.heads[i], info.dim};
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferEagleSplitNormRopeConcatType(gert::InferDataTypeContext *ctx) {
    if (ctx == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = ctx->GetRequiredInputDataType(0);
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) || ctx->GetRequiredInputDataType(1) != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 2; i < 6; ++i) {
        const auto inputType = i < 4 ? ctx->GetOptionalInputDataType(i) : ctx->GetRequiredInputDataType(i);
        if (i < 4 && inputType == ge::DT_UNDEFINED) {
            continue;
        }
        if (inputType != dtype) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t i = 0; i < 3; ++i) {
        ctx->SetOutputDataType(i, dtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EagleSplitNormRopeConcat)
    .InferShape(InferEagleSplitNormRopeConcat)
    .InferDataType(InferEagleSplitNormRopeConcatType)
    .InputsDataDependency({1});
} // namespace ops
