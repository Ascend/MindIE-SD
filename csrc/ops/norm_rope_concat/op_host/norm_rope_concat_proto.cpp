/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
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
 * \file norm_rope_concat_proto.cpp
 * \brief Shape and DataType inference for NormRopeConcat operator
 */

#include <register/op_impl_registry.h>
#include "norm_rope_concat_base.h"

using namespace ge;
using namespace nrc;
namespace ops {
static ge::graphStatus CheckShape(gert::InferShapeContext *context, const gert::Shape *shape, int64_t batch, int64_t head,
                           int64_t dim, int64_t &seq)
{
    if (shape->GetDimNum() != INPUT_DIM_NUM) {
        return ge::GRAPH_FAILED;
    }
    if (shape->GetDim(BATCH_DIM) != batch || shape->GetDim(HEAD_DIM) != head || shape->GetDim(DIM_DIM) != dim) {
        return ge::GRAPH_FAILED;
    }
    seq = shape->GetDim(SEQ_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4NormRopeConcat(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(static_cast<size_t>(InputIndexForward::QUERY_INDEX));
    const gert::Shape *keyShape = context->GetInputShape(static_cast<size_t>(InputIndexForward::KEY_INDEX));
    const gert::Shape *valueShape = context->GetInputShape(static_cast<size_t>(InputIndexForward::VALUE_INDEX));
    if (queryShape == nullptr) { return ge::GRAPH_FAILED; }
    if (keyShape == nullptr) { return ge::GRAPH_FAILED; }
    if (valueShape == nullptr) { return ge::GRAPH_FAILED; }

    gert::Shape *queryOutputShape =
        context->GetOutputShape(static_cast<size_t>(OutputIndexForward::QUERY_OUTPUT_INDEX));
    gert::Shape *keyOutputShape = context->GetOutputShape(static_cast<size_t>(OutputIndexForward::KEY_OUTPUT_INDEX));
    gert::Shape *valueOutputShape =
        context->GetOutputShape(static_cast<size_t>(OutputIndexForward::VALUE_OUTPUT_INDEX));

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) { return ge::GRAPH_FAILED; }
    auto normType = attrs->GetInt(static_cast<size_t>(AttrIndexForward::NORM_TYPE_INDEX));
    auto normAddedType = attrs->GetInt(static_cast<size_t>(AttrIndexForward::NORM_ADDED_TYPE_INDEX));
    auto isTraining = attrs->GetBool(static_cast<size_t>(AttrIndexForward::IS_TRAINING_INDEX));
    if (normType == nullptr) { return ge::GRAPH_FAILED; }
    if (normAddedType == nullptr) { return ge::GRAPH_FAILED; }
    if (isTraining == nullptr) { return ge::GRAPH_FAILED; }

    if (queryShape->GetDimNum() != INPUT_DIM_NUM) {
            return ge::GRAPH_FAILED;
    }
    int64_t batch = queryShape->GetDim(BATCH_DIM);
    int64_t head = queryShape->GetDim(HEAD_DIM);
    int64_t dim = queryShape->GetDim(DIM_DIM);
    int64_t querySeq = queryShape->GetDim(SEQ_DIM);
    int64_t keySeq = 0;
    int64_t valueSeq = 0;
    int64_t encoderQuerySeq = 0;
    int64_t encoderKeySeq = 0;
    int64_t encoderValueSeq = 0;

    if (CheckShape(context, keyShape, batch, head, dim, keySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShape(context, valueShape, batch, head, dim, valueSeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *encoderQueryShape =
        context->GetInputShape(static_cast<size_t>(InputIndexForward::ENCODER_QUERY_INDEX));
    const gert::Shape *encoderKeyShape =
        context->GetInputShape(static_cast<size_t>(InputIndexForward::ENCODER_KEY_INDEX));
    const gert::Shape *encoderValueShape =
        context->GetInputShape(static_cast<size_t>(InputIndexForward::ENCODER_VALUE_INDEX));
    if (encoderQueryShape != nullptr &&
        CheckShape(context, encoderQueryShape, batch, head, dim, encoderQuerySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (encoderKeyShape != nullptr &&
        CheckShape(context, encoderKeyShape, batch, head, dim, encoderKeySeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (encoderValueShape != nullptr &&
        CheckShape(context, encoderValueShape, batch, head, dim, encoderValueSeq) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    *queryOutputShape = {batch, head, querySeq + encoderQuerySeq, dim}; // B, H, S, D
    *keyOutputShape = {batch, head, keySeq + encoderKeySeq, dim};       // B, H, S, D
    *valueOutputShape = {batch, head, valueSeq + encoderValueSeq, dim}; // B, H, S, D

    if (*isTraining) {
        if (*normType == static_cast<int64_t>(NormType::LAYER_NORM) ||
            *normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
            gert::Shape *normQueryMeanShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_QUERY_MEAN_INDEX));
            gert::Shape *normQueryRstdShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_QUERY_RSTD_INDEX));
            gert::Shape *normKeyMeanShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_KEY_MEAN_INDEX));
            gert::Shape *normKeyRstdShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_KEY_RSTD_INDEX));
            *normQueryMeanShape = {batch, querySeq, head, 1};
            *normQueryRstdShape = {batch, querySeq, head, 1};
            *normKeyMeanShape = {batch, keySeq, head, 1};
            *normKeyRstdShape = {batch, keySeq, head, 1};
        }
        if (*normAddedType == static_cast<int64_t>(NormType::LAYER_NORM) ||
            *normAddedType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
            gert::Shape *normAddedQueryMeanShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_MEAN_INDEX));
            gert::Shape *normAddedQueryRstdShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_RSTD_INDEX));
            gert::Shape *normAddedKeyMeanShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_MEAN_INDEX));
            gert::Shape *normAddedKeyRstdShape =
                context->GetOutputShape(static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_RSTD_INDEX));
            *normAddedQueryMeanShape = {batch, encoderQuerySeq, head, 1};
            *normAddedQueryRstdShape = {batch, encoderQuerySeq, head, 1};
            *normAddedKeyMeanShape = {batch, encoderKeySeq, head, 1};
            *normAddedKeyRstdShape = {batch, encoderKeySeq, head, 1};
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4NormRopeConcat(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto inputDtype = context->GetInputDataType(static_cast<size_t>(InputIndexForward::QUERY_INDEX));
    context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::QUERY_OUTPUT_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::KEY_OUTPUT_INDEX), inputDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::VALUE_OUTPUT_INDEX), inputDtype);
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) { return ge::GRAPH_FAILED; }
    auto normType = attrs->GetInt(static_cast<size_t>(AttrIndexForward::NORM_TYPE_INDEX));
    auto normAddedType = attrs->GetInt(static_cast<size_t>(AttrIndexForward::NORM_ADDED_TYPE_INDEX));
    auto isTraining = attrs->GetBool(static_cast<size_t>(AttrIndexForward::IS_TRAINING_INDEX));
    if (normType == nullptr) { return ge::GRAPH_FAILED; }
    if (normAddedType == nullptr) { return ge::GRAPH_FAILED; }
    if (isTraining == nullptr) { return ge::GRAPH_FAILED; }
    if (*isTraining) {
        if (*normType == static_cast<int64_t>(NormType::LAYER_NORM) ||
            *normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_QUERY_MEAN_INDEX), ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_QUERY_RSTD_INDEX), ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_KEY_MEAN_INDEX), ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_KEY_RSTD_INDEX), ge::DT_FLOAT);
        }
        if (*normAddedType == static_cast<int64_t>(NormType::LAYER_NORM) ||
            *normAddedType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_MEAN_INDEX),
                                       ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_RSTD_INDEX),
                                       ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_MEAN_INDEX),
                                       ge::DT_FLOAT);
            context->SetOutputDataType(static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_RSTD_INDEX),
                                       ge::DT_FLOAT);
        }
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NormRopeConcat).InferShape(InferShape4NormRopeConcat).InferDataType(InferDataType4NormRopeConcat);
} // namespace ops

