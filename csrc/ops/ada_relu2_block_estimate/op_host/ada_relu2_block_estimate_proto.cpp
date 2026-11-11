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

#include <algorithm>
#include <string>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

namespace {
constexpr size_t QUERY_INDEX = 0;
constexpr size_t KEY_INDEX = 1;
constexpr size_t SCORE_OUTPUT_INDEX = 0;
constexpr size_t ROW_SUM_OUTPUT_INDEX = 1;
constexpr size_t INPUT_LAYOUT_ATTR_INDEX = 0;
constexpr size_t STRIDE_ATTR_INDEX = 1;
constexpr size_t SPARSE_SIZE_ATTR_INDEX = 2;
constexpr size_t NUM_HEADS_ATTR_INDEX = 3;

int64_t CeilDiv(int64_t value, int64_t divisor) {
    if (value < 0) {
        return value;
    }
    return (value + divisor - 1) / divisor;
}
} // namespace

namespace ops {
static ge::graphStatus AdaRelu2BlockEstimateInferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    gert::Shape *scoreShape = context->GetOutputShape(SCORE_OUTPUT_INDEX);
    gert::Shape *rowSumShape = context->GetOutputShape(ROW_SUM_OUTPUT_INDEX);
    const auto attrs = context->GetAttrs();
    if (queryShape == nullptr || keyShape == nullptr || scoreShape == nullptr || rowSumShape == nullptr ||
        attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const char *layoutValue = attrs->GetAttrPointer<char>(INPUT_LAYOUT_ATTR_INDEX);
    const int32_t *strideValue = attrs->GetAttrPointer<int32_t>(STRIDE_ATTR_INDEX);
    const int32_t *sparseSizeValue = attrs->GetAttrPointer<int32_t>(SPARSE_SIZE_ATTR_INDEX);
    const int32_t *numHeadsValue = attrs->GetAttrPointer<int32_t>(NUM_HEADS_ATTR_INDEX);
    if (layoutValue == nullptr || strideValue == nullptr || sparseSizeValue == nullptr || numHeadsValue == nullptr ||
        *strideValue <= 0 || *sparseSizeValue <= 0) {
        return ge::GRAPH_FAILED;
    }

    const std::string layout(layoutValue);
    int64_t batchSize = 0;
    int64_t queryHeads = 0;
    int64_t querySeqLen = 0;
    int64_t keySeqLen = 0;
    if (layout == "BNSD") {
        if (queryShape->GetDimNum() != 4 || keyShape->GetDimNum() != 4) {
            return ge::GRAPH_FAILED;
        }
        batchSize = queryShape->GetDim(0);
        queryHeads = queryShape->GetDim(1);
        querySeqLen = queryShape->GetDim(2);
        keySeqLen = keyShape->GetDim(2);
    } else if (layout == "BSND") {
        if (queryShape->GetDimNum() != 4 || keyShape->GetDimNum() != 4) {
            return ge::GRAPH_FAILED;
        }
        batchSize = queryShape->GetDim(0);
        queryHeads = queryShape->GetDim(2);
        querySeqLen = queryShape->GetDim(1);
        keySeqLen = keyShape->GetDim(1);
    } else if (layout == "BSH") {
        if (queryShape->GetDimNum() != 3 || keyShape->GetDimNum() != 3 || *numHeadsValue <= 0) {
            return ge::GRAPH_FAILED;
        }
        batchSize = queryShape->GetDim(0);
        queryHeads = *numHeadsValue;
        querySeqLen = queryShape->GetDim(1);
        keySeqLen = keyShape->GetDim(1);
    } else {
        return ge::GRAPH_FAILED;
    }

    const int64_t queryBlocks = CeilDiv(querySeqLen, *sparseSizeValue);
    const int64_t keyBlocks = CeilDiv(keySeqLen, *sparseSizeValue);
    const int64_t scoredKeyBlocks = keyBlocks < 0 ? keyBlocks : std::max(keyBlocks - 1, static_cast<int64_t>(0));
    const int64_t reduceSize = CeilDiv(*sparseSizeValue, *strideValue);

    scoreShape->SetDimNum(5);
    scoreShape->SetDim(0, batchSize);
    scoreShape->SetDim(1, queryHeads);
    scoreShape->SetDim(2, queryBlocks);
    scoreShape->SetDim(3, scoredKeyBlocks);
    scoreShape->SetDim(4, reduceSize);

    rowSumShape->SetDimNum(4);
    rowSumShape->SetDim(0, batchSize);
    rowSumShape->SetDim(1, queryHeads);
    rowSumShape->SetDim(2, queryBlocks);
    rowSumShape->SetDim(3, reduceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AdaRelu2BlockEstimateInferDtype(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(SCORE_OUTPUT_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(ROW_SUM_OUTPUT_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaRelu2BlockEstimate)
    .InferShape(AdaRelu2BlockEstimateInferShape)
    .InferDataType(AdaRelu2BlockEstimateInferDtype);
} // namespace ops
