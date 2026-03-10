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

#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"

namespace {
constexpr int OUTPUTINDEX0 = 0;
constexpr int OUTPUTINDEX1 = 1;
constexpr int OUTPUTINDEX2 = 2;
constexpr int SEQ_LEN_DIM = 2;
constexpr int HEAD_DIM_DIM = 3;
constexpr int INPUT_HEAD_NUM_DIM = 2;
constexpr int INPUT_HEAD_DIM_DIM = 3;
}


namespace ops {

static ge::graphStatus LaPreprocessInferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *qShape = context->GetInputShape(0);
    const gert::Shape *kShape = context->GetInputShape(1);
    const gert::Shape *vShape = context->GetInputShape(2);
    gert::Shape *outQShape = context->GetOutputShape(0);
    gert::Shape *outKShape = context->GetOutputShape(1);
    gert::Shape *outVShape = context->GetOutputShape(2);

    if (qShape == nullptr || kShape == nullptr || vShape == nullptr ||
        outQShape == nullptr || outKShape == nullptr || outVShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto alignLen = *context->GetAttrs()->GetAttrPointer<int32_t>(0);

    // 输出形状: [batch, head_num, padded_seq_len, head_dim]
    outQShape->SetDimNum(qShape->GetDimNum());
    outQShape->SetDim(0, qShape->GetDim(0));  // batch
    outQShape->SetDim(1, qShape->GetDim(2));  // head_num (从第2维移到第1维)
    int32_t qPadDim = (qShape->GetDim(1) + alignLen - 1) / alignLen * alignLen;  // padded seq_len
    outQShape->SetDim(SEQ_LEN_DIM, qPadDim);
    outQShape->SetDim(HEAD_DIM_DIM, qShape->GetDim(INPUT_HEAD_DIM_DIM));  // head_dim

    outKShape->SetDimNum(kShape->GetDimNum());
    outKShape->SetDim(0, kShape->GetDim(0));
    outKShape->SetDim(1, kShape->GetDim(INPUT_HEAD_NUM_DIM));
    int32_t kPadDim = (kShape->GetDim(1) + alignLen - 1) / alignLen * alignLen;
    outKShape->SetDim(SEQ_LEN_DIM, kPadDim);
    outKShape->SetDim(HEAD_DIM_DIM, kShape->GetDim(INPUT_HEAD_DIM_DIM));

    outVShape->SetDimNum(vShape->GetDimNum());
    outVShape->SetDim(0, vShape->GetDim(0));
    outVShape->SetDim(1, vShape->GetDim(INPUT_HEAD_NUM_DIM));
    int32_t vPadDim = (vShape->GetDim(1) + alignLen - 1) / alignLen * alignLen;
    outVShape->SetDim(SEQ_LEN_DIM, vPadDim);
    outVShape->SetDim(HEAD_DIM_DIM, vShape->GetDim(INPUT_HEAD_DIM_DIM));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LaPreprocessInferDtype(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUTINDEX0, ge::DT_FLOAT16);
    context->SetOutputDataType(OUTPUTINDEX1, ge::DT_FLOAT16);
    context->SetOutputDataType(OUTPUTINDEX2, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LaPreprocess)
    .InferShape(LaPreprocessInferShape)
    .InferDataType(LaPreprocessInferDtype);

} // namespace ops
