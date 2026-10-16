/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {

static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t KEY_INDEX = 1;
static constexpr uint32_t VALUE_INDEX = 2;
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

static constexpr uint32_t ATTR_Q_INPUT_LAYOUT_INDEX = 0;
static constexpr uint32_t ATTR_KV_INPUT_LAYOUT_INDEX = 1;
static constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 2;

static constexpr uint32_t DIM_BSH = 3;
static constexpr uint32_t DIM_TND = 3;
static constexpr uint32_t DIM_BNSD = 4;

static constexpr uint32_t BSH_DIM_B = 0;
static constexpr uint32_t BSH_DIM_S = 1;
static constexpr uint32_t BSH_DIM_H = 2;

static constexpr uint32_t TND_DIM_T = 0;
static constexpr uint32_t TND_DIM_N = 1;
static constexpr uint32_t TND_DIM_D = 2;
static constexpr uint32_t TND_DIM_NUM = 3;

static constexpr uint32_t BNSD_DIM_B = 0;
static constexpr uint32_t BNSD_DIM_N = 1;
static constexpr uint32_t BNSD_DIM_S = 2;
static constexpr uint32_t BNSD_DIM_D = 3;
static constexpr uint32_t BNSD_DIM_NUM = 4;
static constexpr uint32_t LSE_DIM_D = 1;

static constexpr int32_t UNKNOWN_DIMS = -2;

static ge::graphStatus InferShapeEagleQuantBlockSparseAttention(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("EagleQuantBlockSparseAttention", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    
    OP_LOGD(context->GetNodeName(), "Enter EagleQuantBlockSparseAttention InferShape impl.");
    
    // 获取Query shape
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    
    // 获取Key shape
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    
    // 获取Value shape
    const gert::Shape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);
    
    // 获取输出shape
    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    
    gert::Shape *softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);
    
    // 获取属性
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    
    const char *qInputLayoutPtr = attrs->GetAttrPointer<char>(ATTR_Q_INPUT_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputLayoutPtr);
    
    const char *kvInputLayoutPtr = attrs->GetAttrPointer<char>(ATTR_KV_INPUT_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvInputLayoutPtr);
    
    const int64_t *numKvHeadsPtr = attrs->GetInt(ATTR_NUM_KV_HEADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numKvHeadsPtr);
    
    // UNKNOWN DIM处理
    if ((queryShape->GetDimNum() == 1 && queryShape->GetDim(0) == UNKNOWN_DIMS) ||
        (keyShape->GetDimNum() == 1 && keyShape->GetDim(0) == UNKNOWN_DIMS) ||
        (valueShape->GetDimNum() == 1 && valueShape->GetDim(0) == UNKNOWN_DIMS)) {
        attentionOutShape->SetDimNum(1);
        (*attentionOutShape)[0] = UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(1);
        (*softmaxLseShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }
    
    // 设置AttentionOut shape (与Query shape相同)
    *attentionOutShape = *queryShape;
    
    // 验证Q layout和KV layout
    std::string qLayout(qInputLayoutPtr);
    std::string kvLayout(kvInputLayoutPtr);
    
    // 验证Q layout (只支持TND和BNSD)
    if (qLayout == "TND") {
        if (queryShape->GetDimNum() != DIM_TND) {
            OP_LOGE(context->GetNodeName(), "Layout TND, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (qLayout == "BNSD") {
        if (queryShape->GetDimNum() != DIM_BNSD) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context->GetNodeName(), "Unsupported Q layout: %s. Only TND and BNSD are supported.", qInputLayoutPtr);
        return ge::GRAPH_FAILED;
    }
    
    // 验证Q和KV格式一致性：如果其中一个是BNSD，另一个也必须是BNSD
    bool isQBNSD = (qLayout == "BNSD");
    bool isKvBNSD = (kvLayout == "BNSD");
    
    if (isQBNSD != isKvBNSD) {
        OP_LOGE(context->GetNodeName(), 
                "Q and KV layouts must match: if one is BNSD, the other must also be BNSD. "
                "Q layout: %s, KV layout: %s", qLayout.c_str(), kvLayout.c_str());
        return ge::GRAPH_FAILED;
    }
    
    // 验证KV layout
    if (kvLayout == "TND") {
        if (keyShape->GetDimNum() != DIM_TND || valueShape->GetDimNum() != DIM_TND) {
            OP_LOGE(context->GetNodeName(), "Layout TND, KV dims must be 3!");
            return ge::GRAPH_FAILED;
        }
        
        // TND格式: [T, N, D]
        int64_t kvN = keyShape->GetDim(TND_DIM_N);
        int64_t kvD = keyShape->GetDim(TND_DIM_D);
        
        if (*numKvHeadsPtr != 0 && kvN != *numKvHeadsPtr) {
            OP_LOGE(context->GetNodeName(), "KV heads mismatch in TND format: %ld != %ld", kvN, *numKvHeadsPtr);
            return ge::GRAPH_FAILED;
        }
        
        if (valueShape->GetDim(TND_DIM_D) != kvD) {
            OP_LOGE(context->GetNodeName(), "K and V head dimension mismatch in TND format");
            return ge::GRAPH_FAILED;
        }
    } else if (kvLayout == "BNSD") {
        if (keyShape->GetDimNum() != DIM_BNSD || valueShape->GetDimNum() != DIM_BNSD) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD, KV dims must be 4!");
            return ge::GRAPH_FAILED;
        }
        
        // BNSD格式: [B, N, S, D]
        int64_t kvB = keyShape->GetDim(BNSD_DIM_B);
        int64_t kvN = keyShape->GetDim(BNSD_DIM_N);
        int64_t kvD = keyShape->GetDim(BNSD_DIM_D);
        
        if (*numKvHeadsPtr != 0 && kvN != *numKvHeadsPtr) {
            OP_LOGE(context->GetNodeName(), "KV heads mismatch in BNSD format: %ld != %ld", kvN, *numKvHeadsPtr);
            return ge::GRAPH_FAILED;
        }
        
        if (valueShape->GetDim(BNSD_DIM_D) != kvD) {
            OP_LOGE(context->GetNodeName(), "K and V head dimension mismatch in BNSD format");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context->GetNodeName(), "Unsupported KV layout: %s. Only TND format is supported.", kvInputLayoutPtr);
        return ge::GRAPH_FAILED;
    }
    
    // 设置SoftmaxLse shape (如果需要)
    // SoftmaxLse shape通常是 [batch, num_heads, q_seqlen, 1] 或类似维度
    if (qLayout == "TND") {
        // TND格式
        softmaxLseShape->SetDimNum(TND_DIM_NUM);
        (*softmaxLseShape)[TND_DIM_T] = queryShape->GetDim(TND_DIM_T);
        (*softmaxLseShape)[TND_DIM_N] = queryShape->GetDim(TND_DIM_N);
        (*softmaxLseShape)[TND_DIM_D] = LSE_DIM_D;
    } else if (qLayout == "BNSD") {
        // BNSD格式
        softmaxLseShape->SetDimNum(BNSD_DIM_NUM);
        (*softmaxLseShape)[BNSD_DIM_B] = queryShape->GetDim(BNSD_DIM_B);
        (*softmaxLseShape)[BNSD_DIM_N] = queryShape->GetDim(BNSD_DIM_N);
        (*softmaxLseShape)[BNSD_DIM_S] = queryShape->GetDim(BNSD_DIM_S);
        (*softmaxLseShape)[BNSD_DIM_D] = LSE_DIM_D;
    } else {
        OP_LOGE(context->GetNodeName(), "Unexpected Q layout in softmaxLse shape calculation: %s", qInputLayoutPtr);
        return ge::GRAPH_FAILED;
    }
    
    OP_LOGD(context->GetNodeName(), "EagleQuantBlockSparseAttention InferShape success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeEagleQuantBlockSparseAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EagleQuantBlockSparseAttention)
    .InferShape(InferShapeEagleQuantBlockSparseAttention)
    .InferDataType(InferDataTypeEagleQuantBlockSparseAttention);

}  // namespace ops

