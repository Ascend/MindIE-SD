/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "eagle_quant_block_sparse_attention_tiling.h"
#include <cmath>
#include <cstring>
#include "log/log.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_base.h"  // provides ASCENDC_EXTERN_C

using namespace ge;
using namespace std;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;
constexpr int TND_DIM_NUM = 3;

constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;
constexpr int BNSD_DIM_NUM = 4;

constexpr int BSH_DIM_B = 0;
constexpr int BSH_DIM_S = 1;
constexpr int BSH_DIM_H = 2;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int BLOCK_SPARSE_MASK_INDEX = 3;
constexpr int ATTEN_MASK_INDEX = 4;
constexpr int BLOCK_SHAPE_INDEX = 5;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 7;
constexpr int BLOCK_TABLE_INDEX = 8;
constexpr int SOFTMAX_LSE_INDEX = 10;
constexpr int MAX_BLOCK_NUM_INDEX = 2;


constexpr int Q_INPUT_LAYOUT_INDEX = 0;
constexpr int KV_INPUT_LAYOUT_INDEX = 1;
constexpr int NUM_KEY_VALUE_HEADS_INDEX = 2;
constexpr int MASK_TYPE_INDEX = 3;
constexpr int SCALE_VALUE_INDEX = 4;
constexpr int INNER_PRECISE_INDEX = 5;
constexpr int BLOCK_SIZE_INDEX = 6;
constexpr int PRE_TOKENS_INDEX = 7;
constexpr int NEXT_TOKENS_INDEX = 8;
constexpr int SOFTMAX_LSE_FLAG_INDEX = 9;

constexpr int VALID_EMBEDDING_SIZE_64 = 64;
constexpr int VALID_EMBEDDING_SIZE_128 = 128;

constexpr int LSE_NO_OUT = 0;
constexpr int LSE_OUT = 1;

namespace optiling {

constexpr uint32_t BASIC_BLOCK_SIZE = 128;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
constexpr uint32_t NUM3 = 3;
constexpr uint32_t SOC_VER_950_CODE = 4;
constexpr uint32_t INF_WINDOW_SIZE_PRE_NEXT = 2147483647;
constexpr uint32_t SPARSE_PATTERN_MODE_MASK = 0;
constexpr uint32_t SPARSE_PATTERN_MODE_TABLE = 1;

constexpr uint32_t TILE_SIZE_128 = 128;
constexpr uint32_t TILE_SIZE_256 = 256;
constexpr uint32_t TILE_SIZE_512 = 512;
constexpr uint32_t D_SIZE_128 = 128;
constexpr uint32_t D_SIZE_256 = 256;

constexpr uint32_t SOLO_BUF = 1;
constexpr uint32_t DUO_BUF = 2;
constexpr uint32_t TRIO_BUF = 3;

std::unordered_map<std::string, std::string> inputLayoutMapQ2Kv = {
    {"TND", "TND"},
    {"BNSD", "BNSD"}
};

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint32_t GetQBlocks(int32_t qseqlen, int32_t x)
{
    uint32_t qBlocksInX = CeilDiv(x, BASIC_BLOCK_SIZE);
    uint32_t completeXBlocks = x != 0 ? qseqlen / x : qseqlen / BASIC_BLOCK_SIZE;
    uint32_t remainingSeqlen = x != 0 ? qseqlen - completeXBlocks * x : qseqlen % BASIC_BLOCK_SIZE;
    uint32_t remainingBlocks = CeilDiv(remainingSeqlen, BASIC_BLOCK_SIZE);
    return qBlocksInX * completeXBlocks + remainingBlocks;
}

static inline uint32_t GetQNBlockTile()
{
    uint32_t qNBlockTile = 1;
    return qNBlockTile;
}

ge::graphStatus BSATiling::GetNpuInfo(gert::TilingContext *bsaContext)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(bsaContext->GetPlatformInfo());
    
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    socVer_ = static_cast<uint32_t>(ascendcPlatform.GetSocVersion());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ValidateTNDSeqlenSum(gert::TilingContext *bsaContext)
{
    // 只在TND格式时进行校验
    if (qInputLayout_ != RFAQInputLayout::TND_Q || kvCacheLayout_ != RFAKvCacheLayout::TND) {
        return ge::GRAPH_SUCCESS;
    }
    
    // 计算所有batch的qseqlen之和
    int64_t sumQSeqlen = 0;
    int64_t sumKvSeqlen = 0;

    for (uint32_t i = 0; i < batch_; i++) {
        sumQSeqlen += qSeqLenList_[i];
        sumKvSeqlen += kvSeqLenList_[i];
    }
    
    // 校验qseqlen之和是否等于Q的T
    if (sumQSeqlen != totalTokensT_) {
        OP_LOGE(bsaContext->GetNodeName(), 
                "TND format validation failed: sum of qseqlen across all batches (%ld) != Q T (%ld)", 
                sumQSeqlen, totalTokensT_);
        return ge::GRAPH_FAILED;
    }
    
    // 校验kvseqlen之和是否等于KV的T
    if (sumKvSeqlen != totalTokensKv_) {
        OP_LOGE(bsaContext->GetNodeName(), 
                "TND format validation failed: sum of kvseqlen across all batches (%ld) != KV T (%ld)", 
                sumKvSeqlen, totalTokensKv_);
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::GetInputLayout(gert::TilingContext *bsaContext)
{
    auto attrs = bsaContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, attrs);
    auto attrLayoutQ = attrs->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX);
    auto attrLayoutKv = attrs->GetAttrPointer<char>(KV_INPUT_LAYOUT_INDEX);
    if (attrLayoutQ == nullptr || attrLayoutKv == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "qInputLayout, kvInputLayout must be provided.");
        return ge::GRAPH_FAILED;
    }
    std::string qLayout(attrLayoutQ);
    std::string kvLayout(attrLayoutKv);
    auto it = inputLayoutMapQ2Kv.find(qLayout);
    auto itRev = inputLayoutMapQ2Kv.find(kvLayout);
    if (it == inputLayoutMapQ2Kv.end() || itRev == inputLayoutMapQ2Kv.end()) {
        OP_LOGE(bsaContext->GetNodeName(), "The inputLayout attrs only support TND/BNSD.");
        return ge::GRAPH_FAILED;
    } else if (it->second != kvLayout) {
        OP_LOGE(bsaContext->GetNodeName(), "QInputLayout and kvInputLayout must be the same.");
        return ge::GRAPH_FAILED;
    } else if (it->first == "TND") {
        qInputLayout_ = RFAQInputLayout::TND_Q;
        kvCacheLayout_ = RFAKvCacheLayout::TND;
    } else if (it->first == "BNSD") {
        qInputLayout_ = RFAQInputLayout::BNSD_Q;
        kvCacheLayout_ = RFAKvCacheLayout::BNSD;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::CheckQKVDtype(gert::TilingContext *bsaContext)
{
    auto qInputDesc = bsaContext->GetInputDesc(QUERY_INDEX);
    auto kInputDesc = bsaContext->GetInputDesc(KEY_INDEX);
    auto vInputDesc = bsaContext->GetInputDesc(VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, qInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, kInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, vInputDesc);
    dataType_ = qInputDesc->GetDataType();
    auto kDataType = kInputDesc->GetDataType();
    vDataType_ = vInputDesc->GetDataType();

    // 原有FP16/BF16校验保留
    if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16 && dataType_ != ge::DT_INT8) {
        OP_LOGE(bsaContext->GetNodeName(), 
            "The supported dtype of query/key is float16, bfloat16 or int8.");
        return ge::GRAPH_FAILED;
    }
    
    // 950 混合精度特判：Q/K=int8, V=FLOAT8_E4M3FN
    if (socVer_ == SOC_VER_950_CODE && 
        dataType_ == ge::DT_INT8 && 
        kDataType == ge::DT_INT8 ) {
        return ge::GRAPH_SUCCESS;
    } else if (dataType_ != kDataType || dataType_ != vDataType_) {
        // 非混合精度场景保持原有严格一致性校验
        OP_LOGE(bsaContext->GetNodeName(), 
            "Tensor query/key/value must have consistent dtype, or use 950 mixed prec (Q/K=int8, V=fp8).");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::CheckQKVDimVal(
    gert::TilingContext *bsaContext, uint32_t kHeads, uint32_t vHeads, uint32_t kHeadDim, uint32_t vHeadDim)
{
    if (embeddingSize_ != kHeadDim || embeddingSize_ != vHeadDim) {
        OP_LOGE(bsaContext->GetNodeName(), "Tensor query/key/value must have consistent headDim with each other, "
            "but got qHeadDim %u, kHeadDim %u, vHeadDim %u.", embeddingSize_, kHeadDim, vHeadDim);
        return ge::GRAPH_FAILED;
    }
    if (kvHeads_ != kHeads || kvHeads_ != vHeads) {
        OP_LOGE(bsaContext->GetNodeName(),
            "Tensor key/value must have consistent headNum with each other and attr kvHeads, "
            "but got kHeads %u, vHeads %u, kvHeads(attr) %u.", kHeads, vHeads, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    // temporary regulations of D
    if ((embeddingSize_ != VALID_EMBEDDING_SIZE_128) && (embeddingSize_ != VALID_EMBEDDING_SIZE_64)) {
        OP_LOGE(bsaContext->GetNodeName(),
            "The supported headDim so far is 64 or 128, but got %u.", embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseQKVInTND(gert::TilingContext *bsaContext)
{
    const auto *queryShape = bsaContext->GetInputShape(QUERY_INDEX);
    const auto *keyShape = bsaContext->GetInputShape(KEY_INDEX);
    const auto *valueShape = bsaContext->GetInputShape(VALUE_INDEX);
    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "Query/Key/Value shape is null");
        return ge::GRAPH_FAILED;
    }
    if (queryShape->GetStorageShape().GetDimNum() != TND_DIM_NUM ||
        keyShape->GetStorageShape().GetDimNum() != TND_DIM_NUM ||
        valueShape->GetStorageShape().GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(bsaContext->GetNodeName(), "Tensor query/key/value must have 3 dimensions when layout is 'TND'.");
        return ge::GRAPH_FAILED;
    }
    totalTokensT_ = queryShape->GetStorageShape().GetDim(TND_DIM_T);
    totalTokensKv_ = keyShape->GetStorageShape().GetDim(TND_DIM_T);
    numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
    auto kHeads = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(TND_DIM_N));
    auto vHeads = static_cast<uint32_t>(valueShape->GetStorageShape().GetDim(TND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_D));
    auto embeddingSizeK = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(TND_DIM_D));
    auto embeddingSizeV = static_cast<uint32_t>(valueShape->GetStorageShape().GetDim(TND_DIM_D));
    if (CheckQKVDimVal(bsaContext, kHeads, vHeads, embeddingSizeK, embeddingSizeV) != ge::GRAPH_SUCCESS) {
        OP_LOGE(bsaContext->GetNodeName(), "Check query/key/value dim values failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseQKVInBNSD(gert::TilingContext *bsaContext)
{
    const auto *queryShape = bsaContext->GetInputShape(QUERY_INDEX);
    const auto *keyShape = bsaContext->GetInputShape(KEY_INDEX);
    const auto *valueShape = bsaContext->GetInputShape(VALUE_INDEX);
    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "Query/Key/Value shape is null");
        return ge::GRAPH_FAILED;
    }
    if (queryShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM ||
        keyShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM ||
        valueShape->GetStorageShape().GetDimNum() != BNSD_DIM_NUM) {
        OP_LOGE(bsaContext->GetNodeName(), "Tensor query/key/value must have 4 dimensions when layout is 'BNSD'.");
        return ge::GRAPH_FAILED;
    }
    batch_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_B));
    auto kBatch = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BNSD_DIM_B));
    auto vBatch = static_cast<uint32_t>(valueShape->GetStorageShape().GetDim(BNSD_DIM_B));
    numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_N));
    auto kHeads = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BNSD_DIM_N));
    auto vHeads = static_cast<uint32_t>(valueShape->GetStorageShape().GetDim(BNSD_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_D));
    auto embeddingSizeK = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BNSD_DIM_D));
    auto embeddingSizeV = static_cast<uint32_t>(valueShape->GetStorageShape().GetDim(BNSD_DIM_D));
    maxQSeqlen_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BNSD_DIM_S));
    maxKvSeqlen_ = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BNSD_DIM_S));
    if (batch_ != kBatch || batch_ != vBatch) {
        OP_LOGE(bsaContext->GetNodeName(), "Tensor query/key/value must have consistent batch with each other, "
            "but got qBatch %u, kBatch %u, vBatch %u.", batch_, kBatch, vBatch);
        return ge::GRAPH_FAILED;
    }
    if (CheckQKVDimVal(bsaContext, kHeads, vHeads, embeddingSizeK, embeddingSizeV) != ge::GRAPH_SUCCESS) {
        OP_LOGE(bsaContext->GetNodeName(), "Check query/key/value dim values failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseRequiredTensors(gert::TilingContext *bsaContext)
{
    if (CheckQKVDtype(bsaContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (qInputLayout_ == RFAQInputLayout::TND_Q) {
        ret = ParseQKVInTND(bsaContext);
    } else if (qInputLayout_ == RFAQInputLayout::BNSD_Q) {
        ret = ParseQKVInBNSD(bsaContext);
    }
    return ret;
}

ge::graphStatus BSATiling::ParseSeqlensInTND(gert::TilingContext *bsaContext)
{
    const auto *actualSeqLengths = bsaContext->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
    const auto *actualSeqLengthsKv = bsaContext->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);
    if (actualSeqLengths == nullptr || actualSeqLengthsKv == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(),
            "ActualSeqLengths/actualSeqLengthsKv must be provided when corresponding layout is 'TND'.");
        return ge::GRAPH_FAILED;
    }
    batch_ = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
    auto batchKvS = static_cast<uint32_t>(actualSeqLengthsKv->GetShapeSize());
    if (batch_ != batchKvS) {
        OP_LOGE(bsaContext->GetNodeName(),
            "ActualSeqLengths & actualSeqLengthsKv must have consistent batch size, "
            "but got batch in actualSeqLengths: %u, batch in actualSeqLengthsKv: %u", batch_, batchKvS);
        return ge::GRAPH_FAILED;
    }
    qSeqLenList_ = actualSeqLengths->GetData<int64_t>();
    kvSeqLenList_ = actualSeqLengthsKv->GetData<int64_t>();
    useUniformQSeqlen_ = false;
    useUniformKvSeqlen_ = false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseSeqlensInBNSD(gert::TilingContext *bsaContext)
{
    const auto *actualSeqLengths = bsaContext->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
    const auto *actualSeqLengthsKv = bsaContext->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);
    if (actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr) {
        uint32_t batchQS = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
        uint32_t batchKvS = static_cast<uint32_t>(actualSeqLengthsKv->GetShapeSize());
        if (batch_ != batchKvS || batch_ != batchQS) {
            OP_LOGE(bsaContext->GetNodeName(),
                "ActualSeqLengths & actualSeqLengthsKv must have consistent batch size with each other and context,"
                "but got batch in actualSeqLengths: %u, batch in actualSeqLengthsKv: %u, "
                "batch in context(from query dim0): %u",
                batchQS, batchKvS, batch_);
            return ge::GRAPH_FAILED;
        }
        qSeqLenList_ = actualSeqLengths->GetData<int64_t>();
        kvSeqLenList_ = actualSeqLengthsKv->GetData<int64_t>();
        useUniformQSeqlen_ = false;
        useUniformKvSeqlen_ = false;
    } else if (actualSeqLengths == nullptr && actualSeqLengthsKv == nullptr) {
        useUniformQSeqlen_ = true;
        useUniformKvSeqlen_ = true;
    } else {
        OP_LOGE(bsaContext->GetNodeName(),
            "ActualSeqLengths & actualSeqLengthsKv must be either both provided, or neither provided.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseSeqlens(gert::TilingContext *bsaContext)
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (qInputLayout_ == RFAQInputLayout::TND_Q) {
        ret = ParseSeqlensInTND(bsaContext);
    } else if (qInputLayout_ == RFAQInputLayout::BNSD_Q) {
        ret = ParseSeqlensInBNSD(bsaContext);
    }
    return ret;
}

ge::graphStatus BSATiling::CheckSparsePattern(gert::TilingContext *bsaContext, const int64_t defaultShape)
{
    const auto *blockSparseMaskDesc = bsaContext->GetInputDesc(BLOCK_SPARSE_MASK_INDEX);
    const auto *blockSparseMaskShape = bsaContext->GetInputShape(BLOCK_SPARSE_MASK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, blockSparseMaskDesc);
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, blockSparseMaskShape);
    auto sparsePatternDtype = blockSparseMaskDesc->GetDataType();
    if (sparsePatternDtype == ge::DT_INT8 || sparsePatternDtype == ge::DT_BOOL ) {
        sparsePatternMode_ = SPARSE_PATTERN_MODE_MASK;
    } else if (sparsePatternDtype == ge::DT_INT32) {
        sparsePatternMode_ = SPARSE_PATTERN_MODE_TABLE;
    } else {
        OP_LOGE(bsaContext->GetNodeName(),
            "BlockSparseMask only supports int8 mask or int32 sparse table, but got dtype %d.",
            static_cast<int32_t>(sparsePatternDtype));
        return ge::GRAPH_FAILED;
    }
    if (blockShapeX_ <= 0 || blockShapeY_ <= 0) {
        OP_LOGE(bsaContext->GetNodeName(), "BlockShape elems must be greater than 0, "
            "but got elem0: %ld, elem1: %ld.", blockShapeX_, blockShapeY_);
        return ge::GRAPH_FAILED;
    }
    // temporary regulation of blockShapeY
    if (blockShapeY_ % defaultShape != 0) {
        OP_LOGE(bsaContext->GetNodeName(), "BlockShape elem1 must be a multiple of 128 so far, "
            "but got elem1: %ld.", blockShapeY_);
        return ge::GRAPH_FAILED;
    }
    if (blockSparseMaskShape->GetStorageShape().GetDimNum() != 4) {
        printf("Current DimNum%d\n",blockSparseMaskShape->GetStorageShape().GetDimNum());
        //OP_LOGE(bsaContext->GetNodeName(), "BlockSparseMask must have 4 dims.");
        //return ge::GRAPH_FAILED;
    }
    uint32_t bsmBatch = blockSparseMaskShape->GetStorageShape().GetDim(0);  // batch_size
    uint32_t bsmNumHead = blockSparseMaskShape->GetStorageShape().GetDim(1);  // num_heads
    if (bsmBatch != batch_) {
        OP_LOGE(bsaContext->GetNodeName(), "BlockSparseMask must have consistent batch with context,"
            "but got BlockSparseMask batch(dim0): %u, context batch: %u.", bsmBatch, batch_);
        return ge::GRAPH_FAILED;        
    }
    if (bsmNumHead != numHeads_) {
        OP_LOGE(bsaContext->GetNodeName(), "BlockSparseMask must have consistent numHeads with context,"
            "but got BlockSparseMask numHeads(dim1): %u, context numHeads: %u.", bsmNumHead, numHeads_);
        return ge::GRAPH_FAILED; 
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseSparsePattern(gert::TilingContext *bsaContext)
{
    constexpr int64_t DEFAULT_BLOCK_SHAPE = 128;
    blockShapeX_ = DEFAULT_BLOCK_SHAPE;
    blockShapeY_ = DEFAULT_BLOCK_SHAPE;
    const auto *blockSparseMaskTensor = bsaContext->GetOptionalInputTensor(BLOCK_SPARSE_MASK_INDEX);
    const auto *blockShapeTensor = bsaContext->GetOptionalInputTensor(BLOCK_SHAPE_INDEX);
    
    if (blockSparseMaskTensor == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "BlockSparseMask should be provided so far.");
        return ge::GRAPH_FAILED;
    }
    if (blockShapeTensor != nullptr) {
        uint32_t blockShapeElemNum = static_cast<uint32_t>(blockShapeTensor->GetShapeSize());
        if (blockShapeElemNum != 2) {
            OP_LOGE(bsaContext->GetNodeName(), "BlockShape elem num must be 2.");
            return ge::GRAPH_FAILED;
        }
        blockShapeList = blockShapeTensor->GetData<int64_t>();
        if (blockShapeList != nullptr) {
            blockShapeX_ = blockShapeList[0];
            blockShapeY_ = blockShapeList[1];
        }
    }
    if (CheckSparsePattern(bsaContext, DEFAULT_BLOCK_SHAPE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseAttenMask(gert::TilingContext *bsaContext)
{
    const auto *attenMaskTensor = bsaContext->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    if (attenMaskTensor != nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "AttenMask is NOT YET supported.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseBlockTable(gert::TilingContext *bsaContext)
{
    const auto *blockTableTensor = bsaContext->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    if (blockTableTensor != nullptr) {
        OP_LOGE(bsaContext->GetNodeName(),
            "Paged cache is NOT YET supported, therefore blockTable should be nullptr.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseOptionalTensors(gert::TilingContext *bsaContext)
{
    if (ParseSeqlens(bsaContext) != ge::GRAPH_SUCCESS ||
        ParseSparsePattern(bsaContext) != ge::GRAPH_SUCCESS ||
        ParseAttenMask(bsaContext) != ge::GRAPH_SUCCESS ||
        ParseBlockTable(bsaContext)  != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::ParseAttrs(gert::TilingContext *bsaContext)
{
    auto attrs = bsaContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, attrs);
    if (attrs->GetAttrPointer<uint32_t>(NUM_KEY_VALUE_HEADS_INDEX) == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "numKeyValueHeads is null");
        return ge::GRAPH_FAILED;
    }
    kvHeads_ = *attrs->GetAttrPointer<uint32_t>(NUM_KEY_VALUE_HEADS_INDEX);
    
    if (attrs->GetAttrPointer<float>(SCALE_VALUE_INDEX) == nullptr) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    } else {
        scaleValue_ = *attrs->GetAttrPointer<float>(SCALE_VALUE_INDEX);
    }
    
    if (attrs->GetAttrPointer<uint32_t>(MASK_TYPE_INDEX) != nullptr) {
        maskType_ = *attrs->GetAttrPointer<uint32_t>(MASK_TYPE_INDEX);
    }
    
    // 获取innerPrecise参数
    if (attrs->GetAttrPointer<uint32_t>(INNER_PRECISE_INDEX) != nullptr) {
        innerPrecise_ = *attrs->GetAttrPointer<uint32_t>(INNER_PRECISE_INDEX);
    }
    if (socVer_ == SOC_VER_950_CODE) {
        if (innerPrecise_ != BsaInnerCalcPrec::LOW_HIGH_MIXED) {
            OP_LOGE(bsaContext->GetNodeName(), "On chip 950, only innerPrec = 4 is supported, "
                "but got %u.", innerPrecise_);
            return ge::GRAPH_FAILED;
        }
    } else {
        auto qInputDesc = bsaContext->GetInputDesc(QUERY_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(bsaContext, qInputDesc);
        auto dtypeQ = qInputDesc->GetDataType();
        if (innerPrecise_ != BsaInnerCalcPrec::ALL_HIGH && innerPrecise_ != BsaInnerCalcPrec::ALL_LOW) {
            OP_LOGE(bsaContext->GetNodeName(), "On chip 910 & 910_93, only innerPrec = 0 or 1 is supported, "
                "but got %u.", innerPrecise_);
            return ge::GRAPH_FAILED;
        } else if (innerPrecise_ == BsaInnerCalcPrec::ALL_LOW && dtypeQ == ge::DT_BF16) {
            OP_LOGE(bsaContext->GetNodeName(), "On chip 910 & 910_93, when query dtype is bfloat16, "
                "only innerPrec = 0 is supported, but got %u.", innerPrecise_);
            return ge::GRAPH_FAILED;
        }
    }
    // reserved yet non-configurable attrs
    int64_t blockSize = *attrs->GetAttrPointer<int64_t>(BLOCK_SIZE_INDEX);
    if (blockSize != 0) {
        OP_LOGE(bsaContext->GetNodeName(), "Since paged cache is not yet supported, "
                "blocksize must be 0, but got %ld.", blockSize);
        return ge::GRAPH_FAILED;
    }
    int64_t preTokens = *attrs->GetAttrPointer<int64_t>(PRE_TOKENS_INDEX);
    int64_t nextTokens = *attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_INDEX);
    if (preTokens != INF_WINDOW_SIZE_PRE_NEXT || nextTokens != INF_WINDOW_SIZE_PRE_NEXT) {
        OP_LOGE(bsaContext->GetNodeName(), "Since windowed atten mask is not yet supported, "
                "preTokens & nextTokens must be 2147483647, but got %ld, %ld.", preTokens, nextTokens);
        return ge::GRAPH_FAILED;
    }
    auto softmaxLsePtr = attrs->GetAttrPointer<int64_t>(SOFTMAX_LSE_FLAG_INDEX);
    if (softmaxLsePtr == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "Attr softmaxLseFlag is nullptr.");
        return ge::GRAPH_FAILED;
    } else if (*softmaxLsePtr == LSE_OUT) {
        if (socVer_ == SOC_VER_950_CODE) {
            OP_LOGE(bsaContext->GetNodeName(), "Attr softmaxLseFlag must be 0 on chip 950.");
            return ge::GRAPH_FAILED;
        }
        softmaxLseFlag_ = true;
    } else if (*softmaxLsePtr == LSE_NO_OUT) {
        softmaxLseFlag_ = false;
    } else {
        OP_LOGE(bsaContext->GetNodeName(), "Attr softmaxLseFlag must be 0 or 1, but got: %ld.", *softmaxLsePtr);
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

void BSATiling::CalculateBatchTaskSplit(int64_t qSeqlen, uint32_t groupSize,
                                        uint32_t &curTaskNum, uint32_t &curQBlockNum)
{
    uint32_t curQBlockTile = GetQNBlockTile();
    uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQBlockTile);
    uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads_;
    curTaskNum = GetQBlocks(qSeqlen, blockShapeX_) * curQNBlockNum;
    curQBlockNum = CeilDiv(qSeqlen, blockShapeX_) * numHeads_;
}

uint32_t BSATiling::GetCurQSTileNum950(int64_t curQSeqlen)
{
    uint32_t fullXBlockNum = curQSeqlen / blockShapeX_;
    uint32_t tailXBlockSize = curQSeqlen % blockShapeX_;
    uint32_t qSTileNumPerFullXBlock = (blockShapeX_ + qBaseTile_ - 1) / qBaseTile_;
    uint32_t qSTileNumTailXBlock = (tailXBlockSize + qBaseTile_ - 1) / qBaseTile_;
    uint32_t curQSTileNum = qSTileNumPerFullXBlock * fullXBlockNum + qSTileNumTailXBlock;
    return curQSTileNum;
}

void BSATiling::CalcBaseTileTilingParams950()
{
    qBaseTile_ = (blockShapeX_ > TILE_SIZE_128) ? TILE_SIZE_128 : blockShapeX_;
    if (innerPrecise_ == BsaInnerCalcPrec::LOW_HIGH_MIXED && embeddingSize_ <= D_SIZE_128) {
        kvBaseTile_ = TILE_SIZE_512;
    } else {
        kvBaseTile_ = TILE_SIZE_128;
    }
}

void BSATiling::CalcSplitCoreTilingParams950()
{
    CalcBaseTileTilingParams950();
    for (uint32_t bIdx = 0; bIdx < batch_; bIdx++) {
        int64_t curQSeqlen = useUniformQSeqlen_ ? maxQSeqlen_ : qSeqLenList_[bIdx];
        int64_t curKvSeqlen = useUniformKvSeqlen_ ? maxKvSeqlen_ : kvSeqLenList_[bIdx];
        uint32_t curQSTileNum = GetCurQSTileNum950(curQSeqlen);
        uint32_t curBatchTaskNum = curQSTileNum * numHeads_;
        totalTaskNum_ += curBatchTaskNum;
        if (bIdx == 0) {
            firstBatchTaskNum_ = curBatchTaskNum;
        }
        maxQSeqlen_ = (curQSeqlen > maxQSeqlen_) ? curQSeqlen : maxQSeqlen_;
        maxKvSeqlen_ = (curKvSeqlen > maxKvSeqlen_) ? curKvSeqlen : maxKvSeqlen_;
    }
    blockDim_ = aicNum_;
    // mask2idx split core
    xBlockNumAligned_ = (maxQSeqlen_ + blockShapeX_ - 1) / blockShapeX_;
    yBlockNumAligned_ = (maxKvSeqlen_ + blockShapeY_ - 1) / blockShapeY_;
    uint32_t totalRowNumBlockMask = batch_ * numHeads_ * xBlockNumAligned_;
    avgRowPerSubCore_ = (totalRowNumBlockMask + aivNum_ - 1) / aivNum_;
    preActiveSubCoreNum_ = (totalRowNumBlockMask + avgRowPerSubCore_ - 1) / avgRowPerSubCore_;
}

ge::graphStatus BSATiling::CalculateTaskSplit(gert::TilingContext *bsaContext)
{
    // 计算总的Q块数量和最大KV块数量
    totalQBlocks_ = 0;

    if (kvHeads_ == 0) {
        OP_LOGE(bsaContext->GetNodeName(), "kvHeads_ is 0, cannot calculate groupSize");
        return ge::GRAPH_FAILED;
    }
    if (batch_ == 0) {
        OP_LOGE(bsaContext->GetNodeName(), "batch_ is 0 in CalculateTaskSplit");
        return ge::GRAPH_FAILED;
    }
    
    // 根据useUniformQSeqlen_标志位决定分核时使用actualSeqLengths数组还是maxQSeqlen_
    uint32_t groupSize = numHeads_ / kvHeads_;
    
    // 遍历每个batch进行分核计算
    for (auto i = 0; i < batch_; i++) {
        // 根据useUniformQSeqlen_标志位决定使用actualSeqLengths数组还是maxQSeqlen_
        int64_t qSeqlen;
        if (useUniformQSeqlen_) {
            // BNSD格式下actualSeqLengths为nullptr，使用maxQSeqlen_作为统一的qseqlen值
            qSeqlen = static_cast<int64_t>(maxQSeqlen_);
        } else {
            // 使用actualSeqLengths数组（TND格式或BNSD格式但提供了actualSeqLengths）
            if (qSeqLenList_ == nullptr) {
                OP_LOGE(bsaContext->GetNodeName(), "qSeqLenList_ is nullptr, cannot calculate task split");
                return ge::GRAPH_FAILED;
            }
            qSeqlen = qSeqLenList_[i];
        }

        uint32_t curTaskNum = 0;
        uint32_t curQBlockNum = 0;
        CalculateBatchTaskSplit(qSeqlen, groupSize, curTaskNum, curQBlockNum);
        
        if (i == 0) {
            firstBatchTaskNum_ = curTaskNum;
            firstQBlockNum_ = curQBlockNum;
        }
        totalTaskNum_ += curTaskNum;
        totalQBlocks_ += curQBlockNum;
    }
    blockDim_ = std::min(aicNum_, totalTaskNum_);
    return ge::GRAPH_SUCCESS;
}

void BSATiling::CalcWorkspaceTilingParams950(gert::TilingContext *bsaContext)
{
    selectIdxSize_ = (sparsePatternMode_ == SPARSE_PATTERN_MODE_TABLE) ?
 	0 : batch_ * numHeads_ * xBlockNumAligned_ * yBlockNumAligned_ * sizeof(int32_t);
    selectNumIdxSize_ = batch_ * numHeads_ * xBlockNumAligned_ * sizeof(int32_t);
    workSpaceSize_ = libapiSize_ + selectIdxSize_ + selectNumIdxSize_;
    bsaContext->GetWorkspaceSizes(1)[0] = workSpaceSize_;
}

ge::graphStatus BSATiling::CalculateWorkSpace(gert::TilingContext *bsaContext)
{
    if (blockDim_ == 0) {
        OP_LOGE(bsaContext->GetNodeName(), "blockDim is 0");
        return ge::GRAPH_FAILED;
    }

    const auto *blockSparseMaskShape = bsaContext->GetInputShape(BLOCK_SPARSE_MASK_INDEX);
    maxKvBlockNum_ = blockSparseMaskShape->GetStorageShape().GetDim(3);
    maxQBlockNum_ = blockSparseMaskShape->GetStorageShape().GetDim(2);
    selectIdxSize_ = (sparsePatternMode_ == SPARSE_PATTERN_MODE_TABLE) ?
 	        0 : CeilDiv(blockShapeX_, 128) * CeilDiv(maxKvBlockNum_, 32) * 32 * sizeof(uint32_t) * batch_ * numHeads_ * maxQBlockNum_;
    selectNumIdxSize_ = CeilDiv(blockShapeX_, 128) * sizeof(uint32_t) * 32 * batch_ * numHeads_ * maxQBlockNum_;
    int32_t syncSize_ = sizeof(uint32_t) * 256;
    
    mm1OutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    smOnlineOutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
    mm2OutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    updateSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    
    workSpaceSize_ = libapiSize_ + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_ + selectNumIdxSize_ + selectIdxSize_ + syncSize_;
    bsaContext->GetWorkspaceSizes(1)[0] = workSpaceSize_;
    uint32_t totalTaskNumMask = batch_ * numHeads_ * maxQBlockNum_;
    avgRowNumPerSubCore_ = CeilDiv(totalTaskNumMask, blockDim_ * 2);
    preActivateSubCoreNum_ = CeilDiv(totalTaskNumMask, avgRowNumPerSubCore_);
    
    return ge::GRAPH_SUCCESS;
}

void BSATiling::CalcMatmulPhaseL1TileInfo950()
{
    uint32_t qBaseTileAligned128 = (qBaseTile_ + TILE_SIZE_128 - 1) / TILE_SIZE_128 * TILE_SIZE_128;
    uint32_t embeddingSizeAligned128 = (embeddingSize_ + TILE_SIZE_128 - 1) / TILE_SIZE_128 * TILE_SIZE_128;
    uint32_t kvBaseTileAligned128 = (kvBaseTile_ + TILE_SIZE_128 - 1) / TILE_SIZE_128 * TILE_SIZE_128;

    // 基本原则，Q的基块常驻在L1
    mm1L1TileM_ = qBaseTileAligned128;
    mm1L1TileKLeft_ = embeddingSizeAligned128;
    qL1BufNum_ = SOLO_BUF;
    if (embeddingSizeAligned128 == D_SIZE_256) {
        // K矩阵开启2buf，D按256分割，S2按128分割
        // 可以证明，当Q常驻在L1时，无论D和kvBaseTile_为多少，均不会有QK的MTE2重复搬运
        mm1L1TileN_ = TILE_SIZE_128;
        mm1L1TileKRight_ = TILE_SIZE_256;
        kL1BufNum_ = DUO_BUF;
        // V矩阵D按256分割，kvBaseTile_不分割，指令提前于核间同步下发，是否开启db取决于kvBaseTile_的大小
        mm2L1TileN_ = TILE_SIZE_256;
        mm2L1TileKLeft_ = kvBaseTileAligned128;
        vL1BufNum_ = TILE_SIZE_256 / kvBaseTileAligned128;
        // P矩阵在950上会常驻L1，由于基块的prelaunch为2，因此最好有3 buf，以免基块间流水阻塞
        mm2L1TileM_ = qBaseTileAligned128;
        mm2L1TileKRight_ = kvBaseTileAligned128;
        pL1BufNum_ = TRIO_BUF;
    } else if (embeddingSizeAligned128 == D_SIZE_128) {
        // K矩阵开启2buf，D按128分割，S2按512分割
        mm1L1TileN_ = TILE_SIZE_512;
        mm1L1TileKRight_ = TILE_SIZE_128;
        kL1BufNum_ = DUO_BUF;
        // V矩阵开启db，D按128分割，kvBaseTile_不分割，指令同样提前于核间同步下发
        // 如果kvBaseTile_进一步增大，考虑关闭db，使得kvBaseTile_不分割
        mm2L1TileN_ = TILE_SIZE_128;
        mm2L1TileKLeft_ = kvBaseTileAligned128;
        vL1BufNum_ = DUO_BUF;
        // P矩阵在950上会常驻L1，由于基块的prelaunch为2，因此最好有3 buf，以免基块间流水阻塞
        mm2L1TileM_ = qBaseTileAligned128;
        mm2L1TileKRight_ = kvBaseTileAligned128;
        pL1BufNum_ = TRIO_BUF;
    }
}

ge::graphStatus BSATiling::FillTilingData(gert::TilingContext *bsaContext)
{
    if (tilingData_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_batch(batch_);
    tilingData_->set_maxNumBlocksPerBatch(maxNumBlocksPerBatch_);
    tilingData_->set_firstBatchTaskNum(firstBatchTaskNum_);
    tilingData_->set_totalTaskNum(totalTaskNum_);
    tilingData_->set_maskType(maskType_);
    
    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);
    
    tilingData_->set_firstQBlockNum(firstQBlockNum_);
    tilingData_->set_totalQBlocks(totalQBlocks_);
    tilingData_->set_maxKvBlockNum(maxKvBlockNum_);
    tilingData_->set_maxQBlockNum(maxQBlockNum_);
    tilingData_->set_avgRowNumPerSubCore(avgRowNumPerSubCore_);
    tilingData_->set_preActivateSubCoreNum(preActivateSubCoreNum_);
    
    tilingData_->set_kvCacheLayout(static_cast<uint32_t>(kvCacheLayout_));
    tilingData_->set_queryLayout(static_cast<uint32_t>(qInputLayout_));
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_maxKvSeqlen(maxKvSeqlen_);
    tilingData_->set_sparsePatternMode(sparsePatternMode_);
    // BNSD格式下当actualSeqLengths为nullptr时，使用maxQSeqlen和maxKvSeqlen作为统一值
    tilingData_->set_useUniformQSeqlen(useUniformQSeqlen_ ? 1 : 0);
    tilingData_->set_useUniformKvSeqlen(useUniformKvSeqlen_ ? 1 : 0);
    
    // 生成tilingKey（按照开发规范：在tiling层生成）
    uint64_t tilingKey = GenerateTilingKey(bsaContext);
    tilingData_->set_tilingKey(tilingKey);
    bsaContext->SetTilingKey(tilingKey);
    bsaContext->SetBlockDim(blockDim_);
    
    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_selectNumIdxSize(selectNumIdxSize_);
    tilingData_->set_selectIdxSize(selectIdxSize_);
    // fill 950 mask2idx tile info
    tilingData_->BsaMask2IdxTileInfo.set_xBlockNumAligned(xBlockNumAligned_);
    tilingData_->BsaMask2IdxTileInfo.set_yBlockNumAligned(yBlockNumAligned_);
    tilingData_->BsaMask2IdxTileInfo.set_avgRowPerSubCore(avgRowPerSubCore_);
    tilingData_->BsaMask2IdxTileInfo.set_preActiveSubCoreNum(preActiveSubCoreNum_);
    // fill 950 base tile info
    tilingData_->BsaBaseTileInfo.set_qBaseTile(qBaseTile_);
    tilingData_->BsaBaseTileInfo.set_kvBaseTile(kvBaseTile_);
    // fill 950 matmul phase L1 tile info
    tilingData_->BsaMmPhaseL1TileInfo.set_mm1L1TileM(mm1L1TileM_);
    tilingData_->BsaMmPhaseL1TileInfo.set_mm1L1TileN(mm1L1TileN_);
    tilingData_->BsaMmPhaseL1TileInfo.set_mm1L1TileKLeft(mm1L1TileKLeft_);
    tilingData_->BsaMmPhaseL1TileInfo.set_mm1L1TileKRight(mm1L1TileKRight_);
    tilingData_->BsaMmPhaseL1TileInfo.set_mm2L1TileM(mm2L1TileM_);
    tilingData_->BsaMmPhaseL1TileInfo.set_mm2L1TileN(mm2L1TileN_);
    tilingData_->BsaMmPhaseL1TileInfo.set_mm2L1TileKLeft(mm2L1TileKLeft_);
    tilingData_->BsaMmPhaseL1TileInfo.set_mm2L1TileKRight(mm2L1TileKRight_);
    tilingData_->BsaMmPhaseL1TileInfo.set_qL1BufNum(qL1BufNum_);
    tilingData_->BsaMmPhaseL1TileInfo.set_kL1BufNum(kL1BufNum_);
    tilingData_->BsaMmPhaseL1TileInfo.set_vL1BufNum(vL1BufNum_);
    tilingData_->BsaMmPhaseL1TileInfo.set_pL1BufNum(pL1BufNum_);
    return ge::GRAPH_SUCCESS;
}

uint64_t BSATiling::GenerateTilingKey(gert::TilingContext *bsaContext)
{
    /**
     * 64位整数，使用十进制位域表示：
     * AAAABBBBCCCCDDDDEEEE
     * - 位0-1:   Q Layout（个位）      2=TND, 3=BNSD
     * - 位2-4:   Mask Type（千位）    0=NoMask, 3=CausalMask
     * - 位5-7:   Softmax Precision（十万位） 0=Float, 1=Half
     * - 位8-10:  PagedCache Flag（千万位）  0=NoCache, 1=WithCache
     * - 位11-13: KV Layout（十亿位）   00=TND, 20=BNSD
     * - 位14-15: Data Type（百亿位）  00=FP16, 22=BF16
     * - 位16-18: Operator Category（千万亿位） 900=EagleQuantBlockSparseAttention910, 905=EagleQuantBlockSparseAttention950
     * 
     * 示例：
     * - FP16, TND, TND, NoCache, Half, NoMask = 9000000030100002
     * - FP16, TND, TND, NoCache, Float, NoMask = 9000000030000002
     */
    
    uint64_t tilingKey = 9000000000000000ULL;  // RFA基础值（Operator Category = 900）
    if (socVer_ == SOC_VER_950_CODE) {
        tilingKey = 9050000000000000ULL;
    }
    
    // [位14-15] Data Type（百亿位）
    if (dataType_ == ge::DT_FLOAT16) {
        tilingKey += 0;  // 00 for FP16
    } else if (dataType_ == ge::DT_BF16) {
        tilingKey += 22220ULL;  // 22 for BF16 -> 9000000030000002 + 22220 = 9000000030022222
    } else if (dataType_ == ge::DT_INT8) {
        auto oOutputDesc = bsaContext->GetOutputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(bsaContext, oOutputDesc);
        auto dtypeO = oOutputDesc->GetDataType();
        if (dtypeO == ge::DT_FLOAT16) {
            tilingKey += 44440ULL;  // 11 for INT8 output with FP16
        } else if (dtypeO == ge::DT_BF16) {
            tilingKey += 55550ULL;  // 33 for INT8 output with BF16
        }
    }
    
    // [位11-13] KV Layout（十亿位）
    if (kvCacheLayout_ == RFAKvCacheLayout::TND) {
        tilingKey += 30000000ULL;  // 00 for TND
    } else if (kvCacheLayout_ == RFAKvCacheLayout::BNSD) {
        tilingKey += 50000000ULL;  // 20 for BNSD
    }
    
    // [位8-10] PagedCache Flag（千万位）
    bool hasPagedCache = (bsaContext->GetOptionalInputTensor(BLOCK_TABLE_INDEX) != nullptr);
    if (hasPagedCache) {
        tilingKey += 1000000ULL;  // 1 for WithCache
    }
    
    // [位5-7] Softmax Precision（十万位）
    if (innerPrecise_ == 1) {
        tilingKey += 100000ULL;  // 1 for Half (FP16) Softmax
    } else if (innerPrecise_ == 4) {
        tilingKey += 400000ULL; // 4 for low prec online softmax & high prec rescale O
    }
    // innerPrecise_ == 0: 0 for Float Softmax（默认值）
    
    // [位2-4] Mask Type（千位）
    if (maskType_ == 3) {  // Causal mask
        tilingKey += 3000ULL;
    }
    // maskType_ == 0: 0 for NoMask（默认值）
    
    // [位0-1] Q Layout（个位）
    if (qInputLayout_ == RFAQInputLayout::TND_Q) {
        tilingKey += 2;  // 2 for TND
    } else if (qInputLayout_ == RFAQInputLayout::BNSD_Q) {
        tilingKey += 3;  // 3 for BNSD
    }

    // Softmax LSE（亿位）
    if (softmaxLseFlag_) {
        tilingKey += 100000000ULL; // 1 for lse out
    }

    constexpr uint64_t V_DTYPE_OFFSET = 10000000000ULL;  // 10^10
    if (vDataType_ == ge::DT_FLOAT8_E4M3FN || vDataType_ == ge::DT_INT8) {
        tilingKey += V_DTYPE_OFFSET * 1ULL;
    } else if (vDataType_ == ge::DT_FLOAT8_E5M2) {
        tilingKey += V_DTYPE_OFFSET * 2ULL;
    }
    // vDataType 与 Q/K 相同时，加 0，保持兼容
  
    
    return tilingKey;
}

ge::graphStatus BSATiling::GetBsaTiling(gert::TilingContext *bsaContext,
                                         EagleQuantBlockSparseAttentionTilingData &tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(bsaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(bsaContext->GetNodeName(), "GetNpuInfo failed");
        return ret;
    }
    if (GetInputLayout(bsaContext) != ge::GRAPH_SUCCESS ||
        ParseAttrs(bsaContext) != ge::GRAPH_SUCCESS ||
        ParseRequiredTensors(bsaContext) != ge::GRAPH_SUCCESS ||
        ParseOptionalTensors(bsaContext) != ge::GRAPH_SUCCESS) {
        ret = ge::GRAPH_FAILED;
        return ret;
    }
    // 校验TND格式下qseqlen和kvseqlen之和是否分别等于Q和KV的T
    ret = ValidateTNDSeqlenSum(bsaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(bsaContext->GetNodeName(), "ValidateTNDSeqlenSum failed");
        return ret;
    }
    
    if (socVer_ == SOC_VER_950_CODE) {
        CalcSplitCoreTilingParams950();
        CalcMatmulPhaseL1TileInfo950();
        CalcWorkspaceTilingParams950(bsaContext);
    } else {
        ret = CalculateTaskSplit(bsaContext);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(bsaContext->GetNodeName(), "CalculateTaskSplit failed");
            return ret;
        }
        ret = CalculateWorkSpace(bsaContext);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(bsaContext->GetNodeName(), "CalculateWorkSpace failed");
            return ret;
        }
    }

    ret = FillTilingData(bsaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(bsaContext->GetNodeName(), "FillTilingData failed");
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSATiling::BsaSetTilingData(gert::TilingContext *context,
    EagleQuantBlockSparseAttentionTilingData &tilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("EagleQuantBlockSparseAttention",
        "RawTilingData got from GE context is nullptr."), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingEagleQuantBlockSparseAttention(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("EagleQuantBlockSparseAttention",
        "Context is nullptr."), return ge::GRAPH_FAILED);
    EagleQuantBlockSparseAttentionTilingData tilingData;
    BSATiling bsaTiling;
    if (bsaTiling.GetBsaTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        bsaTiling.BsaSetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetBsaTiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForEagleQuantBlockSparseAttention(gert::TilingParseContext* context)
{
    (void) context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EagleQuantBlockSparseAttention)
    .Tiling(TilingEagleQuantBlockSparseAttention)
    .TilingInputsDataDependency({5, 6, 7}, {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .TilingParse<EagleQuantBlockSparseAttentionCompileInfo>(TilingPrepareForEagleQuantBlockSparseAttention); // 向框架注册入口函数;

}  // namespace optiling
