/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "eagle_block_sparse_attention_tiling.h"
#include <algorithm>
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
#include "tiling/tiling_base.h" // provides ASCENDC_EXTERN_C

using namespace ge;
using namespace std;

constexpr int NUM_1 = 1;
constexpr int NUM_2 = 2;
constexpr int NUM_64 = 64;
// mxfp4 默认 dstTypeMax:CX 方案 dstTypeMax=0 时默认 log2(6.0),OCP 方案固定 2.0
constexpr float MXFP4_DEFAULT_DST_TYPE_MAX_LOG2 = 2.584962500721156f;
constexpr float MXFP4_OCP_DST_TYPE_MAX = 2.0f;

constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;
constexpr int DIM_3 = 3;
constexpr int DIM_NUM_4 = 4;
constexpr int DIM_NUM_5 = 5;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;
constexpr int TND_DIM_NUM = 3;

constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;
constexpr int BNSD_DIM_NUM = 4;

constexpr int BSND_DIM_B = 0;
constexpr int BSND_DIM_S = 1;
constexpr int BSND_DIM_N = 2;
constexpr int BSND_DIM_D = 3;
constexpr int BSND_DIM_NUM = 4;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int BLOCK_SPARSE_MASK_INDEX = 3;
constexpr int ATTEN_MASK_INDEX = 4;
constexpr int BLOCK_SHAPE_INDEX = 5;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 7;
constexpr int BLOCK_TABLE_INDEX = 8;
constexpr int Q_DEQUANT_SCALE_INDEX = 9;
constexpr int K_DEQUANT_SCALE_INDEX = 10;
constexpr int V_DEQUANT_SCALE_INDEX = 11;

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
// V3 新增:量化参数属性索引(追加在 softmaxLseFlag 之后,与 def.cpp 属性顺序一致)
constexpr int QUANT_MODE_INDEX = 10;
constexpr int DST_TYPE_MAX_INDEX = 11;

constexpr int ATTENTION_OUT_INDEX = 0;

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

constexpr uint32_t TILE_SIZE_16 = 16;
constexpr uint32_t TILE_SIZE_128 = 128;
constexpr uint32_t TILE_SIZE_256 = 256;
constexpr uint32_t D_SIZE_128 = 128;
constexpr uint32_t D_SIZE_256 = 256;

constexpr uint32_t SOLO_BUF = 1;
constexpr uint32_t DUO_BUF = 2;
constexpr uint32_t TRIO_BUF = 3;

static std::unordered_map<std::string, std::string> inputLayoutMapQ2Kv = {{"TND", "TND"}, {"BNSD", "BNSD"}, {"BSND", "BSND"}};

enum QuantMode : int64_t {
    NO_QUANT = 0,
    FP8_QUANT = 1,
    MXFP4_OCP_QUANT = 2,
    MXFP4_CX_QUANT = 3,
};

// 判断是否为 mxfp4 量化(OCP 或 CX)
static bool IsMxfp4Quant(int64_t qm) { return qm == MXFP4_OCP_QUANT || qm == MXFP4_CX_QUANT; }

static std::string DataTypeToString(ge::DataType dataType)
{
    static const ::unordered_map<ge::DataType, std::string> DataTypeToStringMap = {
        {ge::DT_FLOAT8_E4M3FN, "ge::DT_FLOAT8_E4M3FN"},
        {ge::DT_FLOAT8_E8M0, "ge::DT_FLOAT8_E8M0"},
        {ge::DT_FLOAT4_E2M1, "ge::DT_FLOAT4_E2M1"},
        {ge::DT_BF16, "ge::DT_BF16"},
        {ge::DT_FLOAT16, "ge::DT_FLOAT16"},
        {ge::DT_FLOAT, "ge::DT_FLOAT"},
        {ge::DT_INT8, "ge::DT_INT8"},
        {ge::DT_INT16, "ge::DT_INT16"},
        {ge::DT_INT32, "ge::DT_INT32"},
        {ge::DT_INT64, "ge::DT_INT64"},
        {ge::DT_BOOL, "ge::DT_BOOL"},
        {ge::DT_DOUBLE, "ge::DT_DOUBLE"}};

    const auto it = DataTypeToStringMap.find(dataType);
    if (it != DataTypeToStringMap.end()) {
        return it->second;
    }
    OP_LOGE("EagleBlockSparseAttention", "Data type %d is not supported", dataType);
    return "UNDEFINED";
}

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

ge::graphStatus EagleBsaTiling::GetNpuInfo(gert::TilingContext *bsaContext)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(bsaContext->GetPlatformInfo());

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    socVer_ = static_cast<uint32_t>(ascendcPlatform.GetSocVersion());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckFormat(ge::Format format)
{
    if (format != ge::FORMAT_ND && format != ge::FORMAT_NCL && format != ge::FORMAT_NCHW &&
        format != ge::FORMAT_NCDHW) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::CheckNDFormat(gert::TilingContext *bsaContext)
{
    auto queryFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(bsaContext->GetInputDesc(QUERY_INDEX)->GetStorageFormat()));
    OP_CHECK_IF(CheckFormat(queryFormat) != ge::GRAPH_SUCCESS,
                OP_LOGE(bsaContext->GetNodeName(), "query check format failed"), return ge::GRAPH_FAILED);
    auto keyFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(bsaContext->GetInputDesc(KEY_INDEX)->GetStorageFormat()));
    OP_CHECK_IF(CheckFormat(keyFormat) != ge::GRAPH_SUCCESS,
                OP_LOGE(bsaContext->GetNodeName(), "key check format failed"), return ge::GRAPH_FAILED);
    auto valueFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(bsaContext->GetInputDesc(VALUE_INDEX)->GetStorageFormat()));
    OP_CHECK_IF(CheckFormat(valueFormat) != ge::GRAPH_SUCCESS,
                OP_LOGE(bsaContext->GetNodeName(), "value check format failed"), return ge::GRAPH_FAILED);

    const auto *blockSparseMaskTensor = bsaContext->GetOptionalInputTensor(BLOCK_SPARSE_MASK_INDEX);
    if (blockSparseMaskTensor != nullptr) {
        auto blockSparseMaskFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(blockSparseMaskTensor->GetStorageFormat()));
        OP_CHECK_IF(CheckFormat(blockSparseMaskFormat) != ge::GRAPH_SUCCESS,
                    OP_LOGE(bsaContext->GetNodeName(), "blockSparseMask check format failed"), return ge::GRAPH_FAILED);
    }

    const auto *attenMaskTensor = bsaContext->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    if (attenMaskTensor != nullptr) {
        auto attenMaskFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(attenMaskTensor->GetStorageFormat()));
        OP_CHECK_IF(CheckFormat(attenMaskFormat) != ge::GRAPH_SUCCESS,
                    OP_LOGE(bsaContext->GetNodeName(), "attenMask check format failed"), return ge::GRAPH_FAILED);
    }

    const auto *blockTableTensor = bsaContext->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    if (blockTableTensor != nullptr) {
        auto blockTableFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(blockTableTensor->GetStorageFormat()));
        OP_CHECK_IF(CheckFormat(blockTableFormat) != ge::GRAPH_SUCCESS,
                    OP_LOGE(bsaContext->GetNodeName(), "blockTable check format failed"), return ge::GRAPH_FAILED);
    }

    const auto *qDequantScaleTensor = bsaContext->GetOptionalInputTensor(Q_DEQUANT_SCALE_INDEX);
    if (qDequantScaleTensor != nullptr) {
        auto qDequantScaleFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(qDequantScaleTensor->GetStorageFormat()));
        OP_CHECK_IF(CheckFormat(qDequantScaleFormat) != ge::GRAPH_SUCCESS,
                    OP_LOGE(bsaContext->GetNodeName(), "qDequantScale check format failed"), return ge::GRAPH_FAILED);
    }
    const auto *kDequantScaleTensor = bsaContext->GetOptionalInputTensor(K_DEQUANT_SCALE_INDEX);
    if (kDequantScaleTensor != nullptr) {
        auto kDequantScaleFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(kDequantScaleTensor->GetStorageFormat()));
        OP_CHECK_IF(CheckFormat(kDequantScaleFormat) != ge::GRAPH_SUCCESS,
                    OP_LOGE(bsaContext->GetNodeName(), "kDequantScale check format failed"), return ge::GRAPH_FAILED);
    }
    const auto *vDequantScaleTensor = bsaContext->GetOptionalInputTensor(V_DEQUANT_SCALE_INDEX);
    if (vDequantScaleTensor != nullptr) {
        auto vDequantScaleFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(vDequantScaleTensor->GetStorageFormat()));
        OP_CHECK_IF(CheckFormat(vDequantScaleFormat) != ge::GRAPH_SUCCESS,
                    OP_LOGE(bsaContext->GetNodeName(), "vDequantScale check format failed"), return ge::GRAPH_FAILED);
    }

    auto attentionOutFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(bsaContext->GetOutputDesc(ATTENTION_OUT_INDEX)->GetStorageFormat()));
    OP_CHECK_IF(CheckFormat(attentionOutFormat) != ge::GRAPH_SUCCESS,
                OP_LOGE(bsaContext->GetNodeName(), "attentionOut check format failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ValidateTNDSeqlenSum(gert::TilingContext *bsaContext)
{
    // 只在TND格式时进行校验
    if (qInputLayout_ != BSAQInputLayout::TND_Q || kvCacheLayout_ != BSAKvCacheLayout::TND_KV) {
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
                "TND format validation failed: sum of qseqlen across all batches (%ld) != Q T (%ld)", sumQSeqlen,
                totalTokensT_);
        return ge::GRAPH_FAILED;
    }

    // 校验kvseqlen之和是否等于KV的T
    if (sumKvSeqlen != totalTokensKv_) {
        OP_LOGE(bsaContext->GetNodeName(),
                "TND format validation failed: sum of kvseqlen across all batches (%ld) != KV T (%ld)", sumKvSeqlen,
                totalTokensKv_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::GetInputLayout(gert::TilingContext *bsaContext)
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
        OP_LOGE(bsaContext->GetNodeName(), "The inputLayout attrs only support TND/BNSD/BSND.");
        return ge::GRAPH_FAILED;
    } else if (it->second != kvLayout) {
        OP_LOGE(bsaContext->GetNodeName(), "QInputLayout and kvInputLayout must be the same.");
        return ge::GRAPH_FAILED;
    } else if (it->first == "TND") {
        qInputLayout_ = BSAQInputLayout::TND_Q;
        kvCacheLayout_ = BSAKvCacheLayout::TND_KV;
    } else if (it->first == "BNSD") {
        qInputLayout_ = BSAQInputLayout::BNSD_Q;
        kvCacheLayout_ = BSAKvCacheLayout::BNSD_KV;
    } else if (it->first == "BSND") {
        if (socVer_ != SOC_VER_950_CODE) {
            OP_LOGE(bsaContext->GetNodeName(), "Only the chip 950 supports BSND inputLayout.");
            return ge::GRAPH_FAILED;
        }
        qInputLayout_ = BSAQInputLayout::BSND_Q;
        kvCacheLayout_ = BSAKvCacheLayout::BSND_KV;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::CheckQKVDtype(gert::TilingContext *bsaContext)
{
    auto qInputDesc = bsaContext->GetInputDesc(QUERY_INDEX);
    auto kInputDesc = bsaContext->GetInputDesc(KEY_INDEX);
    auto vInputDesc = bsaContext->GetInputDesc(VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, qInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, kInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, vInputDesc);
    dataType_ = qInputDesc->GetDataType();
    auto kDataType = kInputDesc->GetDataType();
    auto vDataType = vInputDesc->GetDataType();

    if (socVer_ == SOC_VER_950_CODE) {
        if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16 && dataType_ != ge::DT_FLOAT8_E4M3FN &&
            dataType_ != ge::DT_FLOAT4_E2M1) {
            OP_LOGE(bsaContext->GetNodeName(), "On chip 950, the supported dtype of query/key/value is float16, "
                                               "bfloat16, float8_e4m3fn or float4_e2m1.");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16) {
            OP_LOGE(bsaContext->GetNodeName(),
                    "On chip 910 & 910_93, the supported dtype of query/key/value is float16 or bfloat16.");
            return ge::GRAPH_FAILED;
        }
    }

    if (dataType_ != kDataType || dataType_ != vDataType) {
        OP_LOGE(bsaContext->GetNodeName(), "Tensor query/key/value must have consistent dtype with each other.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::CheckQKVDimVal(gert::TilingContext *bsaContext, uint32_t kHeads, uint32_t vHeads,
                                          uint32_t kHeadDim, uint32_t vHeadDim)
{
    if (embeddingSize_ != kHeadDim || embeddingSize_ != vHeadDim) {
        OP_LOGE(bsaContext->GetNodeName(),
                "Tensor query/key/value must have consistent headDim with each other, "
                "but got qHeadDim %u, kHeadDim %u, vHeadDim %u.",
                embeddingSize_, kHeadDim, vHeadDim);
        return ge::GRAPH_FAILED;
    }
    if (kvHeads_ != kHeads || kvHeads_ != vHeads) {
        OP_LOGE(bsaContext->GetNodeName(),
                "Tensor key/value must have consistent headNum with each other and attr kvHeads, "
                "but got kHeads %u, vHeads %u, kvHeads(attr) %u.",
                kHeads, vHeads, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    if (numHeads_ % kvHeads_ != 0) {
        OP_LOGE(bsaContext->GetNodeName(),
                "The query's headNum must be a multiple of the key/value's headNum, but now the query's headNum is %u "
                "and the key/value's headNum is %u.",
                numHeads_, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    // temporary regulations of D
    if ((embeddingSize_ != VALID_EMBEDDING_SIZE_128) && (embeddingSize_ != VALID_EMBEDDING_SIZE_64)) {
        OP_LOGE(bsaContext->GetNodeName(), "The supported headDim so far is 64 or 128, but got %u.", embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ParseQKVInTND(gert::TilingContext *bsaContext)
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

ge::graphStatus EagleBsaTiling::ParseQKVInBNSD(gert::TilingContext *bsaContext)
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
        OP_LOGE(bsaContext->GetNodeName(),
                "Tensor query/key/value must have consistent batch with each other, "
                "but got qBatch %u, kBatch %u, vBatch %u.",
                batch_, kBatch, vBatch);
        return ge::GRAPH_FAILED;
    }
    if (CheckQKVDimVal(bsaContext, kHeads, vHeads, embeddingSizeK, embeddingSizeV) != ge::GRAPH_SUCCESS) {
        OP_LOGE(bsaContext->GetNodeName(), "Check query/key/value dim values failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ParseQKVInBSND(gert::TilingContext *bsaContext)
{
    const auto *queryShape = bsaContext->GetInputShape(QUERY_INDEX);
    const auto *keyShape = bsaContext->GetInputShape(KEY_INDEX);
    const auto *valueShape = bsaContext->GetInputShape(VALUE_INDEX);
    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "Query/Key/Value shape is null");
        return ge::GRAPH_FAILED;
    }
    if (queryShape->GetStorageShape().GetDimNum() != BSND_DIM_NUM ||
        keyShape->GetStorageShape().GetDimNum() != BSND_DIM_NUM ||
        valueShape->GetStorageShape().GetDimNum() != BSND_DIM_NUM) {
        OP_LOGE(bsaContext->GetNodeName(), "Tensor query/key/value must have 4 dimensions when layout is 'BSND'.");
        return ge::GRAPH_FAILED;
    }
    batch_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BSND_DIM_B));
    auto kBatch = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BSND_DIM_B));
    auto vBatch = static_cast<uint32_t>(valueShape->GetStorageShape().GetDim(BSND_DIM_B));
    numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BSND_DIM_N));
    auto kHeads = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BSND_DIM_N));
    auto vHeads = static_cast<uint32_t>(valueShape->GetStorageShape().GetDim(BSND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BSND_DIM_D));
    auto embeddingSizeK = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BSND_DIM_D));
    auto embeddingSizeV = static_cast<uint32_t>(valueShape->GetStorageShape().GetDim(BSND_DIM_D));
    maxQSeqlen_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(BSND_DIM_S));
    maxKvSeqlen_ = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(BSND_DIM_S));
    if (batch_ != kBatch || batch_ != vBatch) {
        OP_LOGE(bsaContext->GetNodeName(),
                "Tensor query/key/value must have consistent batch with each other, "
                "but got qBatch %u, kBatch %u, vBatch %u.",
                batch_, kBatch, vBatch);
        return ge::GRAPH_FAILED;
    }
    if (CheckQKVDimVal(bsaContext, kHeads, vHeads, embeddingSizeK, embeddingSizeV) != ge::GRAPH_SUCCESS) {
        OP_LOGE(bsaContext->GetNodeName(), "Check query/key/value dim values failed.");
        return ge::GRAPH_FAILED;
    }
    if (IsMxfp4Quant(quantMode_) &&
        (embeddingSize_ != VALID_EMBEDDING_SIZE_64 && embeddingSize_ != VALID_EMBEDDING_SIZE_128)) {
        OP_LOGE(bsaContext->GetNodeName(),
                "(quantMode:%d)mxfp4 requires embeddingSize to be multiple of 64 or 128, but got %u.", quantMode_,
                embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::CheckAttentionOutDtype(gert::TilingContext *bsaContext)
{
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        attentionOutDataType_ = bsaContext->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
        if (attentionOutDataType_ != ge::DT_FLOAT16 && attentionOutDataType_ != ge::DT_BF16) {
            OP_LOGE(bsaContext->GetNodeName(),
                    "The supported dtype of attentionOut is float16 or bfloat16 when the dtype of query/key/value is "
                    "all float8_e4m3fn, but now it is %s.",
                    DataTypeToString(attentionOutDataType_).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    // V3 新增:mxfp4 (DT_FLOAT4_E2M1) 输入时,attentionOut 支持 float16 或 bfloat16
    if (IsMxfp4Quant(quantMode_)) {
        attentionOutDataType_ = bsaContext->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
        if (attentionOutDataType_ != ge::DT_FLOAT16 && attentionOutDataType_ != ge::DT_BF16) {
            OP_LOGE(bsaContext->GetNodeName(),
                    "The supported dtype of attentionOut is float16 or bfloat16 when the dtype of query/key/value is "
                    "all float4_e2m1 (mxfp4), but now it is %s.",
                    DataTypeToString(attentionOutDataType_).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ParseRequiredTensors(gert::TilingContext *bsaContext)
{
    if (CheckQKVDtype(bsaContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;

    if (qInputLayout_ == BSAQInputLayout::TND_Q) {
        ret = ParseQKVInTND(bsaContext);
    } else if (qInputLayout_ == BSAQInputLayout::BNSD_Q) {
        ret = ParseQKVInBNSD(bsaContext);
    } else if (qInputLayout_ == BSAQInputLayout::BSND_Q) {
        ret = ParseQKVInBSND(bsaContext);
    }

    if (socVer_ == SOC_VER_950_CODE) {
        if (CheckAttentionOutDtype(bsaContext) != GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ret;
}

ge::graphStatus EagleBsaTiling::ParseSeqlensInTND(gert::TilingContext *bsaContext)
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
                "but got batch in actualSeqLengths: %u, batch in actualSeqLengthsKv: %u",
                batch_, batchKvS);
        return ge::GRAPH_FAILED;
    }
    qSeqLenList_ = actualSeqLengths->GetData<int64_t>();
    kvSeqLenList_ = actualSeqLengthsKv->GetData<int64_t>();
    btKvTokens_ = 0; // [TND mxfp4] 重置(防御复用), 累加 sum_b CeilDiv(S2_b, 64) 供 V-scale TND 校验
    for (uint32_t bIdx = 0; bIdx < batch_; bIdx++) {
        maxQSeqlen_ = (qSeqLenList_[bIdx] > maxQSeqlen_) ? qSeqLenList_[bIdx] : maxQSeqlen_;
        maxKvSeqlen_ = (kvSeqLenList_[bIdx] > maxKvSeqlen_) ? kvSeqLenList_[bIdx] : maxKvSeqlen_;
        btKvTokens_ += static_cast<uint32_t>((kvSeqLenList_[bIdx] + NUM_64 - 1) / NUM_64); // CeilDiv(S2_b, 64)
    }
    useUniformQSeqlen_ = false;
    useUniformKvSeqlen_ = false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ParseSeqlensInNonTND(gert::TilingContext *bsaContext)
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

        // 校验BNSD/BSND格式时，每个batch的actualSeqLength须小于等于maxSeqLength
        for (uint32_t i = 0; i < batch_; i++) {
            if (qSeqLenList_[i] > maxQSeqlen_) {
                OP_LOGE(bsaContext->GetNodeName(),
                        "The actualSeqLength of each batch must be less than or equal to the maxSeqLength. The "
                        "maxSeqLength is %u, but the value at index %u of the actualSeqLengths is %ld.",
                        maxQSeqlen_, i, qSeqLenList_[i]);
                return ge::GRAPH_FAILED;
            }
            if (kvSeqLenList_[i] > maxKvSeqlen_) {
                OP_LOGE(bsaContext->GetNodeName(),
                        "The actualSeqLengthKv of each batch must be less or equal to than the maxSeqLengthKv. The "
                        "maxSeqLengthKv is %u, but the value at index %u of the actualSeqLengthsKv is %ld.",
                        maxKvSeqlen_, i, kvSeqLenList_[i]);
                return ge::GRAPH_FAILED;
            }
        }
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

ge::graphStatus EagleBsaTiling::ParseSeqlens(gert::TilingContext *bsaContext)
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (qInputLayout_ == BSAQInputLayout::TND_Q) {
        ret = ParseSeqlensInTND(bsaContext);
    } else if (qInputLayout_ == BSAQInputLayout::BNSD_Q || qInputLayout_ == BSAQInputLayout::BSND_Q) {
        ret = ParseSeqlensInNonTND(bsaContext);
    }
    return ret;
}

ge::graphStatus EagleBsaTiling::CheckSparsePattern(gert::TilingContext *bsaContext, const int64_t defaultShape)
{
    const auto *blockSparseMaskShape = bsaContext->GetOptionalInputShape(BLOCK_SPARSE_MASK_INDEX);
    if (blockShapeX_ <= 0 || blockShapeY_ <= 0) {
        OP_LOGE(bsaContext->GetNodeName(),
                "BlockShape elems must be greater than 0, "
                "but got elem0: %ld, elem1: %ld.",
                blockShapeX_, blockShapeY_);
        return ge::GRAPH_FAILED;
    }
    // Only blockShapeY requires alignment (A950: 16; A910/A910B: 16). blockShapeX has no align constraint.
    if (blockShapeY_ % defaultShape != 0) {
        OP_LOGE(bsaContext->GetNodeName(),
                "BlockShape elem1 must be a multiple of %ld, "
                "but got elem1: %ld.",
                defaultShape, blockShapeY_);
        return ge::GRAPH_FAILED;
    }
    if (blockSparseMaskShape->GetStorageShape().GetDimNum() != 4) {
        OP_LOGE(bsaContext->GetNodeName(), "BlockSparseMask must have 4 dims.");
        return ge::GRAPH_FAILED;
    }

    uint32_t blockSparseMaskBatch = static_cast<uint32_t>(blockSparseMaskShape->GetStorageShape().GetDim(DIM_0));
    uint32_t blockSparseMaskNumHeads = static_cast<uint32_t>(blockSparseMaskShape->GetStorageShape().GetDim(DIM_1));
    maxQBlockNum_ = static_cast<uint32_t>(blockSparseMaskShape->GetStorageShape().GetDim(DIM_2));
    maxKvBlockNum_ = static_cast<uint32_t>(blockSparseMaskShape->GetStorageShape().GetDim(DIM_3));

    uint32_t expectedMaxQBlockNum = CeilDiv(maxQSeqlen_, blockShapeX_);
    uint32_t expectedMaxKVBlockNum = CeilDiv(maxKvSeqlen_, blockShapeY_);
    std::string shapeDescription = "(batch, numHeads, maxQBlockNum, maxKVBlockNum)";

    if (blockSparseMaskBatch != batch_ || blockSparseMaskNumHeads != numHeads_ ||
        maxQBlockNum_ != expectedMaxQBlockNum || maxKvBlockNum_ != expectedMaxKVBlockNum) {
        OP_LOGE(bsaContext->GetNodeName(),
                "The shape of blockSparseMask must be consistent with %s. The expected shape is (%u, %u, %u, %u), but "
                "the current shape is (%u, %u, %u, %u).",
                shapeDescription.c_str(), batch_, numHeads_, expectedMaxQBlockNum, expectedMaxKVBlockNum,
                blockSparseMaskBatch, blockSparseMaskNumHeads, maxQBlockNum_, maxKvBlockNum_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ParseSparsePattern(gert::TilingContext *bsaContext)
{
    constexpr int64_t MIN_BLOCK_ALIGN = 16;
    constexpr int64_t A2_A3_BLOCK_ALIGN = 16;
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
    const int64_t defaultShape = (socVer_ == SOC_VER_950_CODE) ? MIN_BLOCK_ALIGN : A2_A3_BLOCK_ALIGN;
    if (CheckSparsePattern(bsaContext, defaultShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // V3 新增:mxfp4 要求 blockShapeX 为 64 的倍数
    if (IsMxfp4Quant(quantMode_) && blockShapeX_ % 64 != 0) {
        OP_LOGE(bsaContext->GetNodeName(), "mxfp4 requires blockShapeX to be multiple of 64, but got %u.",
                blockShapeX_);
        return ge::GRAPH_FAILED;
    }
    // V3 新增:mxfp4 要求 blockShapeY 为 64 的倍数
    if (IsMxfp4Quant(quantMode_) && blockShapeY_ % 64 != 0) {
        OP_LOGE(bsaContext->GetNodeName(), "mxfp4 requires blockShapeY to be multiple of 64, but got %u.",
                blockShapeY_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ParseAttenMask(gert::TilingContext *bsaContext)
{
    const auto *attenMaskTensor = bsaContext->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    if (attenMaskTensor != nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "AttenMask is NOT YET supported.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ParseBlockTable(gert::TilingContext *bsaContext)
{
    const auto *blockTableTensor = bsaContext->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    if (blockTableTensor != nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "Paged cache is NOT YET supported, therefore blockTable should be nullptr.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ValidateGenericDequantScale(gert::TilingContext *bsaContext, const int parameterIndex)
{
    std::string parameterName;
    switch (parameterIndex) {
        case Q_DEQUANT_SCALE_INDEX:
            parameterName = "qDequantScale";
            break;
        case K_DEQUANT_SCALE_INDEX:
            parameterName = "kDequantScale";
            break;
        case V_DEQUANT_SCALE_INDEX:
            parameterName = "vDequantScale";
            break;
        default:
            parameterName = "UNDEFINED";
            break;
    }

    const auto *dequantScaleTensor = bsaContext->GetOptionalInputTensor(parameterIndex);
    if (quantMode_ == FP8_QUANT) {
        if (dequantScaleTensor == nullptr) {
            OP_LOGE(bsaContext->GetNodeName(), "Parameter %s must not be nullptr when the dtype is float8_e4m3fn.",
                    parameterName.c_str());
            return ge::GRAPH_FAILED;
        }

        auto dequantScaleDType = bsaContext->GetOptionalInputDesc(parameterIndex)->GetDataType();
        if (dequantScaleDType != ge::DT_FLOAT) {
            OP_LOGE(bsaContext->GetNodeName(), "The supported dtype of %s is float32, but now it is %s.",
                    parameterName.c_str(), DataTypeToString(dequantScaleDType).c_str());
            return ge::GRAPH_FAILED;
        }

        auto dequantScaleShape = bsaContext->GetOptionalInputShape(parameterIndex);
        int64_t dequantScaleDimNum = dequantScaleShape->GetStorageShape().GetDimNum();
        if (dequantScaleDimNum != DIM_NUM_4) {
            OP_LOGE(bsaContext->GetNodeName(), "The expected shape dim of %s is 4, but now it is %ld.",
                    parameterName.c_str(), dequantScaleDimNum);
            return ge::GRAPH_FAILED;
        }

        uint32_t dequantScaleBatch = static_cast<uint32_t>(dequantScaleShape->GetStorageShape().GetDim(DIM_0));
        uint32_t dequantScaleNumHeads = static_cast<uint32_t>(dequantScaleShape->GetStorageShape().GetDim(DIM_1));
        uint32_t dequantScaleBlockNum = static_cast<uint32_t>(dequantScaleShape->GetStorageShape().GetDim(DIM_2));
        uint32_t dequantScaleLastDim = static_cast<uint32_t>(dequantScaleShape->GetStorageShape().GetDim(DIM_3));

        uint32_t specificNumHeads = parameterIndex == Q_DEQUANT_SCALE_INDEX ? numHeads_ : kvHeads_;
        uint32_t specificMaxBlockNum = parameterIndex == Q_DEQUANT_SCALE_INDEX ? maxQBlockNum_ : maxKvBlockNum_;
        std::string shapeDescription = parameterIndex == Q_DEQUANT_SCALE_INDEX ?
                                           "(totalQToken, numHeads, maxQBlockNum, 1)" :
                                           "(totalKVToken, numKVHeads, maxKVBlockNum, 1)";

        if (dequantScaleBatch != batch_ || dequantScaleNumHeads != specificNumHeads ||
            dequantScaleBlockNum != specificMaxBlockNum || dequantScaleLastDim != NUM_1) {
            OP_LOGE(bsaContext->GetNodeName(),
                    "The shape of %s must be consistent with %s. The expected shape is (%u, %u, %u, %d), but the "
                    "current shape is (%u, %u, %u, %u).",
                    parameterName.c_str(), shapeDescription.c_str(), batch_, specificNumHeads, specificMaxBlockNum,
                    NUM_1, dequantScaleBatch, dequantScaleNumHeads, dequantScaleBlockNum, dequantScaleLastDim);
            return ge::GRAPH_FAILED;
        }
    } else if (IsMxfp4Quant(quantMode_)) {
        return ValidateMxfp4DequantScale(bsaContext, parameterIndex);
    } else {
        if (dequantScaleTensor != nullptr) {
            OP_LOGE(bsaContext->GetNodeName(), "Parameter %s must be nullptr when quantMode is 0(NoQuant).",
                    parameterName.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// V3 新增:mxfp4 量化下 Q/K/V scale 形状校验 (仅 BNSD/BSND,TND 暂不校验)
//   QScale:  BNSD=[B,N1,S1,D//64,2]  BSND=[B,S1,N1,D//64,2]
//   KScale:  BNSD=[B,N2,S2,D//64,2]  BSND=[B,S2,N2,D//64,2]
//   VScale:  BNSD=[B,N2,S2//64,D,2]  BSND=[B,S2//64,N2,D,2]  (S2//64 向上取整)
ge::graphStatus EagleBsaTiling::ValidateMxfp4DequantScale(gert::TilingContext *bsaContext, const int parameterIndex)
{
    const auto *scaleShape = bsaContext->GetOptionalInputShape(parameterIndex);
    if (scaleShape == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "(quantMode =2/3) mxfp4 dequantScale (index=%d) must not be nullptr.",
                parameterIndex);
        return ge::GRAPH_FAILED;
    }
    auto &storageShape = scaleShape->GetStorageShape();

    bool isTND = (qInputLayout_ == BSAQInputLayout::TND_Q);
    bool isBNSD = (qInputLayout_ == BSAQInputLayout::BNSD_Q);
    const char *layoutName = isTND ? "TND" : (isBNSD ? "BNSD" : "BSND");
    uint32_t dDiv64 = embeddingSize_ / NUM_64;

    auto checkDim = [&](uint32_t expected, int idx) -> bool {
        uint32_t actual = static_cast<uint32_t>(storageShape.GetDim(idx));
        if (actual != expected) {
            OP_LOGE(bsaContext->GetNodeName(),
                    "(quantMode =2/3) mxfp4 scale dim[%d] mismatch (layout=%s, index=%d): expected %u, got %u.", idx,
                    layoutName, parameterIndex, expected, actual);
            return false;
        }
        return true;
    };

    if (isTND) {
        // [TND mxfp4] 4D scale shapes(无 batch 维, T/BTkv 在 dim0), 对齐 V3 doc:
        //   QScale=[Tq,Nq,D//64,2]  KScale=[Tkv,Nkv,D//64,2]  VScale=[BTkv,Nkv,D,2]
        //   Tq=totalTokensT_, Tkv=totalTokensKv_, BTkv=btKvTokens_(sum_b CeilDiv(S2_b,64), ParseSeqlensInTND 累加)
        if (storageShape.GetDimNum() != DIM_NUM_4) {
            OP_LOGE(bsaContext->GetNodeName(), "(quantMode =2/3) mxfp4 TND scale must be 4D, got %zuD.",
                    storageShape.GetDimNum());
            return ge::GRAPH_FAILED;
        }
        uint32_t expDim0, expDim1, expDim2;
        if (parameterIndex == Q_DEQUANT_SCALE_INDEX) {
            expDim0 = static_cast<uint32_t>(totalTokensT_);
            expDim1 = numHeads_;
            expDim2 = dDiv64;
        } else if (parameterIndex == V_DEQUANT_SCALE_INDEX) {
            expDim0 = btKvTokens_;
            expDim1 = kvHeads_;
            expDim2 = embeddingSize_;
        } else { // K_DEQUANT_SCALE_INDEX
            expDim0 = static_cast<uint32_t>(totalTokensKv_);
            expDim1 = kvHeads_;
            expDim2 = dDiv64;
        }
        if (!checkDim(expDim0, 0) || !checkDim(expDim1, 1) || !checkDim(expDim2, 2) || !checkDim(NUM_2, 3)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // BNSD/BSND: 5D scale shapes (existing)
    if (storageShape.GetDimNum() != DIM_NUM_5) {
        OP_LOGE(bsaContext->GetNodeName(), "(quantMode =2/3) mxfp4 scale must be 5D, got %zuD.",
                storageShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    uint32_t kvsDiv64 = (maxKvSeqlen_ + NUM_64 - 1) / NUM_64; // CeilDiv(maxKVS, 64); BNSD/BSND uniform 用 max
    uint32_t expDim1, expDim2, expDim3;
    if (parameterIndex == Q_DEQUANT_SCALE_INDEX) {
        // QScale: BNSD=[B,QN,QS,D//64,2]  BSND=[B,QS,QN,D//64,2]
        expDim1 = isBNSD ? numHeads_ : maxQSeqlen_;
        expDim2 = isBNSD ? maxQSeqlen_ : numHeads_;
        expDim3 = dDiv64;
    } else if (parameterIndex == V_DEQUANT_SCALE_INDEX) {
        // VScale: BNSD=[B,KVN,CeilDiv(KVS, 64),D,2]  BSND=[B,CeilDiv(KVS, 64),KVN,D,2]
        expDim1 = isBNSD ? kvHeads_ : kvsDiv64;
        expDim2 = isBNSD ? kvsDiv64 : kvHeads_;
        expDim3 = embeddingSize_;
    } else {
        // KScale: BNSD=[B,KVN,KVS,D//64,2]  BSND=[B,KVS,KVN,D//64,2]
        expDim1 = isBNSD ? kvHeads_ : maxKvSeqlen_;
        expDim2 = isBNSD ? maxKvSeqlen_ : kvHeads_;
        expDim3 = dDiv64;
    }
    if (!checkDim(batch_, 0) || !checkDim(expDim1, 1) || !checkDim(expDim2, 2) || !checkDim(expDim3, 3) ||
        !checkDim(NUM_2, 4)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ValidateQDequantScale(gert::TilingContext *bsaContext)
{
    return ValidateGenericDequantScale(bsaContext, Q_DEQUANT_SCALE_INDEX);
}

ge::graphStatus EagleBsaTiling::ValidateKDequantScale(gert::TilingContext *bsaContext)
{
    return ValidateGenericDequantScale(bsaContext, K_DEQUANT_SCALE_INDEX);
}

ge::graphStatus EagleBsaTiling::ValidateVDequantScale(gert::TilingContext *bsaContext)
{
    return ValidateGenericDequantScale(bsaContext, V_DEQUANT_SCALE_INDEX);
}

ge::graphStatus EagleBsaTiling::ParseOptionalTensors(gert::TilingContext *bsaContext)
{
    if (ParseSeqlens(bsaContext) != ge::GRAPH_SUCCESS || ParseSparsePattern(bsaContext) != ge::GRAPH_SUCCESS ||
        ParseAttenMask(bsaContext) != ge::GRAPH_SUCCESS || ParseBlockTable(bsaContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (socVer_ == SOC_VER_950_CODE) {
        if (ValidateQDequantScale(bsaContext) != ge::GRAPH_SUCCESS ||
            ValidateKDequantScale(bsaContext) != ge::GRAPH_SUCCESS ||
            ValidateVDequantScale(bsaContext) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EagleBsaTiling::ParseAttrs(gert::TilingContext *bsaContext)
{
    auto attrs = bsaContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(bsaContext, attrs);
    if (attrs->GetAttrPointer<uint32_t>(NUM_KEY_VALUE_HEADS_INDEX) == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "numKeyValueHeads is null");
        return ge::GRAPH_FAILED;
    }
    kvHeads_ = *attrs->GetAttrPointer<uint32_t>(NUM_KEY_VALUE_HEADS_INDEX);
    if (kvHeads_ == 0) {
        OP_LOGE(bsaContext->GetNodeName(), "Attr numKeyValueHeads must not be 0.");
        return ge::GRAPH_FAILED;
    }

    if (attrs->GetAttrPointer<float>(SCALE_VALUE_INDEX) == nullptr) {
        OP_LOGW(bsaContext->GetNodeName(), "Attr scaleValue is not provided and will be set as 1.0.");
        scaleValue_ = 1.0f;
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
        if (innerPrecise_ != EagleBsaInnerCalcPrec::LOW_HIGH_MIXED) {
            OP_LOGE(bsaContext->GetNodeName(),
                    "On chip 950, only innerPrec = 4 is supported, "
                    "but got %u.",
                    innerPrecise_);
            return ge::GRAPH_FAILED;
        }
    } else {
        auto qInputDesc = bsaContext->GetInputDesc(QUERY_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(bsaContext, qInputDesc);
        auto dtypeQ = qInputDesc->GetDataType();
        if (innerPrecise_ != EagleBsaInnerCalcPrec::ALL_HIGH && innerPrecise_ != EagleBsaInnerCalcPrec::ALL_LOW) {
            OP_LOGE(bsaContext->GetNodeName(),
                    "On chip 910 & 910_93, only innerPrec = 0 or 1 is supported, "
                    "but got %u.",
                    innerPrecise_);
            return ge::GRAPH_FAILED;
        } else if (innerPrecise_ == EagleBsaInnerCalcPrec::ALL_LOW && dtypeQ == ge::DT_BF16) {
            OP_LOGE(bsaContext->GetNodeName(),
                    "On chip 910 & 910_93, when query dtype is bfloat16, "
                    "only innerPrec = 0 is supported, but got %u.",
                    innerPrecise_);
            return ge::GRAPH_FAILED;
        }
    }
    // reserved yet non-configurable attrs
    int64_t blockSize = *attrs->GetAttrPointer<int64_t>(BLOCK_SIZE_INDEX);
    if (blockSize != 0) {
        OP_LOGE(bsaContext->GetNodeName(),
                "Since paged cache is not yet supported, "
                "blocksize must be 0, but got %ld.",
                blockSize);
        return ge::GRAPH_FAILED;
    }
    int64_t preTokens = *attrs->GetAttrPointer<int64_t>(PRE_TOKENS_INDEX);
    int64_t nextTokens = *attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_INDEX);
    if (preTokens != INF_WINDOW_SIZE_PRE_NEXT || nextTokens != INF_WINDOW_SIZE_PRE_NEXT) {
        OP_LOGE(bsaContext->GetNodeName(),
                "Since windowed atten mask is not yet supported, "
                "preTokens & nextTokens must be 2147483647, but got %ld, %ld.",
                preTokens, nextTokens);
        return ge::GRAPH_FAILED;
    }
    auto softmaxLsePtr = attrs->GetAttrPointer<int64_t>(SOFTMAX_LSE_FLAG_INDEX);
    if (softmaxLsePtr == nullptr) {
        OP_LOGE(bsaContext->GetNodeName(), "Attr softmaxLseFlag is nullptr.");
        return ge::GRAPH_FAILED;
    } else if (*softmaxLsePtr == LSE_OUT) {
        softmaxLseFlag_ = true;
    } else if (*softmaxLsePtr == LSE_NO_OUT) {
        softmaxLseFlag_ = false;
    } else {
        OP_LOGE(bsaContext->GetNodeName(), "Attr softmaxLseFlag must be 0 or 1, but got: %ld.", *softmaxLsePtr);
        return ge::GRAPH_FAILED;
    }

    // V3 新增:读取量化参数 quantMode / dstTypeMax
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(QUANT_MODE_INDEX);
    if (quantModePtr == nullptr) {
        quantMode_ = 0; // 默认值,与 def.cpp 中 .Int(0) 一致
    } else {
        quantMode_ = static_cast<uint32_t>(*quantModePtr);
    }
    auto dstTypeMaxPtr = attrs->GetAttrPointer<float>(DST_TYPE_MAX_INDEX);
    if (dstTypeMaxPtr == nullptr) {
        dstTypeMax_ = 0.0f; // 默认值,与 def.cpp 中 .Float(0) 一致
    } else {
        dstTypeMax_ = *dstTypeMaxPtr;
    }
    if (quantMode_ == MXFP4_CX_QUANT) {
        if (dstTypeMax_ == 0.0f) {
            dstTypeMax_ = MXFP4_DEFAULT_DST_TYPE_MAX_LOG2;
        } else {
            dstTypeMax_ = std::log2(dstTypeMax_);
        }
        // 派生量:device 侧无法调用 host 库函数,host 侧预算好
        // dstTypeMax_ 此处已是 log2 域(CX) 或固定值(OCP)
        // 负号在 kernel 侧加,此处只传原值
        log2Cx_ = dstTypeMax_;
        log2CxCeil_ = std::ceil(dstTypeMax_);
    }

    return ge::GRAPH_SUCCESS;
}

void EagleBsaTiling::CalculateBatchTaskSplit(int64_t qSeqlen, uint32_t groupSize, uint32_t &curTaskNum,
                                        uint32_t &curQBlockNum)
{
    uint32_t curQBlockTile = GetQNBlockTile();
    uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQBlockTile);
    uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads_;
    curTaskNum = GetQBlocks(qSeqlen, blockShapeX_) * curQNBlockNum;
    curQBlockNum = CeilDiv(qSeqlen, blockShapeX_) * numHeads_;
}

uint32_t EagleBsaTiling::GetCurQSTileNum950(int64_t curQSeqlen)
{
    uint32_t fullXBlockNum = curQSeqlen / blockShapeX_;
    uint32_t tailXBlockSize = curQSeqlen % blockShapeX_;
    uint32_t qSTileNumPerFullXBlock = (blockShapeX_ + qBaseTile_ - 1) / qBaseTile_;
    uint32_t qSTileNumTailXBlock = (tailXBlockSize + qBaseTile_ - 1) / qBaseTile_;
    uint32_t curQSTileNum = qSTileNumPerFullXBlock * fullXBlockNum + qSTileNumTailXBlock;
    return curQSTileNum;
}

static inline uint32_t GetMaxDiv(uint32_t value, uint32_t maxBound)
{
    if (value == 0 || maxBound == 0) {
        return 1;
    }
    if (maxBound >= value) {
        return value;
    }
    uint32_t result = 1;
    for (uint32_t d = 1; d * d <= value; ++d) {
        if (value % d == 0) {
            if (d <= maxBound) {
                result = std::max(result, d);
            }
            const uint32_t paired = value / d;
            if (paired <= maxBound) {
                result = std::max(result, paired);
            }
        }
    }
    return result;
}

// FP8: pick largest preferred tile that divides blockShapeY (tile <= Y, Y % tile == 0).
static inline uint32_t CalcFp8KvBaseTile(uint32_t blockShapeY, uint32_t preferredTile)
{
    if (blockShapeY % preferredTile == 0) {
        return preferredTile;
    }
    // Mixed path prefers 256; fall back to 128 when Y is a multiple of 128 but not 256.
    if (preferredTile > TILE_SIZE_128 && blockShapeY % TILE_SIZE_128 == 0) {
        return TILE_SIZE_128;
    }
    if (blockShapeY > preferredTile) {
        return GetMaxDiv(blockShapeY, preferredTile);
    }
    return blockShapeY;
}

void EagleBsaTiling::CalcBaseTileTilingParams950()
{
    qBaseTile_ = (blockShapeX_ > TILE_SIZE_128) ? TILE_SIZE_128 : static_cast<uint32_t>(blockShapeX_);
    const bool isMixedPrecision = (innerPrecise_ == EagleBsaInnerCalcPrec::LOW_HIGH_MIXED && embeddingSize_ <= D_SIZE_128);
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        const uint32_t preferredTile = isMixedPrecision ? TILE_SIZE_256 : TILE_SIZE_128;
        kvBaseTile_ = CalcFp8KvBaseTile(static_cast<uint32_t>(blockShapeY_), preferredTile);
    } else {
        kvBaseTile_ = isMixedPrecision ? TILE_SIZE_256 : TILE_SIZE_128;
    }
}

void EagleBsaTiling::CalcSplitCoreTilingParams950()
{
    CalcBaseTileTilingParams950();
    for (uint32_t bIdx = 0; bIdx < batch_; bIdx++) {
        int64_t curQSeqlen = useUniformQSeqlen_ ? maxQSeqlen_ : qSeqLenList_[bIdx];
        uint32_t curQSTileNum = GetCurQSTileNum950(curQSeqlen);
        uint32_t curBatchTaskNum = curQSTileNum * numHeads_;
        totalTaskNum_ += curBatchTaskNum;
        if (bIdx == 0) {
            firstBatchTaskNum_ = curBatchTaskNum;
        }
    }
    blockDim_ = aicNum_;
    // mask2idx split core
    xBlockNumAligned_ = (maxQSeqlen_ + blockShapeX_ - 1) / blockShapeX_;
    yBlockNumAligned_ = (maxKvSeqlen_ + blockShapeY_ - 1) / blockShapeY_;
    uint32_t totalRowNumBlockMask = batch_ * numHeads_ * xBlockNumAligned_;
    avgRowPerSubCore_ = (totalRowNumBlockMask + aivNum_ - 1) / aivNum_;
    preActiveSubCoreNum_ = (totalRowNumBlockMask + avgRowPerSubCore_ - 1) / avgRowPerSubCore_;
}

ge::graphStatus EagleBsaTiling::CalculateTaskSplit(gert::TilingContext *bsaContext)
{
    // 计算总的Q块数量和最大KV块数量
    totalQBlocks_ = 0;

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
            // BNSD/BSND格式下actualSeqLengths为nullptr，使用maxQSeqlen_作为统一的qseqlen值
            qSeqlen = static_cast<int64_t>(maxQSeqlen_);
        } else {
            // 使用actualSeqLengths数组（TND/BNSD/BSND格式但提供了actualSeqLengths）
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

void EagleBsaTiling::CalcWorkspaceTilingParams950(gert::TilingContext *bsaContext)
{
    selectIdxSize_ =
        static_cast<uint64_t>(batch_) * numHeads_ * xBlockNumAligned_ * yBlockNumAligned_ * sizeof(int32_t);
    selectNumIdxSize_ = static_cast<uint64_t>(batch_) * numHeads_ * xBlockNumAligned_ * sizeof(int32_t);
    workSpaceSize_ = libapiSize_ + selectIdxSize_ + selectNumIdxSize_;
    bsaContext->GetWorkspaceSizes(1)[0] = workSpaceSize_;
}

ge::graphStatus EagleBsaTiling::CalculateWorkSpace(gert::TilingContext *bsaContext)
{
    if (blockDim_ == 0) {
        OP_LOGE(bsaContext->GetNodeName(), "blockDim is 0");
        return ge::GRAPH_FAILED;
    }

    selectIdxSize_ = CeilDiv(blockShapeX_, 128) * CeilDiv(maxKvBlockNum_, 32) * 32 * sizeof(uint32_t) * batch_ *
                     numHeads_ * maxQBlockNum_;
    selectNumIdxSize_ = CeilDiv(blockShapeX_, 128) * sizeof(uint32_t) * 32 * batch_ * numHeads_ * maxQBlockNum_;
    int32_t syncSize_ = sizeof(uint32_t) * 256;

    mm1OutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    smOnlineOutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
    mm2OutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    updateSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;

    workSpaceSize_ = libapiSize_ + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_ + selectNumIdxSize_ +
                     selectIdxSize_ + syncSize_;
    bsaContext->GetWorkspaceSizes(1)[0] = workSpaceSize_;
    uint32_t totalTaskNumMask = batch_ * numHeads_ * maxQBlockNum_;
    avgRowNumPerSubCore_ = CeilDiv(totalTaskNumMask, blockDim_ * 2);
    preActivateSubCoreNum_ = CeilDiv(totalTaskNumMask, avgRowNumPerSubCore_);

    return ge::GRAPH_SUCCESS;
}

void EagleBsaTiling::CalcMatmulPhaseL1TileInfo950()
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
        // K矩阵开启2buf，D按128分割，S2按256分割
        mm1L1TileN_ = TILE_SIZE_256;
        mm1L1TileKRight_ = TILE_SIZE_128;
        kL1BufNum_ = DUO_BUF;
        // V矩阵开启db，D按128分割，kvBaseTile_不分割，指令同样提前于核间同步下发
        // 如果kvBaseTile_进一步增大，考虑关闭db，使得kvBaseTile_不分割
        mm2L1TileN_ = TILE_SIZE_128;
        mm2L1TileKLeft_ = TILE_SIZE_256;
        vL1BufNum_ = DUO_BUF;
        // P矩阵在950上会常驻L1，由于基块的prelaunch为2，因此最好有3 buf，以免基块间流水阻塞
        mm2L1TileM_ = qBaseTileAligned128;
        mm2L1TileKRight_ = kvBaseTileAligned128;
        pL1BufNum_ = TRIO_BUF;
    }
}

ge::graphStatus EagleBsaTiling::FillTilingData(gert::TilingContext *bsaContext)
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
    // 向下取整: 每个核至少处理的 task 数（余数由前 totalTaskNum_ % blockDim_ 个核各多处理一个）
    coreTaskNum_ = (blockDim_ > 0) ? (totalTaskNum_ / blockDim_) : 0;
    tilingData_->set_coreTaskNum(coreTaskNum_);
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

    // BNSD/BSND格式下当actualSeqLengths为nullptr时，使用maxQSeqlen和maxKvSeqlen作为统一值
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
    // V3 新增:填充量化参数
    tilingData_->set_log2Cx(log2Cx_);
    tilingData_->set_log2CxCeil(log2CxCeil_);
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

uint64_t EagleBsaTiling::GenerateTilingKey(gert::TilingContext *bsaContext)
{
    /**
     * 64位整数，使用十进制位域表示：
     * AAAABBBBCCCCDDDDEEEE
     * - 位0-1:   Q Layout（个位）      2=TND, 3=BNSD, 4=BSND
     * - 位2-4:   Mask Type（千位）    0=NoMask, 3=CausalMask
     * - 位5-7:   Softmax Precision（十万位） 0=Float, 1=Half
     * - 位8-10:  PagedCache Flag（千万位）  0=NoCache, 1=WithCache
     * - 位11-13: KV Layout（十亿位）   30=TND, 50=BNSD, 60=BSND
     * - 位14-15: Data Type（百亿位）  00=FP16, 22=BF16
     * - 位16-18: Operator Category（千万亿位） 900=EagleBlockSparseAttention910, 905=EagleBlockSparseAttention950
     *
     * 示例：
     * - FP16, TND, TND, NoCache, Half, NoMask = 9000000030100002
     * - FP16, TND, TND, NoCache, Float, NoMask = 9000000030000002
     */

    uint64_t tilingKey = 9000000000000000ULL; // BSA基础值（Operator Category = 900）
    if (socVer_ == SOC_VER_950_CODE) {
        tilingKey = 9050000000000000ULL;
    }

    // [位14-15] Data Type（百亿位）
    if (dataType_ == ge::DT_FLOAT16) {
        tilingKey += 0; // 00 for FP16
    } else if (dataType_ == ge::DT_BF16) {
        tilingKey += 22220ULL; // 22 for BF16 -> 9000000030000002 + 22220 = 9000000030022222
    } else if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        if (attentionOutDataType_ == ge::DT_FLOAT16) {
            tilingKey += 10;
        } else if (attentionOutDataType_ == ge::DT_BF16) {
            tilingKey += 20;
        }
    }
    if (quantMode_ == MXFP4_OCP_QUANT) {
        if (attentionOutDataType_ == ge::DT_FLOAT16) {
            tilingKey += 31;
        } else {
            tilingKey += 30;
        }
    } else if (quantMode_ == MXFP4_CX_QUANT) {
        if (attentionOutDataType_ == ge::DT_FLOAT16) {
            tilingKey += 41;
        } else {
            tilingKey += 40;
        }
    }

    // [位11-13] KV Layout（十亿位）
    if (kvCacheLayout_ == BSAKvCacheLayout::TND_KV) {
        tilingKey += 30000000ULL; // 30 for TND
    } else if (kvCacheLayout_ == BSAKvCacheLayout::BNSD_KV) {
        tilingKey += 50000000ULL; // 50 for BNSD
    } else if (kvCacheLayout_ == BSAKvCacheLayout::BSND_KV) {
        tilingKey += 60000000ULL; // 60 for BSND
    }

    // [位8-10] PagedCache Flag（千万位）
    bool hasPagedCache = (bsaContext->GetOptionalInputTensor(BLOCK_TABLE_INDEX) != nullptr);
    if (hasPagedCache) {
        tilingKey += 1000000ULL; // 1 for WithCache
    }

    // [位5-7] Softmax Precision（十万位）
    if (innerPrecise_ == 1) {
        tilingKey += 100000ULL; // 1 for Half (FP16) Softmax
    } else if (innerPrecise_ == 4) {
        tilingKey += 400000ULL; // 4 for low prec online softmax & high prec rescale O
    }
    // innerPrecise_ == 0: 0 for Float Softmax（默认值）

    // [位2-4] Mask Type（千位）
    if (maskType_ == 3) { // Causal mask
        tilingKey += 3000ULL;
    }
    // maskType_ == 0: 0 for NoMask（默认值）

    // [位0-1] Q Layout（个位）
    if (qInputLayout_ == BSAQInputLayout::TND_Q) {
        tilingKey += 2; // 2 for TND
    } else if (qInputLayout_ == BSAQInputLayout::BNSD_Q) {
        tilingKey += 3; // 3 for BNSD
    } else if (qInputLayout_ == BSAQInputLayout::BSND_Q) {
        tilingKey += 4; // 4 for BSND
    }

    // Softmax LSE（亿位）
    if (softmaxLseFlag_) {
        tilingKey += 100000000ULL; // 1 for lse out
    }

    return tilingKey;
}

ge::graphStatus EagleBsaTiling::GetBsaTiling(gert::TilingContext *bsaContext, EagleBlockSparseAttentionTilingData &tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(bsaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(bsaContext->GetNodeName(), "GetNpuInfo failed");
        return ret;
    }
    if (socVer_ == SOC_VER_950_CODE) {
        ret = CheckNDFormat(bsaContext);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(bsaContext->GetNodeName(), "CheckNDFormat failed");
            return ret;
        }
    }
    if (GetInputLayout(bsaContext) != ge::GRAPH_SUCCESS || ParseAttrs(bsaContext) != ge::GRAPH_SUCCESS ||
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

ge::graphStatus EagleBsaTiling::BsaSetTilingData(gert::TilingContext *context, EagleBlockSparseAttentionTilingData &tilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("EagleBlockSparseAttention", "RawTilingData got from GE context is nullptr."),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingEagleBlockSparseAttention(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("EagleBlockSparseAttention", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    EagleBlockSparseAttentionTilingData tilingData;
    EagleBsaTiling bsaTiling;
    if (bsaTiling.GetBsaTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        bsaTiling.BsaSetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetBsaTiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForEagleBlockSparseAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EagleBlockSparseAttention)
    .Tiling(TilingEagleBlockSparseAttention)
    .TilingInputsDataDependency({5, 6, 7},
                                {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .TilingParse<EagleBlockSparseAttentionCompileInfo>(TilingPrepareForEagleBlockSparseAttention); // 向框架注册入口函数;

} // namespace optiling
