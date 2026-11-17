/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 */

#include <algorithm>
#include <cstdint>
#include <string>

#include "err/ops_err.h"
#include "graph/types.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ada_relu2_block_estimate_tiling.h"

namespace {
constexpr size_t QUERY_INDEX = 0;
constexpr size_t KEY_INDEX = 1;
constexpr size_t ACTUAL_SEQ_Q_INDEX = 2;
constexpr size_t ACTUAL_SEQ_KV_INDEX = 3;
constexpr size_t Q_SCALE_INDEX = 6;
constexpr size_t K_SCALE_INDEX = 7;
constexpr size_t INPUT_LAYOUT_ATTR_INDEX = 0;
constexpr size_t STRIDE_ATTR_INDEX = 1;
constexpr size_t SPARSE_SIZE_ATTR_INDEX = 2;
constexpr size_t NUM_HEADS_ATTR_INDEX = 3;
constexpr size_t NUM_KV_HEADS_ATTR_INDEX = 4;
constexpr size_t SCALE_VALUE_ATTR_INDEX = 5;
constexpr size_t THRESHOLD_ATTR_INDEX = 6;

constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t SPARSE_SIZE = 128;
constexpr uint32_t Q_TILE_SIZE = 1024;
constexpr uint32_t Q_LOAD_CHUNK = 256;
constexpr uint32_t KV_TILE_SIZE = 2048;
constexpr uint32_t M_L1_SLICE_NUM = 2;
constexpr uint32_t SCORE_ROWS = Q_TILE_SIZE / SPARSE_SIZE;

constexpr uint32_t CeilDiv(uint32_t value, uint32_t divisor) { return (value + divisor - 1U) / divisor; }

} // namespace

namespace optiling {
static ge::graphStatus AdaRelu2BlockEstimateTilingFunc(gert::TilingContext *context) {
    if (context == nullptr || context->GetInputShape(QUERY_INDEX) == nullptr ||
        context->GetInputShape(KEY_INDEX) == nullptr || context->GetInputDesc(QUERY_INDEX) == nullptr ||
        context->GetInputDesc(KEY_INDEX) == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    const char *layoutValue = attrs->GetAttrPointer<char>(INPUT_LAYOUT_ATTR_INDEX);
    const int32_t *strideValue = attrs->GetAttrPointer<int32_t>(STRIDE_ATTR_INDEX);
    const int32_t *sparseSizeValue = attrs->GetAttrPointer<int32_t>(SPARSE_SIZE_ATTR_INDEX);
    const int32_t *numHeadsValue = attrs->GetAttrPointer<int32_t>(NUM_HEADS_ATTR_INDEX);
    const int32_t *numKvHeadsValue = attrs->GetAttrPointer<int32_t>(NUM_KV_HEADS_ATTR_INDEX);
    const float *scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    const float *thresholdValue = attrs->GetAttrPointer<float>(THRESHOLD_ATTR_INDEX);
    if (layoutValue == nullptr || strideValue == nullptr || sparseSizeValue == nullptr || numHeadsValue == nullptr ||
        numKvHeadsValue == nullptr || scaleValue == nullptr || thresholdValue == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (*strideValue != 8 && *strideValue != 16) {
        OP_LOGE(context->GetNodeName(), "stride must be 8 or 16, but got %d.", *strideValue);
        return ge::GRAPH_FAILED;
    }

    const std::string layout(layoutValue);
    if (layout != "BNSD") {
        OP_LOGE(context->GetNodeName(), "inputLayout %s is reserved but is not supported; only BNSD is enabled.",
            layout.c_str());
        return ge::GRAPH_FAILED;
    }

    const auto &qShape = context->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto &kShape = context->GetInputShape(KEY_INDEX)->GetStorageShape();
    if (qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4) {
        OP_LOGE(context->GetNodeName(), "BNSD query and key must both be rank 4.");
        return ge::GRAPH_FAILED;
    }

    const ge::DataType qType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    const ge::DataType kType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    const uint32_t batchSize = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t headNumQ = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t seqLenQ = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t headDimQ = static_cast<uint32_t>(qShape.GetDim(3));
    const uint32_t batchSizeK = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t headNumKV = static_cast<uint32_t>(kShape.GetDim(1));
    const uint32_t seqLenKV = static_cast<uint32_t>(kShape.GetDim(2));
    const uint32_t headDimK = static_cast<uint32_t>(kShape.GetDim(3));

    const bool isRegular =
        (qType == ge::DT_FLOAT16 && kType == ge::DT_FLOAT16) || (qType == ge::DT_BF16 && kType == ge::DT_BF16);
    const bool isInt8 = qType == ge::DT_INT8 && kType == ge::DT_INT8;
    if ((!isRegular && !isInt8) || batchSize == 0 || batchSize != batchSizeK || headNumQ == 0 || headNumKV == 0 ||
        headNumQ != static_cast<uint32_t>(*numHeadsValue) || headNumKV != static_cast<uint32_t>(*numKvHeadsValue) ||
        headNumQ % headNumKV != 0 || headDimQ != HEAD_DIM || headDimK != HEAD_DIM ||
        *sparseSizeValue != static_cast<int32_t>(SPARSE_SIZE) || *thresholdValue < 0.0F || *thresholdValue > 1.0F) {
        OP_LOGE(context->GetNodeName(),
            "template requires BNSD FP16/FP16, BF16/BF16 or INT8/INT8, D=128, stride in {8,16}, "
            "sparseSize=128, Nq%%Nkv=0 and threshold in [0,1].");
        return ge::GRAPH_FAILED;
    }
    if (isInt8 &&
        (context->GetOptionalInputDesc(Q_SCALE_INDEX) == nullptr ||
            context->GetOptionalInputDesc(K_SCALE_INDEX) == nullptr)) {
        OP_LOGE(context->GetNodeName(), "q_scale and k_scale are required for INT8 query and key.");
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor *actualSeqQ = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    const gert::Tensor *actualSeqKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    // Some aclnn paths materialize an omitted optional IntArray as a zero-length tensor descriptor.
    if (actualSeqQ != nullptr && actualSeqQ->GetShapeSize() == 0) {
        actualSeqQ = nullptr;
    }
    // Treat an empty optional KV-length descriptor exactly like nullptr and use the physical KV length.
    if (actualSeqKV != nullptr && actualSeqKV->GetShapeSize() == 0) {
        actualSeqKV = nullptr;
    }
    const int64_t *qSeqLenList = nullptr;
    const int64_t *kvSeqLenList = nullptr;
    if (actualSeqQ != nullptr) {
        if (static_cast<uint32_t>(actualSeqQ->GetShapeSize()) != batchSize) {
            OP_LOGE(context->GetNodeName(), "actual_seq_lengths size must equal batch size, but got %u and %u.",
                static_cast<uint32_t>(actualSeqQ->GetShapeSize()), batchSize);
            return ge::GRAPH_FAILED;
        }
        qSeqLenList = actualSeqQ->GetData<int64_t>();
        if (qSeqLenList == nullptr) {
            OP_LOGE(context->GetNodeName(), "failed to read actual_seq_lengths in host tiling.");
            return ge::GRAPH_FAILED;
        }
    }
    if (actualSeqKV != nullptr) {
        if (static_cast<uint32_t>(actualSeqKV->GetShapeSize()) != batchSize) {
            OP_LOGE(context->GetNodeName(), "actual_seq_lengths_kv size must equal batch size, but got %u and %u.",
                static_cast<uint32_t>(actualSeqKV->GetShapeSize()), batchSize);
            return ge::GRAPH_FAILED;
        }
        kvSeqLenList = actualSeqKV->GetData<int64_t>();
        if (kvSeqLenList == nullptr) {
            OP_LOGE(context->GetNodeName(), "failed to read actual_seq_lengths_kv in host tiling.");
            return ge::GRAPH_FAILED;
        }
    }

    const uint32_t qBlockNum = CeilDiv(seqLenQ, SPARSE_SIZE);
    const uint32_t qTileNum = CeilDiv(qBlockNum, SCORE_ROWS);
    const uint32_t kvBlockNum = CeilDiv(seqLenKV, SPARSE_SIZE);
    const uint32_t scoredKvBlockNum = kvBlockNum > 0 ? kvBlockNum - 1U : 0U;
    uint64_t totalTaskNum = 0;
    for (uint32_t bIdx = 0; bIdx < batchSize; ++bIdx) {
        const int64_t curQSeqlen = qSeqLenList == nullptr ? static_cast<int64_t>(seqLenQ) : qSeqLenList[bIdx];
        const int64_t curKvSeqlen = kvSeqLenList == nullptr ? static_cast<int64_t>(seqLenKV) : kvSeqLenList[bIdx];
        if (curQSeqlen < 0 || curQSeqlen > static_cast<int64_t>(seqLenQ) || curKvSeqlen < 0 ||
            curKvSeqlen > static_cast<int64_t>(seqLenKV)) {
            OP_LOGE(context->GetNodeName(),
                "actual sequence length of batch %u is out of the physical Q/K sequence range.", bIdx);
            return ge::GRAPH_FAILED;
        }
        const uint32_t curQSTileNum = CeilDiv(static_cast<uint32_t>(curQSeqlen), Q_TILE_SIZE);
        const uint32_t curTotalTaskNum = curQSTileNum * headNumQ;
        totalTaskNum += curTotalTaskNum;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicCores = platform.GetCoreNumAic();
    if (aicCores == 0 || totalTaskNum == 0) {
        OP_LOGE(context->GetNodeName(), "invalid AIC core count or empty task set.");
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum =
        static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(aicCores), totalTaskNum));

    const uint32_t stride = static_cast<uint32_t>(*strideValue);
    const uint32_t mL0TileSize = stride == 8U ? 128U : 64U;
    const uint32_t nL0TileSize = stride == 8U ? 128U : 64U;
    const uint32_t kL0TileSize = (stride == 8U ? 128U : 256U) * (isInt8 ? 2U : 1U);
    const uint32_t kL0LoopNum = stride * HEAD_DIM / kL0TileSize;

    AdaRelu2BlockEstimateTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_seqLenQ(seqLenQ);
    tiling.set_seqLenKV(seqLenKV);
    tiling.set_headNumQ(headNumQ);
    tiling.set_headNumKV(headNumKV);
    tiling.set_headDim(HEAD_DIM);
    tiling.set_actualSeqLengthsSize(actualSeqQ == nullptr ? 0U : actualSeqQ->GetShapeSize());
    tiling.set_actualSeqLengthsKVSize(actualSeqKV == nullptr ? 0U : actualSeqKV->GetShapeSize());
    tiling.set_stride(stride);
    tiling.set_sparseSize(SPARSE_SIZE);
    tiling.set_scaleValue(*scaleValue);
    tiling.set_threshold(*thresholdValue);
    const uint32_t tilingKey = isInt8 ? ADA_RELU2_QBNSD_KVBNSD_INT8_KEY : ADA_RELU2_QBNSD_KVBNSD_KEY;
    tiling.set_inputLayout(tilingKey);
    tiling.set_qTileSize(Q_TILE_SIZE);
    tiling.set_qLoadChunk(Q_LOAD_CHUNK);
    tiling.set_kvTileSize(KV_TILE_SIZE);
    tiling.set_mL0TileSize(mL0TileSize);
    tiling.set_nL0TileSize(nL0TileSize);
    tiling.set_kL0TileSize(kL0TileSize);
    tiling.set_kL0LoopNum(kL0LoopNum);
    tiling.set_mL1SliceNum(M_L1_SLICE_NUM);
    tiling.set_qBlockNum(qBlockNum);
    tiling.set_qTileNum(qTileNum);
    tiling.set_kvBlockNum(kvBlockNum);
    tiling.set_scoredKvBlockNum(scoredKvBlockNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_totalTaskNum(totalTaskNum);
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForAdaRelu2BlockEstimate(gert::TilingParseContext *context) {
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaRelu2BlockEstimate)
    .Tiling(AdaRelu2BlockEstimateTilingFunc)
    .TilingParse<AdaRelu2BlockEstimateCompileInfo>(TilingPrepareForAdaRelu2BlockEstimate);
} // namespace optiling
