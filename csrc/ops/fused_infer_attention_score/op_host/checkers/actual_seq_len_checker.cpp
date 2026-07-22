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

/*!
 * \file actual_seq_len_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "actual_seq_len_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// single para
ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQDim(const FiaTilingInfo &fiaInfo) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验query的actualSeqLengths的维度
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    uint32_t batchSize = fiaInfo.bSize;
    FiaLayout qLayout = fiaInfo.qLayout;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        // query的layout为TND/NTD时，actualSeqLengthsQ的长度为query的batch值
        OP_CHECK_IF((actualSeqLengthsQDimNum != batchSize),
            OP_LOGE(fiaInfo.opName,
                "The size(%u) of actualSeqLengthQ is not equal to the batchSize(%u) of query. "
                "The size of actualSeqLengthQ must be equal to the batchSize of query when "
                "the layout of query is TND or NTD.",
                actualSeqLengthsQDimNum, batchSize),
            return ge::GRAPH_FAILED);
    } else {
        // query为非TND/NTD，actualSeqLengthsQ的长度为1或大于等于query的batch值
        if (actualSeqLengthsQDimNum != DIM_NUM_1 && actualSeqLengthsQDimNum < batchSize) {
            std::string correctStr =
                "greater than or equal to the batchSize: " + std::to_string(batchSize) + " of query or equal to 1.";
            OP_LOGE_FOR_INVALID_LISTSIZE(fiaInfo.opName, "actual_seq_lengths",
                std::to_string(actualSeqLengthsQDimNum).c_str(), correctStr.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQData(const FiaTilingInfo &fiaInfo) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验query的actualSeqLengthData的数值约束
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    uint32_t batchSize = fiaInfo.bSize;
    FiaLayout qLayout = fiaInfo.qLayout;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        // query的layout为TND/NTD时，其值应递增，且为非负数
        for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            if (actualSeqLengthsQTensor->GetData<int64_t>() == nullptr) {
                return ge::GRAPH_SUCCESS;
            }
            int64_t curSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx];
            // 其值应为递增
            if (bIdx != 0U) {
                int64_t lastSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx - 1];
                OP_CHECK_IF((curSeqLengthData < lastSeqLengthData),
                    OP_LOGE(fiaInfo.opName,
                        "actualSeqLengthsQ[%u](%ld) < actualSeqLengthQ[%u](%ld). "
                        "actualSeqLengthsQ must be increasing when the layout of query is "
                        "TND or NTD.",
                        bIdx, curSeqLengthData, bIdx - 1, lastSeqLengthData),
                    return ge::GRAPH_FAILED);
            }
            // curSeqLengthData应为非负数
            if (curSeqLengthData < 0) {
                std::string valueStr = "actual_seq_lengths[" + std::to_string(bIdx) + "]";
                std::string reason =
                    "The value of actual_seq_lengths[" + std::to_string(bIdx) + "] cannot be less than 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    fiaInfo.opName, valueStr.c_str(), std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        // query的layout为非TND/NTD，其值应不大于Q_S，且为非负数
        int64_t sOfQuery = static_cast<int64_t>(fiaInfo.s1Size);
        uint32_t actualSeqLengthsSize = std::min(actualSeqLengthsQDimNum, batchSize);
        for (uint32_t i = 0; i < actualSeqLengthsSize; i++) {
            if (actualSeqLengthsQTensor->GetData<int64_t>() == nullptr) {
                return ge::GRAPH_SUCCESS;
            }
            int64_t curSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[i];
            // curSeqLengthData应不大于Q_S
            OP_CHECK_IF(curSeqLengthData > sOfQuery,
                OP_LOGE(fiaInfo.opName,
                    "actualSeqLengthsQ[%u](%ld) is larger than Q_S(%ld). The elements of actualSeqLengthsQ should not "
                    "be larger than Q_S when "
                    "the layout of query is not TND/NTD.",
                    i, curSeqLengthData, sOfQuery),
                return ge::GRAPH_FAILED);

            // curSeqLengthData应为非负数
            if (curSeqLengthData < 0) {
                std::string valueStr = "actual_seq_lengths[" + std::to_string(i) + "]";
                std::string reason = "The value of actual_seq_lengths[" + std::to_string(i) + "] cannot be less than 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    fiaInfo.opName, valueStr.c_str(), std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvDim(const FiaTilingInfo &fiaInfo) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验key/value的actualSeqLengths的维度
    auto &actualSeqLengthsKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    uint32_t batchSize = fiaInfo.bSize;
    FiaLayout qLayout = fiaInfo.qLayout;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        // key/value的layout为TND/NTD时，actualSeqLengthsKv的长度为batchSize
        OP_CHECK_IF((actualSeqLengthsKvDimNum != batchSize),
            OP_LOGE(fiaInfo.opName,
                "The size(%u) of actualSeqLengthsKv is not equal to the batchSize(%u). "
                "The size of actualSeqLengthsKv must be equal to the batchSize when "
                "the layout of key/value is TND or NTD.",
                actualSeqLengthsKvDimNum, batchSize),
            return ge::GRAPH_FAILED);
    } else {
        // key/value的layout为非TND/NTD，actualSeqLengthsKv的长度为1或大于等于batchSize
        if (actualSeqLengthsKvDimNum != DIM_NUM_1 && actualSeqLengthsKvDimNum < batchSize) {
            std::string correctStr =
                "greater than or equal to the batchSize: " + std::to_string(batchSize) + " or equal to 1.";
            OP_LOGE_FOR_INVALID_LISTSIZE(fiaInfo.opName, "actual_seq_lengths_kv",
                std::to_string(actualSeqLengthsKvDimNum).c_str(), correctStr.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvData(const FiaTilingInfo &fiaInfo) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验key/value的actualSeqLengthData的数值约束
    auto &actualSeqLengthsKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    int64_t batchSize = fiaInfo.bSize;
    FiaLayout qLayout = fiaInfo.qLayout;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        // key/value的layout为TND或NTD时，非page attention场景时，其值应递增，且为非负数
        for (int64_t bIdx = 0; bIdx < batchSize; bIdx++) {
            if (actualSeqLengthsKvTensor->GetData<int64_t>() == nullptr) {
                return ge::GRAPH_SUCCESS;
            }
            int64_t curSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx];
            // 非page attention场景时，其值应为递增
            if (bIdx != 0) {
                int64_t lastSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx - 1];
                OP_CHECK_IF(
                    (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION && curSeqLengthData < lastSeqLengthData),
                    OP_LOGE(fiaInfo.opName,
                        "actualSeqLengthsKv[%ld](%ld) < actualSeqLengthsKv[%ld](%ld). "
                        "actualSeqLengthsKv must be increasing when the layout of key/value is "
                        "TND or NTD.",
                        bIdx, curSeqLengthData, bIdx - 1, lastSeqLengthData),
                    return ge::GRAPH_FAILED);
            }
            // curSeqLengthData应为非负数
            if (curSeqLengthData < 0) {
                std::string valueStr = "actual_seq_lengths_kv[" + std::to_string(bIdx) + "]";
                std::string reason =
                    "The value of actual_seq_lengths_kv[" + std::to_string(bIdx) + "] cannot be less than 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    fiaInfo.opName, valueStr.c_str(), std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        // key/value的layout为非TND/NTD，其值应不大于KV_S，且为非负数
        int64_t sOfKeyValue = static_cast<int64_t>(fiaInfo.s2Size);
        int64_t actualSeqLengthsSize = std::min(static_cast<int64_t>(actualSeqLengthsKvDimNum), batchSize);
        for (int64_t i = 0; i < actualSeqLengthsSize; i++) {
            if (actualSeqLengthsKvTensor->GetData<int64_t>() == nullptr) {
                return ge::GRAPH_SUCCESS;
            }
            int64_t curSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[i];
            // curSeqLengthData应不大于KV_S
            if (curSeqLengthData > sOfKeyValue) {
                std::string valueStr = "actual_seq_lengths_kv[" + std::to_string(i) + "]";
                std::string reason = "The value of actual_seq_lengths_kv[" + std::to_string(i) +
                    "] cannot be greater than KV_S: " + std::to_string(sOfKeyValue);
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    fiaInfo.opName, valueStr.c_str(), std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            // curSeqLengthData应为非负数
            if (curSeqLengthData < 0) {
                std::string valueStr = "actual_seq_lengths_kv[" + std::to_string(i) + "]";
                std::string reason =
                    "The value of actual_seq_lengths_kv[" + std::to_string(i) + "] cannot be less than 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    fiaInfo.opName, valueStr.c_str(), std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQTNDLastData(const FiaTilingInfo &fiaInfo) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验query的输入为TND/NTD时，actualSeqLengthQ的最后一个元素与T相等
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    auto &queryShape = fiaInfo.opParamInfo.query.shape->GetStorageShape();
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    int64_t actualSeqLengthsQLastData = actualSeqLengthsQTensor->GetData<int64_t>()[actualSeqLengthsQDimNum - 1];
    if (fiaInfo.qLayout == FiaLayout::TND) {
        OP_CHECK_IF(actualSeqLengthsQLastData != queryShape.GetDim(DIM_NUM_0),
            OP_LOGE(fiaInfo.opName,
                "The last element(%ld) of actualSeqLengthsQ is not equal to the T(%ld) of query. "
                "The last element of actualSeqLengthsQ must be equal to the T of query when "
                "the layout of query is TND.",
                actualSeqLengthsQLastData, queryShape.GetDim(DIM_NUM_0)),
            return ge::GRAPH_FAILED);
    }
    if (fiaInfo.qLayout == FiaLayout::NTD) {
        OP_CHECK_IF(actualSeqLengthsQLastData != queryShape.GetDim(DIM_NUM_1),
            OP_LOGE(fiaInfo.opName,
                "The last element(%ld) of actualSeqLengthsQ is not equal to the T(%ld) of query. "
                "The last element of actualSeqLengthsQ must be equal to the T of query when "
                "the layout of query is NTD.",
                actualSeqLengthsQLastData, queryShape.GetDim(DIM_NUM_1)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvTNDLastData(const FiaTilingInfo &fiaInfo) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验key/value的输入为TND/NTD时，actualSeqLengthsKv的最后一个元素与T相等
    auto &actualSeqLengthsKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    auto &keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        // 若使能page attention，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    int64_t actualSeqLengthsKvLastData = actualSeqLengthsKvTensor->GetData<int64_t>()[actualSeqLengthsKvDimNum - 1];
    if (fiaInfo.kvLayout == FiaLayout::TND) {
        OP_CHECK_IF(actualSeqLengthsKvLastData != keyShape.GetDim(DIM_NUM_0),
            OP_LOGE(fiaInfo.opName,
                "The last element(%ld) of actualSeqLengthsKv is not equal to the T(%ld) of "
                "key/value. The last element of actualSeqLengthsKv must be equal to the T of key/value "
                "when the layout of key/value is TND.",
                actualSeqLengthsKvLastData, keyShape.GetDim(DIM_NUM_0)),
            return ge::GRAPH_FAILED);
    }
    if (fiaInfo.kvLayout == FiaLayout::NTD) {
        OP_CHECK_IF(actualSeqLengthsKvLastData != keyShape.GetDim(DIM_NUM_1),
            OP_LOGE(fiaInfo.opName,
                "The last element(%ld) of actualSeqLengthsKv is not equal to the T(%ld) of "
                "key/value. The last element of actualSeqLengthsKv must be equal to the T of key/value "
                "when the layout of key/value is NTD.",
                actualSeqLengthsKvLastData, keyShape.GetDim(DIM_NUM_1)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus ActualSeqLenChecker::CheckExistenceActualSeqLenQ(const FiaTilingInfo &fiaInfo) {
    // 校验actualSeqLenQ的存在性
    FiaLayout qLayout = fiaInfo.qLayout;
    // query的Layout为TND/NTD时，actualSeqLengthsQ必须传入
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        OP_CHECK_IF(actualSeqLengthsQTensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsQ",
                "actualSeqLengthsQ cannot be empty when the layout of query is TND or NTD"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckExistenceActualSeqLenKv(const FiaTilingInfo &fiaInfo) {
    FiaLayout kvLayout = fiaInfo.kvLayout;
    // key、value的layout为TND/NTD时，actualSeqLengthsKv必须传入
    if (kvLayout == FiaLayout::TND || kvLayout == FiaLayout::NTD) {
        OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsKv",
                "actualSeqLengthsKv cannot be empty when the layout of key and value is TND or NTD"),
            return ge::GRAPH_FAILED);
    }
    // PagedAttention场景下，必须传入actualSeqLengthsKv
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsKv",
                "actualSeqLengthsKv cannot be empty when page attention is enabled"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus ActualSeqLenChecker::CheckFeatureAlibi(const FiaTilingInfo &fiaInfo) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 使能alibi pse时，query和key每个batch的seqlength需要相等
    int64_t actualSeqLengthsQData = 0;
    int64_t actualSeqLengthsKvData = 0;
    uint32_t batchSize = fiaInfo.bSize;
    if (fiaInfo.enableAlibiPse) {
        for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            actualSeqLengthsQData = GetActualSeqLengthsQData(fiaInfo, bIdx);
            actualSeqLengthsKvData = GetActualSeqLengthsKvData(fiaInfo, bIdx);
            if (actualSeqLengthsQData != actualSeqLengthsKvData) {
                std::string attrName = "actualSeqLengthsQ[" + std::to_string(bIdx) + "] and actualSeqLengthsKvData[" +
                    std::to_string(bIdx) + "]";
                std::string valueStr =
                    std::to_string(actualSeqLengthsQData) + " and " + std::to_string(actualSeqLengthsKvData);
                std::string reason = attrName + " must be the same when pseType is 2 or 3";
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    fiaInfo.opName, attrName.c_str(), valueStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeatureIFAMLA(const FiaTilingInfo &fiaInfo) {
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    FiaLayout qLayout = fiaInfo.qLayout;
    // IFAMLA全量化场景，仅query的layout为TND/NTD时支持传入actualSeqLengthsQ
    bool enableIFAMLA = (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA && actualSeqLengthsQTensor != nullptr) {
        OP_CHECK_IF((qLayout != FiaLayout::TND) && (qLayout != FiaLayout::NTD),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsQ",
                "actualSeqLengthsQ must be empty in IFA MLA and non-TND/NTD scenarios"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeatureFP8GQAFullquant(const FiaTilingInfo &fiaInfo) {
    if (fiaInfo.fullQuantMode != FiaFullQuantMode::QK_PER_TOKEN_HEAD_V_PER_HEAD) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }

    constexpr int64_t minQuerySeqLen = 1;
    constexpr int64_t maxQuerySeqLen = 65536; // 64K
    constexpr int64_t maxKVSeqLen = 262144; // 256K

    uint32_t batchSize = fiaInfo.bSize;
    for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
        int64_t qSeqLen = GetActualSeqLengthsQData(fiaInfo, bIdx);
        OP_CHECK_IF(qSeqLen < minQuerySeqLen || qSeqLen > maxQuerySeqLen,
            OP_LOGE(fiaInfo.opName,
                "In FP8 GQA fullquant scenario, query sequence length(%ld) should be in range of [%ld, %ld].", qSeqLen,
                minQuerySeqLen, maxQuerySeqLen),
            return ge::GRAPH_FAILED);

        int64_t kvSeqLen = GetActualSeqLengthsKvData(fiaInfo, bIdx);
        OP_CHECK_IF(kvSeqLen < 0 || kvSeqLen > maxKVSeqLen,
            OP_LOGE(fiaInfo.opName,
                "In FP8 GQA fullquant scenario, kv sequence length(%ld) should be in range of [0, %ld].", kvSeqLen,
                maxKVSeqLen),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// general
int64_t ActualSeqLenChecker::GetActualSeqLengthsQData(const FiaTilingInfo &fiaInfo, uint32_t bIdx) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 获取bIdx对应batch的query的seqLength
    int64_t actualSeqLengthsQData = 0;
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (fiaInfo.isAccumQSeq) {
        // actualSeqLengths为累加形式
        if (bIdx == 0U) {
            actualSeqLengthsQData = actualSeqLengthsQTensor->GetData<int64_t>()[0];
        } else {
            actualSeqLengthsQData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx] -
                actualSeqLengthsQTensor->GetData<int64_t>()[bIdx - 1];
        }
    } else {
        // 非累加形式
        if (actualSeqLengthsQTensor != nullptr) {
            int64_t actSeqLenDims = actualSeqLengthsQTensor->GetShapeSize();
            actualSeqLengthsQData = actSeqLenDims > 1 ? actualSeqLengthsQTensor->GetData<int64_t>()[bIdx]
                                                      : actualSeqLengthsQTensor->GetData<int64_t>()[0];
        } else {
            actualSeqLengthsQData = fiaInfo.s1Size;
        }
    }
    return actualSeqLengthsQData;
}

int64_t ActualSeqLenChecker::GetActualSeqLengthsKvData(const FiaTilingInfo &fiaInfo, uint32_t bIdx) {
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 获取bIdx对应batch的key和value的seqLength
    int64_t actualSeqLengthsKvData = 0;
    auto &actualSeqLengthsKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    if (fiaInfo.isAccumKVSeq) {
        // actualSeqLengths为累加形式
        if (bIdx == 0U) {
            actualSeqLengthsKvData = actualSeqLengthsKvTensor->GetData<int64_t>()[0];
        } else {
            actualSeqLengthsKvData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx] -
                actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx - 1];
        }
    } else {
        // 非累加形式
        if (actualSeqLengthsKvTensor != nullptr) {
            int64_t actSeqLenKVDims = actualSeqLengthsKvTensor->GetShapeSize();
            actualSeqLengthsKvData = actSeqLenKVDims > 1 ? actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx]
                                                         : actualSeqLengthsKvTensor->GetData<int64_t>()[0];
        } else {
            actualSeqLengthsKvData = fiaInfo.s2Size;
        }
    }
    return actualSeqLengthsKvData;
}

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != CheckActualSeqLenQDim(fiaInfo) || ge::GRAPH_SUCCESS != CheckActualSeqLenQData(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenKvDim(fiaInfo) || ge::GRAPH_SUCCESS != CheckActualSeqLenKvData(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenQTNDLastData(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenKvTNDLastData(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != CheckExistenceActualSeqLenQ(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckExistenceActualSeqLenKv(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != CheckFeatureAlibi(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureIFAMLA(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckFeatureFP8GQAFullquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) {
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
