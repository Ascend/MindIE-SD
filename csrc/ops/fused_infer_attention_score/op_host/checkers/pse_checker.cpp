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
 * \file pse_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "pse_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

constexpr int32_t PSE_SHIFT_MAX = 1048576; // 2^20
constexpr int32_t PSE_SHIFT_MIN = -1048576; // -2^20
constexpr int64_t PSE_OUTER_MUL_ADD_TYPE = 0;
constexpr int64_t PSE_OUTER_ADD_MUL_TYPE = 1;
constexpr int64_t PSE_INNER_MUL_ADD_TYPE = 2;
constexpr int64_t PSE_INNER_MUL_ADD_SQRT_TYPE = 3;
constexpr int64_t PSE_NONE_TYPE = 9;

// singlepara
ge::graphStatus PSEChecker::CheckPseType(const FiaTilingInfo &fiaInfo) {
    // 校验pseType的合法性
    if (fiaInfo.opParamInfo.pseType == nullptr) {
        // 若pseType不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    int64_t pseType = *fiaInfo.opParamInfo.pseType;
    // pseType支持范围为0
    if (pseType != PSE_OUTER_MUL_ADD_TYPE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            fiaInfo.opName, "pse_type", std::to_string(pseType).c_str(), "The value of pse_type must be 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckPseShiftDataType(const FiaTilingInfo &fiaInfo) {
    // 校验pseShift的数据类型
    if (fiaInfo.opParamInfo.pseType == nullptr || fiaInfo.opParamInfo.pseShift.desc == nullptr) {
        // 若pseType或者pseShift的数据类型不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    int64_t pseType = *fiaInfo.opParamInfo.pseType;
    auto &pseShiftDesc = fiaInfo.opParamInfo.pseShift.desc;
    if (fiaInfo.enableAlibiPse) {
        // 使能alibi，pseShift的数据类型必须为FLOAT32
        if (pseShiftDesc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "pse_shift",
                ToString(pseShiftDesc->GetDataType()).c_str(),
                "The dtype of pse_shift must be FLOAT32 when pse_type is 2 or 3");
            return ge::GRAPH_FAILED;
        }
    } else {
        // 不使能alibi
        // query的datatype为FLOAT16、INT8时，pseShift的datatype应为FLOAT16
        if (fiaInfo.inputQType == ge::DT_FLOAT16 || fiaInfo.inputQType == ge::DT_INT8) {
            if (pseShiftDesc->GetDataType() != ge::DT_FLOAT16) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "pse_shift",
                    ToString(pseShiftDesc->GetDataType()).c_str(),
                    "The dtype of pse_shift must be FLOAT16 when the dtype of query "
                    "is FLOAT16 or INT8 and pse_type is 0");
                return ge::GRAPH_FAILED;
            }
        }
        // query的datatype为BFLOAT16时，pseShift的datatype应为BFLOAT16
        if (fiaInfo.inputQType == ge::DT_BF16) {
            if (pseShiftDesc->GetDataType() != ge::DT_BF16) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "pse_shift",
                    ToString(pseShiftDesc->GetDataType()).c_str(),
                    "The dtype of pse_shift must be BFLOAT16 when the dtype of query is BFLOAT16 and pseType is 0");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckPseShiftShape(const FiaTilingInfo &fiaInfo) {
    // 校验pseShift的shape
    auto &pseShiftTensor = fiaInfo.opParamInfo.pseShift.tensor;
    if (pseShiftTensor == nullptr) {
        // 若pseShiftTensor不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    auto pseShiftShape = pseShiftTensor->GetStorageShape();
    if (fiaInfo.enableAlibiPse) {
        // pseType为2或3时，使能alibi，pseShist的shape应为(N)
        uint32_t numHeads = *fiaInfo.opParamInfo.numHeads;
        gert::Shape expectedShapeN = gert::Shape({numHeads});
        if (pseShiftShape != expectedShapeN) {
            std::string reason =
                "The shape of pse_shift must be [N]([" + std::to_string(numHeads) + "]) when pse_type is 2 or 3";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                fiaInfo.opName, "pse_shift", ToStringRaw(pseShiftShape).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        uint32_t batchSize = fiaInfo.bSize;
        uint32_t n1Size = fiaInfo.n1Size;
        uint32_t s1Size = fiaInfo.s1Size;
        int64_t s2Size = fiaInfo.s2Size;
        // pseShift的维度必须为4
        uint32_t pseShiftDimNum = pseShiftShape.GetDimNum();
        if (pseShiftDimNum != DIM_NUM_4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "pse_shift",
                std::to_string(pseShiftDimNum).c_str(), "The shape dim of pse_shift must be 4 when pse_type is 0");
            return ge::GRAPH_FAILED;
        }
        uint32_t pseShiftBatch = pseShiftShape.GetDim(DIM_NUM_0);
        uint32_t pseShiftN = pseShiftShape.GetDim(DIM_NUM_1);
        uint32_t pseShiftS1 = pseShiftShape.GetDim(DIM_NUM_2);
        uint32_t pseShiftS2 = pseShiftShape.GetDim(DIM_NUM_3);
        uint32_t actualSharedPrefixLen = 0;
        if (fiaInfo.opParamInfo.actualSharedPrefixLen.tensor != nullptr &&
            fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetStorageShape().GetShapeSize() != 0) {
            if (fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetData<int64_t>() != nullptr) {
                actualSharedPrefixLen = fiaInfo.opParamInfo.actualSharedPrefixLen.tensor->GetData<int64_t>()[0];
            }
        }
        if (pseShiftS1 > 1) {
            // P_S1 > 1分支
            if ((pseShiftBatch != 1 && pseShiftBatch != batchSize) || (pseShiftN != n1Size) || (pseShiftS1 < s1Size) ||
                (pseShiftS2 < s2Size + actualSharedPrefixLen)) {
                std::string reason = "The shape of pse_shift must be [1 or " + std::to_string(batchSize) + ", " +
                    std::to_string(n1Size) + ", >=" + std::to_string(s1Size) +
                    ", >=" + std::to_string(s2Size + actualSharedPrefixLen) + "]";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    fiaInfo.opName, "pse_shift", ToStringRaw(pseShiftShape).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            // P_S1 = 1分支
            if ((pseShiftBatch != 1 && pseShiftBatch != batchSize) || (pseShiftN != n1Size) || (pseShiftS1 != 1) ||
                (pseShiftS2 < s2Size + actualSharedPrefixLen)) {
                std::string reason = "The shape of pse_shift must be [1 or " + std::to_string(batchSize) + ", " +
                    std::to_string(n1Size) + ", 1, >=" + std::to_string(s2Size + actualSharedPrefixLen) + "]";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    fiaInfo.opName, "pse_shift", ToStringRaw(pseShiftShape).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus PSEChecker::CheckPseShiftExistence(const FiaTilingInfo &fiaInfo) {
    // 校验pseShift的存在性
    if (fiaInfo.enableAlibiPse) {
        // pseType为2或3时，使能alibi，pseShift必须传
        OP_CHECK_IF((fiaInfo.opParamInfo.pseShift.tensor == nullptr),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                fiaInfo.opName, "pseShift", "pseShift cannot be empty when pseType is 2 or 3"),
            return ge::GRAPH_FAILED);
    }
    // 校验pseShift的desc的存在性，若pseShift存在，其desc也必须存在
    if (fiaInfo.opParamInfo.pseShift.tensor != nullptr) {
        OP_CHECK_IF((fiaInfo.opParamInfo.pseShift.desc == nullptr),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift", "dtype of pseShift cannot be empty"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus PSEChecker::CheckFeaturePA(const FiaTilingInfo &fiaInfo) {
    if (fiaInfo.opParamInfo.pseType == nullptr || fiaInfo.enableAlibiPse) {
        // 若不存在pseType或pseType = 2/3，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    // Page attention使能场景下，传入的PseShift的最后一维需要大于等于maxBlockNumPerSeq * blockSize
    if (*fiaInfo.opParamInfo.pseType != 0 && fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        uint32_t pseShiftS2 = fiaInfo.pseShiftS2;
        int32_t blockSize = fiaInfo.blockSize;
        uint32_t maxBlockNumPerBatch = fiaInfo.maxBlockNumPerBatch;
        if (pseShiftS2 < maxBlockNumPerBatch * blockSize) {
            std::string reason = "The last axis of pse_shift must be greater than or equal to maxBlockNumPerBatch(" +
                std::to_string(maxBlockNumPerBatch) + ") * blockSize(" + std::to_string(blockSize) +
                ") when page attention is enabled";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "pse_shift",
                ToStringRaw(fiaInfo.opParamInfo.pseShift.tensor->GetStorageShape()).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckerFeatureCrossover(const FiaTilingInfo &fiaInfo) {
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    // 校验使能alibi时与其他特性交叉的约束
    if (fiaInfo.enableAlibiPse || fiaInfo.pseShiftFlag) {
        // D不等长时，不支持pse
        OP_CHECK_IF(fiaInfo.isQKVDDifferent,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
                "pseShift must be empty when D of query and key is not equal to D of value"),
            return ge::GRAPH_FAILED);
    }
    if (fiaInfo.enableAlibiPse) {
        // 使能alibi时，MLA不支持pse
        OP_CHECK_IF(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128 || fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128 ||
                fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                fiaInfo.opName, "pseShift", "pseShift must be empty in the MLA scenario"),
            return ge::GRAPH_FAILED);
    } else if (fiaInfo.pseShiftFlag) {
        if (fiaInfo.isMaxWorkspace) {
            return ge::GRAPH_SUCCESS;
        }
        // 非alibi时，MLA，不支持pse
        OP_CHECK_IF(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128 || fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128 ||
                fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                fiaInfo.opName, "pseShift", "pseShift must be empty in the MLA scenario"),
            return ge::GRAPH_FAILED);
        // pse使能时，若inputLayout为BSH_BNSD/BSND_BNSD/TND/NTD/NTD_TND/TND_NTD，不支持pse
        OP_CHECK_IF(layoutStr == "BSH_BNSD" || layoutStr == "BSND_BNSD" || layoutStr == "TND" || layoutStr == "NTD" ||
                layoutStr == "NTD_TND" || layoutStr == "TND_NTD",
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
                "pseShift must be empty when the inputLayout is BSH_BNSD/BSND_BNSD/TND/NTD/NTD_TND/TND_NTD"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// multipara
ge::graphStatus PSEChecker::CheckAlibiStartIdx(const FiaTilingInfo &fiaInfo) {
    // 使能alibi时，校验qStartIdx和kvStartIdx的取值范围，和kvStartIdx - qStartIdx的取值范围
    // qstartIdx的取值范围应满足[-2147483648, 2147483647]
    auto &qStartIdxTensor = fiaInfo.opParamInfo.qStartIdx.tensor;
    auto &kvStartIdxTensor = fiaInfo.opParamInfo.kvStartIdx.tensor;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    if (qStartIdxTensor != nullptr && qStartIdxTensor->GetShapeSize() >= 1) {
        const int64_t *qStartIdxs = qStartIdxTensor->GetData<int64_t>();
        if (qStartIdxs != nullptr) {
            qStartIdx = qStartIdxs[0];
            if (qStartIdx > INT32_MAX || qStartIdx < INT32_MIN) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "q_start_idx", std::to_string(qStartIdx).c_str(),
                    "The value of q_start_idx must be [-2147483648, 2147483647]");
                return ge::GRAPH_FAILED;
            }
        }
    }
    // kvStartIdx的取值范围应满足[-2147483648, 2147483647]
    if (kvStartIdxTensor != nullptr && kvStartIdxTensor->GetShapeSize() >= 1) {
        const int64_t *kvStartIdxs = kvStartIdxTensor->GetData<int64_t>();
        if (kvStartIdxs != nullptr) {
            kvStartIdx = kvStartIdxs[0];
            if (kvStartIdx > INT32_MAX || kvStartIdx < INT32_MIN) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "kv_start_idx",
                    std::to_string(kvStartIdx).c_str(), "The value of kv_start_idx must be [-2147483648, 2147483647]");
                return ge::GRAPH_FAILED;
            }
        }
    }
    // kvStartIdx - qStartIdx的取值范围应满足[-1048576, 1048576]
    if ((kvStartIdx - qStartIdx > PSE_SHIFT_MAX) || (kvStartIdx - qStartIdx) < PSE_SHIFT_MIN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "kv_start_idx - q_start_idx",
            std::to_string(kvStartIdx - qStartIdx).c_str(),
            "The value of kv_start_idx - q_start_idx must be [-1048576, 1048576]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != CheckPseType(fiaInfo) || ge::GRAPH_SUCCESS != CheckPseShiftDataType(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckPseShiftShape(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != CheckPseShiftExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != CheckFeaturePA(fiaInfo) || ge::GRAPH_SUCCESS != CheckerFeatureCrossover(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PSEChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != CheckAlibiStartIdx(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
