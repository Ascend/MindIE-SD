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
 * \file fia_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "fia_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;

ge::graphStatus FIAChecker::Init(const FiaTilingInfo &fiaInfo) {
    if (fiaInfo.inputQType != fiaInfo.inputKvType) {
        enableAntiQuant_ = true;
    } else if (fiaInfo.inputQType == ge::DT_FLOAT16 || fiaInfo.inputQType == ge::DT_BF16) {
        enableNonQuant_ = true;
    } else {
        enableFullQuant_ = true;
    }

    actualSeqLenChecker_.reset(new ActualSeqLenChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    dequantChecker_.reset(new DequantChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    learnableSinkChecker_.reset(new LearnableSinkChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    leftPaddingChecker_.reset(new LeftPaddingChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    maskChecker_.reset(new MaskChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    pagedAttentionChecker_.reset(new PagedAttentionChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    postQuantChecker_.reset(new PostQuantChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    pseChecker_.reset(new PSEChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    ropeChecker_.reset(new RopeChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    commonChecker_.reset(new CommonChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    softmaxLSEChecker_.reset(new SoftmaxLSEChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));
    systemPrefixChecker_.reset(new SystemPrefixChecker(enableNonQuant_, enableFullQuant_, enableAntiQuant_));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckMindIESDFp8PerblockScope(const FiaTilingInfo &fiaInfo) const {
    OP_CHECK_IF(fiaInfo.quantMode != FiaQuantMode::FULL_QUANT ||
            fiaInfo.fullQuantMode != FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT,
        OP_LOGE(fiaInfo.opName, "MindIE-SD FIA only supports FP8 E4M3FN per-block full quantization."),
        return ge::GRAPH_FAILED);

    const bool qkvDtypeSupported = fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN &&
        fiaInfo.inputKvType == ge::DT_FLOAT8_E4M3FN && fiaInfo.opParamInfo.value.desc != nullptr &&
        fiaInfo.opParamInfo.value.desc->GetDataType() == ge::DT_FLOAT8_E4M3FN;
    OP_CHECK_IF(!qkvDtypeSupported,
        OP_LOGE(fiaInfo.opName, "MindIE-SD FIA only supports FLOAT8_E4M3FN query/key/value."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.outputType != ge::DT_FLOAT16 && fiaInfo.outputType != ge::DT_BF16,
        OP_LOGE(fiaInfo.opName, "MindIE-SD FIA only supports FLOAT16 or BF16 attention_out."), return ge::GRAPH_FAILED);

    const bool quantModeSupported = fiaInfo.opParamInfo.queryQuantMode != nullptr &&
        fiaInfo.opParamInfo.keyAntiquantMode != nullptr && fiaInfo.opParamInfo.valueAntiquantMode != nullptr &&
        *fiaInfo.opParamInfo.queryQuantMode == 7 && *fiaInfo.opParamInfo.keyAntiquantMode == 7 &&
        *fiaInfo.opParamInfo.valueAntiquantMode == 7;
    OP_CHECK_IF(!quantModeSupported,
        OP_LOGE(fiaInfo.opName, "MindIE-SD FIA only supports query/key/value quant mode 7/7/7."),
        return ge::GRAPH_FAILED);

    const std::string inputLayout(fiaInfo.opParamInfo.layOut == nullptr ? "" : fiaInfo.opParamInfo.layOut);
    const bool layoutSupported =
        inputLayout == "BNSD" || inputLayout == "BSH" || inputLayout == "BSND" || inputLayout == "NTD";
    OP_CHECK_IF(!layoutSupported,
        OP_LOGE(fiaInfo.opName, "MindIE-SD FIA only supports BNSD, BSH, BSND or NTD input_layout."),
        return ge::GRAPH_FAILED);

    const bool dSupported =
        fiaInfo.qkHeadDim == fiaInfo.vHeadDim && (fiaInfo.qkHeadDim == 64U || fiaInfo.qkHeadDim == 128U);
    OP_CHECK_IF(!dSupported, OP_LOGE(fiaInfo.opName, "MindIE-SD FIA only supports D/DV both 64 or both 128."),
        return ge::GRAPH_FAILED);

    const bool unsupportedFeature = fiaInfo.attenMaskFlag || fiaInfo.pseShiftFlag || fiaInfo.enableAlibiPse ||
        fiaInfo.ropeMode != RopeMode::NO_ROPE || fiaInfo.kvStorageMode != KvStorageMode::BATCH_CONTINUOUS ||
        fiaInfo.sysPrefixFlag || fiaInfo.softmaxLseFlag || fiaInfo.isOutQuantEnable || fiaInfo.qPaddingSizeFlag ||
        fiaInfo.kvPaddingSizeFlag || fiaInfo.learnableSinkFlag || fiaInfo.sparseMode != SPARSE_MODE_NO_MASK;
    OP_CHECK_IF(unsupportedFeature,
        OP_LOGE(fiaInfo.opName,
            "MindIE-SD FIA FP8 per-block scope does not support mask, PSE, rope, PA, prefix, softmax_lse, "
            "post quant, padding, learnable sink or sparse mode."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != commonChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != postQuantChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ropeChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pseChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != leftPaddingChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != systemPrefixChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != learnableSinkChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != dequantChecker_->CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != commonChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != postQuantChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ropeChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pseChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != leftPaddingChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != systemPrefixChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != learnableSinkChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != dequantChecker_->CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != commonChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != postQuantChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ropeChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pseChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != leftPaddingChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != systemPrefixChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != learnableSinkChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != dequantChecker_->CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) {
    if (ge::GRAPH_SUCCESS != commonChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != maskChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != actualSeqLenChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pagedAttentionChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != postQuantChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ropeChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != pseChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != leftPaddingChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != systemPrefixChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != softmaxLSEChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != learnableSinkChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != dequantChecker_->CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FIAChecker::Process(const FiaTilingInfo &fiaInfo) {
    if (fiaInfo.emptyTensorFlag) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckMindIESDFp8PerblockScope(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckSinglePara(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckParaExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckCrossFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckMultiParaConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
