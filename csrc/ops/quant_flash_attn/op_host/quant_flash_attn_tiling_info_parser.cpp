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
 * \file quant_flash_attn_tiling_info_parser.cpp
 * \brief
 */

#include <cmath>
#include <map>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "quant_flash_attn_tiling_info_parser.h"

using std::map;
using std::pair;
using std::string;
using namespace ge;
// using namespace AscendC;
namespace optiling {

ge::graphStatus QuantFlashAttnTilingInfoParser::GetOpName() {
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("quant_flash_attn", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    tilingInfo_.opName = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetNpuInfo() {
    tilingInfo_.platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(tilingInfo_.platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(tilingInfo_.opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo_.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(tilingInfo_.opName, "num of core obtained is 0."), return GRAPH_FAILED);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_3510) {
        OPS_REPORT_VECTOR_INNER_ERR(tilingInfo_.opName, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttnTilingInfoParser::GetOptionalInputParaInfo() {
    tilingInfo_.opParamInfo.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    tilingInfo_.opParamInfo.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    tilingInfo_.opParamInfo.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    tilingInfo_.opParamInfo.cuSeqlensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    tilingInfo_.opParamInfo.cuSeqlensKv.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_KV_INDEX);
    tilingInfo_.opParamInfo.cuSeqlensKv.desc = context_->GetOptionalInputDesc(CU_SEQLENS_KV_INDEX);
    tilingInfo_.opParamInfo.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    tilingInfo_.opParamInfo.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    tilingInfo_.opParamInfo.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
    tilingInfo_.opParamInfo.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);
    tilingInfo_.opParamInfo.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    tilingInfo_.opParamInfo.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
    tilingInfo_.opParamInfo.attnMask.tensor = context_->GetOptionalInputTensor(ATTN_MASK_INDEX);
    tilingInfo_.opParamInfo.attnMask.desc = context_->GetOptionalInputDesc(ATTN_MASK_INDEX);
    tilingInfo_.opParamInfo.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    tilingInfo_.opParamInfo.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void QuantFlashAttnTilingInfoParser::GetInputParaInfo() {
    tilingInfo_.opParamInfo.query.desc = context_->GetInputDesc(QUERY_INDEX);
    tilingInfo_.opParamInfo.query.shape = context_->GetInputShape(QUERY_INDEX);
    tilingInfo_.opParamInfo.key.desc = context_->GetInputDesc(KEY_INDEX);
    tilingInfo_.opParamInfo.key.shape = context_->GetInputShape(KEY_INDEX);
    tilingInfo_.opParamInfo.value.desc = context_->GetInputDesc(VALUE_INDEX);
    tilingInfo_.opParamInfo.value.shape = context_->GetInputShape(VALUE_INDEX);
    tilingInfo_.opParamInfo.qDescale.shape = context_->GetInputShape(QUERY_DESCALE_INDEX);
    tilingInfo_.opParamInfo.qDescale.desc = context_->GetInputDesc(QUERY_DESCALE_INDEX);
    tilingInfo_.opParamInfo.kDescale.shape = context_->GetInputShape(KEY_DESCALE_INDEX);
    tilingInfo_.opParamInfo.kDescale.desc = context_->GetInputDesc(KEY_DESCALE_INDEX);
    tilingInfo_.opParamInfo.vDescale.shape = context_->GetInputShape(VALUE_DESCALE_INDEX);
    tilingInfo_.opParamInfo.vDescale.desc = context_->GetInputDesc(VALUE_DESCALE_INDEX);
    GetOptionalInputParaInfo();
}

void QuantFlashAttnTilingInfoParser::GetOutputParaInfo() {
    tilingInfo_.opParamInfo.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    tilingInfo_.opParamInfo.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    tilingInfo_.opParamInfo.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    tilingInfo_.opParamInfo.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetAttrParaInfo() {
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
        return ge::GRAPH_FAILED);

    tilingInfo_.opParamInfo.qQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    tilingInfo_.opParamInfo.kQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_KEY_QUANT_MODE_INDEX);
    tilingInfo_.opParamInfo.vQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_VALUE_QUANT_MODE_INDEX);
    tilingInfo_.opParamInfo.quantBlockSizeQs = attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_BLOCK_SIZE_INDEX);
    tilingInfo_.opParamInfo.quantBlockSizeKs = attrs->GetAttrPointer<int64_t>(ATTR_KEY_QUANT_BLOCK_SIZE_INDEX);
    tilingInfo_.opParamInfo.quantBlockSizeVs = attrs->GetAttrPointer<int64_t>(ATTR_VALUE_QUANT_BLOCK_SIZE_INDEX);
    tilingInfo_.opParamInfo.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    tilingInfo_.opParamInfo.maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);
    tilingInfo_.opParamInfo.winLeft = attrs->GetAttrPointer<int64_t>(ATTR_WIN_LEFT_INDEX);
    tilingInfo_.opParamInfo.winRight = attrs->GetAttrPointer<int64_t>(ATTR_WIN_RIGHT_INDEX);
    tilingInfo_.opParamInfo.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    tilingInfo_.opParamInfo.maxSeqlenKV = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);
    tilingInfo_.opParamInfo.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    tilingInfo_.opParamInfo.layoutKV = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);
    tilingInfo_.opParamInfo.layoutOut = attrs->GetStr(ATTR_LAYOUT_OUT_INDEX);
    tilingInfo_.opParamInfo.softmaxPrecision = attrs->GetAttrPointer<int64_t>(ATTR_SOFTMAX_PRECISION_INDEX);
    tilingInfo_.opParamInfo.returnSoftMaxLse = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_LSE_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetOpParaInfo() {
    GetInputParaInfo();
    GetOutputParaInfo();
    return GetAttrParaInfo();
}

ge::graphStatus QuantFlashAttnTilingInfoParser::CheckRequiredInOutExistence() const {
    OP_CHECK_IF(tilingInfo_.opParamInfo.query.shape == nullptr,
        OP_LOGE(tilingInfo_.opName, "Shape of tensor query is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.query.desc == nullptr,
        OP_LOGE(tilingInfo_.opName, "Desc of tensor query is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.key.shape == nullptr,
        OP_LOGE(tilingInfo_.opName, "Shape of tensor key is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.key.desc == nullptr,
        OP_LOGE(tilingInfo_.opName, "Desc of tensor key is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.value.shape == nullptr,
        OP_LOGE(tilingInfo_.opName, "Shape of tensor value is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.value.desc == nullptr,
        OP_LOGE(tilingInfo_.opName, "Desc of tensor value is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.qDescale.shape == nullptr,
        OP_LOGE(tilingInfo_.opName, "Shape of tensor q_descale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.qDescale.desc == nullptr,
        OP_LOGE(tilingInfo_.opName, "Desc of tensor q_descale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.kDescale.shape == nullptr,
        OP_LOGE(tilingInfo_.opName, "Shape of tensor k_descale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.kDescale.desc == nullptr,
        OP_LOGE(tilingInfo_.opName, "Desc of tensor k_descale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.vDescale.shape == nullptr,
        OP_LOGE(tilingInfo_.opName, "Shape of tensor v_descale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.vDescale.desc == nullptr,
        OP_LOGE(tilingInfo_.opName, "Desc of tensor v_descale is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.attnOut.shape == nullptr,
        OP_LOGE(tilingInfo_.opName, "Shape of tensor attn_out is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.attnOut.desc == nullptr,
        OP_LOGE(tilingInfo_.opName, "Desc of tensor attn_out is nullptr"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::CheckOptionalInputExistence() const {
    if ((tilingInfo_.opParamInfo.blockTable.tensor == nullptr && tilingInfo_.opParamInfo.blockTable.desc != nullptr) ||
        (tilingInfo_.opParamInfo.blockTable.tensor != nullptr && tilingInfo_.opParamInfo.blockTable.desc == nullptr)) {
        OP_LOGE(tilingInfo_.opName, "tensor and desc of %s are either both not nullptr or both nullptr.",
            BLOCK_TABLE_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((tilingInfo_.opParamInfo.cuSeqlensQ.tensor == nullptr && tilingInfo_.opParamInfo.cuSeqlensQ.desc != nullptr) ||
        (tilingInfo_.opParamInfo.cuSeqlensQ.tensor != nullptr && tilingInfo_.opParamInfo.cuSeqlensQ.desc == nullptr)) {
        OP_LOGE(tilingInfo_.opName, "tensor and desc of %s are either both not nullptr or both nullptr.",
            CU_SEQLENS_Q_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((tilingInfo_.opParamInfo.cuSeqlensKv.tensor == nullptr &&
            tilingInfo_.opParamInfo.cuSeqlensKv.desc != nullptr) ||
        (tilingInfo_.opParamInfo.cuSeqlensKv.tensor != nullptr &&
            tilingInfo_.opParamInfo.cuSeqlensKv.desc == nullptr)) {
        OP_LOGE(tilingInfo_.opName, "tensor and desc of %s are either both not nullptr or both nullptr.",
            CU_SEQLENS_KV_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((tilingInfo_.opParamInfo.sequsedQ.tensor == nullptr && tilingInfo_.opParamInfo.sequsedQ.desc != nullptr) ||
        (tilingInfo_.opParamInfo.sequsedQ.tensor != nullptr && tilingInfo_.opParamInfo.sequsedQ.desc == nullptr)) {
        OP_LOGE(tilingInfo_.opName, "tensor and desc of %s are either both not nullptr or both nullptr.",
            SEQUSED_Q_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((tilingInfo_.opParamInfo.sequsedKv.tensor == nullptr && tilingInfo_.opParamInfo.sequsedKv.desc != nullptr) ||
        (tilingInfo_.opParamInfo.sequsedKv.tensor != nullptr && tilingInfo_.opParamInfo.sequsedKv.desc == nullptr)) {
        OP_LOGE(tilingInfo_.opName, "tensor and desc of %s are either both not nullptr or both nullptr.",
            SEQUSED_KV_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((tilingInfo_.opParamInfo.sinks.tensor == nullptr && tilingInfo_.opParamInfo.sinks.desc != nullptr) ||
        (tilingInfo_.opParamInfo.sinks.tensor != nullptr && tilingInfo_.opParamInfo.sinks.desc == nullptr)) {
        OP_LOGE(tilingInfo_.opName, "tensor and desc of %s are either both not nullptr or both nullptr.",
            SINKS_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((tilingInfo_.opParamInfo.attnMask.tensor == nullptr && tilingInfo_.opParamInfo.attnMask.desc != nullptr) ||
        (tilingInfo_.opParamInfo.attnMask.tensor != nullptr && tilingInfo_.opParamInfo.attnMask.desc == nullptr)) {
        OP_LOGE(tilingInfo_.opName, "tensor and desc of %s are either both not nullptr or both nullptr.",
            ATTEN_MASK_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((tilingInfo_.opParamInfo.metadata.tensor == nullptr && tilingInfo_.opParamInfo.metadata.desc != nullptr) ||
        (tilingInfo_.opParamInfo.metadata.tensor != nullptr && tilingInfo_.opParamInfo.metadata.desc == nullptr)) {
        OP_LOGE(tilingInfo_.opName, "tensor and desc of %s are either both not nullptr or both nullptr.",
            METADATA_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::CheckRequiredAttrExistence() const {
    OP_CHECK_IF(tilingInfo_.opParamInfo.qQuantMode == nullptr,
        OP_LOGE(tilingInfo_.opName, "attr q_quant_mode is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.kQuantMode == nullptr,
        OP_LOGE(tilingInfo_.opName, "attr k_quant_mode is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo_.opParamInfo.vQuantMode == nullptr,
        OP_LOGE(tilingInfo_.opName, "attr v_quant_mode is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::CheckRequiredParaExistence() const {
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS ||
        CheckOptionalInputExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetCuSeqLenQDims() {
    if (tilingInfo_.layoutQ == FiaLayout::TND) {
        if (tilingInfo_.opParamInfo.cuSeqlensQ.tensor == nullptr) {
            OP_LOGE(tilingInfo_.opName, "when %s's layout is %s, %s must be provided.", QUERY_NAME.c_str(),
                LayoutToSerialString(tilingInfo_.layoutQ).c_str(), CU_SEQLENS_Q_NAME.c_str());
            return ge::GRAPH_FAILED;
        }
        int64_t shapeSize = tilingInfo_.opParamInfo.cuSeqlensQ.tensor->GetShapeSize();
        if (shapeSize <= 1) {
            OP_LOGE(tilingInfo_.opName, "%s's shape size is %ld, it should be greater than 1.",
                CU_SEQLENS_Q_NAME.c_str(), shapeSize);
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.qCuSeqLensSize = static_cast<uint32_t>(shapeSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetCuSeqLenKvDims() {
    if (tilingInfo_.layoutKV == FiaLayout::TND) {
        if (tilingInfo_.opParamInfo.cuSeqlensKv.tensor == nullptr) {
            OP_LOGE(tilingInfo_.opName, "when key/value's layout is %s, %s must be provided.",
                LayoutToSerialString(tilingInfo_.layoutKV).c_str(), CU_SEQLENS_KV_NAME.c_str());
            return ge::GRAPH_FAILED;
        }
        int64_t shapeSize = tilingInfo_.opParamInfo.cuSeqlensKv.tensor->GetShapeSize();
        if (shapeSize <= 1) {
            OP_LOGE(tilingInfo_.opName, "%s's shape size is %ld, it should be greater than 1.",
                CU_SEQLENS_KV_NAME.c_str(), shapeSize);
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.kvCuSeqLensSize = static_cast<uint32_t>(shapeSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetSeqUsedQDims() {
    tilingInfo_.qSeqUsedSize = 0;
    if (tilingInfo_.opParamInfo.sequsedQ.tensor != nullptr) {
        int64_t shapeSize = tilingInfo_.opParamInfo.sequsedQ.tensor->GetShapeSize();
        if (shapeSize <= 0) {
            OP_LOGE(tilingInfo_.opName, "%s's shape size is %ld, it should be greater than 0 when seqused_q exists.",
                SEQUSED_Q_NAME.c_str(), shapeSize, SEQUSED_Q_NAME.c_str());
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.qSeqUsedSize = static_cast<uint32_t>(shapeSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetSeqUsedKvDims() {
    tilingInfo_.kvSeqUsedSize = 0;
    if (tilingInfo_.opParamInfo.sequsedKv.tensor != nullptr) {
        int64_t shapeSize = tilingInfo_.opParamInfo.sequsedKv.tensor->GetShapeSize();
        if (shapeSize <= 0) {
            OP_LOGE(tilingInfo_.opName, "%s's shape size is %ld, it should be greater than 0 when %s exists.",
                SEQUSED_KV_NAME.c_str(), shapeSize, SEQUSED_KV_NAME.c_str());
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.kvSeqUsedSize = static_cast<uint32_t>(shapeSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetBatchSize() {
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, cu_seqlens_q必须传入, 以cu_seqlens_q数组的长度为B+1
    if (tilingInfo_.layoutQ == FiaLayout::TND) {
        return tilingInfo_.qCuSeqLensSize - 1;
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasShapeB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.bSize = queryShape_->GetShapeB();
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetN1Size() {
    // 从 Q 形状获取 N1 值
    if (queryShape_ != nullptr && queryShape_->HasShapeN()) {
        tilingInfo_.n1Size = static_cast<uint32_t>(queryShape_->GetShapeN());
    } else {
        OP_LOGE(tilingInfo_.opName, "Failed to get N1 size from query shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetN2Size() {
    // 从 K 形状获取 N2 值
    if (keyShape_ != nullptr && keyShape_->HasShapeN()) {
        tilingInfo_.n2Size = static_cast<uint32_t>(keyShape_->GetShapeN());
    } else {
        OP_LOGE(tilingInfo_.opName, "Failed to get N2 size from key shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetGSize() {
    // 获取G基准值
    if (tilingInfo_.n1Size % tilingInfo_.n2Size != 0U) {
        OP_LOGE(tilingInfo_.opName, "Q numHeads(%u) should be a multiple of Kv Heads(%u).", tilingInfo_.n1Size,
            tilingInfo_.n2Size);
        return ge::GRAPH_FAILED;
    }
    tilingInfo_.gSize = tilingInfo_.n1Size / tilingInfo_.n2Size;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetQkHeadDim() {
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingInfo_.qkHeadDim = static_cast<uint32_t>(queryShape_->GetShapeD()); // 后面需要把qkHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetValueHeadDim() {
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    if (valueShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingInfo_.vHeadDim = static_cast<uint32_t>(valueShape_->GetShapeD()); // 后面需要把vHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttnTilingInfoParser::GetQueryTSize() {
    // 获取query的T基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    tilingInfo_.queryTSize = (queryShape_->HasShapeT()) ? static_cast<uint32_t>(queryShape_->GetShapeT()) : 0;
}

void QuantFlashAttnTilingInfoParser::GetKeyTSize() {
    tilingInfo_.keyTSize = (keyShape_->HasShapeT()) ? static_cast<uint32_t>(keyShape_->GetShapeT()) : 0;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetMaxSeqLenQ() {
    tilingInfo_.maxSeqLenQ = -1;
    if (tilingInfo_.opParamInfo.maxSeqlenQ != nullptr) {
        tilingInfo_.maxSeqLenQ = *tilingInfo_.opParamInfo.maxSeqlenQ;
    }
    if (tilingInfo_.maxSeqLenQ < -1) {
        OP_LOGE(tilingInfo_.opName, "max_seqlen_q must be >= -1, but got %lld", tilingInfo_.maxSeqLenQ);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetMaxSeqLenKv() {
    tilingInfo_.maxSeqLenKv = -1;
    if (tilingInfo_.opParamInfo.maxSeqlenKV != nullptr) {
        tilingInfo_.maxSeqLenKv = *tilingInfo_.opParamInfo.maxSeqlenKV;
    }
    if (tilingInfo_.maxSeqLenKv < -1) {
        OP_LOGE(tilingInfo_.opName, "max_seqlen_kv must be >= -1, but got %d", tilingInfo_.maxSeqLenKv);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetS1Size() {
    // 获取S1基准值
    // BSH/BSND/BNSD: 以query的S维度为基准
    if (tilingInfo_.layoutQ == FiaLayout::TND) {
        GetQueryTSize();
        if (tilingInfo_.maxSeqLenQ >= 0) {
            tilingInfo_.s1Size = tilingInfo_.maxSeqLenQ;
        } else {
            tilingInfo_.s1Size = tilingInfo_.queryTSize;
        }
    } else {
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.s1Size = static_cast<uint32_t>(queryShape_->GetShapeS());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetS2SizeForBatchContinuous() {
    if (tilingInfo_.layoutKV == FiaLayout::TND) {
        GetKeyTSize();
        if (tilingInfo_.maxSeqLenKv >= 0) {
            tilingInfo_.s2Size = tilingInfo_.maxSeqLenKv;
        } else {
            tilingInfo_.s2Size = tilingInfo_.keyTSize;
        }
    } else {
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.s2Size = keyShape_->GetShapeS();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetBlockSize() {
    if (keyShape_->CheckHasShapeBlockSize(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingInfo_.blockSize = keyShape_->GetShapeBlockSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetMaxBlockNumPerBatch() {
    if (tilingInfo_.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        uint32_t dimNum = tilingInfo_.opParamInfo.blockTable.tensor->GetStorageShape().GetDimNum();
        if (dimNum != 2U) {
            OP_LOGE(tilingInfo_.opName, "the dim num of %s is %u, it should be 2.", BLOCK_TABLE_NAME.c_str(), dimNum);
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.maxBlockNumPerBatch = tilingInfo_.opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    } else {
        tilingInfo_.maxBlockNumPerBatch = 0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetS2SizeForPageAttention() {
    if (GetBlockSize() != ge::GRAPH_SUCCESS && GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingInfo_.s2Size = tilingInfo_.blockSize * tilingInfo_.maxBlockNumPerBatch;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetS2Size() {
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、TENSOR_LIST时, 从kCache_的所有Tensor的S轴的最大值
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (tilingInfo_.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetInAndOutLayout() {
    if (tilingInfo_.opParamInfo.layoutQ == nullptr) {
        tilingInfo_.layoutQ = FiaLayout::BSND;
    } else {
        const std::map<std::string, FiaLayout> qLayoutMap = {
            {"BSND", FiaLayout::BSND}, {"BNSD", FiaLayout::BNSD}, {"TND", FiaLayout::TND}};
        auto itQ = qLayoutMap.find(tilingInfo_.opParamInfo.layoutQ);
        if (itQ == qLayoutMap.end()) {
            OP_LOGE(tilingInfo_.opName, "Invalid layout_q: %s, only support BSND/BNSD/TND",
                tilingInfo_.opParamInfo.layoutQ);
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.layoutQ = itQ->second;
    }

    if (tilingInfo_.opParamInfo.layoutKV == nullptr) {
        tilingInfo_.layoutKV = FiaLayout::BSND;
    } else {
        const std::map<std::string, FiaLayout> kvLayoutMap = {{"BSND", FiaLayout::BSND}, {"BNSD", FiaLayout::BNSD},
            {"TND", FiaLayout::TND}, {"PA_BBND", FiaLayout::BnBsH}, {"PA_BNBD", FiaLayout::BnNBsD}};
        auto itKV = kvLayoutMap.find(tilingInfo_.opParamInfo.layoutKV);
        if (itKV == kvLayoutMap.end()) {
            OP_LOGE(tilingInfo_.opName, "Invalid layoutKV: %s, only support BSND/BNSD/TND/PA_BBND/PA_BNBD",
                tilingInfo_.opParamInfo.layoutKV);
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.layoutKV = itKV->second;
    }

    if (tilingInfo_.opParamInfo.layoutOut == nullptr) {
        tilingInfo_.layoutOut = FiaLayout::BSND;
    } else {
        const std::map<std::string, FiaLayout> outLayoutMap = {
            {"BSND", FiaLayout::BSND}, {"BNSD", FiaLayout::BNSD}, {"TND", FiaLayout::TND}};
        auto itOut = outLayoutMap.find(tilingInfo_.opParamInfo.layoutOut);
        if (itOut == outLayoutMap.end()) {
            OP_LOGE(tilingInfo_.opName, "Invalid layoutOut: %s, only support BSND/BNSD/TND",
                tilingInfo_.opParamInfo.layoutOut);
            return ge::GRAPH_FAILED;
        }
        tilingInfo_.layoutOut = itOut->second;
    }

    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttnTilingInfoParser::GetPreNextToken() {
    // 特殊场景下需要更新值
    tilingInfo_.maskMode = tilingInfo_.opParamInfo.maskMode == nullptr ? 0 : *tilingInfo_.opParamInfo.maskMode;

    // 从输入读取参数值
    tilingInfo_.winLeft = tilingInfo_.opParamInfo.winLeft == nullptr ? 0 : *tilingInfo_.opParamInfo.winLeft;
    tilingInfo_.winRight = tilingInfo_.opParamInfo.winRight == nullptr ? 0 : *tilingInfo_.opParamInfo.winRight;

    // 边界场景需要更新值
    if (tilingInfo_.winLeft == -1) {
        tilingInfo_.winLeft = MASK_MODE_INT_MAX;
    }
    if (tilingInfo_.winRight == -1) {
        tilingInfo_.winRight = MASK_MODE_INT_MAX;
    }
}

ge::graphStatus QuantFlashAttnTilingInfoParser::GetQkvDataType() {
    tilingInfo_.inputQType = tilingInfo_.opParamInfo.query.desc->GetDataType();
    tilingInfo_.inputKType = tilingInfo_.opParamInfo.key.desc->GetDataType();
    tilingInfo_.inputVType = tilingInfo_.opParamInfo.value.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttnTilingInfoParser::SetFaShape() {
    queryShape_ = std::make_shared<FiaTilingShape>(
        tilingInfo_.opParamInfo.query.shape->GetStorageShape(), tilingInfo_.layoutQ, QUERY_NAME, tilingInfo_.opName);
    keyShape_ = std::make_shared<FiaTilingShape>(
        tilingInfo_.opParamInfo.key.shape->GetStorageShape(), tilingInfo_.layoutKV, KEY_NAME, tilingInfo_.opName);
    valueShape_ = std::make_shared<FiaTilingShape>(
        tilingInfo_.opParamInfo.value.shape->GetStorageShape(), tilingInfo_.layoutKV, VALUE_NAME, tilingInfo_.opName);
    // qDescaleShape_ =
    //     std::make_shared<FiaTilingShape>(tilingInfo_.opParamInfo.qDescale.shape->GetStorageShape(), tilingInfo_.layoutQ, Q_DESCALE_NAME, tilingInfo_.opName);
    // kDescaleShape_ =
    //     std::make_shared<FiaTilingShape>(tilingInfo_.opParamInfo.kDescale.shape->GetStorageShape(), tilingInfo_.layoutKV, K_DESCALE_NAME, tilingInfo_.opName);
    // vDescaleShape_ =
    //     std::make_shared<FiaTilingShape>(tilingInfo_.opParamInfo.vDescale.shape->GetStorageShape(), tilingInfo_.layoutKV, V_DESCALE_NAME, tilingInfo_.opName);
}

void QuantFlashAttnTilingInfoParser::GetKvStorageMode() {
    if (tilingInfo_.opParamInfo.blockTable.tensor != nullptr) {
        tilingInfo_.kvStorageMode = KvStorageMode::PAGE_ATTENTION;
    } else {
        tilingInfo_.kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;
    }
}

void QuantFlashAttnTilingInfoParser::GetSoftmaxScale() {
    if (tilingInfo_.opParamInfo.softmaxScale != nullptr) {
        tilingInfo_.softmaxScale = *tilingInfo_.opParamInfo.softmaxScale;
    } else {
        tilingInfo_.softmaxScale = ((float)1.0) / std::sqrt(tilingInfo_.qkHeadDim);
    }
}

ge::graphStatus QuantFlashAttnTilingInfoParser::ParseAxisInfo() {
    if (ge::GRAPH_SUCCESS != GetSeqUsedQDims() || ge::GRAPH_SUCCESS != GetSeqUsedKvDims() ||
        ge::GRAPH_SUCCESS != GetCuSeqLenQDims() || ge::GRAPH_SUCCESS != GetCuSeqLenKvDims() ||
        ge::GRAPH_SUCCESS != GetBatchSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetQkHeadDim() || ge::GRAPH_SUCCESS != GetValueHeadDim()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetMaxSeqLenQ() || ge::GRAPH_SUCCESS != GetMaxSeqLenKv() ||
        ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }

    tilingInfo_.qQuantMode = static_cast<QuantMode>(*tilingInfo_.opParamInfo.qQuantMode);
    tilingInfo_.kQuantMode = static_cast<QuantMode>(*tilingInfo_.opParamInfo.kQuantMode);
    tilingInfo_.vQuantMode = static_cast<QuantMode>(*tilingInfo_.opParamInfo.vQuantMode);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::ParseFeatureInfo() {
    tilingInfo_.attnMaskFlag = (tilingInfo_.opParamInfo.attnMask.tensor != nullptr);
    tilingInfo_.learnableSinkFlag = (tilingInfo_.opParamInfo.sinks.tensor != nullptr);
    tilingInfo_.softmaxPresision = (tilingInfo_.opParamInfo.softmaxPrecision == nullptr)
        ? 1
        : *tilingInfo_.opParamInfo.softmaxPrecision; // 1:默认值
    tilingInfo_.returnSoftmaxLse = (tilingInfo_.opParamInfo.returnSoftMaxLse == nullptr)
        ? 0
        : *tilingInfo_.opParamInfo.returnSoftMaxLse; // 0:默认值

    tilingInfo_.quantBlockSizeQs =
        (tilingInfo_.opParamInfo.quantBlockSizeQs == nullptr) ? 0 : *tilingInfo_.opParamInfo.quantBlockSizeQs;
    tilingInfo_.quantBlockSizeKs =
        (tilingInfo_.opParamInfo.quantBlockSizeKs == nullptr) ? 0 : *tilingInfo_.opParamInfo.quantBlockSizeKs;
    tilingInfo_.quantBlockSizeVs =
        (tilingInfo_.opParamInfo.quantBlockSizeVs == nullptr) ? 0 : *tilingInfo_.opParamInfo.quantBlockSizeVs;

    GetSoftmaxScale();
    GetPreNextToken();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingInfoParser::Parse() {
    if (context_ == nullptr) {
        OP_LOGE("quant_flash_attn", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInAndOutLayout() || ge::GRAPH_SUCCESS != GetQkvDataType()) {
        return ge::GRAPH_FAILED;
    }
    SetFaShape();
    GetKvStorageMode();

    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != ParseFeatureInfo()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
