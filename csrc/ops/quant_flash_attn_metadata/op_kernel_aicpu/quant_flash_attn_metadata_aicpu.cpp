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
 * \file quant_flash_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include "quant_flash_attn_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t QuantFlashAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx) {
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SectionStreamKResult splitRes;
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantFlashAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx) {
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool hasSocVersion =
        GetAttrValue(ctx, "custom_soc_version", socVersion_) || GetAttrValue(ctx, "soc_version", socVersion_);
    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
        GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) && GetAttrValue(ctx, "head_dim", headDim_) &&
        GetAttrValue(ctx, "q_quant_mode", qQuantMode_) && GetAttrValue(ctx, "k_quant_mode", kQuantMode_) &&
        GetAttrValue(ctx, "v_quant_mode", vQuantMode_) && GetAttrValue(ctx, "q_dtype", qDtype_) &&
        GetAttrValue(ctx, "k_dtype", kDtype_) && GetAttrValue(ctx, "v_dtype", vDtype_) && hasSocVersion &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    return ParamsInit();
    // return true;
}

std::vector<int64_t> QuantFlashAttnMetadataCpuKernel::GetTensorDataAsInt64(Tensor *tensor, size_t size) {
    std::vector<int64_t> result(size);
    if (tensor == nullptr || tensor->GetData() == nullptr || size == 0) {
        return result;
    }

    DataType dataType = tensor->GetDataType();
    void *data = tensor->GetData();

    switch (dataType) {
    case DT_INT32: {
        int32_t *ptr = static_cast<int32_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            result[i] = static_cast<int64_t>(ptr[i]);
        }
        break;
    }
    case DT_INT64: {
        int64_t *ptr = static_cast<int64_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            result[i] = ptr[i];
        }
        break;
    }
    case DT_INT16: {
        int16_t *ptr = static_cast<int16_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            result[i] = static_cast<int64_t>(ptr[i]);
        }
        break;
    }
    case DT_UINT32: {
        uint32_t *ptr = static_cast<uint32_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            result[i] = static_cast<int64_t>(ptr[i]);
        }
        break;
    }
    case DT_UINT64: {
        uint64_t *ptr = static_cast<uint64_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            result[i] = static_cast<int64_t>(ptr[i]);
        }
        break;
    }
    case DT_UINT16: {
        uint16_t *ptr = static_cast<uint16_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            result[i] = static_cast<int64_t>(ptr[i]);
        }
        break;
    }
    default:
        break;
    }
    return result;
}

bool QuantFlashAttnMetadataCpuKernel::ParamsInit() {
    // Device info
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
    deviceInfo.cvRatio = aivCoreNum_ / aicCoreNum_;
    // deviceInfo.socVersion = socVersion_;
    // baseInfo
    // actual seq size
    baseInfo.isCumulativeQuerySeq = layoutQ_ == "TND" || layoutQ_ == "NTD";
    baseInfo.isCumulativeKvSeq = layoutKv_ == "TND" || layoutKv_ == "NTD";
    if (batchSize_ > 0) {
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        if (baseInfo.isCumulativeQuerySeq) {
            for (uint32_t i = 1; i < batchSize_; ++i) {
                baseInfo.actualQuerySeqSize[i] += baseInfo.actualQuerySeqSize[i - 1];
            }
        }
        if (baseInfo.isCumulativeKvSeq) {
            for (uint32_t i = 1; i < batchSize_; ++i) {
                baseInfo.actualKvSeqSize[i] += baseInfo.actualKvSeqSize[i - 1];
            }
        }
    }
    if (!baseInfo.isCumulativeQuerySeq && sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        batchSize_ = sequsedQ_->GetTensorShape()->GetDimSize(0);
        auto sequsedQ = GetTensorDataAsInt64(sequsedQ_, batchSize_);
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = sequsedQ[i];
            maxSeqlenQ_ = std::max(static_cast<int64_t>(maxSeqlenQ_), sequsedQ[i]);
        }
    } else if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        batchSize_ = cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
        auto cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_, batchSize_ + 1);
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = cuSeqlensQ[i + 1];
            maxSeqlenQ_ = std::max(static_cast<int64_t>(maxSeqlenQ_), cuSeqlensQ[i + 1] - cuSeqlensQ[i]);
        }
    }
    if (!baseInfo.isCumulativeKvSeq && sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        batchSize_ = sequsedKv_->GetTensorShape()->GetDimSize(0);
        auto sequsedKv = GetTensorDataAsInt64(sequsedKv_, batchSize_);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = sequsedKv[i];
            maxSeqlenKv_ = std::max(static_cast<int64_t>(maxSeqlenKv_), sequsedKv[i]);
        }
    } else if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        batchSize_ = cuSeqlensKv_->GetTensorShape()->GetDimSize(0) - 1;
        auto cuSeqlensKv = GetTensorDataAsInt64(cuSeqlensKv_, batchSize_ + 1);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = cuSeqlensKv[i + 1];
            maxSeqlenKv_ = std::max(static_cast<int64_t>(maxSeqlenKv_), cuSeqlensKv[i + 1] - cuSeqlensKv[i]);
        }
    }
    baseInfo.batchSize = batchSize_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.headDim = headDim_;
    baseInfo.attenMaskFlag = maskMode_ = true; // todo
    baseInfo.sparseMode = maskMode_;
    baseInfo.preToken = winLeft_ == -1 ? std::numeric_limits<uint32_t>::max() : winLeft_;
    baseInfo.nextToken = winRight_ == -1 ? std::numeric_limits<uint32_t>::max() : winRight_;
    baseInfo.layoutQuery = ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = ConvertToLayout(layoutKv_);
    baseInfo.queryType = static_cast<load_balance::DataType>(qDtype_);
    baseInfo.kvType = static_cast<load_balance::DataType>(kDtype_);

    // param
    if (numHeadsKv_ == 0) {
        numHeadsKv_ = numHeadsQ_;
        groupSize_ = 1;
    } else {
        groupSize_ = numHeadsQ_ / numHeadsKv_;
    }
    mBaseSize_ = 128;
    s2BaseSize_ = 256;
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 0U; // sectionNum = 1
    param.fdOn = false;
    return true;
}

bool QuantFlashAttnMetadataCpuKernel::BalanceSchedule(SectionStreamKResult &splitRes) {
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == LOAD_BALANCE_SUCCESS;
}

bool QuantFlashAttnMetadataCpuKernel::GenMetaData(SectionStreamKResult &splitRes) {
    uint32_t sectionNum = splitRes.sectionNum;
    detail::QFaMetaData faMetadata(metaData_->GetData(), sectionNum);
    uint32_t *ptr = (uint32_t *)metaData_->GetData();
    ptr[1] = mBaseSize_;
    ptr[2] = s2BaseSize_;
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
            // faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_CORE_ENABLE_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_START_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_START_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_START_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_END_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_END_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_END_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_FIRST_QFD_DATA_WORKSPACE_IDX_INDEX, 0U);
        }
        for (uint32_t i = 0; i < AIV_CORE_NUM; ++i) {
            // faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_CORE_ENABLE_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_BN2_IDX_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_IDX_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_WORKSPACE_IDX_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_WORKSPACE_NUM_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_START_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_NUM_INDEX, 0U);
        }
        // QFA Metadata Generate
        auto faSplitRes = splitRes.sectionFaResult[sectionId];
        for (uint32_t i = 0; i < faSplitRes.usedCoreNum; ++i) {
            // faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_CORE_ENABLE_INDEX, 1U);
            // QFA start
            if (i > 0) {
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_START_INDEX, faSplitRes.bN2End[i - 1]);
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_START_INDEX, faSplitRes.gS1End[i - 1]);
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_START_INDEX, faSplitRes.s2End[i - 1]);
            } else if (sectionId > 0) {
                auto preQFaSplitRes = splitRes.sectionFaResult[sectionId - 1];
                faMetadata.setQFaMetadata(
                    sectionId, i, optiling::QFA_BN2_START_INDEX, preQFaSplitRes.bN2End[preQFaSplitRes.usedCoreNum - 1]);
                faMetadata.setQFaMetadata(
                    sectionId, i, optiling::QFA_M_START_INDEX, preQFaSplitRes.gS1End[preQFaSplitRes.usedCoreNum - 1]);
                faMetadata.setQFaMetadata(
                    sectionId, i, optiling::QFA_S2_START_INDEX, preQFaSplitRes.s2End[preQFaSplitRes.usedCoreNum - 1]);
            }
            // QFA end
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_END_INDEX, faSplitRes.bN2End[i]);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_END_INDEX, faSplitRes.gS1End[i]);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_END_INDEX, faSplitRes.s2End[i]);
            // QFA idx
            faMetadata.setQFaMetadata(
                sectionId, i, optiling::QFA_FIRST_QFD_DATA_WORKSPACE_IDX_INDEX, faSplitRes.firstFdDataWorkspaceIdx[i]);
        }
        // QFD Metadata Generate
        auto fdSplitRes = splitRes.sectionFdResult[sectionId];
        for (uint32_t i = 0; i < fdSplitRes.usedVecNum; ++i) {
            // faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_CORE_ENABLE_INDEX, 1U);
            uint32_t curTaskIdx = fdSplitRes.taskIdx[i];
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_BN2_IDX_INDEX, fdSplitRes.bN2Idx[curTaskIdx]);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_IDX_INDEX, fdSplitRes.gS1Idx[curTaskIdx]);
            faMetadata.setQFdMetadata(
                sectionId, i, optiling::QFD_WORKSPACE_IDX_INDEX, fdSplitRes.workspaceIdx[curTaskIdx]);
            faMetadata.setQFdMetadata(
                sectionId, i, optiling::QFD_WORKSPACE_NUM_INDEX, fdSplitRes.s2SplitNum[curTaskIdx]);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_START_INDEX, fdSplitRes.mStart[i]);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_NUM_INDEX, fdSplitRes.mLen[i]);
        }
    }
    return true;
}

namespace {
static const char *kernelType = "QuantFlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, QuantFlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
