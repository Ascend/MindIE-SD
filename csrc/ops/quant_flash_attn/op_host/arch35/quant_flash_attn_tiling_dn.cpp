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
 * \file quant_flash_attn_tiling_dn.cpp
 * \brief
 */

#include "quant_flash_attn_tiling_dn.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"
#include "../../op_kernel/arch35/quant_flash_attn_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

void QuantFlashAttnTilingDn::InitTilingInfo(TilingInfo *tilingInfo) {
    tilingInfo_ = static_cast<QuantFlashAttnTilingInfo *>(tilingInfo);
}

bool QuantFlashAttnTilingDn::IsCapable() {
    if (tilingInfo_ == nullptr) {
        return false;
    }

    return true;
}

void QuantFlashAttnTilingDn::CalcScheduleMode() {
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(tilingInfo_->opName, "FIA schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus QuantFlashAttnTilingDn::DoOpTiling() {
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(tilingInfo_->opName, "Set plat memory info fail."),
        return ge::GRAPH_FAILED);

    InitImplParam();
    SplitPolicy();
    FillTiling();
    CalcScheduleMode();
    CalcWorkspaceSize();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingDn::SetPlatMemoryInfo() {
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(tilingInfo_->opName, "The platformInfoPtr is null!"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.cvRatio = platformInfo_.aivNum / platformInfo_.aicNum;
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(tilingInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
        platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize, platformInfo_.ubSize,
        platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttnTilingDn::InitImplParam() {}

void QuantFlashAttnTilingDn::SplitPolicy() {
    CalcNumBlocks(platformInfo_.aicNum);
    // flashDecodeFlag_ = true; // TODO 原本是根据分核，当前先写死
}

void QuantFlashAttnTilingDn::GenTilingKey() {
    uint8_t queryOutLayout = LAYOUT_ENUM_BSND;
    if (tilingInfo_->layoutQ == FiaLayout::BSND) {
        if (tilingInfo_->layoutOut == FiaLayout::BNSD) {
            queryOutLayout = LAYOUT_ENUM_BSND;
        }
    } else if (tilingInfo_->layoutQ == FiaLayout::BNSD) {
        if (tilingInfo_->layoutOut == FiaLayout::BNSD) {
            queryOutLayout = LAYOUT_ENUM_BNSD;
        } else if (tilingInfo_->layoutOut == FiaLayout::BSND) {
            queryOutLayout = LAYOUT_ENUM_BNSD_BSND;
        }
    } else if (tilingInfo_->layoutQ == FiaLayout::TND) {
        if (tilingInfo_->layoutOut == FiaLayout::TND) {
            queryOutLayout = LAYOUT_ENUM_TND;
        }
    }

    uint8_t kvStorageMode = KV_STORAGE_MODE_CONTINUE;
    if (tilingInfo_->kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        if (tilingInfo_->layoutKV == FiaLayout::BSND) {
            kvStorageMode = KV_STORAGE_MODE_PA_BSND;
        } else if (tilingInfo_->layoutKV == FiaLayout::BNSD) {
            kvStorageMode = KV_STORAGE_MODE_PA_BSND;
        }
    }

    bool hasMask = tilingInfo_->attnMaskFlag;

    tilingKey_ = GET_TPL_TILING_KEY(queryOutLayout, kvStorageMode, hasMask);
}

void QuantFlashAttnTilingDn::CalcNumBlocks(uint32_t aicNum) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(tilingInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void QuantFlashAttnTilingDn::CalcWorkspaceSize() {
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    workspaceSize_ = sysWorkspaceSize;
    workspaceSize_ += 0;
}

void QuantFlashAttnTilingDn::FillTiling() {
    tilingData_.baseTiling.flashAttnBaseParams.bSize = tilingInfo_->bSize;
    tilingData_.baseTiling.flashAttnBaseParams.t1Size = tilingInfo_->queryTSize;
    tilingData_.baseTiling.flashAttnBaseParams.t2Size = tilingInfo_->keyTSize;
    tilingData_.baseTiling.flashAttnBaseParams.n2Size = tilingInfo_->n2Size;
    tilingData_.baseTiling.flashAttnBaseParams.gSize = tilingInfo_->gSize;
    tilingData_.baseTiling.flashAttnBaseParams.s1Size = tilingInfo_->s1Size;
    tilingData_.baseTiling.flashAttnBaseParams.s2Size = tilingInfo_->s2Size;
    tilingData_.baseTiling.flashAttnBaseParams.dSize = tilingInfo_->qkHeadDim;
    tilingData_.baseTiling.flashAttnBaseParams.dSizeV = tilingInfo_->vHeadDim;
    tilingData_.baseTiling.flashAttnBaseParams.qCuSeqLensSize = tilingInfo_->qCuSeqLensSize;
    tilingData_.baseTiling.flashAttnBaseParams.kvCuSeqLensSize = tilingInfo_->kvCuSeqLensSize;
    tilingData_.baseTiling.flashAttnBaseParams.qSeqUsedSize = tilingInfo_->qSeqUsedSize;
    tilingData_.baseTiling.flashAttnBaseParams.kvSeqUsedSize = tilingInfo_->kvSeqUsedSize;
    tilingData_.baseTiling.flashAttnBaseParams.scaleValue = tilingInfo_->softmaxScale;
    tilingData_.baseTiling.flashAttnBaseParams.isSoftMaxLseEnable = (tilingInfo_->returnSoftmaxLse > 0);
    tilingData_.baseTiling.flashAttnBaseParams.coreNum = numBlocks_;
    tilingData_.baseTiling.flashAttnBaseParams.outputLayout = (uint32_t)(tilingInfo_->layoutOut);

    tilingData_.baseTiling.flashAttnAttenMaskParams.sparseMode = tilingInfo_->maskMode;
    tilingData_.baseTiling.flashAttnAttenMaskParams.winLefts = tilingInfo_->winLeft;
    tilingData_.baseTiling.flashAttnAttenMaskParams.winRights = tilingInfo_->winRight;
    tilingData_.baseTiling.flashAttnAttenMaskParams.attenMaskS1Size = (tilingInfo_->maskMode != 0) ? 2048 : 0;
    tilingData_.baseTiling.flashAttnAttenMaskParams.attenMaskS2Size = (tilingInfo_->maskMode != 0) ? 2048 : 0;
    tilingData_.baseTiling.flashAttnAttenMaskParams.isRowInvalid = true;

    tilingData_.baseTiling.flashAttnPageAttentionParams.paLayoutType = (uint8_t)(tilingInfo_->layoutKV);
    tilingData_.baseTiling.flashAttnPageAttentionParams.blockSize = tilingInfo_->blockSize;
    tilingData_.baseTiling.flashAttnPageAttentionParams.maxBlockNumPerBatch = tilingInfo_->maxBlockNumPerBatch;

    int64_t outSize = tilingInfo_->opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize =
        tilingInfo_->returnSoftmaxLse ? tilingInfo_->opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    tilingData_.baseTiling.flashAttnEmptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.totalOutputSize = outSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.totalSoftMaxLseOutputSize = lseSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.needInit = false;
}

ge::graphStatus QuantFlashAttnTilingDn::SetTilingData(QuantFlashAttnTilingData &tilingData) {
    QuantFlashAttnTilingData *tiling = context_->GetTilingData<QuantFlashAttnTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(tilingInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

// 值越小表示优先级越高. 对于FIA, 使用3位数表示优先级, 优先级编码含义为:
// 1. 百位代表非量化、伪量化、全量化等场景, 即: 0xx-非量化，1xx-伪量化, 2xx-全量化
// 2. 十位表示gqa、mla、泛化，即: x0x-mla, x1x-gpa, x2x-泛化
// 3. 个位代表特化模板到泛化模板的优先级排序
REGISTER_TILING_TEMPLATE_FIA(
    QuantFlashAttn, QuantFlashAttnTilingDn, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 210);
} // namespace optiling
