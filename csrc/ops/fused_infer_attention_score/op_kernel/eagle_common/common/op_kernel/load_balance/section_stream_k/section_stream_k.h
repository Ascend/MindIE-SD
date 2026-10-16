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
 * \file section_stream_k.h
 * \brief SectionStreamK对外接口
 */

#ifndef SECTION_STREAM_K_H
#define SECTION_STREAM_K_H

#include <vector>
#include "../base_info.h"
#include "section_stream_k_def.h"
#include "section_stream_k_impl.h"

namespace load_balance {

class SectionStreamK {
  public:
    inline static uint32_t Compute(const DeviceInfo &deviceInfo, const IBaseInfo &baseInfo,
        const SectionStreamKParam &param, SectionStreamKResult &result);
};

inline uint32_t SectionStreamK::Compute(const DeviceInfo &deviceInfo, const IBaseInfo &baseInfo,
    const SectionStreamKParam &param, SectionStreamKResult &result) {
    SectionStreamKImpl impl{};
    auto ret = impl.SetParam(param);
    if (ret != SECTION_STREAM_K_SUCCESS) {
        return ret;
    }

    if (deviceInfo.aivCoreMaxNum < deviceInfo.aicCoreMaxNum) {
        return SECTION_STREAM_K_ERROR_AIV_LESS_THAN_AIC;
    }

    auto implResult = impl.Compute(deviceInfo, baseInfo);

    result.sectionNum = implResult.size();
    for (size_t i = 0; i < result.sectionNum; ++i) {
        SectionStreamKFaResult tmpFa(deviceInfo.aicCoreMaxNum);
        tmpFa.usedCoreNum = implResult[i].usedCoreNum;
        tmpFa.bN2End = implResult[i].bN2End;
        tmpFa.gS1End = implResult[i].gS1End;
        tmpFa.s2End = implResult[i].s2End;
        tmpFa.firstFdDataWorkspaceIdx = implResult[i].firstFdDataWorkspaceIdx;
        result.sectionFaResult.emplace_back(tmpFa);

        SectionStreamKFdResult tmpFd(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
        tmpFd.usedVecNum = implResult[i].usedVecNum;
        tmpFd.bN2Idx = implResult[i].bN2Idx;
        tmpFd.gS1Idx = implResult[i].mIdx;
        tmpFd.workspaceIdx = implResult[i].workspaceIdx;
        tmpFd.s2SplitNum = implResult[i].s2SplitNum;
        tmpFd.taskIdx = implResult[i].taskIdx;
        tmpFd.mStart = implResult[i].mStart;
        tmpFd.mLen = implResult[i].mLen;
        result.sectionFdResult.emplace_back(tmpFd);
    }

    return SECTION_STREAM_K_SUCCESS;
}

}

#endif
