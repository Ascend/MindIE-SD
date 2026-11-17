/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * SPDX-License-Identifier: MulanPSL-2.0
 */

#ifndef QFA_HEAD_SPLIT_H
#define QFA_HEAD_SPLIT_H

#include "../../common/op_kernel/load_balance/section_stream_k/section_stream_k.h"
#include "../../quant_flash_attn/op_kernel/arch35/qfa_query_tile.h"

namespace qfa_metadata {

inline void RemapHeadBoundary(uint32_t &bn2, uint32_t &m, const load_balance::BaseInfo &baseInfo, uint32_t tileSize) {
    uint32_t batch = bn2 / baseInfo.GetQueryHeadNum();
    uint32_t group = bn2 % baseInfo.GetGroupSize();
    if (batch < baseInfo.GetBatchSize()) {
        uint64_t tilesPerHead = QFA_KERNEL::QueryTileCount(baseInfo.GetQuerySeqSize(batch), 1, tileSize, false);
        m += group * tilesPerHead;
    }
    bn2 /= baseInfo.GetGroupSize();
}

inline uint32_t BalanceSchedule(const load_balance::DeviceInfo &deviceInfo, const load_balance::BaseInfo &baseInfo,
    const load_balance::SectionStreamKParam &param, load_balance::SectionStreamKResult &result) {
    // BSND Q data and BNGSD Q scales must use the same head-major M traversal,
    // including aligned lengths: the old S1G traversal silently mismatched scales.
    bool splitHeads = baseInfo.GetQueryLayout() == load_balance::Layout::BSND && baseInfo.GetGroupSize() > 1;
    if (baseInfo.GetQueryLayout() == load_balance::Layout::BNSD && baseInfo.GetGroupSize() > 1) {
        for (uint32_t batch = 0; batch < baseInfo.GetBatchSize(); ++batch) {
            splitHeads |= baseInfo.GetQuerySeqSize(batch) % param.mBaseSize != 0;
        }
    }
    if (!splitHeads) {
        return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, result);
    }

    // Scheduling-only view: each Q head is an independent M axis. Tensor shapes,
    // actual sequence lengths and the kernel's KV-head addressing stay unchanged.
    load_balance::BaseInfo headInfo = baseInfo;
    headInfo.kvHeadNum = headInfo.queryHeadNum;
    headInfo.layoutQuery = load_balance::Layout::BNSD;
    uint32_t status = load_balance::SectionStreamK::Compute(deviceInfo, headInfo, param, result);
    if (status != LOAD_BALANCE_SUCCESS) {
        return status;
    }
    for (auto &fa : result.sectionFaResult) {
        for (uint32_t core = 0; core < fa.usedCoreNum; ++core) {
            RemapHeadBoundary(fa.bN2End[core], fa.gS1End[core], baseInfo, param.mBaseSize);
        }
    }
    for (auto &fd : result.sectionFdResult) {
        for (uint32_t task = 0; task < fd.bN2Idx.size(); ++task) {
            RemapHeadBoundary(fd.bN2Idx[task], fd.gS1Idx[task], baseInfo, param.mBaseSize);
        }
    }
    return status;
}

} // namespace qfa_metadata

#endif
