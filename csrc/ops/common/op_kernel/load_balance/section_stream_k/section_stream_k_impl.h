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
 * \file section_stream_k_impl.h
 * \brief SectionStreamK具体实现
 */

#ifndef SECTION_STREAM_K_IMPL_H
#define SECTION_STREAM_K_IMPL_H

#include <cmath>
#include <algorithm>
#include "section_stream_k_def.h"
#include "../load_balance_common.h"

namespace load_balance {

class SectionStreamKImpl {
  public:
    struct SectionStreamKImplResult {
        int64_t maxCost{0}; // 慢核开销

        // FA
        uint32_t usedCoreNum{0U}; // 使用的核数量
        std::vector<uint32_t> bN2End{}; // 每个核处理数据的BN2结束点
        std::vector<uint32_t> gS1End{}; // 每个核处理数据的GS1结束点
        std::vector<uint32_t> s2End{}; // 每个核处理数据的S2结束点
        uint32_t maxS2SplitNum{0U}; // 单个归约任务最大分核数量
        uint32_t fdTaskNum{0U}; // 归约任务数量
        std::vector<uint32_t> firstFdDataWorkspaceIdx{}; // 每个核第一份归约任务的存放地址

        // FD
        uint32_t usedVecNum{0U}; // 归约过程使用的vector数量
        // 1、归约任务的索引信息
        std::vector<uint32_t> bN2Idx{}; // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
        std::vector<uint32_t> mIdx{}; // 每个归约任务的GS1索引，脚标为归约任务的序号
        std::vector<uint32_t> workspaceIdx{}; // 每个归约任务在workspace中的存放位置
        std::vector<uint32_t> s2SplitNum{}; // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
        std::vector<uint32_t> mSize{}; // 每个归约任务的M轴大小，脚标为归约任务的序号
        // 2、FD kernel阶段，归约任务的分核信息
        std::vector<uint32_t> taskIdx{}; // 每个vector处理的归约任务对应ID
        std::vector<uint32_t> mStart{}; // 每个vector处理的归约任务的M轴起点
        std::vector<uint32_t> mLen{}; // 每个vector处理的归约任务的M轴行数

        explicit SectionStreamKImplResult(uint32_t aicNum, uint32_t aivNum)
            : bN2End(aicNum), gS1End(aicNum), s2End(aicNum), firstFdDataWorkspaceIdx(aicNum), bN2Idx(aicNum),
              mIdx(aicNum), workspaceIdx(aicNum), s2SplitNum(aicNum), mSize(aicNum), taskIdx(aivNum), mStart(aivNum),
              mLen(aivNum) {}

        inline void Clear() {
            maxCost = INT64_MIN;
            usedCoreNum = 0U;
            maxS2SplitNum = 0U;
            fdTaskNum = 0U;
            std::fill(bN2End.begin(), bN2End.end(), 0U);
            std::fill(gS1End.begin(), gS1End.end(), 0U);
            std::fill(s2End.begin(), s2End.end(), 0U);
            std::fill(firstFdDataWorkspaceIdx.begin(), firstFdDataWorkspaceIdx.end(), 0U);

            usedVecNum = 0U;
            std::fill(bN2Idx.begin(), bN2Idx.end(), 0U);
            std::fill(mIdx.begin(), mIdx.end(), 0U);
            std::fill(workspaceIdx.begin(), workspaceIdx.end(), 0U);
            std::fill(s2SplitNum.begin(), s2SplitNum.end(), 0U);
            std::fill(mSize.begin(), mSize.end(), 0U);
            std::fill(taskIdx.begin(), taskIdx.end(), 0U);
            std::fill(mStart.begin(), mStart.end(), 0U);
            std::fill(mLen.begin(), mLen.end(), 0U);
        }
    };

  public:
    inline std::vector<SectionStreamKImplResult> Compute(const DeviceInfo &deviceInfo, const IBaseInfo &baseInfo);
    inline void SetParam(const SectionStreamKParam &param);

  private:
    enum BlockType : uint32_t { NORMAL_BLOCK = 0, TAIL_BLOCK, BLOCK_MAX_TYPE };

    struct GridInfo {
        uint32_t sectionNum{0U}; // section
        std::vector<uint32_t> sectionBn2Idx{}; // BN2方向，section的切分点
        std::vector<uint32_t> mBaseNum{}; // S1G方向，切了多少个基本块
        std::vector<uint32_t> s2BaseNum{}; // S2方向，切了多少个基本块
        std::vector<uint32_t> mTailSize{}; // S1G方向，尾块size
        std::vector<uint32_t> s2TailSize{}; // S2方向，尾块size
        bool isEmpty{true};

        explicit GridInfo(uint32_t batchSize)
            : mBaseNum(batchSize), s2BaseNum(batchSize), mTailSize(batchSize), s2TailSize(batchSize) {}
    };

    struct CostInfo {
        std::vector<int64_t> bN2CostOfEachBatch{}; // 整个batch的开销
        std::vector<uint32_t> bN2BlockOfEachBatch{}; // 整个batch的开销
        std::vector<int64_t> bN2LastBlockCostOfEachBatch{}; // batch最后一块的开销
        std::vector<uint32_t> sectionBlockNum{};
        std::vector<uint32_t> sectionCost{};

        explicit CostInfo(uint32_t batchSize)
            : bN2CostOfEachBatch(batchSize), bN2BlockOfEachBatch(batchSize), bN2LastBlockCostOfEachBatch(batchSize) {}
    };

    struct ComputeContext {
        const DeviceInfo &deviceInfo;
        const IBaseInfo &baseInfo;
        GridInfo gridInfo;
        CostInfo costInfo;

        explicit ComputeContext(const DeviceInfo &dInfo, const IBaseInfo &bInfo)
            : deviceInfo(dInfo), baseInfo(bInfo), gridInfo(bInfo.GetBatchSize()), costInfo(bInfo.GetBatchSize()) {}
    };

    // 分核功能模块内部使用：记录batch相关的临时信息
    struct BatchCache {
        uint32_t bIdx{0U};
        uint32_t s1Size{0U};
        uint32_t s2Size{0U};
        int64_t preTokenLeftUp{0};
        int64_t nextTokenLeftUp{0};
        Table<int64_t, BLOCK_MAX_TYPE, BLOCK_MAX_TYPE> typeCost{};
    };

    // 分核功能模块内部使用：记录当前行（S1G）的临时信息
    struct S1GCache {
        uint32_t bIdx{0U};
        uint32_t s1GIdx{0U};
        uint32_t s2Start{0U};
        uint32_t s2End{0U};
        int64_t s1GCost{0};
        int64_t s1GLastBlockCost{0};
        uint32_t s1GBlock{0U};
        int64_t s1GNormalBlockCost{0};
    };

    // 分核功能模块内部使用：记录分配过程中，当前核的负载信息
    struct CoreCache {
        int64_t costLimit{0}; // 负载上限
        int64_t cost{0}; // 已分配负载
        uint32_t block{0U}; // 已分配块数
    };

    struct AssignContext {
        uint32_t curSectionIdx{0U};
        uint32_t curBIdx{0U};
        uint32_t curBN2Idx{0U};
        uint32_t curS1GIdx{0U};
        uint32_t curS2Idx{0U};
        uint32_t curCoreIdx{0U};
        int64_t unassignedCost{0};
        uint32_t usedCoreNum{0U};
        uint32_t curKvSplitPart{1U};
        uint32_t preFdDataNum{0U};
        int64_t bN2Cost{0};
        uint32_t bN2Block{0U};
        bool isFinished{false};
        BatchCache batchCache{};
        S1GCache s1GCache{};
        CoreCache coreCache{};
    };

    struct FaConfig {
        bool fdOn{true};
        uint32_t coreNum{0U};
        int64_t coreLimit{0U};
        int64_t costLimit{INT64_MIN};
        uint32_t sectionIdx{0U};
    };

    using CostTable = Table<int64_t, BLOCK_MAX_TYPE, BLOCK_MAX_TYPE>;

  private:
    inline static int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    inline CostTable CalcCostTable(
        uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize);
    inline Range<uint32_t> CalcS2Range(uint32_t s1GIdx, const IBaseInfo &baseInfo, const BatchCache &batchCache);
    inline void CalcGridInfo(ComputeContext &computeContext);
    inline void CalcCostInfo(ComputeContext &computeContext);
    inline void CalcBatchCost(uint32_t bIdx, const ComputeContext &computeContext, CostInfo &costInfo);
    inline void CalcBatchCache(uint32_t bIdx, const ComputeContext &computeContext, BatchCache &batchCache);
    inline void CalcS1GCache(
        uint32_t s1GIdx, const ComputeContext &computeContext, const BatchCache &batchCache, S1GCache &s1GCache);
    inline void ScheduleFa(
        const FaConfig &faConfig, const ComputeContext &computeContext, SectionStreamKImplResult &result);
    inline void ScheduleFd(SectionStreamKImplResult &result);
    inline bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SectionStreamKImplResult &result);
    inline void RecordFDInfo(
        const ComputeContext &computeContext, const AssignContext &assignContext, SectionStreamKImplResult &result);

    // assign
    inline void AssignByBatch(const ComputeContext &computeContext, AssignContext &assignContext);
    inline void AssignByRow(const ComputeContext &computeContext, AssignContext &assignContext);
    inline void AssignByBlock(const ComputeContext &computeContext, AssignContext &assignContext);

  private:
    SectionStreamKParam m_param{};
};

inline std::vector<SectionStreamKImpl::SectionStreamKImplResult> SectionStreamKImpl::Compute(
    const DeviceInfo &deviceInfo, const IBaseInfo &baseInfo) {
    ComputeContext computeContext{deviceInfo, baseInfo};
    std::vector<SectionStreamKImplResult> result{};

    CalcGridInfo(computeContext);
    // 全空case
    if (computeContext.gridInfo.isEmpty) {
        result.emplace_back(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
        result[0].bN2End[0] = NumToIndex(baseInfo.GetBatchSize() * baseInfo.GetQueryHeadNum());
        result[0].gS1End[0] = 0U;
        result[0].s2End[0] = 0U;
        return result;
    }

    CalcCostInfo(computeContext);
    for (uint32_t sectionIdx = 0; sectionIdx < computeContext.gridInfo.sectionNum; ++sectionIdx) {
        uint32_t maxCore = std::min(deviceInfo.aicCoreMaxNum, computeContext.costInfo.sectionBlockNum[sectionIdx]);
        uint32_t minCore = static_cast<uint32_t>(
            std::sqrt(static_cast<float>(computeContext.costInfo.sectionBlockNum[sectionIdx]) + 0.25f) + 0.5f);
        minCore = std::min(minCore, maxCore);
        minCore = std::max(minCore, deviceInfo.aicCoreMinNum);

        SectionStreamKImplResult actualResult(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
        actualResult.maxCost = INT64_MAX;
        actualResult.usedCoreNum = deviceInfo.aicCoreMaxNum;

        SectionStreamKImplResult tmpResult(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
        FaConfig faConfig{};
        faConfig.sectionIdx = sectionIdx;

        if (m_param.fdOn) {
            faConfig.fdOn = true;
            for (uint32_t i = minCore; i <= maxCore; ++i) {
                faConfig.coreNum = i;
                faConfig.coreLimit = actualResult.usedCoreNum;
                faConfig.costLimit = actualResult.maxCost;
                ScheduleFa(faConfig, computeContext, tmpResult);
                if (tmpResult.maxCost < actualResult.maxCost) {
                    actualResult = tmpResult;
                }

                tmpResult.Clear();
            }

            if (actualResult.fdTaskNum > 0U) {
                ScheduleFd(actualResult);
            }
        }

        faConfig.coreNum = maxCore;
        faConfig.fdOn = false;
        faConfig.coreLimit = actualResult.usedCoreNum;
        faConfig.costLimit = actualResult.maxCost;
        ScheduleFa(faConfig, computeContext, tmpResult);
        if (!IsWithinTolerance(tmpResult.maxCost, 0L, actualResult.maxCost)) {
            actualResult = tmpResult;
        }

        result.emplace_back(actualResult);
    }

    return result;
}

inline void SectionStreamKImpl::SetParam(const SectionStreamKParam &param) {
    m_param = param;
    if (m_param.faToleranceRatio == 0) {
        m_param.faToleranceRatio = 2U;
    }
    if (m_param.mBaseSize == 0) {
        m_param.mBaseSize = 1U; // at least one
    }
    if (m_param.s2BaseSize == 0) {
        m_param.s2BaseSize = 1U; // at least
    }
    if (m_param.costFunc == nullptr) {
        m_param.costFunc = CalcCost;
    }
}

inline void SectionStreamKImpl::CalcGridInfo(ComputeContext &computeContext) {
    const IBaseInfo &baseInfo = computeContext.baseInfo;

    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    GridInfo &gridInfo = computeContext.gridInfo;
    for (uint32_t bIdx = 0; bIdx < baseInfo.GetBatchSize(); bIdx++) {
        uint32_t s1Size = baseInfo.GetQuerySeqSize(bIdx);
        uint32_t s2Size = baseInfo.GetKvSeqSize(bIdx);

        gridInfo.mBaseNum[bIdx] = CeilDiv(s1Size * baseInfo.GetGroupSize(), m_param.mBaseSize);
        gridInfo.mTailSize[bIdx] = (s1Size * baseInfo.GetGroupSize()) % m_param.mBaseSize;
        gridInfo.s2BaseNum[bIdx] = CeilDiv(s2Size, m_param.s2BaseSize);
        gridInfo.s2TailSize[bIdx] = s2Size % m_param.s2BaseSize;
        if (gridInfo.mBaseNum[bIdx] != 0U && gridInfo.s2BaseNum[bIdx] != 0U) {
            gridInfo.isEmpty = false;
        }
    }

    // calc section
    if (m_param.l2Byte == 0) {
        gridInfo.sectionBn2Idx.emplace_back(baseInfo.GetBatchSize() * baseInfo.GetKvHeadNum());
        gridInfo.sectionNum = 1;
        return;
    }
    uint32_t tokenSize = 0U;
    uint32_t bn2Idx = 0U;
    uint32_t tokenCost;
    uint32_t headDim = baseInfo.GetHeadDim();
    for (uint32_t bIdx = 0; bIdx < baseInfo.GetBatchSize(); bIdx++) {
        uint32_t s2Size = baseInfo.GetKvSeqSize(bIdx);
        for (uint32_t n2Idx = 0; n2Idx < baseInfo.GetKvHeadNum(); ++n2Idx) {
            tokenCost = s2Size * headDim * GetDataTypeByteSize(baseInfo.GetKvDataType()) * 2U; // 2: k and v
            if (!IsWithinTolerance(m_param.l2Byte, 0U, tokenSize + tokenCost) && tokenSize != 0U) {
                gridInfo.sectionBn2Idx.emplace_back(bn2Idx);
                gridInfo.sectionNum++;
                tokenSize = 0U;
            }
            tokenSize += tokenCost;
            bn2Idx++;
        }
    }

    // final section
    gridInfo.sectionBn2Idx.emplace_back(baseInfo.GetBatchSize() * baseInfo.GetKvHeadNum());
    gridInfo.sectionNum++;
}

inline void SectionStreamKImpl::CalcBatchCost(uint32_t bIdx, const ComputeContext &computeContext, CostInfo &costInfo) {
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (baseInfo.GetQuerySeqSize(bIdx) == 0U || baseInfo.GetKvSeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, computeContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < gridInfo.mBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, computeContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;

        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

inline SectionStreamKImpl::CostTable SectionStreamKImpl::CalcCostTable(
    uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize) {
    CostTable typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = m_param.costFunc(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : m_param.costFunc(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : m_param.costFunc(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] =
        (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : m_param.costFunc(s1GTailSize, s2TailSize);
    return typeCost;
}

inline void SectionStreamKImpl::CalcBatchCache(
    uint32_t bIdx, const ComputeContext &computeContext, BatchCache &batchCache) {
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = baseInfo.GetQuerySeqSize(bIdx);
    batchCache.s2Size = baseInfo.GetKvSeqSize(bIdx);
    batchCache.preTokenLeftUp = baseInfo.GetPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = baseInfo.GetNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.typeCost =
        CalcCostTable(m_param.mBaseSize, m_param.s2BaseSize, gridInfo.mTailSize[bIdx], gridInfo.s2TailSize[bIdx]);
}

inline void SectionStreamKImpl::CalcS1GCache(
    uint32_t s1GIdx, const ComputeContext &computeContext, const BatchCache &batchCache, S1GCache &s1GCache) {
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    auto s2Range = CalcS2Range(s1GIdx, baseInfo, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;

    if (s1GCache.s2Start >= s1GCache.s2End || gridInfo.mBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num =
        (gridInfo.s2TailSize[batchCache.bIdx] != 0U && s1GCache.s2End == gridInfo.s2BaseNum[batchCache.bIdx]) ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;
    if (s1GIdx == (gridInfo.mBaseNum[batchCache.bIdx] - 1U) && gridInfo.mTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK]
                                                      : batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK]
                                                      : batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }

    if (s1GCache.s1GBlock == 1) {
        s1GCache.s1GLastBlockCost += m_param.v0Cost;
    }
    s1GCache.s1GCost += m_param.v0Cost;
}

inline void SectionStreamKImpl::CalcCostInfo(ComputeContext &computeContext) {
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    CostInfo &costInfo = computeContext.costInfo;
    costInfo.sectionBlockNum.resize(gridInfo.sectionNum);
    costInfo.sectionCost.resize(gridInfo.sectionNum);

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < baseInfo.GetBatchSize(); bIdx++) {
        CalcBatchCost(bIdx, computeContext, costInfo);
    }

    for (uint32_t sectionIdx = 0; sectionIdx < gridInfo.sectionBn2Idx.size(); ++sectionIdx) {
        uint32_t bn2StartIdx = (sectionIdx == 0U) ? 0U : gridInfo.sectionBn2Idx[sectionIdx - 1U];
        uint32_t bn2EndIdx = gridInfo.sectionBn2Idx[sectionIdx];

        uint32_t bIdx = FloorDiv(bn2StartIdx, baseInfo.GetKvHeadNum());
        for (uint32_t bn2Idx = bn2StartIdx; bn2Idx < bn2EndIdx; ++bn2Idx) {
            costInfo.sectionBlockNum[sectionIdx] += costInfo.bN2BlockOfEachBatch[bIdx];
            costInfo.sectionCost[sectionIdx] += costInfo.bN2CostOfEachBatch[bIdx];
            bIdx = FloorDiv(bn2Idx, baseInfo.GetKvHeadNum());
        }
    }
}

inline int64_t SectionStreamKImpl::CalcCost(uint32_t basicM, uint32_t basicS2) {
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2); // 6：M轴系数，10：S2轴系数
}

inline Range<uint32_t> SectionStreamKImpl::CalcS2Range(
    uint32_t s1GIdx, const IBaseInfo &baseInfo, const BatchCache &batchCache) {
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    // actual seq == 0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    // no mask
    if (baseInfo.GetSparseMode() == SparseMode::BUTT) {
        s2Start = 0U;
        s2End = CeilDiv(batchCache.s2Size, m_param.s2BaseSize);
        return std::make_pair(s2Start, s2End);
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(m_param.mBaseSize);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(m_param.mBaseSize),
        NumToIndex(static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(baseInfo.GetGroupSize())));

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (baseInfo.GetIsS1G()) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(baseInfo.GetGroupSize());
        s1LastToken = s1GLastToken / static_cast<int64_t>(baseInfo.GetGroupSize());
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    // 2. trans index of token to index of block
    // no valid token
    if (s2FirstToken >= batchCache.s2Size || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }

    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(NumToIndex(batchCache.s2Size)));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(NumToIndex(batchCache.s2Size)));
    s2Start = static_cast<uint32_t>(s2FirstToken) / m_param.s2BaseSize;
    s2End = ToOpenInterval(static_cast<uint32_t>(s2LastToken) / m_param.s2BaseSize); // end of block index

    return std::make_pair(s2Start, s2End);
}

void SectionStreamKImpl::ScheduleFa(
    const FaConfig &faConfig, const ComputeContext &computeContext, SectionStreamKImplResult &result) {
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const CostInfo &costInfo = computeContext.costInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    if (faConfig.coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext{};
    assignContext.curSectionIdx = faConfig.sectionIdx;
    assignContext.curBIdx = (faConfig.sectionIdx == 0)
        ? 0U
        : FloorDiv(gridInfo.sectionBn2Idx[faConfig.sectionIdx - 1U], baseInfo.GetKvHeadNum());
    assignContext.curBN2Idx = (faConfig.sectionIdx == 0) ? 0U : gridInfo.sectionBn2Idx[faConfig.sectionIdx - 1U];
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.sectionCost[faConfig.sectionIdx];
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, computeContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, computeContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;

    for (uint32_t i = 0; i < faConfig.coreNum; ++i) {
        if (result.maxCost > faConfig.costLimit && i > faConfig.coreLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
            assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;

        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = assignContext.unassignedCost / (faConfig.coreNum - i);

        if (!faConfig.fdOn) {
            assignContext.coreCache.costLimit =
                std::max(assignContext.coreCache.costLimit, assignContext.s1GCache.s1GCost);
        } else {
            int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
            if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
                curCost = assignContext.s1GCache.s1GLastBlockCost;
            };
            if (assignContext.curS2Idx == 0 && assignContext.s1GCache.s1GBlock != 1) {
                curCost += m_param.v0Cost;
            }
            assignContext.coreCache.costLimit = std::max(assignContext.coreCache.costLimit, curCost);
        }

        AssignByBatch(computeContext, assignContext);
        AssignByRow(computeContext, assignContext);
        if (faConfig.fdOn) {
            AssignByBlock(computeContext, assignContext);
        }

        result.bN2End[i] = assignContext.curBN2Idx;
        result.gS1End[i] = assignContext.curS1GIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;
        if (faConfig.fdOn && IsNeedRecordFDInfo(assignContext, result)) {
            RecordFDInfo(computeContext, assignContext, result);
            assignContext.preFdDataNum += assignContext.curKvSplitPart;
            assignContext.curKvSplitPart = 1U;
        }

        if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
            assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
            assignContext.curKvSplitPart++;
        }
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

inline void SectionStreamKImpl::ScheduleFd(SectionStreamKImplResult &result) {
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0U;
    for (uint32_t i = 0; i < result.fdTaskNum; i++) {
        totalFDLoad += result.s2SplitNum[i] * result.mSize[i];
    }
    // 计算每个核处理的load
    if (result.taskIdx.empty()) {
        return;
    }
    uint64_t averageLoad = CeilDiv(totalFDLoad, result.taskIdx.size()); //向上取整，避免核负载为0
    uint32_t curCoreIndex = 0U;
    for (uint32_t i = 0; i < result.fdTaskNum; ++i) {
        // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数
        uint32_t curFDVectorNum = result.s2SplitNum[i] * result.mSize[i] / averageLoad;
        curFDVectorNum = std::max(curFDVectorNum, 1U);
        uint32_t curAvgMSize =
            CeilDiv(result.mSize[i], curFDVectorNum); // 计算当前归约任务每个核的行数，向上取整，避免行数为0
        curFDVectorNum = CeilDiv(result.mSize[i], curAvgMSize); // 重新计算正确的核数，避免因为向上取整导致有核为空
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            result.taskIdx[curCoreIndex] = i;
            result.mStart[curCoreIndex] = vid * curAvgMSize;
            result.mLen[curCoreIndex] =
                (vid < curFDVectorNum - 1U) ? curAvgMSize : (result.mSize[i] - vid * curAvgMSize);
            curCoreIndex++;
        }
    }
    result.usedVecNum = curCoreIndex;
}

inline bool SectionStreamKImpl::IsNeedRecordFDInfo(
    const AssignContext &assignContext, const SectionStreamKImplResult &result) {
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == result.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == result.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

inline void SectionStreamKImpl::RecordFDInfo(
    const ComputeContext &computeContext, const AssignContext &assignContext, SectionStreamKImplResult &result) {
    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;

    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / baseInfo.GetKvHeadNum();
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = baseInfo.GetQuerySeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == NumToIndex(gridInfo.mBaseNum[splitBIdx]))
        ? (s1Size * baseInfo.GetGroupSize() - splitS1GIdx * m_param.mBaseSize)
        : m_param.mBaseSize;
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.bN2Idx[result.fdTaskNum] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.mIdx[result.fdTaskNum] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.workspaceIdx[result.fdTaskNum] = assignContext.preFdDataNum;
    result.s2SplitNum[result.fdTaskNum] = assignContext.curKvSplitPart;
    result.mSize[result.fdTaskNum] = curFdS1gSize;
    result.fdTaskNum++;
}

inline void SectionStreamKImpl::AssignByBatch(const ComputeContext &computeContext, AssignContext &assignContext) {
    if (assignContext.isFinished) {
        return;
    }

    const IBaseInfo &baseInfo = computeContext.baseInfo;
    const GridInfo &gridInfo = computeContext.gridInfo;
    const CostInfo &costInfo = computeContext.costInfo;

    while (assignContext.bN2Cost == 0 ||
        IsWithinTolerance(assignContext.coreCache.costLimit,
            SafeFloorDiv(costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx], m_param.faToleranceRatio, 0L),
            assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == gridInfo.sectionBn2Idx[assignContext.curSectionIdx]) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / baseInfo.GetKvHeadNum() != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.GetKvHeadNum();
            CalcBatchCache(assignContext.curBIdx, computeContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, computeContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

inline void SectionStreamKImpl::AssignByRow(const ComputeContext &computeContext, AssignContext &assignContext) {
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
        SafeFloorDiv(assignContext.s1GCache.s1GLastBlockCost, m_param.faToleranceRatio, 0L),
        assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost
            ? assignContext.bN2Cost - assignContext.s1GCache.s1GCost
            : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock
            ? assignContext.bN2Block - assignContext.s1GCache.s1GBlock
            : 0U;
        // 计算新一行的信息
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, computeContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

inline void SectionStreamKImpl::AssignByBlock(const ComputeContext &computeContext, AssignContext &assignContext) {
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    int64_t realCost = curCost;
    if (assignContext.curS2Idx == 0U && assignContext.s1GCache.s1GBlock != 1U) {
        realCost += m_param.v0Cost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, SafeFloorDiv(realCost, m_param.faToleranceRatio, 0L),
        assignContext.coreCache.cost + realCost)) {
        assignContext.coreCache.cost += realCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - realCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - realCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;

        realCost = curCost;
        if (assignContext.curS2Idx == 0U && assignContext.s1GCache.s1GBlock != 1U) {
            realCost += m_param.v0Cost;
        }
    }
}

}
#endif
