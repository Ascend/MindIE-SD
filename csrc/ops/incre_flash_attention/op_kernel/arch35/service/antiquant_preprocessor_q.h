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
 * \file antiquant_preprocessor_q.h
 * \brief
 */
#ifndef ANTIQUANT_PREPROCESSOR_Q_H
#define ANTIQUANT_PREPROCESSOR_Q_H

#include "../incre_flash_attention_pub.h"

struct AntiquantQPreprocessorTaskParam {
    uint32_t queryInputQueBufferSize;
    uint32_t dealRowCount;
    uint32_t columnCount;
    uint32_t tensorAOffset;
    uint32_t tensorAStride;
    uint32_t qHeadNum;
    bool isPFABSH;
    bool isFirstSInnerLoop;
};
template <typename IFAT> class AntiQuantQPreprocessor {
  public:
    using Q_T = typename IFAT::queryType;
    static constexpr IFAProfile PROFILE = IFAT::profile;

    __aicore__ inline AntiQuantQPreprocessor(){};
    __aicore__ inline void Process(TSCM<TPosition::VECIN, 1> &queryScm, TQue<QuePosition::VECIN, 1> &queryInputQue,
        TQue<QuePosition::VECOUT, 1> &queryOutputQue, GlobalTensor<Q_T> queryGm,
        const AntiquantQPreprocessorTaskParam &taskParam);

  private:
    __aicore__ inline void CopyAntiqQuery(LocalTensor<Q_T> &queryUb, GlobalTensor<Q_T> queryGm, uint64_t qOffset,
        uint32_t dealRowCount, const AntiquantQPreprocessorTaskParam &taskParam, uint16_t elementTypeSize);
};
template <typename IFAT>
__aicore__ inline void AntiQuantQPreprocessor<IFAT>::CopyAntiqQuery(LocalTensor<Q_T> &queryUb,
    GlobalTensor<Q_T> queryGm, uint64_t qOffset, uint32_t dealRowCount,
    const AntiquantQPreprocessorTaskParam &taskParam, uint16_t elementTypeSize) {
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = taskParam.columnCount * sizeof(Q_T);
    copyInParams.dstStride = (PROFILE.D - taskParam.columnCount) / elementTypeSize;

    if (taskParam.isPFABSH) {
        copyInParams.srcStride = (taskParam.qHeadNum - 1) * taskParam.columnCount * sizeof(Q_T);
    } else {
        copyInParams.srcStride = 0;
    }

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    DataCopyPad(queryUb, queryGm[qOffset], copyInParams, copyInPadParams);
}

template <typename IFAT>
__aicore__ inline void AntiQuantQPreprocessor<IFAT>::Process(TSCM<TPosition::VECIN, 1> &queryScm,
    TQue<QuePosition::VECIN, 1> &queryInputQue, TQue<QuePosition::VECOUT, 1> &queryOutputQue, GlobalTensor<Q_T> queryGm,
    const AntiquantQPreprocessorTaskParam &taskParam) {
    LocalTensor<Q_T> queryScmTensor = queryScm.AllocTensor<Q_T>();
    if (taskParam.isFirstSInnerLoop) {
        uint32_t gSplitSize = taskParam.queryInputQueBufferSize / PROFILE.D / sizeof(Q_T);
        uint32_t loopCount = (taskParam.dealRowCount + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = taskParam.dealRowCount - (loopCount - 1) * gSplitSize;
        uint16_t elementTypeSize = ONE_BLK_SIZE / sizeof(Q_T);
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if (i + 1 == loopCount) {
                gSplitSize = tailSplitSize;
            }
            LocalTensor<Q_T> querytmpUb = queryInputQue.template AllocTensor<Q_T>();
            CopyAntiqQuery(querytmpUb, queryGm, taskParam.tensorAOffset + startRow * taskParam.tensorAStride,
                gSplitSize, taskParam, elementTypeSize);
            queryInputQue.template EnQue(querytmpUb);
            queryInputQue.DeQue<Q_T>();
            LocalTensor<Q_T> queryUb = queryOutputQue.template AllocTensor<Q_T>();
            DataCopyParams intriParams;
            intriParams.blockCount = (taskParam.columnCount + elementTypeSize - 1) / elementTypeSize;
            intriParams.blockLen = 1;
            intriParams.srcStride = 0;
            intriParams.dstStride = gSplitSize - 1;

            for (uint32_t j = 0; j < gSplitSize; j++) {
                DataCopy(queryUb[j * elementTypeSize], querytmpUb[j * PROFILE.D], intriParams);
            }
            queryInputQue.FreeTensor(querytmpUb);

            queryOutputQue.template EnQue(queryUb);
            queryOutputQue.DeQue<Q_T>();

            DataCopyParams intriParams2;
            intriParams2.blockCount = (taskParam.columnCount + elementTypeSize - 1) / elementTypeSize;
            intriParams2.blockLen = gSplitSize;
            intriParams2.srcStride = 0;
            intriParams2.dstStride = 16 - gSplitSize; // L1最小要求16对齐，D64带pse PROFILE.G为8
            DataCopy(queryScmTensor[startRow * elementTypeSize], queryUb, intriParams2);
            queryOutputQue.FreeTensor(queryUb);
        }
    }
    queryScm.EnQue(queryScmTensor);
}
#endif
