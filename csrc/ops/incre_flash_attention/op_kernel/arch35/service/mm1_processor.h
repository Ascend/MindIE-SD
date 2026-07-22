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
 * \file mm1_processor.h
 * \brief
 */
#ifndef MM1_PROCESSOR_H
#define MM1_PROCESSOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../matmul_modules/ifa_flag_data.h"

using namespace matmul;

struct Mm1TaskParam {
    uint32_t sigM;
    uint32_t sigN;
    uint32_t sigK;
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
    uint32_t tscmIdx;
    int64_t tensorAOffset;
    int64_t tensorBOffset;
};
template <typename IFAT, typename mmType> class Mm1Processor {
  public:
    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using MM_OUT_T = T;
    using MM_IN_T = Q_T;
    __aicore__ inline Mm1Processor(){};
    // 非量化
    __aicore__ inline void Send(LocalTensor<MM_OUT_T> &mm1OutResUb, GlobalTensor<Q_T> &queryGm,
        GlobalTensor<KV_T> &keyGm, const Mm1TaskParam &taskParam);
    // 伪量化
    __aicore__ inline void Send(LocalTensor<MM_OUT_T> &mm1OutResUb, TSCM<TPosition::VECIN, 1> &queryScmQueue,
        TSCM<TPosition::VECIN, 1> &keyScmQueue, const Mm1TaskParam &taskParam);
    // 非量化L1自管理
    __aicore__ inline void Send(LocalTensor<MM_OUT_T> &mm1OutResUb, TSCM<TPosition::GM, 1> &queryScmQueue,
        TSCM<TPosition::GM, 1> &keyScmQueue, const Mm1TaskParam &taskParam);
    __aicore__ inline void Wait();
    mmType mm;
};

template <typename IFAT, typename mmType>
__aicore__ inline void Mm1Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T> &mm1OutResUb, GlobalTensor<Q_T> &queryGm,
    GlobalTensor<KV_T> &keyGm, const Mm1TaskParam &taskParam) {
    mm.SetOrgShape(taskParam.orgM, taskParam.orgN, taskParam.orgKa, taskParam.orgKb, taskParam.orgKc);
    mm.SetTensorA(queryGm[taskParam.tensorAOffset]);
    mm.SetTensorB(keyGm[taskParam.tensorBOffset], true);
    mm.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);

    mm.template IterateAll<false>(mm1OutResUb, false, false, true);
}

template <typename IFAT, typename mmType>
__aicore__ inline void Mm1Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T> &mm1OutResUb,
    TSCM<TPosition::GM, 1> &queryScmQueue, TSCM<TPosition::GM, 1> &keyScmQueue, const Mm1TaskParam &taskParam) {
    LocalTensor<Q_T> queryScmTensor = queryScmQueue.DeQue<Q_T>();
    LocalTensor<KV_T> keyScmTensor = keyScmQueue.DeQue<KV_T>();
    mm.SetTensorA(queryScmTensor);
    mm.SetTensorB(keyScmTensor, true);
    mm.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);
    mm.template IterateAll<false>(mm1OutResUb, false, false, true);

    keyScmQueue.FreeTensor(keyScmTensor);
    queryScmQueue.FreeTensor(queryScmTensor);
}

template <typename IFAT, typename mmType>
__aicore__ inline void Mm1Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T> &mm1OutResUb,
    TSCM<TPosition::VECIN, 1> &queryScmQueue, TSCM<TPosition::VECIN, 1> &keyScmQueue, const Mm1TaskParam &taskParam) {
    LocalTensor<Q_T> queryScmTensor = queryScmQueue.DeQue<Q_T>();
    LocalTensor<Q_T> keyScmTensor = keyScmQueue.DeQue<Q_T>();
    mm.SetTensorA(queryScmTensor);
    mm.SetTensorB(keyScmTensor, true);
    mm.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);
    mm.template IterateAll<false>(mm1OutResUb, false, false, true);

    keyScmQueue.FreeTensor(keyScmTensor);
    queryScmQueue.FreeTensor(queryScmTensor);
}

template <typename IFAT, typename mmType> __aicore__ inline void Mm1Processor<IFAT, mmType>::Wait() {
    mm.WaitIterateAll();
    mm.End();
}
#endif
