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
#ifndef MM2_PROCESSOR_H
#define MM2_PROCESSOR_H

struct Mm2TaskParam {
    uint32_t sigM;
    uint32_t sigN;
    uint32_t sigK;
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
    int64_t tensorBOffset;
};
template <typename IFAT, typename mmType> class Mm2Processor {
  public:
    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using MM_OUT_T = T;
    using MM_IN_T = Q_T;
    __aicore__ inline Mm2Processor(){};
    // 非量化
    __aicore__ inline void Send(LocalTensor<MM_OUT_T> &mm2OutResUb, TSCM<TPosition::VECIN, 1> &softmaxResScmQueue,
        GlobalTensor<KV_T> &valueGm, const Mm2TaskParam &taskParam);
    // 伪量化
    __aicore__ inline void Send(LocalTensor<MM_OUT_T> &mm2OutResUb, TSCM<TPosition::VECIN, 1> &softmaxResScmQueue,
        TSCM<TPosition::VECIN, 1> &valueScmQueue, const Mm2TaskParam &taskParam);

    __aicore__ inline void Wait();
    mmType bmm2;
};

template <typename IFAT, typename mmType>
__aicore__ inline void Mm2Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T> &mm2OutResUb,
    TSCM<TPosition::VECIN, 1> &softmaxResScmQueue, GlobalTensor<KV_T> &valueGm, const Mm2TaskParam &taskParam) {
    LocalTensor<Q_T> softmaxResScmTensor = softmaxResScmQueue.DeQue<Q_T>();
    bmm2.SetOrgShape(taskParam.orgM, taskParam.orgN, taskParam.orgKa, taskParam.orgKb, taskParam.orgKc);
    bmm2.SetTensorA(softmaxResScmTensor);
    bmm2.SetTensorB(valueGm[taskParam.tensorBOffset]);
    bmm2.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);
    bmm2.template IterateAll<false>(mm2OutResUb, false, false, true);

    softmaxResScmQueue.FreeTensor(softmaxResScmTensor);
}

template <typename IFAT, typename mmType>
__aicore__ inline void Mm2Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T> &mm2OutResUb,
    TSCM<TPosition::VECIN, 1> &softmaxResScmQueue, TSCM<TPosition::VECIN, 1> &valueScmQueue,
    const Mm2TaskParam &taskParam) {
    LocalTensor<Q_T> softmaxResScmTensor = softmaxResScmQueue.DeQue<Q_T>();
    LocalTensor<Q_T> valueScmTensor = valueScmQueue.DeQue<Q_T>();

    bmm2.SetTensorA(softmaxResScmTensor);
    bmm2.SetTensorB(valueScmTensor);
    bmm2.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);
    bmm2.template IterateAll<false>(mm2OutResUb, false, false, true);

    valueScmQueue.FreeTensor(valueScmTensor);
    softmaxResScmQueue.FreeTensor(softmaxResScmTensor);
}

template <typename IFAT, typename mmType> __aicore__ inline void Mm2Processor<IFAT, mmType>::Wait() {
    bmm2.WaitIterateAll();
    bmm2.End();
}
#endif
