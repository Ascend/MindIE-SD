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
 * \file l1_to_ub_iterator.h
 * \brief
 */

#ifndef L1_TO_UB_ITERATOR_H
#define L1_TO_UB_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l1_to_ub
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType> struct l1_to_ub {
    __aicore__ l1_to_ub(AscendC::LocalTensor<DataType> ubTensor, AscendC::LocalTensor<DataType> l1Tensor,
        uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride) {
        AscendC::DataCopy(ubTensor, l1Tensor, AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
    };
};

/////////////////////////////////////////////////////
// ub_to_l1
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType> struct ub_to_l1 {
    __aicore__ ub_to_l1(AscendC::LocalTensor<DataType> l1Tensor, AscendC::LocalTensor<DataType> ubTensor,
        uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride) {
        AscendC::DataCopy(l1Tensor, ubTensor, AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
    };
};
#endif // L1_TO_UB_ITERATOR_H
