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
 * \file l1_to_fb_iterator.h
 * \brief
 */

#ifndef L1_TO_FB_ITERATOR_H
#define L1_TO_FB_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l1_to_fb
/////////////////////////////////////////////////////

// Partial specialization for V220
template <ArchType ArchTag, typename DataType> struct l1_to_fb {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::fbBlockSize / sizeof(DataType);

    __aicore__ l1_to_fb(
        AscendC::LocalTensor<DataType> fbTensor, AscendC::LocalTensor<DataType> l1Tensor, uint32_t ntileActual) {
        copy_cbuf_to_fbuf((__fbuf__ DataType *)fbTensor.GetPhyAddr(), (__cbuf__ DataType *)l1Tensor.GetPhyAddr(), 1,
            CeilDiv<BLOCK_SIZE>(ntileActual), 0, 0);
    };
};

#endif // L1_TO_FB_ITERATOR_H
