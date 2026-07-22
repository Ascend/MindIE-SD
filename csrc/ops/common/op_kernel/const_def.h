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
 * \file const_def.h
 * \brief Common constants
 */
#ifndef CONST_DEF_H
#define CONST_DEF_H

namespace AttentionCommon {
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);

enum class TASK_DEAL_MODE : uint32_t {
    DEAL_ZERO = 0,
    SKIP = 1,
    CREATE_TASK = 2,
    SKIP_S1OUT = 3,
    SKIP_ZERO = 4,
    S2_END = 5,
};

template <typename T> __aicore__ inline T Align(T num, T rnd) {
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b) { return (a > b) ? (b) : (a); }

template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b) { return (a > b) ? (a) : (b); }

template <typename T> __aicore__ inline T CeilDiv(T num, T rnd) {
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

} // namespace const_def
#endif
