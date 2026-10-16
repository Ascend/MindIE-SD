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
 * \file kernel_utils.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_ASCENDC_KERNEL_UTILS_H_
#define OPS_BUILT_IN_OP_ASCENDC_KERNEL_UTILS_H_

namespace ops {

template<typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template<typename, typename>
struct IsSame
    : public falseType {
};
template<typename Tp>
struct IsSame<Tp, Tp>
    : public trueType {
};

template <typename T>
__aicore__ inline T Ceil(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T FloorDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b;
}

template <typename T>
__aicore__ inline T Aligned(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

}
#endif  // OPS_BUILT_IN_OP_ASCENDC_KERNEL_UTILS_H_