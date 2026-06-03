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
 * \file quant_flash_attn_tiling_utils.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_TILING_UTILS_H
#define QUANT_FLASH_ATTN_TILING_UTILS_H

namespace optiling {
template <typename T> inline auto CeilDivision(T num1, T num2) -> T {
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T> inline auto CalcTailSize(T num1, T num2) -> T {
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

template <typename T> inline auto AlignUp(T num1, T num2) -> T {
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T> inline auto increGcd(T a, T b) -> T {
    if (b == 0) {
        return a;
    }
    if (a % b == 0) {
        return b;
    }
    return increGcd(b, a % b);
}

} // namespace optiling

#endif // QUANT_FLASH_ATTN_TILING_UTILS_H
