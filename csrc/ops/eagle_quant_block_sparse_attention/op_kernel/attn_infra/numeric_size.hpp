/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_ARCH_NUMERIC_SIZE_HPP
#define NPU_ARCH_NUMERIC_SIZE_HPP

#include "base_defs.hpp"

#include <cstddef>
#include <type_traits>

namespace NpuArch {

#if defined(__CCE__)
#if (__CCE_AICORE__ == 310)
template <typename T> struct SizeOfBits {
    static constexpr size_t value = (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<float4_e2m1x2_t>> ||
                                        std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<float4_e1m2x2_t>>)
        ? size_t{4}
        : AscendC::SizeOfBits<T>::value;
};
#else
using AscendC::SizeOfBits;
#endif
#else
template <typename T> struct SizeOfBits {
    static constexpr size_t value = sizeof(T) * 8;
};
#endif

#if !defined(__CCE__)
#if (__CCE_AICORE__ == 310)
template <> struct SizeOfBits<float4_e2m1x2_t> {
    static constexpr size_t value = 4;
};

template <> struct SizeOfBits<float4_e1m2x2_t> {
    static constexpr size_t value = 4;
};
#endif
#endif

template <typename ReturnType = size_t, typename T> HOST_DEVICE constexpr ReturnType BitsToBytes(T bits) {
    return (static_cast<ReturnType>(bits) + static_cast<ReturnType>(7)) / static_cast<ReturnType>(8);
}

template <typename ReturnType = size_t, typename T> HOST_DEVICE constexpr ReturnType BytesToBits(T bytes) {
    return static_cast<ReturnType>(bytes) * static_cast<ReturnType>(8);
}

} // namespace NpuArch

#endif // NPU_ARCH_NUMERIC_SIZE_HPP
