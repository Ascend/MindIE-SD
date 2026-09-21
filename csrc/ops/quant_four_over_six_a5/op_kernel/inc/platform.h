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
 * \file platform.h
 * \brief platform apator
 */
#ifndef OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_
#define OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

#ifndef KERNEL_API
#define KERNEL_API extern "C" __global__ __aicore__
#endif

namespace platform {

#define MID_THREAD_NUM 1024

__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
#if __CCE_AICORE__ == 220
    return true;
#else
    return false;
#endif
}

/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

/**
 * Check whether the type is supported by atomic add for simd
 */
template<typename T>
__aicore__ inline constexpr bool IsSupportAtomicAddTypeSIMD()
{
#if __CCE_AICORE__ == 310
    return ops::IsSame<T, float>::value || ops::IsSame<T, half>::value || ops::IsSame<T, int16_t>::value ||
        ops::IsSame<T, int32_t>::value || ops::IsSame<T, int8_t>::value || ops::IsSame<T, bfloat16_t>::value;
#else
    return false;
#endif
}

} // namespace platform

namespace PlatformSocInfo {
__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
    return platform::IsDataCopyPadSupport();
}

}

#endif  // OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_