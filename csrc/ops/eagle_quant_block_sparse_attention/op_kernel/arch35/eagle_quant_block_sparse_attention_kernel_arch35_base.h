/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_ARCH35_KERNEL_BASE
#define BSA_ARCH35_KERNEL_BASE

#include "../arch35/kernel_utils.hpp"

using namespace NpuArch;
using namespace tla;

namespace BsaKernelArch35 {

struct BsaRegularKernelArch35QMode1 {};

template <class DispatchPolicy, class EpilogueMask2Idx, class BlockMmadQK, class EpilogueOnlineSoftmax,
    class BlockMmadPV, class EpilogueRescaleO, Format qFormat, Format kvFormat>
class BsaRegularKernelArch35 {};

} // namespace BsaKernelArch35

#endif // BSA_ARCH35_KERNEL_BASE
