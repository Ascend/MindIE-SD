/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_compile_info.h
 * \brief
 */
#ifndef __OP_HOST_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_COMPILE_INFO_H__
#define __OP_HOST_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_COMPILE_INFO_H__

#include <cstdint>
#include <string>

#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"

namespace optiling {

struct EagleQuantBatchMatmulSwigluCompileInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0cSize;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint32_t workspaceNum;
    uint32_t aivNum;
    uint32_t aicNum;
    bool supportL0c2Out;
    bool supportL12BtBf16;
    bool supportMmadS8S4;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
    std::string socVersionStr = "";
};

} // namespace optiling
#endif // __OP_HOST_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_COMPILE_INFO_H__
