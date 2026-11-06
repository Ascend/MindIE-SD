/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform_util.h
 * \brief
 */
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <tiling/platform/platform_ascendc.h>
#include "platform/soc_spec.h"
#include "eagle_quant_batch_matmul_swiglu_compile_info.h"

namespace optiling {

class EagleQbmmPlatformUtil {
  public:
    static void ParseRuntimePlatformInfo(optiling::EagleQuantBatchMatmulSwigluCompileInfo &compileInfo,
        const char *op_name, fe::PlatFormInfos &platform_info);

  private:
    static void GetLocalMemSize(
        fe::PlatFormInfos &platform_info, const std::string &lable, const std::string &mem_type, uint64_t &size);
};
} // namespace optiling
