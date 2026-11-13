/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
 * \file platform_util.cc
 * \brief function of platform_util
 */

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include "platform/platform_info.h"
#include "log/log.h"
#include "platform_util.h"

namespace optiling {
void EagleQbmmPlatformUtil::GetLocalMemSize(
    fe::PlatFormInfos &platform_info, const string &lable, const string &mem_type, uint64_t &size) {
    std::string temp_str;
    size = 0;
    if (platform_info.GetPlatformRes(lable, mem_type, temp_str)) {
        OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", mem_type.c_str(), temp_str.c_str());
        errno = 0;
        char *end_ptr = nullptr;
        const unsigned long long parsed = std::strtoull(temp_str.c_str(), &end_ptr, 10);
        const size_t first_non_space = temp_str.find_first_not_of(" \t\n\r\f\v");
        const bool is_negative = first_non_space != std::string::npos && temp_str[first_non_space] == '-';
        if (temp_str.empty() || is_negative || end_ptr == temp_str.c_str() || *end_ptr != '\0' || errno == ERANGE) {
            OP_LOGE_WITHOUT_REPORT("NO_OP_NAME", "illegal PLATFORM INFO %s: %s", mem_type.c_str(), temp_str.c_str());
            size = 0;
        } else {
            size = static_cast<uint64_t>(parsed);
        }
    } else {
        OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", mem_type.c_str());
    }
}

void EagleQbmmPlatformUtil::ParseRuntimePlatformInfo(optiling::EagleQuantBatchMatmulSwigluCompileInfo &compileInfo,
    const char *op_name, fe::PlatFormInfos &platform_info) {
    (void)op_name;
    platform_info.GetLocalMemSize(fe::LocalMemType::UB, compileInfo.ubSize);
    // SD 910B reserves 8192 bytes for framework use and 256 bytes for alignment.
    compileInfo.ubSize = std::min<uint64_t>(compileInfo.ubSize, 188160);
    platform_info.GetLocalMemSize(fe::LocalMemType::L1, compileInfo.l1Size);
    platform_info.GetLocalMemSize(fe::LocalMemType::L2, compileInfo.l2Size);
    platform_info.GetLocalMemSize(fe::LocalMemType::L0_A, compileInfo.l0aSize);
    platform_info.GetLocalMemSize(fe::LocalMemType::L0_B, compileInfo.l0bSize);
    platform_info.GetLocalMemSize(fe::LocalMemType::L0_C, compileInfo.l0cSize);
}
} // namespace optiling
