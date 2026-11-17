/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under Mulan PSL v2. See LICENSE for details.
 */
#ifndef MINDIESD_FREQUENCY_OP_API_H
#define MINDIESD_FREQUENCY_OP_API_H

#include <dlfcn.h>
#include <cstring>
#include <initializer_list>

namespace mindiesd {

inline void *GetFrequencyRegulatorApi(const char *apiName) {
    constexpr const char *executeApi = "aclnnFrequencyRegulator";
    constexpr const char *workspaceApi = "aclnnFrequencyRegulatorGetWorkspaceSize";
    if (std::strcmp(apiName, executeApi) != 0 && std::strcmp(apiName, workspaceApi) != 0) {
        return nullptr;
    }
    // The executor and workspace API must come from the same library/ABI.
    static void *handler = []() -> void * {
        for (const char *library : {"libopapi_oam.so", "libopapi.so"}) {
            void *candidate = dlopen(library, RTLD_LAZY | RTLD_LOCAL);
            if (candidate == nullptr) {
                continue;
            }
            if (dlsym(candidate, "aclnnFrequencyRegulator") != nullptr &&
                dlsym(candidate, "aclnnFrequencyRegulatorGetWorkspaceSize") != nullptr) {
                return candidate;
            }
            dlclose(candidate);
        }
        return nullptr;
    }();
    return handler == nullptr ? nullptr : dlsym(handler, apiName);
}

} // namespace mindiesd

#endif
