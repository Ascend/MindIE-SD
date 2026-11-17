/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../inc/legacy_common_manager.h"

#include <set>
#include <dlfcn.h>
#include <dirent.h>
#include <limits.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "log/log.h"

// 兼容opp整包场景：整包不编译本文件，仅子包编译
namespace {
static const std::string OPHOST_BUILTIN_SO_NAME = "libophost_nn.so";
static const std::string OPHOST_CUSTOM_SO_NAME = "libcust_opmaster_rt2.0.so";
static const std::string OPAPI_BUILTIN_SO_NAME = "libopapi_nn.so";
static const std::string OPAPI_CUSTOM_SO_NAME = "libcust_opapi.so";
static const std::string LEGACY_SO_NAME = "libophost_comm_legacy.so";
static const std::string OPHOST_PATH = "/built-in/op_impl/ai_core/tbe/op_host/lib/linux/";
const static std::set<std::string> SUPPORTED_VERS = {
    "ascend910b", "ascend910_93", "ascend310p", "ascend310b", "ascend910"};
static const Ops::EagleQbmm::LegacyCommonMgr LEGACY_COMMMON_MGR;
} // namespace

namespace Ops {
namespace EagleQbmm {
const LegacyCommonMgr &LegacyCommonMgr::GetInstance() {
    // 简单修改，从单例改为返回静态常量，确保本so在被dlopen时就触发对libophost_comm_legacy.so的dlopen
    // GE的atc转om场景下，om在推理时so被释放到临时目录下，加载完删除，因此不能只靠单例被业务侧调用来触发对legacy的dlopen
    return LEGACY_COMMMON_MGR;
}

LegacyCommonMgr::LegacyCommonMgr() {
    handle_ = nullptr;
#if defined(ASCEND_COMPUTE_UNIT)
    isSupported_ = SUPPORTED_VERS.find(ASCEND_COMPUTE_UNIT) != SUPPORTED_VERS.end();
#endif

    std::string soPath;
    if (isSupported_ && GetLegacyCommonSoPath(soPath)) {
        handle_ = dlopen(soPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle_ == nullptr) {
            OP_LOGW("LegacyCommonMgr", "Fail to dlopen %s, reason: %s.", soPath.c_str(), dlerror());
        }
    }
}

LegacyCommonMgr::~LegacyCommonMgr() {
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

bool LegacyCommonMgr::GetLegacyCommonSoPath(std::string &soPath) const {
    // SD installs its vendor beside the Python package, outside the toolkit's
    // opp tree. Resolve the built-in legacy dependency from the active toolkit.
    const char *opp = std::getenv("ASCEND_OPP_PATH");
    if (opp != nullptr && opp[0] != '\0') {
        soPath = std::string(opp) + OPHOST_PATH + GetCpuArch() + "/" + LEGACY_SO_NAME;
        if (access(soPath.c_str(), R_OK) == 0) {
            return true;
        }
    }
    std::string parentPath;
    std::string currSoName;
    if (!GetParentPath(parentPath, currSoName)) {
        OP_LOGW("LegacyCommonMgr", "Fail to get parent path of ophost so.");
        return false;
    }

    if (currSoName != OPHOST_BUILTIN_SO_NAME && currSoName != OPHOST_CUSTOM_SO_NAME &&
        currSoName != OPAPI_BUILTIN_SO_NAME && currSoName != OPAPI_CUSTOM_SO_NAME) {
        OP_LOGW("LegacyCommonMgr", "Invalid current so name[%s].", currSoName.c_str());
        return false;
    }

    if (!GetSoPathForBuiltin(parentPath, currSoName, soPath) && !GetSoPathForCustomOp(parentPath, currSoName, soPath) &&
        !GetSoPathForOm(parentPath, soPath)) {
        OP_LOGW("LegacyCommonMgr", "Fail to get %s path.", LEGACY_SO_NAME.c_str());
        return false;
    }

    char soRealPath[PATH_MAX] = {0};
    if (realpath(soPath.c_str(), soRealPath) == nullptr) {
        OP_LOGW("LegacyCommonMgr", "Fail to get realpath of [%s].", soPath.c_str());
        return false;
    }
    soPath.assign(soRealPath);
    OP_LOGI("LegacyCommonMgr", "get %s realpath[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
    return true;
}

bool LegacyCommonMgr::GetParentPath(std::string &parentPath, std::string &currSoName) {
    Dl_info dlInfo;
    if (dladdr(reinterpret_cast<const void *>(&LegacyCommonMgr::GetParentPath), &dlInfo) == 0 ||
        dlInfo.dli_fname == nullptr) {
        OP_LOGW("LegacyCommonMgr", "Fail to get current so path from dladdr.");
        return false;
    } else {
        OP_LOGD("LegacyCommonMgr", "current so path[%s].", dlInfo.dli_fname);

        char soPath[PATH_MAX] = {0};
        if (realpath(dlInfo.dli_fname, soPath) == nullptr) {
            OP_LOGW("LegacyCommonMgr", "Fail to get realpath of [%s].", dlInfo.dli_fname);
            return false;
        }

        parentPath.assign(soPath);
        auto pos = parentPath.rfind('/');
        if (pos != std::string::npos) {
            currSoName = parentPath.substr(pos + 1);
            parentPath.erase(pos);
        }
        OP_LOGD("LegacyCommonMgr", "current so[%s] parent path[%s].", currSoName.c_str(), parentPath.c_str());
        return true;
    }
}

bool LegacyCommonMgr::GetSoPathForBuiltin(
    const std::string &parentPath, const std::string &currSoName, std::string &soPath) const {
    if (currSoName == OPHOST_BUILTIN_SO_NAME) {
        // ophost built-in场景两个so在相同路径下
        soPath = parentPath + std::string("/") + LEGACY_SO_NAME;
        OP_LOGD("LegacyCommonMgr", "try to find %s by path[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
        if (access(soPath.c_str(), F_OK) == 0) {
            OP_LOGI("LegacyCommonMgr", "get %s path[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
            return true;
        }
    }

    if (currSoName == OPAPI_BUILTIN_SO_NAME) {
        // opapi场景需要先向上找到opp入口
        const char *oppPath = "/../../opp";
        soPath = parentPath + std::string(oppPath) + OPHOST_PATH + GetCpuArch() + std::string("/") + LEGACY_SO_NAME;
        OP_LOGD("LegacyCommonMgr", "try to find %s by path[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
        if (access(soPath.c_str(), F_OK) == 0) {
            OP_LOGI("LegacyCommonMgr", "get %s path[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
            return true;
        }
    }

    return false;
}

bool LegacyCommonMgr::GetSoPathForCustomOp(
    const std::string &parentPath, const std::string &currSoName, std::string &soPath) const {
    // 自定义算子场景需要先向上找到opp入口
    std::string oppPath;
    if (currSoName == OPHOST_CUSTOM_SO_NAME) {
        // 路径：opp/vendors/custom_nn/op_impl/ai_core/tbe/op_tiling/lib/linux/$(arch)/libcust_opmaster_rt2.0.so
        oppPath = "/../../../../../../../../../../opp";
    } else if (currSoName == OPAPI_CUSTOM_SO_NAME) {
        // 路径：opp/vendors/custom_nn/op_api/lib/libcust_opapi.so
        oppPath = "/../../../../../opp";
    } else {
        return false;
    }

    soPath = parentPath + oppPath + OPHOST_PATH + GetCpuArch() + std::string("/") + LEGACY_SO_NAME;
    OP_LOGD("LegacyCommonMgr", "try to find %s by path[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
    if (access(soPath.c_str(), F_OK) == 0) {
        OP_LOGI("LegacyCommonMgr", "get %s path[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
        return true;
    }

    return false;
}

bool LegacyCommonMgr::GetSoPathForOm(const std::string &parentPath, std::string &soPath) const {
    std::string omSoRootPath(parentPath);
    auto pos = omSoRootPath.rfind('/');
    if (pos != std::string::npos) {
        omSoRootPath.erase(pos);
    }

    struct dirent *entry;
    DIR *dir = opendir(omSoRootPath.c_str());
    if (dir == nullptr) {
        OP_LOGW("LegacyCommonMgr", "Fail to open om so root path[%s].", omSoRootPath.c_str());
        return false;
    }

    // 目录结构：$HOME/.ascend_temp/.om_exe_data/${pid}_${tid}_${opId}/libxxx.so，遍历查找libophost_comm_legacy.so
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        soPath = omSoRootPath + std::string("/") + entry->d_name + std::string("/") + LEGACY_SO_NAME;
        OP_LOGD("LegacyCommonMgr", "try to find %s by path[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
        if (access(soPath.c_str(), F_OK) == 0) {
            OP_LOGI("LegacyCommonMgr", "get %s path[%s].", LEGACY_SO_NAME.c_str(), soPath.c_str());
            (void)closedir(dir);
            return true;
        }
    }

    (void)closedir(dir);
    return false;
}

std::string LegacyCommonMgr::GetCpuArch() const {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#else
    return "x86_64";
#endif
}

} // namespace NN
} // namespace Ops
