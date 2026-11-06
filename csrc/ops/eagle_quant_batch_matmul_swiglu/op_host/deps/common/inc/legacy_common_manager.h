/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEGACY_COMMMON_MANAGER_H
#define LEGACY_COMMMON_MANAGER_H

#include <string>
#include <dlfcn.h>

namespace Ops {
namespace EagleQbmm {
class LegacyCommonMgr {
  public:
    LegacyCommonMgr();
    ~LegacyCommonMgr();

    // 删除复制构造函数和赋值运算符
    LegacyCommonMgr(const LegacyCommonMgr &) = delete;
    LegacyCommonMgr &operator=(const LegacyCommonMgr &) = delete;

    // 获取实例
    static const LegacyCommonMgr &GetInstance();

    bool IsSupport() const { return isSupported_; }

    /**
     * @brief 获取函数指针
     * @tparam FuncType 函数指针类型
     * @param lib_id 库标识符
     * @param symbol_name 符号名称
     * @return 函数指针
     */
    template <typename FuncType> FuncType GetFunc(const char *symbolName) const {
        if (!isSupported_ || handle_ == nullptr || symbolName == nullptr) {
            return nullptr;
        }

        dlerror(); // 清除之前的错误

        void *symbol = dlsym(handle_, symbolName);
        if (symbol == nullptr) {
            return nullptr;
        }

        return reinterpret_cast<FuncType>(symbol);
    }

  private:
    bool GetLegacyCommonSoPath(std::string &soPath) const;

    static bool GetParentPath(std::string &parentPath, std::string &currSoName);

    bool GetSoPathForBuiltin(const std::string &parentPath, const std::string &currSoName, std::string &soPath) const;

    bool GetSoPathForCustomOp(const std::string &parentPath, const std::string &currSoName, std::string &soPath) const;

    bool GetSoPathForOm(const std::string &parentPath, std::string &soPath) const;

    std::string GetCpuArch() const;

    bool isSupported_ = true;

    void *handle_;
};
} // namespace NN
} // namespace Ops

#endif // LEGACY_COMMMON_MANAGER_H
