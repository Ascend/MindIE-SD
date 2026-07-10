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
 * \file aicpu_common.h
 * \brief
 */

#ifndef AICPU_COMMON_H
#define AICPU_COMMON_H

#include "cpu_context.h"

namespace aicpu {
template <typename T> static auto AlignUp(T num1, T num2) -> T {
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, bool>::type GetAttrValue(
    CpuKernelContext &ctx, const std::string &name, T &value) {
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = static_cast<T>(attr->GetInt());
    return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name, std::string &value) {
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetString();
    return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name, bool &value) {
    auto attr = ctx.GetAttr(name);
    if (!attr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetBool();
    return true;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, void>::type GetAttrValueOpt(
    CpuKernelContext &ctx, const std::string &name, T &value) {
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = static_cast<T>(attr->GetInt());
    }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, std::string &value) {
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetString();
    }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, bool &value) {
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetBool();
    }
}
}

#endif
