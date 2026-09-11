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
 * \file tiling_util.cpp
 * \brief
 */

#include "tiling_base/tiling_util.h"
#include "platform/platform_ascendc.h"

#include <sstream>
#include <string>

namespace Ops {
namespace Base {
// vendored matmul tiling 日志引用（原由 ops-transformer 框架库提供）；
// MindIE 无该框架库，缺失会致 libcust_opmaster dlopen 失败。仅日志格式化。
std::string ToString(const gert::Shape &shape)
{
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape.GetDim(i);
    }
    oss << ")";
    return oss.str();
}

std::string ToString(ge::DataType dtype)
{
    std::ostringstream oss;
    oss << static_cast<int32_t>(dtype);
    return oss.str();
}
} // namespace Base

namespace Transformer {
namespace OpTiling {
static const gert::Shape g_vec_1_shape = {1};

static bool IsRegbaseSocVersion(platform_ascendc::SocVersion version)
{
    const static std::set<platform_ascendc::SocVersion> regbaseSocVersions = {
        platform_ascendc::SocVersion::ASCEND950};

    return regbaseSocVersions.find(version) != regbaseSocVersions.end();
}

bool IsRegbaseSocVersion(const gert::TilingParseContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegbaseSocVersion(socVersion);
}

bool IsRegbaseSocVersion(const gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegbaseSocVersion(socVersion);
}

const gert::Shape &EnsureNotScalar(const gert::Shape &inShape) {
  if (inShape.IsScalar()) {
    return g_vec_1_shape;
  }
  return inShape;
}
} // namespace OpTiling
} // namespace Transformer
} // namespace Ops
