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
 * \file op_cache_tiling.cpp
 * \brief
 */

#include "../../inc/op_host/op_cache_tiling.h"
#include "../../inc/legacy_common_manager.h"
#include "log/log.h"

// 兼容opp整包场景：整包不编译本文件，仅子包编译
namespace Ops {
namespace EagleQbmm {
bool TilingPrepareForOpCache(gert::TilingContext *context) {
    if (!Ops::EagleQbmm::LegacyCommonMgr::GetInstance().IsSupport()) {
        return false;
    }
    using FuncType = bool (*)(gert::TilingContext *);
    const char *symbolName = "LegacyTilingPrepareForOpCache";
    static FuncType func = Ops::EagleQbmm::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(context);
    }
}

bool TilingPrepareForOpCache(gert::TilingParseContext *context) {
    if (!Ops::EagleQbmm::LegacyCommonMgr::GetInstance().IsSupport()) {
        return false;
    }
    using FuncType = bool (*)(gert::TilingParseContext *);
    const char *symbolName = "LegacyTilingParsePrepareForOpCache";
    static FuncType func = Ops::EagleQbmm::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(context);
    }
}

bool GenTiling(const std::string &op_type, const optiling::BatchmatmulCompileParas &compile_params,
    optiling::BatchmatmulRunParas &run_params, optiling::CacheTilingData &tiling, gert::TilingContext *context) {
    if (!Ops::EagleQbmm::LegacyCommonMgr::GetInstance().IsSupport()) {
        return false;
    }
    using FuncType = bool (*)(const std::string &, const optiling::BatchmatmulCompileParas &,
        optiling::BatchmatmulRunParas &, optiling::CacheTilingData &, gert::TilingContext *);
    const char *symbolName = "LegacyGenTbeMatmulTiling";
    static FuncType func = Ops::EagleQbmm::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(op_type, compile_params, run_params, tiling, context);
    }
}

bool CheckSupportConditionQbmm(
    optiling::QbmmType type, optiling::QuantBatchMatmulRunParas &inputParams, uint64_t aicNum, bool supportL0c2Out) {
    if (!Ops::EagleQbmm::LegacyCommonMgr::GetInstance().IsSupport()) {
        return false;
    }
    using FuncType = bool (*)(optiling::QbmmType, optiling::QuantBatchMatmulRunParas &, uint64_t, bool);
    const char *symbolName = "LegacyCheckSupportConditionQbmm";
    static FuncType func = Ops::EagleQbmm::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(type, inputParams, aicNum, supportL0c2Out);
    }
}

bool GenWqbmmTiling(const std::string &op_type, const optiling::WeightQuantBatchMatmulCacheTilingParas &compile_params,
    optiling::WeightQuantBatchMatmulCacheTilingData &cacheTiling) {
    if (!Ops::EagleQbmm::LegacyCommonMgr::GetInstance().IsSupport()) {
        return false;
    }
    using FuncType = bool (*)(const std::string &, const optiling::WeightQuantBatchMatmulCacheTilingParas &,
        optiling::WeightQuantBatchMatmulCacheTilingData &);
    const char *symbolName = "LegacyGenWqbmmTiling";
    static FuncType func = Ops::EagleQbmm::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return false;
    } else {
        return func(op_type, compile_params, cacheTiling);
    }
}
} // namespace NN
} // namespace Ops
