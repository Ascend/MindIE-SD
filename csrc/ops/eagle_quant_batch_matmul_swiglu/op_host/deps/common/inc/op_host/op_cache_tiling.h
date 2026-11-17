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
 * \file cop_ache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_OP_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_OP_CACHE_TILING_H

#include <array>
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "op_cache_def_tiling.h"

// 兼容opp整包、静态库和子包场景，向算子业务侧代码屏蔽差异：
// 子包场景：通过适配层dlopen方式找到libophost_comm_legacy.so里的C接口实现，向本仓算子侧提供Ops::EagleQbmm namepsace的接口调用
// 整包和静态库场景：只做optiling namespace下的接口声明，向本仓算子侧提供Ops::EagleQbmm namepsace的接口调用
#ifdef NN_ENABLE_DLOPEN_LEGACY
namespace Ops {
namespace EagleQbmm {
const std::string WQBMM_MSD = "wqbmm_msd";
const std::string WQBMM_CUSTOM = "wqbmm_custom";

bool TilingPrepareForOpCache(gert::TilingContext *context);
bool TilingPrepareForOpCache(gert::TilingParseContext *context);

bool GenTiling(const std::string &op_type, const optiling::BatchmatmulCompileParas &compile_params,
    optiling::BatchmatmulRunParas &run_params, optiling::CacheTilingData &tiling, gert::TilingContext *context);

bool CheckSupportConditionQbmm(
    optiling::QbmmType type, optiling::QuantBatchMatmulRunParas &inputParams, uint64_t aicNum, bool supportL0c2Out);

bool GenWqbmmTiling(const std::string &op_type, const optiling::WeightQuantBatchMatmulCacheTilingParas &compile_params,
    optiling::WeightQuantBatchMatmulCacheTilingData &cacheTiling);
} // namespace NN
} // namespace Ops
#else
namespace optiling {
const std::string WQBMM_MSD = "wqbmm_msd";
const std::string WQBMM_CUSTOM = "wqbmm_custom";
bool TilingPrepareForOpCache(gert::TilingContext *context);
bool TilingPrepareForOpCache(gert::TilingParseContext *context);

bool GenTiling(const std::string &op_type, const BatchmatmulCompileParas &compile_params,
    BatchmatmulRunParas &run_params, CacheTilingData &tiling, gert::TilingContext *context);

bool CheckSupportConditionQbmm(
    QbmmType type, QuantBatchMatmulRunParas &inputParams, uint64_t aicNum, bool supportL0c2Out);

bool GenWqbmmTiling(const std::string &op_type, const WeightQuantBatchMatmulCacheTilingParas &compile_params,
    WeightQuantBatchMatmulCacheTilingData &cacheTiling);
} // namespace optiling

namespace Ops {
namespace EagleQbmm {
using optiling::CheckSupportConditionQbmm;
using optiling::GenTiling;
using optiling::GenWqbmmTiling;
using optiling::TilingPrepareForOpCache;
using optiling::WQBMM_CUSTOM;
using optiling::WQBMM_MSD;
} // namespace NN
} // namespace Ops
#endif
#endif
