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
 * \file base_checker.h
 * \brief
 */

#ifndef BASE_CHECKER_H
#define BASE_CHECKER_H

#include <numeric>
#include "tiling/tiling_api.h"

#include "../../../common/op_host/fia_tiling_info.h"
#include "../../../common/op_host/fia_tiling_shape.h"
#include "../fused_infer_attention_score_tiling_utils.h"

namespace optiling {
class BaseChecker {
  public:
    BaseChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : enableNonQuant_(enableNonQuant), enableFullQuant_(enableFullQuant), enableAntiQuant_(enableAntiQuant) {}
    virtual ~BaseChecker() = default;

  protected:
    virtual ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) = 0;
    virtual ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) = 0;
    virtual ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) = 0;
    virtual ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) = 0;

    // public check funcs
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckFormatSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    template <typename T> ge::graphStatus CheckValueSupport(const T value, const std::vector<T> &expectValList) const;
    ge::graphStatus CheckTensorContiguous(
        const uint32_t &tensorDimNum, const gert::Shape &inputShape, const gert::Stride *Strides, int32_t &index) const;

    // public funcs
    std::string DataTypeToSerialString(ge::DataType type) const;

  protected:
    bool enableNonQuant_ = false;
    bool enableFullQuant_ = false;
    bool enableAntiQuant_ = false;
};
} // namespace optiling
#endif // BASE_CHECKER_H
