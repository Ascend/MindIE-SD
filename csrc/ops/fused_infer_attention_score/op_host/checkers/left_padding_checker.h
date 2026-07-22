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
 * \file left_padding_checker.h
 * \brief
 */

#ifndef LEFT_PADDING_CHECKER_H
#define LEFT_PADDING_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class LeftPaddingChecker : public BaseChecker {
  public:
    LeftPaddingChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~LeftPaddingChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    // 公共校验函数
    ge::graphStatus CheckShapeSupport(const gert::Tensor *tensor, const std::vector<int64_t> &expectShapeList) const;
    ge::graphStatus CheckExistenceDesc(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureActualLen(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureLayout(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureAlibiPse(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeaturePageAttention(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureQueryS(const FiaTilingInfo &fiaInfo);

    // enableNonQuant 相关校验函数
    ge::graphStatus CheckShapeAndDim(const FiaTilingInfo &fiaInfo);

  private:
};

} // namespace optiling
#endif // LEFT_PADDING_CHECKER_H
