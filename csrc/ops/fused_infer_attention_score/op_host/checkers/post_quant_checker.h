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
 * \file post_quant_checker.h
 * \brief
 */

#ifndef POST_QUANT_CHECKER_H
#define POST_QUANT_CHECKER_H

#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class PostQuantChecker : public BaseChecker {
  public:
    PostQuantChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PostQuantChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    // 公共校验函数
    ge::graphStatus CheckSingleDtype(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckExistenceQuantScale2(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureAttenOut(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureQueryDType(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureLayout(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureOutputEqual(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeaturePrefix(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureRowValid(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckAntiquantNotSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckMultiParaQuantOffset2(const FiaTilingInfo &fiaInfo) const;

  private:
};

} // namespace optiling
#endif // POST_QUANT_CHECKER_H
