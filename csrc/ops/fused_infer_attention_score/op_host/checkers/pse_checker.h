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
 * \file pse_checker.h
 * \brief
 */

#ifndef PSE_CHECKER_H
#define PSE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class PSEChecker : public BaseChecker {
  public:
    PSEChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PSEChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    // singlepara
    ge::graphStatus CheckPseType(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckPseShiftDataType(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckPseShiftShape(const FiaTilingInfo &fiaInfo);
    // existence
    ge::graphStatus CheckPseShiftExistence(const FiaTilingInfo &fiaInfo);
    // feature
    ge::graphStatus CheckFeaturePA(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckerFeatureCrossover(const FiaTilingInfo &fiaInfo);
    // multipara
    ge::graphStatus CheckAlibiStartIdx(const FiaTilingInfo &fiaInfo);

  private:
};

} // namespace optiling
#endif // PSE_CHECKER_H
