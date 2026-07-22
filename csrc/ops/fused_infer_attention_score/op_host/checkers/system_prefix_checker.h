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
 * \file system_prefix_checker.h
 * \brief
 */

#ifndef SYSTEM_REPFIX_CHECKER_H
#define SYSTEM_REPFIX_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class SystemPrefixChecker : public BaseChecker {
  public:
    SystemPrefixChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~SystemPrefixChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    // singlepara
    ge::graphStatus CheckSharedPrefixDim(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckSharedPrefixDataType(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckSharedPrefixShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckActualSharedPrefixLenData(const FiaTilingInfo &fiaInfo);

    // existence
    ge::graphStatus CheckSharedPrefixExistence(const FiaTilingInfo &fiaInfo);

    // feature
    ge::graphStatus CheckUnSupportFeature(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureAntiquantS1Gt1(
        const FiaTilingInfo &fiaInfo, int64_t keyAntiquantMode, int64_t valueAntiquantMode);
    ge::graphStatus CheckFeatureAntiquantS1Eq1(
        const FiaTilingInfo &fiaInfo, int64_t keyAntiquantMode, int64_t valueAntiquantMode);

    // multipara
  private:
};

} // namespace optiling
#endif // SYSTEM_REPFIX_CHECKER_H
