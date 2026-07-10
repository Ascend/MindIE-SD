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
 * \file mask_checker.h
 * \brief
 */

#ifndef MASK_CHECKER_H
#define MASK_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class MaskChecker : public BaseChecker {
  public:
    MaskChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~MaskChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    // 公共校验函数
    struct MaskInfo {
        int64_t attenMaskN = 1U;
        uint32_t attenMaskBatch = 1;
        uint32_t attenMaskQSize = 0;
        uint32_t attenMaskSize = 0;
        std::string strMaskShape;
    };
    ge::graphStatus CheckDtypeAndFormat(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckSparseMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckNoQuantIFAMLA(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFullQuantIFAMLA(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckMXFP8FullQuant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFP8GQAFullQuant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQKVDDifferent(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureSparseMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckPretokenAndNexttoken(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckIFADimAndShape(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus GetMaskInfo(const FiaTilingInfo &fiaInfo, MaskInfo &maskInfo) const;
    ge::graphStatus CheckDimAndShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAntiquantSparseMode(const FiaTilingInfo &fiaInfo) const;

  private:
    bool enableIFAMLA = false;
    bool isIFAFlag = false;
    bool enableMXFP8 = false;
    bool enableFP8GQAFullQuant = false;
};

} // namespace optiling
#endif // MASK_CHECKER_H
