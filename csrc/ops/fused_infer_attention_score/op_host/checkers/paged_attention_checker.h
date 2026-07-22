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
 * \file paged_attention_checker.h
 * \brief
 */

#ifndef PAGED_ATTENTION_CHECKER_H
#define PAGED_ATTENTION_CHECKER_H

#include <map>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class PagedAttentionChecker : public BaseChecker {
  public:
    PagedAttentionChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    // 公共校验函数

    ge::graphStatus CheckBlockTableDtype(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckBlockTableShapeSize(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckBlockSize(const FiaTilingInfo &fiaInfo) const;

    ge::graphStatus CheckBlockTableExistence(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckSeqLengthKVExistence(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckQDtypeSupport(const FiaTilingInfo &fiaInfo);

    int64_t GetMaxBlockNumPerBatch(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckMaskShape(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckBlockTableShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckBlockSizeSupport(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKVLayout(const FiaTilingInfo &fiaInfo) const;

    ge::graphStatus CheckPACacheShape(
        const FiaTilingInfo &fiaInfo, const gert::Shape tempShape, const std::string &inputName) const;
    ge::graphStatus CheckFeatureQueryS(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureInputLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const;
};

} // namespace optiling
#endif // PAGED_ATTENTION_CHECKER_H
