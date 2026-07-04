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
 * \file actual_seq_len_checker.h
 * \brief
 */

#ifndef ACTUAL_SEQ_LEN_CHECKER_H
#define ACTUAL_SEQ_LEN_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class ActualSeqLenChecker : public BaseChecker {
  public:
    ActualSeqLenChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~ActualSeqLenChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    // 公共校验函数
    // singlepara
    ge::graphStatus CheckActualSeqLenQDim(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckActualSeqLenQData(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckActualSeqLenKvDim(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckActualSeqLenKvData(const FiaTilingInfo &fiaInfo);

    // existence
    ge::graphStatus CheckExistenceActualSeqLenQ(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckExistenceActualSeqLenKv(const FiaTilingInfo &fiaInfo);

    // feature
    ge::graphStatus CheckFeatureAlibi(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureIFAMLA(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureFP8GQAFullquant(const FiaTilingInfo &fiaInfo);

    // multipara
    ge::graphStatus CheckActualSeqLenQTNDLastData(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckActualSeqLenKvTNDLastData(const FiaTilingInfo &fiaInfo);

    // general
    int64_t GetActualSeqLengthsQData(const FiaTilingInfo &fiaInfo, uint32_t bIdx);
    int64_t GetActualSeqLengthsKvData(const FiaTilingInfo &fiaInfo, uint32_t bIdx);
};

} // namespace optiling
#endif // ACTUAL_SEQ_LEN_CHECKER_H
