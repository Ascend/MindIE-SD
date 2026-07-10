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
 * \file rope_checker.h
 * \brief
 */

#ifndef ROPE_CHECKER_H
#define ROPE_CHECKER_H

#include <map>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class RopeChecker : public BaseChecker {
  public:
    RopeChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~RopeChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    // 公共校验函数
    ge::graphStatus CheckQDsizeSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckRopeDSizeSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckRopeDtype(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckRopeDtypeConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKRopeContiguous(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQKAndQKRopeShapeConsistency(const FiaTilingInfo &fiaInfo, const gert::Shape shape,
        const gert::Shape ropeShape, const std::string &inputName) const;
    ge::graphStatus CheckPAKeyAndKeyRopeShapeConsistency(
        const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape, const gert::Shape &keyRopeShape) const;
    ge::graphStatus CheckTensorlistKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckRopeExistence(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureDecodeMLA(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureAntiQuant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckShapeSupport(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQSSize(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckNSize(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckAxisSupport(const FiaTilingInfo &fiaInfo);

  private:
};

} // namespace optiling
#endif // ROPE_CHECKER_H
