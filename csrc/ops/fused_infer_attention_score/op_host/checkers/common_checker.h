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
 * \file common_checker.h
 * \brief
 */

#ifndef COMMON_CHECKER_H
#define COMMON_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

using std::map;
namespace optiling {
class CommonChecker : public BaseChecker {
  public:
    CommonChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant)
        : BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

  private:
    ge::graphStatus CheckShapeConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKVStorageConsistency(const FiaTilingInfo &fiaInfo);
    // 公共校验函数
    ge::graphStatus CheckInputFormat(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckParaExistenceImpl(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc, const std::string &name,
        std::map<std::string, std::vector<ge::DataType>> dataMap);
    ge::graphStatus CheckPAKeyValue(const FiaTilingInfo &fiaInfo);
    bool CheckEmptyTensorList(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorList(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorListBSH(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorListBNSD(const FiaTilingInfo &fiaInfo);
    bool CheckNormalTensorListBSND(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckTensorList(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckMultiDtype(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAxis(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQueryOutConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKeyValueConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckValueOutDConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQueryShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKeyNHVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape);
    ge::graphStatus CheckKeyDVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape);
    ge::graphStatus CheckKeyShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKVContiguous(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckQueryKeyConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQueryKeyTensorlistConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckMultiAttr(const FiaTilingInfo &fiaInfo);
    void GetQueryDimAndOutDim(const gert::StorageShape *queryShape, const gert::StorageShape *outShape,
        const std::string &layoutStr, int64_t &tmpQueryDim, int64_t &outDim, uint32_t i);

    // enableNonQuant 相关校验函数
    ge::graphStatus CheckNonQuantDataType(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAttr(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckDimNum(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckHeadNum(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckInputLayout(const FiaTilingInfo &fiaInfo);
    ge::graphStatus ValidateNoRopeLayoutDim(const FiaTilingInfo &fiaInfo, const std::string &inputLayout);
    ge::graphStatus CheckInnerPrecise(const FiaTilingInfo &fiaInfo);
    bool CheckTNDLayoutCrossover(const FiaTilingInfo &fiaInfo);
    bool CheckNTDLayoutCrossover(const FiaTilingInfo &fiaInfo);
    bool CheckTransposeLayoutCrossover(const FiaTilingInfo &fiaInfo);

  private:
};

} // namespace optiling
#endif // SHAPE_CHECKER_H
