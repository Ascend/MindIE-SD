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
 * \file flash_attn_tiling_info_parser.h
 * \brief
 */

#pragma once

#include "quant_flash_attn_tiling_info.h"
#include "../../common/op_host/fia_tiling_shape.h"

namespace optiling {
class QuantFlashAttnTilingInfoParser {
  public:
    explicit QuantFlashAttnTilingInfoParser(const gert::TilingContext *context, QuantFlashAttnTilingInfo &faInfo)
        : context_(context), tilingInfo_(faInfo) {}
    ~QuantFlashAttnTilingInfoParser() = default;
    ge::graphStatus Parse();

  private:
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckOptionalInputExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetCuSeqLenQDims();
    ge::graphStatus GetCuSeqLenKvDims();
    ge::graphStatus GetSeqUsedQDims();
    ge::graphStatus GetSeqUsedKvDims();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetValueHeadDim();
    void GetQueryTSize();
    void GetKeyTSize();
    ge::graphStatus GetMaxSeqLenQ();
    ge::graphStatus GetMaxSeqLenKv();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetInAndOutLayout();
    void GetPreNextToken();
    ge::graphStatus GetQkvDataType();
    void SetFaShape();
    void GetKvStorageMode();
    void GetSoftmaxScale();
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus ParseFeatureInfo();

  private:
    const gert::TilingContext *context_ = nullptr;
    QuantFlashAttnTilingInfo &tilingInfo_;

    // NPU信息
    NpuArch npuArch_ = NpuArch::DAV_3510;

    // shape信息
    std::shared_ptr<FiaTilingShape> queryShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> valueShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> qDescaleShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> kDescaleShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> vDescaleShape_ = nullptr;
};
} // namespace optiling
