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
 * \file quant_flash_attn_tiling_dn.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_TILING_H
#define QUANT_FLASH_ATTN_TILING_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../quant_flash_attn_tiling_info.h"
#include "tiling/tiling_api.h" //这个头文件顺序必须在手写的tiling data前
#include "../../op_kernel/arch35/quant_flash_attn_tiling_data.h"

namespace optiling {

struct FaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t config = 0;
    uint64_t pseMode = 0;
    uint64_t quantMode = 31;
    bool hasAttenMask = false;
    bool hasRope = false;
    uint64_t kvLayoutType = 0;
    bool isFd = false;
    bool emptyTensor = false;
    uint64_t maskMode = 0;
    uint64_t matmulMode = 0;
    bool enableKvPrefix = false;
    bool enableS1OutSplit = false;
};

class QuantFlashAttnTilingDn : public FiaTilingBase {
  public:
    explicit QuantFlashAttnTilingDn(gert::TilingContext *context) : FiaTilingBase(context) {}
    ~QuantFlashAttnTilingDn() override = default;

  protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

  private:
    ge::graphStatus SetPlatMemoryInfo();
    void SplitPolicy();
    void GenTilingKey();
    void CalcWorkspaceSize();
    void InitImplParam();
    void CalcScheduleMode();
    void CalcNumBlocks(uint32_t coreNum);
    void FillTiling();
    ge::graphStatus SetTilingData(QuantFlashAttnTilingData &tilingData);

    QuantFlashAttnTilingData tilingData_;
    QfaPlatFormInfo platformInfo_;
    FaTilingKeyInfo tilingKeyInfo_;

    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    int32_t numBlocks_ = 0;

    // Tiling Info
    QuantFlashAttnTilingInfo *tilingInfo_ = nullptr;
};

} // namespace optiling
#endif
