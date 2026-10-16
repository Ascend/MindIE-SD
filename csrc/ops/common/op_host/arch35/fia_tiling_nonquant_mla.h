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
 * \file fia_tiling_noquant_mla.h
 * \brief
*/

#ifndef FIA_TILING_NONQUANT_MLA_H
#define FIA_TILING_NONQUANT_MLA_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../fia_tiling_base.h"
#include "../fia_tiling_info.h"
#include "tiling/tiling_api.h" // 这个头文件顺序必须在手写的tiling data前
#include "../../op_kernel/arch35/fia_tiling_data_noquant_gqa.h"
#include "../split_core_v2.h"
#include "../../../fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tiling_key.h"
#include "../../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_tiling_regbase.h"

namespace optiling {
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

struct FiaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t config = 0;
    uint64_t pseMode = 0;
    uint64_t quantMode = 31;
    bool hasAttenMask = false;
    bool hasRope = false;
    bool isPa = false;
    bool isFd = false;
    bool emptyTensor = false;
    uint64_t maskMode = 0;
    uint64_t matmulMode = 0;
    bool enableKvPrefix = false;
    bool enableS1OutSplit = false;
    bool isReconstructTemp = false;
};

struct FiaPlatFormInfo {
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

class FiaTilingNonQuantMlaArch35 : public FiaTilingBase {
  public:
    explicit FiaTilingNonQuantMlaArch35(gert::TilingContext *context) : FiaTilingBase(context) {}
    ~FiaTilingNonQuantMlaArch35() override = default;

  protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override;
    bool IsCapable() override;
    bool IsCapableBasicCheckMla();
    bool IsCapableFeatureCheckMla();
    bool IsCapableSparseLayoutCheckMla();
    ge::graphStatus DoOpTiling() override;

  private:
    ge::graphStatus SetPlatMemoryInfo();
    void SplitPolicy();
    bool CheckS1OutSplit();
    bool IsQuantStorageMlaExcluded();
    bool IsFeatureFlagExcluded();
    bool IsSparseModeExcluded();
    void SplitOutSeq();
    void ComputeTilingData();
    void SetAttenMaskTilingData();
    void SetStartIdxTilingData();
    void SetPageAttentionLayoutTilingData();
    void CalcWorkspaceSize();
    void GenTilingKey();
    void UpdateTilingKeyConfig();
    void UpdateTilingKeyLayout();
    void UpdateTilingKeyPseMode();
    void UpdateTilingKeyQuantMode();
    void UpdateTilingKeyHasRope();
    void UpdateTilingKeyInfo();
    void SetFATilingData();

    void AdjustSinnerAndSouter();
    void GetPreNextTokensLeftUp(
        int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp);
    void FixParamWithRowInvalid(
        int64_t &actualSeqLength, int64_t actualSeqLengthKV, int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp);
    void SplitNBSeq(const FiaTilingInfo &fiaInfo);
    void InitImplParam();
    bool CheckEnableDN();
    bool CheckQKVActualSeqLengthsRight();
    void PrintAllTilingData();
    void FillTiling();
    void CalcScheduleMode();
    void CalcMaxWorkspaceSize();
    void CalcNumBlocks(uint32_t coreNum);
    ge::graphStatus SetTilingData(FusedInferAttentionScoreTilingData &tilingData);

    FusedInferAttentionScoreTilingData tilingData_;

    FiaTilingKeyInfo tilingKeyInfo_;
    FiaPlatFormInfo platformInfo_;
    uint32_t nLoopTimes_;
    uint32_t gsSize_;
    uint32_t sOuterFactor_;
    uint32_t sInnerFactor_;
    uint32_t sInnerFactorSize_;
    uint32_t blockDim_;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t headDimAlign_ = 0;
    bool flashDecodeFlag_ = false;
    bool dnFlag_ = false;
    bool gsMergeFlag_ = false;
    bool pfaMergeFlag_ = false;
    bool isQKVActualSeqLengthsRightFlag_ = false;
    bool actualSeqLenQFlag_ = false;
    bool actualSeqLenKVFlag_ = false;
    bool actualSharedPrefixLenFlag_ = false;
    std::vector<int64_t> actualSeqLengthsQ_ = {};
    std::vector<int64_t> actualSeqLengthsKV_ = {};
    bool fromPFA_ = false;
    bool isPFAFlag_ = false;
    bool isIFAFlag_ = false;
    bool needInit_ = false;
    bool enableS1OutSplit = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::NORMAL_MODE;
    int32_t numBlocks_ = 0;

    // Tiling Info
    FiaTilingInfo *fiaInfo_ = nullptr;
};

} // namespace optiling
#endif // FIA_TILING_NONQUANT_MLA_H
