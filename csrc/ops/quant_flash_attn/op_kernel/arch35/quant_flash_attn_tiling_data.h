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
 * \file flash_attn_tiling_data.h
 * \brief
 */

#ifndef FLASH_ATTN_TILING_REGBASE_H_
#define FLASH_ATTN_TILING_REGBASE_H_

namespace optiling {

// #if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
constexpr uint32_t FA_AIC_CORE_NUM = 36;
constexpr uint32_t FA_AIV_CORE_NUM = 72;

// AICPU metadata format: 16 fields per core (FA and FD both)
constexpr uint32_t FLASH_ATTN_METADATA_SIZE = 16;
constexpr uint32_t FA_FD_METADATA_SIZE = 16;

// FA Metadata Index Definitions (0-based, matching AICPU flash_attn_metadata.h)
// No CORE_ENABLE field in AICPU format; inactive cores have all-zero data.
constexpr uint32_t FLASH_ATTN_BN2_START_INDEX = 0;
constexpr uint32_t FLASH_ATTN_M_START_INDEX = 1;
constexpr uint32_t FLASH_ATTN_S2_START_INDEX = 2;
constexpr uint32_t FLASH_ATTN_BN2_END_INDEX = 3;
constexpr uint32_t FLASH_ATTN_M_END_INDEX = 4;
constexpr uint32_t FLASH_ATTN_S2_END_INDEX = 5;
constexpr uint32_t FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

// FD Metadata Index Definitions (0-based, matching AICPU flash_attn_metadata.h)
// No CORE_ENABLE field; active state is indicated by FA_FD_M_NUM_INDEX > 0.
constexpr uint32_t FA_FD_BN2_IDX_INDEX = 0;
constexpr uint32_t FA_FD_M_IDX_INDEX = 1;
constexpr uint32_t FA_FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t FA_FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t FA_FD_M_START_INDEX = 4;
constexpr uint32_t FA_FD_M_NUM_INDEX = 5;

constexpr uint32_t FA_METADATA_HEADER_OFFSET = 16U * sizeof(uint32_t);

struct FlashAttnBaseParams {
    uint32_t bSize;
    uint32_t t1Size;
    uint32_t t2Size;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t qCuSeqLensSize; /* 用户输入的cu_seqlens_q的长度 */
    uint32_t kvCuSeqLensSize; /* 用户输入的cu_seqlens_kv的长度 */
    uint32_t qSeqUsedSize; /* 用户输入的seqused_q的长度 */
    uint32_t kvSeqUsedSize; /* 用户输入的seqused_kv的长度 */
    float scaleValue;
    uint8_t isSoftMaxLseEnable;
    uint32_t coreNum;
    uint32_t outputLayout;
};

struct FlashAttnAttenMaskParams {
    uint8_t sparseMode;
    int32_t winLefts;
    int32_t winRights;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    uint8_t isRowInvalid;
};

struct FlashAttnPageAttentionParams {
    uint8_t paLayoutType;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
};

struct FlashAttnWorkspaceParams {
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

struct FlashAttnEmptyTensorParams {
    uint32_t singleCoreSize;
    uint8_t needInit;
    uint64_t totalOutputSize;
    uint64_t totalSoftMaxLseOutputSize;
};

struct FlashAttnMetaData {
    uint32_t FAMetadata[FA_AIC_CORE_NUM][FLASH_ATTN_METADATA_SIZE];
    uint32_t FDMetadata[FA_AIV_CORE_NUM][FA_FD_METADATA_SIZE];
};

class FlashAttnTilingData {
  public:
    FlashAttnBaseParams flashAttnBaseParams;
    FlashAttnAttenMaskParams flashAttnAttenMaskParams;
    FlashAttnPageAttentionParams flashAttnPageAttentionParams;
    FlashAttnWorkspaceParams flashAttnWorkspaceParams;
    FlashAttnEmptyTensorParams flashAttnEmptyTensorParams;
};

class QuantFlashAttnTilingData {
  public:
    FlashAttnTilingData baseTiling;
    FlashAttnMetaData flashAttnMetaData;
};

} // namespace optiling
#endif
