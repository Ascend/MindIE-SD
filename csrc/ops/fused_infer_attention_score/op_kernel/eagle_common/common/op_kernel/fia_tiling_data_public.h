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
 * \file fia_tiling_data_public.h
 * \brief
 */

#ifndef FIA_TILING_DATA_PUBLIC_H_
#define FIA_TILING_DATA_PUBLIC_H_

namespace optiling {

// 数组长度
// TODO，host和device宏定义不一样，如何通过编译宏隔离？
constexpr uint32_t NPU_AIC_CORE_NUM = 36;
constexpr uint32_t NPU_AIV_CORE_NUM = 72;

constexpr uint32_t FA_METADATA_SIZE = 8;
constexpr uint32_t FD_METADATA_SIZE = 8;

// 索引数组含义
//  FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_S2_SPLIT_NUM_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

struct stridesParams {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;

    void set_bnStride(uint64_t bnStride) { this->bnStride = bnStride; }
    uint64_t get_bnStride() const { return bnStride; }
    void set_n2Stride(uint64_t n2Stride) { this->n2Stride = n2Stride; }
    uint64_t get_n2Stride() const { return n2Stride; }
};

struct FiaBaseParams {
    uint32_t bSize = 0;
    uint32_t t1Size = 0;
    uint32_t t2Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dSizeRope = 0;
    uint32_t actualSeqLengthsQSize = 0;
    uint32_t actualSeqLengthsKVSize = 0;
    float scaleValue = 0.0f;
    uint8_t isKvContinuous = 0;
    uint8_t isSoftMaxLseEnable = 0;
    uint32_t coreNum = 0;
    uint32_t outputLayout = 0;
    // 增加strides参数
    stridesParams keyStrides;
    stridesParams valueStrides;
    stridesParams kRopeStrides;
    stridesParams kScaleStrides;
    stridesParams vScaleStrides;
};

struct FiaAttenMaskParams {
    uint8_t sparseMode = 0;
    int32_t preTokens = 0;
    int32_t nextTokens = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    uint8_t isRowInvalidOpen = 0;
    uint8_t isExistRowInvalid = 0;
};

struct FiaPseParams {
    uint8_t pseShiftByBatch = 0;
    uint32_t pseS1Size = 0;
    uint32_t pseS2Size = 0;
    uint32_t pseStride = 0;
    uint32_t qStartIdx = 0;
    uint32_t kvStartIdx = 0;
};

struct FiaSystemPrefixParams {
    uint8_t isActualSharedPrefixLenNull = 0;
    uint32_t prefixSeqInnerSize = 0;
};

struct FiaPageAttentionParams {
    uint8_t paLayoutType = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
};

struct FiaLeftPaddingParams {
    uint8_t isQHasLeftPadding = 0;
    uint8_t isKVHasLeftPadding = 0;
};

struct FiaPostQuantParams {
    uint8_t isPostQuantPerChnl = 0;
    uint8_t isPostQuantBF16 = 0;
};

struct FiaWorkspaceParams {
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct FiaS1OuterSplitCoreParams {
    bool enableS1OutSplit = false;
    uint64_t totalSize = 0;
};

struct FiaEmptyTensorParams {
    uint32_t singleCoreSize = 0;
    uint8_t needInit = 0;
    uint64_t totalOutputSize = 0;
    uint64_t totalSoftMaxLseOutputSize = 0;
};

struct FiaMetaData {
    uint32_t FAMetadata[NPU_AIC_CORE_NUM][FA_METADATA_SIZE] = {0};
    uint32_t FDMetadata[NPU_AIV_CORE_NUM][FD_METADATA_SIZE] = {0};
};
} // namespace optiling
#endif
