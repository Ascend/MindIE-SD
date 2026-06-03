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
 * \file quant_flash_attn_metadata.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_METADATA_H
#define QUANT_FLASH_ATTN_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t QFA_META_SIZE = 1024;
using QFA_METADATA_T = uint32_t;

constexpr uint32_t QFA_METADATA_SIZE = 16;
constexpr uint32_t QFD_METADATA_SIZE = 16;

// QFA Metadata Index Definitions
constexpr uint32_t QFA_BN2_START_INDEX = 0;
constexpr uint32_t QFA_M_START_INDEX = 1;
constexpr uint32_t QFA_S2_START_INDEX = 2;
constexpr uint32_t QFA_BN2_END_INDEX = 3;
constexpr uint32_t QFA_M_END_INDEX = 4;
constexpr uint32_t QFA_S2_END_INDEX = 5;
constexpr uint32_t QFA_FIRST_QFD_DATA_WORKSPACE_IDX_INDEX = 6;

// QFD Metadata Index Definitions
constexpr uint32_t QFD_BN2_IDX_INDEX = 0;
constexpr uint32_t QFD_M_IDX_INDEX = 1;
constexpr uint32_t QFD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t QFD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t QFD_M_START_INDEX = 4;
constexpr uint32_t QFD_M_NUM_INDEX = 5;

#ifdef __CCE_AICORE__
/**
 * @brief 获取sectionNum的绝对索引
 * @return 返回sectionNum的绝对索引
 */
__aicore__ inline uint32_t GetAttrSectionNumIndex() { return 0U; }

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
__aicore__ inline uint32_t GetAttrAbsIndex(
    uint32_t sectionIdx, uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionNum, bool isAIV = false) {
    if (isAIV) {
        return sectionNum * AIC_CORE_NUM * QFA_METADATA_SIZE + QFD_METADATA_SIZE * AIV_CORE_NUM * sectionIdx +
            QFD_METADATA_SIZE * coreIdx + metaIdx + 16U;
    } else {
        return QFA_METADATA_SIZE * AIC_CORE_NUM * sectionIdx + QFA_METADATA_SIZE * coreIdx + metaIdx + 16U;
    }
}
#endif

namespace detail {
struct QFaMetaData {
    uint32_t sectionNum;
    uint32_t *qfaMetadata; // [sectionNum][AIC_CORE_NUM][QFA_METADATA_SIZE];
    uint32_t *qfdMetadata; // [sectionNum][AIV_CORE_NUM][QFD_METADATA_SIZE];
    QFaMetaData(void *metadataPtr, uint32_t sectionNum)
        : sectionNum(sectionNum), qfaMetadata(static_cast<uint32_t *>(metadataPtr) + 16U),
          qfdMetadata(static_cast<uint32_t *>(metadataPtr) + 16U + sectionNum * AIC_CORE_NUM * QFA_METADATA_SIZE) {
        static_cast<uint32_t *>(metadataPtr)[0] = sectionNum;
    }
    void setQFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val) {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < QFA_METADATA_SIZE);
        qfaMetadata[AIC_CORE_NUM * QFA_METADATA_SIZE * sectionIdx + QFA_METADATA_SIZE * aicIdx + metaIdx] = val;
    }
    uint32_t getQFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx) {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < QFA_METADATA_SIZE);
        return qfaMetadata[AIC_CORE_NUM * QFA_METADATA_SIZE * sectionIdx + QFA_METADATA_SIZE * aicIdx + metaIdx];
    }
    void setQFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val) {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < QFD_METADATA_SIZE);
        qfdMetadata[AIV_CORE_NUM * QFD_METADATA_SIZE * sectionIdx + QFD_METADATA_SIZE * aivIdx + metaIdx] = val;
    }
    uint32_t getFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx) {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < QFD_METADATA_SIZE);
        return qfdMetadata[AIV_CORE_NUM * QFD_METADATA_SIZE * sectionIdx + QFD_METADATA_SIZE * aivIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
