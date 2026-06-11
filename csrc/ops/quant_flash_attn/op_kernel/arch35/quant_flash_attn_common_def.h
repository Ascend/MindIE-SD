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
 * \file quant_flash_attn_common_def.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_COMMON_DEF_H_
#define QUANT_FLASH_ATTN_COMMON_DEF_H_

#include "quant_flash_attn_template_tiling_key.h"
#include "../../../common/op_kernel/memcopy/parser.h"

namespace QFA_KERNEL {

constexpr float FLOAT_ZERO = 0;
// // // BUFFER的字节数
static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;
static constexpr uint32_t BUFFER_SIZE_BYTE_128K = 131072;

#ifndef SET_MARK2_JSELF
#define SET_MARK2_JSELF
__aicore__ inline void set_mark(uint64_t v) {
    __asm__ __volatile__("");
    asm volatile("MOV COND, %0\n" : "+l"(v));
    __asm__ __volatile__("");
}
#endif

struct CommonConstInfo {
    /* 轴长度 */
    uint32_t bSize = 0;
    uint64_t t1Size = 0;
    uint64_t t2Size = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dBasicBlock = 0;
    uint32_t gSize = 0; /* g轴的大小 */
    uint32_t n2Size = 0;
    uint64_t s1Size = 0; /* s1总大小 */
    uint64_t s2Size = 0; /* s2总大小 */
    uint32_t qCuSeqLensSize = 0; /* 用户输入的cu_seqlens_q的长度 */
    uint32_t kvCuSeqLensSize = 0; /* 用户输入的cu_seqlens_kv的长度 */
    uint32_t qSeqUsedSize = 0; /* 用户输入的seqused_q的长度 */
    uint32_t kvSeqUsedSize = 0; /* 用户输入的seqused_kv的长度 */

    /* FA kernel meta */
    uint32_t bN2Start = 0;
    uint32_t bN2End = 0;
    uint32_t gS1OStart = 0;
    uint32_t gS1OEnd = 0;
    uint32_t s2OStart = 0;
    uint32_t s2OEnd = 0;
    uint32_t coreFirstTmpOutWsPos = 0;

    /* mask */
    uint32_t sparseMode = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    float scaleValue = 0.0;

    /* 核信息 */
    uint32_t aicIdx = 0;
    uint32_t aivIdx = 0;
    uint8_t subBlockIdx = 0;
    uint32_t coreNum = 0;

    /* FA中间结果写出workspace信息 */
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct PAConstInfo {
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t paLayoutType = 0;
};

struct LseConstInfo {
    bool isSoftmaxLseEnable = false;
};

struct SinkConstInfo {
    bool learnableSinkFlag = false;
};

struct ConstInfo : CommonConstInfo, PAConstInfo, LseConstInfo, SinkConstInfo {};

struct RunInfo {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;
    bool isUpdatePScale = false;
    bool isC2Sync = false; // s2上16个softmax是一个tile，这是每一个tile的第一个softmax 任务
    uint32_t s2FirstStartVecCore = 0; // s2上第一个softmax分给哪个vec core
    uint32_t tileBuffIdx = 0; // 当前tile分给哪个buff
    bool isS2FirstTilePerCore = false; // 16个softmax均分在两个core, 当前任务是否是分在当前core上的第一个，

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t curS2LoopIdx = 0; // 在当前核处理的S2的循环下标
    uint64_t actS1Size = 1; // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1; // 当前处理head的S2轴实际大小
    uint32_t actMSize = 0; // GS1方向上的长度,当前切块M轴长度
    uint32_t actMSizeAlign32 = 0; // GS1 方向上长度对齐
    uint32_t actVecMSize = 0; // VEC 视角, 基本块GS1方向长度，每个核的M长度
    uint32_t vecMbaseIdx = 0; // VEC 对应的M 轴起始位置,V0 为0， V1 为 V0的actVecMSize

    uint32_t actSingleLoopS2Size = 0; // 单个softmaxS2方向长度
    uint32_t actSingleLoopS2SizeAlign = 0; // 对齐到32
    uint32_t actSingleLoopS2SizeAlign64 = 0; // 对齐到64
    uint32_t curS2LoopTimes = 0;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0; // FA阶段，S2外切，需要写到workspace时，写出到第几块M*D的GM块

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;
};

// kernel stream related struct
struct FDparams {
    uint32_t fdCoreEnable = 0;
    uint32_t fdBN2Idx = 0;
    uint32_t fdMIdx = 0;
    uint32_t fdS2SplitNum = 0;
    uint32_t mStart = 0;
    uint32_t mLen = 0;
    uint32_t fdWorkspaceIdx = 0;
};

template <QFA_LAYOUT LAYOUT_Q, typename SEQLEN_T> class SeqLensTool {
  public:
    ActualSeqLensParser<ActualSeqLensMode::ACCUM, SEQLEN_T, true> cuSeqLensParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, SEQLEN_T, false> seqUsedParser;

    __aicore__ inline void Init(__gm__ uint8_t *cuSeqLensGmAddr, uint32_t cuSeqLensDims, __gm__ uint8_t *seqUsedGmAddr,
        uint32_t seqUsedDims, uint64_t defaultSeqUsedVal) {
        cuSeqLensParser.Init(cuSeqLensGmAddr, cuSeqLensDims, seqUsedGmAddr, seqUsedDims);
        seqUsedParser.Init(seqUsedGmAddr, seqUsedDims, defaultSeqUsedVal);
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) {
        if constexpr (LAYOUT_Q != QFA_LAYOUT::TND) {
            return cuSeqLensParser.GetActualSeqLength(bIdx);
        } else {
            return seqUsedParser.GetActualSeqLength(bIdx);
        }
    }
};

__aicore__ inline void DataCopyL1ToUB(__ubuf__ void *ub, __cbuf__ void *l1, uint64_t vecCoreId, uint64_t blockCount,
    uint64_t blockNum, uint64_t srcGap, uint64_t destGap) {
    // #ifdef test_sync
    // return;
    // #endif
    uint64_t cfg = 0;

    cfg |= (vecCoreId & 0x1); // [0] sub_blockid (bit 0)

    cfg |= (blockCount & 0xFFF) << 4; // [15:4] number of bursts (bit 15:4)

    cfg |= (blockNum & 0xFFFF) << 16; // [31:16] length of burst in unit of 32B (bit 31:16)

    cfg |= (srcGap & 0xFFFF) << 32; // [47:32] source gap in unit of 32B (bit 47:32)

    cfg |= (destGap & 0xFFFF) << 48; // [63:48] destination gap in unit of 32B (bit 63:48)

    copy_cbuf_to_ubuf(ub, l1, cfg);
}

}

#endif
