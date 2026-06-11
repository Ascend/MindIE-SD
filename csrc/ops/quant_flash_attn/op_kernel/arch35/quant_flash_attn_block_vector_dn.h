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
 * \file quant_flash_attn_block_vector_dn.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VECTOR_DN_H_
#define QUANT_FLASH_ATTN_BLOCK_VECTOR_DN_H_

#include "vf/vf_softmax_dn_cast_nz_mxfp4_qs128_kvs256.h"
#include "vf/vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32.h"
#include "vf/vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32_multi.h"
#include "vf/vf_nd2nz_indexes_dn_mxfp4.h"
#include "vf/vf_computeScale_dn_mxfp4.h"
#include "vf/vf_updateScale_dn_mxfp4.h"
#include "vf/vf_attenOut_dn_mxfp4.h"
// #include  "vf/vf_init_dubpilcate.h"
#include "vf/vf_mm1_res_pre_padding_align_kvs32_multi.h"

#include "quant_flash_attn_template_tiling_key.h"
#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../../common/op_kernel/memcopy/fa_ub_tensor.h"
#include "../../../common/op_kernel/memcopy/copy_ub_to_gm.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;

namespace QFA_KERNEL {

template <QFA_LAYOUT LAYOUT_OUT> __aicore__ inline constexpr GmFormat GetAttentionOutGmFormat() {
    static_assert(
        (LAYOUT_OUT == QFA_LAYOUT::BSND) || (LAYOUT_OUT == QFA_LAYOUT::BNSD) || (LAYOUT_OUT == QFA_LAYOUT::TND),
        "Get OUT GmFormat fail, LAYOUT_OUT is incorrect");
    if constexpr (LAYOUT_OUT == QFA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_OUT == QFA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_OUT == QFA_LAYOUT::TND) {
        return GmFormat::TNGD;
    }
}

template <QFA_LAYOUT LAYOUT_OUT> __aicore__ inline constexpr UbFormat GetOutUbFormat() {
    static_assert(
        (LAYOUT_OUT == QFA_LAYOUT::BNSD) || (LAYOUT_OUT == QFA_LAYOUT::BSND) || (LAYOUT_OUT == QFA_LAYOUT::TND),
        "Get OutAttention UB GmFormat fail, LAYOUT is incorrect");
    if constexpr (LAYOUT_OUT == QFA_LAYOUT::BSND || LAYOUT_OUT == QFA_LAYOUT::TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT_OUT == QFA_LAYOUT::BNSD) {
        return UbFormat::GS1;
    }
}

template <typename QFAT> class QuantFlashAttnBlockVectorDn {
  public:
    /* =================编译期常量的基本块信息================= */
    using OUT_T = typename QFAT::outputType;
    using SEQLEN_T = uint32_t;
    static constexpr bool SOFTMAX_DN = true;
    static constexpr bool PAGE_ATTENTION = QFAT::pageAttention;
    static constexpr bool HAS_MASK = QFAT::hasMask;

    static constexpr QFA_LAYOUT LAYOUT_Q = QFAT::qLayout;
    static constexpr QFA_LAYOUT LAYOUT_KV = QFAT::kvLayout;
    static constexpr QFA_LAYOUT LAYOUT_OUT = QFAT::outLayout;

  private:
    // 初始化基础常量
    const ConstInfo &constInfo;
    const SeqLensTool<LAYOUT_Q, SEQLEN_T> &qSeqLensTool;
    const SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool;

    static constexpr uint16_t s1BaseSize = 128;
    static constexpr uint16_t s2BaseSize = 256;
    static constexpr uint32_t dBaseSize = 128;
    static constexpr uint32_t dVBaseSize = 128;
    static constexpr uint32_t S00_Offset = 0;
    static constexpr uint32_t S10_Offset = 128;
    static constexpr uint32_t P0_Offset = 0;
    static constexpr uint32_t P1_Offset = 256;
    static constexpr half MINNEG_VALUE = -65504;
    static constexpr uint32_t UPDATE_LEN_SIZE = 64;

    static constexpr uint32_t LOCAL_GROUP_MAX_SIZE = 128 * 128 / 32 * sizeof(half);
    static constexpr uint32_t LOCAL_GLOBAL_MAX_SIZE = 128 * sizeof(half);
    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;
    static constexpr uint32_t SINGLE_PSCALE_SPACE_SIZE = 32 * 5 * 8;
    static constexpr uint32_t SINGLE_GROUP_MAX_SPACE_SIZE = 4 * 128;

    static constexpr uint32_t mm1ResOffset[2] = {S00_Offset, S10_Offset};
    static constexpr uint32_t ve1ResOffset[2] = {P0_Offset, P1_Offset};

    // =================================UB Buffer=================================
    static constexpr uint32_t UB_S_SIZE = 128 * 256;
    static constexpr uint32_t UB_PV_SIZE = 128 * 64;
    static constexpr uint32_t UB_PSCALE_SIZE = 32 * 5 * 8;
    static constexpr uint32_t UB_ROWSUM_SIZE = 64;
    static constexpr uint32_t UB_P_SIZE = 64 * 256;
    static constexpr uint32_t UB_ATTENTIONOUT_SIZE = 128 * 64;
    static constexpr uint32_t UB_ATTENTIONTRANS_SIZE = 128 * 64 * 9 / 8;
    static constexpr uint32_t UB_MAX_SIZE = 128;
    static constexpr uint32_t UB_LOCALGROUPMAX_SIZE = 128 * 4;
    static constexpr uint32_t UB_UPDATE_SIZE = 128;
    static constexpr uint32_t UB_INDEX_SIZE = 256;

    static constexpr uint32_t UB_S_BUFCNT = 2;
    static constexpr uint32_t UB_PV_BUFCNT = 1;
    static constexpr uint32_t UB_PSCALE_BUFCNT = 8;
    static constexpr uint32_t UB_ROWSUM_BUFCNT = 1;
    static constexpr uint32_t UB_P_BUFCNT = 2;
    static constexpr uint32_t UB_ATTENTIONOUT_BUFCNT = 1;
    static constexpr uint32_t UB_MAX_BUFCNT = 1;
    static constexpr uint32_t UB_LOCALGLOBALMAX_BUFCNT = 2;
    static constexpr uint32_t UB_LOCALGROUPMAX_BUFCNT = 20;
    static constexpr uint32_t UB_UPDATE_BUFCNT = 2;
    static constexpr uint32_t UB_INDEX_BUFCNT = 1;
    static constexpr uint32_t UB_FD_BUFCNT = 8;

    LocalTensor<half> mm1ResUB;
    LocalTensor<float> mm2ResUB;
    LocalTensor<uint8_t> pscaleUB;
    LocalTensor<half> peerGlobalMaxUB;
    LocalTensor<uint8_t> vec1ResUB;
    LocalTensor<float> localRowsumUB;
    LocalTensor<float> attentionOutUB;
    LocalTensor<bfloat16_t> attentionTransUB;
    LocalTensor<float> globalRowsumUB;
    LocalTensor<half> localGroupMaxUB;
    LocalTensor<half> localGlobalMaxUB;
    LocalTensor<half> softmaxMaxUB;
    LocalTensor<float> updateScaleUB;
    LocalTensor<uint8_t> nd2nzIndexUB;

    LocalTensor<float> sumFDUB;
    LocalTensor<half> maxFDUB;

    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_P_SIZE = 128 * 256 / 2; // 16K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t L1_P_DESCALE_SIZE = 32 * 5 * 8; // 1.25K
    static constexpr uint32_t L1_P_BUFCNT = 20;

    LocalTensor<uint8_t> pL1Tensor;
    LocalTensor<uint8_t> pScaleL1;
    LocalTensor<half> localGlobalMaxL1;

    /* =====================GM变量==================== */

    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();

    // V2 attentionOut 变量
    static constexpr bool OUT_IS_TND = IS_TND<LAYOUT_OUT>();
    FaGmTensor<OUT_T, OUT_FORMAT, SEQLEN_T, OUT_IS_TND> outGmTensor;
    CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_OUT>()> AttenOutUbToGm;

    // 同步eventID
    static constexpr uint64_t SYNC_VEC1_RES_BUF0_FLAG = 0;
    static constexpr uint64_t SYNC_VEC1_RES_BUF1_FLAG = 1;
    static constexpr uint64_t SYNC_GMAX_UB_TO_L1_BUF0_FLAG = 3;
    static constexpr uint64_t SYNC_GMAX_UB_TO_L1_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_ATTN_BUF_FLAG = 5;

  public:
    // 初始化 Vec Block 层
    __aicore__ inline QuantFlashAttnBlockVectorDn(ConstInfo &constInfo, SeqLensTool<LAYOUT_Q, SEQLEN_T> &qSeqLensTool,
        SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo(constInfo), qSeqLensTool(qSeqLensTool), kvSeqLensTool(kvSeqLensTool){};

    __aicore__ inline void InitInput(__gm__ uint8_t *attentionOut) {
        // 初始化 attentionOut GM Buffer 及 GmTensor
        InitAttentionOutBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSize,
            qSeqLensTool, outGmTensor, attentionOut);
    }

    // 初始化 UB
    __aicore__ inline void InitTensors() {
        AllocEventID();

        // =================================L1 Tensor Init=================================
        uint32_t addrL1Start = 0; // 16K * 20 = 320K
        pL1Tensor = LocalTensor<uint8_t>(TPosition::A1, addrL1Start, L1_P_SIZE * L1_P_BUFCNT);
        addrL1Start += L1_P_SIZE * L1_P_BUFCNT;
        pScaleL1 = LocalTensor<uint8_t>(TPosition::A1, addrL1Start, L1_P_DESCALE_SIZE * L1_P_BUFCNT); // 1K * 20 = 20K

        localGlobalMaxL1 = LocalTensor<half>(TPosition::A1, 498 * 1024, 256);

        // =================================UB Tensor Init=================================
        uint32_t addrUBStart = 0;
        uint32_t addrUBReuseStart = 0;
        mm1ResUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_S_SIZE * UB_S_BUFCNT);

        addrUBStart += UB_S_SIZE * UB_S_BUFCNT * sizeof(half);
        mm2ResUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_PV_SIZE * UB_PV_BUFCNT);

        addrUBStart += UB_PV_SIZE * UB_PV_BUFCNT * sizeof(float);
        localRowsumUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT);

        addrUBStart += UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT * sizeof(float);
        globalRowsumUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT);

        addrUBStart += UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT * sizeof(float);
        vec1ResUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBStart, UB_P_SIZE * UB_P_BUFCNT);

        addrUBReuseStart = addrUBStart;
        attentionTransUB = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, addrUBReuseStart, UB_ATTENTIONTRANS_SIZE * UB_ATTENTIONOUT_BUFCNT);

        addrUBReuseStart += UB_ATTENTIONTRANS_SIZE * UB_ATTENTIONOUT_BUFCNT * sizeof(bfloat16_t);
        pscaleUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBReuseStart, UB_PSCALE_SIZE * UB_PSCALE_BUFCNT);

        addrUBReuseStart += UB_PSCALE_SIZE * UB_PSCALE_BUFCNT * sizeof(uint8_t);
        sumFDUB = LocalTensor<float>(TPosition::VECCALC, addrUBReuseStart, UB_ROWSUM_SIZE * UB_FD_BUFCNT);

        addrUBReuseStart += UB_ROWSUM_SIZE * UB_FD_BUFCNT * sizeof(float);
        maxFDUB = LocalTensor<half>(TPosition::VECCALC, addrUBReuseStart, UB_MAX_SIZE * UB_FD_BUFCNT);

        addrUBStart += UB_P_SIZE * UB_P_BUFCNT * sizeof(uint8_t);
        attentionOutUB =
            LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_ATTENTIONOUT_SIZE * UB_ATTENTIONOUT_BUFCNT);

        addrUBStart += UB_ATTENTIONOUT_SIZE * UB_ATTENTIONOUT_BUFCNT * sizeof(float);
        peerGlobalMaxUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_MAX_SIZE * UB_MAX_BUFCNT);

        addrUBStart += UB_MAX_SIZE * UB_MAX_BUFCNT * sizeof(half);
        softmaxMaxUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_MAX_SIZE * UB_MAX_BUFCNT);

        addrUBStart += UB_MAX_SIZE * UB_MAX_BUFCNT * sizeof(half);
        localGroupMaxUB =
            LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_LOCALGROUPMAX_SIZE * UB_LOCALGROUPMAX_BUFCNT);

        addrUBStart += UB_LOCALGROUPMAX_SIZE * UB_LOCALGROUPMAX_BUFCNT * sizeof(half);
        localGlobalMaxUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_MAX_SIZE * UB_LOCALGLOBALMAX_BUFCNT);

        addrUBStart += UB_MAX_SIZE * UB_LOCALGLOBALMAX_BUFCNT * sizeof(half);
        updateScaleUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_UPDATE_SIZE * UB_UPDATE_BUFCNT);

        addrUBStart += UB_UPDATE_SIZE * UB_UPDATE_BUFCNT * sizeof(float);
        nd2nzIndexUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBStart, UB_INDEX_SIZE * UB_INDEX_BUFCNT);

        // ================================= Init Value =================================
        Mxfp4Api::InitIndexesAndDuplicateCallVF<half>(nd2nzIndexUB, localGlobalMaxUB);
    }

    __aicore__ inline void ReleaseTensors() { FreeEventID(); }

    __aicore__ inline void ComputeVec1(const RunInfo &runInfo) {
        uint32_t buffIdx = GetBufferIdx(runInfo.loop);
        if (runInfo.isS2FirstTilePerCore) { // 每个vector core都需要一个跨tile的同步
            WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_GMAX_UB_TO_L1_BUF0_FLAG + runInfo.tileBuffIdx);
        }
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF0_FLAG + buffIdx);

        if (runInfo.isFirstS2Loop) {
            if (runInfo.actMSize == s1BaseSize && runInfo.actSingleLoopS2Size == s2BaseSize) { // s1=128, s2=256 softmax
                Mxfp4Api::softmaxWithGroupMaxQs128Kvs256CallVF<true, half, uint8_t, false, s2BaseSize, s1BaseSize>(
                    vec1ResUB[ve1ResOffset[buffIdx]], mm1ResUB[mm1ResOffset[buffIdx]],
                    GetLocalGrpMaxUbByLoopIdx(runInfo.loop), GetLocalGlobalMaxUbByCurIdx(runInfo.tileBuffIdx),
                    nd2nzIndexUB, static_cast<half>(constInfo.scaleValue));
            } else {
                // s2 padding 32 multi
                if (runInfo.actSingleLoopS2Size != runInfo.actSingleLoopS2SizeAlign) {
                    Mxfp4Api::Mm1ResPrePaddingAlignKvs32MultiCallVF<half>(mm1ResUB[mm1ResOffset[buffIdx]],
                        static_cast<uint16_t>(runInfo.actSingleLoopS2Size),
                        static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign));
                    PipeBarrier<PIPE_V>();
                }
                // softmax
                if (runInfo.actSingleLoopS2SizeAlign == AttentionCommon::BYTE_BLOCK) { // softmax_padding_32
                    Mxfp4Api::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<true, half, uint8_t, false>(
                        vec1ResUB[ve1ResOffset[buffIdx]], mm1ResUB[mm1ResOffset[buffIdx]],
                        GetLocalGrpMaxUbByLoopIdx(runInfo.loop), GetLocalGlobalMaxUbByCurIdx(runInfo.tileBuffIdx),
                        nd2nzIndexUB, static_cast<half>(constInfo.scaleValue));
                } else { // softmax_padding_32_multi >= 64
                    Mxfp4Api::SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF<true, half, uint8_t, false>(
                        vec1ResUB[ve1ResOffset[buffIdx]], mm1ResUB[mm1ResOffset[buffIdx]],
                        GetLocalGrpMaxUbByLoopIdx(runInfo.loop), GetLocalGlobalMaxUbByCurIdx(runInfo.tileBuffIdx),
                        nd2nzIndexUB, static_cast<half>(constInfo.scaleValue),
                        static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                        static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign64));
                }
            }
        } else {
            if (runInfo.actMSize == s1BaseSize && runInfo.actSingleLoopS2Size == s2BaseSize) { // s1=128, s2=256 softmax
                Mxfp4Api::softmaxWithGroupMaxQs128Kvs256CallVF<false, half, uint8_t, false, s2BaseSize, s1BaseSize>(
                    vec1ResUB[ve1ResOffset[buffIdx]], mm1ResUB[mm1ResOffset[buffIdx]],
                    GetLocalGrpMaxUbByLoopIdx(runInfo.loop), GetLocalGlobalMaxUbByCurIdx(runInfo.tileBuffIdx),
                    nd2nzIndexUB, static_cast<half>(constInfo.scaleValue));
            } else {
                // s2 padding 32 multi
                if (runInfo.actSingleLoopS2Size != runInfo.actSingleLoopS2SizeAlign) {
                    Mxfp4Api::Mm1ResPrePaddingAlignKvs32MultiCallVF<half>(mm1ResUB[mm1ResOffset[buffIdx]],
                        static_cast<uint16_t>(runInfo.actSingleLoopS2Size),
                        static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign));
                    PipeBarrier<PIPE_V>();
                }
                // softmax
                if (runInfo.actSingleLoopS2SizeAlign == AttentionCommon::BYTE_BLOCK) { // softmax_padding_32
                    Mxfp4Api::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<false, half, uint8_t, false>(
                        vec1ResUB[ve1ResOffset[buffIdx]], mm1ResUB[mm1ResOffset[buffIdx]],
                        GetLocalGrpMaxUbByLoopIdx(runInfo.loop), GetLocalGlobalMaxUbByCurIdx(runInfo.tileBuffIdx),
                        nd2nzIndexUB, static_cast<half>(constInfo.scaleValue));
                } else { // softmax_padding_32_multi >= 64
                    Mxfp4Api::SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF<false, half, uint8_t, false>(
                        vec1ResUB[ve1ResOffset[buffIdx]], mm1ResUB[mm1ResOffset[buffIdx]],
                        GetLocalGrpMaxUbByLoopIdx(runInfo.loop), GetLocalGlobalMaxUbByCurIdx(runInfo.tileBuffIdx),
                        nd2nzIndexUB, static_cast<half>(constInfo.scaleValue),
                        static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                        static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign64));
                }
            }
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_VEC1_RES_BUF0_FLAG + buffIdx);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_VEC1_RES_BUF0_FLAG + buffIdx);

        if (runInfo.actMSize == s1BaseSize && runInfo.actSingleLoopS2Size == s2BaseSize) { // s1=128, s2=256 softmax
            DataCopy(pL1Tensor[(runInfo.loop % 20) * (128 * 256 / 2)], vec1ResUB[ve1ResOffset[buffIdx]],
                {static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign / 4), 8, 8, 0});
        } else {
            // 1 整块搬运
            // DataCopy(pL1Tensor[(runInfo.loop % 20) * (128 * 256 / 2)], vec1ResUB[ve1ResOffset[buffIdx]], {static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign / 4), 8, 8, 0});
            // 2 消去高位 低位中间的间隙搬运
            DataCopy(pL1Tensor[(runInfo.loop % 20) * (128 * 256 / 2)], vec1ResUB[ve1ResOffset[buffIdx]],
                {static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign / AttentionCommon::BYTE_BLOCK * 8 / 2), 8, 8,
                    0});
            DataCopy(pL1Tensor[(runInfo.loop % 20) * (128 * 256 / 2) +
                         runInfo.actSingleLoopS2SizeAlign64 / AttentionCommon::BYTE_BLOCK * 8 / 2 * 256],
                vec1ResUB[ve1ResOffset[buffIdx] + BUFFER_SIZE_BYTE_16K],
                {static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign / AttentionCommon::BYTE_BLOCK * 8 / 2), 8, 8,
                    0});
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF0_FLAG + buffIdx);
    }

    __aicore__ inline void CopyGMaxUbToL1(const RunInfo &runInfo) {
        if (runInfo.s2FirstStartVecCore != constInfo.subBlockIdx &&
            runInfo.curS2LoopIdx == 0) { // // 如果当前s2只有一个softmax， 就另外一个核不用做交换操作
            return;
        }
        // uint32_t buffIdx = runInfo.curS2LoopIdx / 16 % 2;
        LocalTensor<half> localGlobalMax = GetLocalGlobalMaxUbByCurIdx(runInfo.tileBuffIdx);

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_GMAX_UB_TO_L1_BUF0_FLAG + runInfo.tileBuffIdx);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_GMAX_UB_TO_L1_BUF0_FLAG + runInfo.tileBuffIdx);

        DataCopy(
            localGlobalMaxL1[(1 - constInfo.subBlockIdx) * L1_SINGLE_GLOBAL_MAX_SIZE], localGlobalMax, {1, 8, 0, 0});
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_GMAX_UB_TO_L1_BUF0_FLAG + runInfo.tileBuffIdx);
    }

    __aicore__ inline void UpdatePScale(const RunInfo &runInfo) {
        if (runInfo.s2FirstStartVecCore != constInfo.subBlockIdx &&
            runInfo.curS2LoopIdx == 0) { // 如果当前s2只有一个softmax， 就另外一个核不用做pscale操作
            return;
        }

        uint16_t firstLoop = 0;
        uint16_t secondLoop = 0;
        uint16_t firstLoopStart = 0;
        uint16_t secondLoopStart = 0;

        GetPScaleParams(runInfo, firstLoopStart, firstLoop, secondLoopStart, secondLoop);

        LocalTensor<uint8_t> pscale1 = this->pscaleUB;
        LocalTensor<uint8_t> pscale2 = this->pscaleUB[firstLoop * SINGLE_PSCALE_SPACE_SIZE];
        LocalTensor<half> localGroupMax1 = this->localGroupMaxUB[firstLoopStart * SINGLE_GROUP_MAX_SPACE_SIZE];
        LocalTensor<half> localGroupMax2 = this->localGroupMaxUB[secondLoopStart * SINGLE_GROUP_MAX_SPACE_SIZE];

        LocalTensor<half> localGlobalMax1 = GetLocalGlobalMaxUbByCurIdx(runInfo.tileBuffIdx);
        LocalTensor<half> localGlobalMax2 = peerGlobalMaxUB;
        LocalTensor<half> softmaxMaxOld = softmaxMaxUB;
        LocalTensor<float> urs = GetUpdateScaleByCurIdx(runInfo.tileBuffIdx);

        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF0_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF1_FLAG);
        if (runInfo.curS2LoopIdx / 16 == 0) {
            Mxfp4Api::computePscale<true, half, s1BaseSize>(pscale1, pscale2, localGroupMax1, localGroupMax2,
                localGlobalMax1, localGlobalMax2, softmaxMaxOld, urs, firstLoop, secondLoop);
        } else {
            Mxfp4Api::computePscale<false, half, s1BaseSize>(pscale1, pscale2, localGroupMax1, localGroupMax2,
                localGlobalMax1, localGlobalMax2, softmaxMaxOld, urs, firstLoop, secondLoop);
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_VEC1_RES_BUF0_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_VEC1_RES_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_VEC1_RES_BUF0_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_VEC1_RES_BUF1_FLAG);

        DataCopy(pScaleL1[constInfo.subBlockIdx * (32 * 5 * 8) + firstLoopStart * (32 * 5 * 8)], pscale1,
            {static_cast<uint16_t>(firstLoop), 40, 0, 40});
        if (secondLoop != 0) {
            DataCopy(pScaleL1[constInfo.subBlockIdx * (32 * 5 * 8) + secondLoopStart * (32 * 5 * 8)],
                pscale1[firstLoop * (32 * 5 * 8)], {static_cast<uint16_t>(secondLoop), 40, 0, 40});
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF0_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF1_FLAG);
    }

    __aicore__ inline void ComputeVec2(const RunInfo &runInfo) {
        LocalTensor<float> urs = GetUpdateScaleByCurIdx(runInfo.tileBuffIdx);

        if (runInfo.curS2LoopIdx / 16 == 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_ATTN_BUF_FLAG);
            Mxfp4Api::processUpdate<false>(
                attentionOutUB, mm2ResUB, urs[constInfo.subBlockIdx * UPDATE_LEN_SIZE], globalRowsumUB, localRowsumUB);
        } else {
            Mxfp4Api::processUpdate<true>(
                attentionOutUB, mm2ResUB, urs[constInfo.subBlockIdx * UPDATE_LEN_SIZE], globalRowsumUB, localRowsumUB);
        }

        if (runInfo.isLastS2Loop) { // s1泛化128
            uint32_t vec2MBaseSize = (s1BaseSize + 1) / 2;
            uint32_t actDealMSize = runInfo.actMSize < vec2MBaseSize ? runInfo.actMSize : vec2MBaseSize;
            if (constInfo.subBlockIdx != 0) {
                actDealMSize = runInfo.actMSize < vec2MBaseSize ? 0 : (runInfo.actMSize - vec2MBaseSize);
            }
            PipeBarrier<PIPE_V>();
            if (actDealMSize != 0) {
                WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF0_FLAG);
                WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF1_FLAG);
                LocalTensor<float> outTensorFp32Ub = attentionTransUB.template ReinterpretCast<float>();
                Mxfp4Api::processOut(outTensorFp32Ub, attentionOutUB, globalRowsumUB);
                PipeBarrier<PIPE_V>();
                LocalTensor<OUT_T> outTensorBf16Ub = attentionTransUB.template ReinterpretCast<OUT_T>();
                LocalTensor<OUT_T> attentionOutUBBf16 = attentionOutUB.template ReinterpretCast<OUT_T>();
                uint32_t columnCnt = constInfo.dSize + AttentionCommon::BYTE_BLOCK / sizeof(OUT_T);
                DataCopy(attentionOutUBBf16, outTensorBf16Ub, actDealMSize * columnCnt);
                PipeBarrier<PIPE_V>();

                SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF0_FLAG);
                SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF1_FLAG);

                SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_ATTN_BUF_FLAG);
                WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_ATTN_BUF_FLAG);

                GmCoord gmCoord;
                gmCoord.bIdx = runInfo.bIdx;
                gmCoord.n2Idx = runInfo.n2Idx;
                gmCoord.gS1Idx = runInfo.gS1Idx + constInfo.subBlockIdx * vec2MBaseSize;
                gmCoord.dIdx = 0;
                gmCoord.gS1DealSize = actDealMSize;
                gmCoord.dDealSize = constInfo.dSize;

                FaUbTensor<OUT_T> ubTensor{
                    .tensor = attentionOutUBBf16, .rowCount = actDealMSize, .colCount = columnCnt};

                AttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_ATTN_BUF_FLAG);
        }
    }

  private:
    // // 同步初始化及释放
    __aicore__ inline void AllocEventID() {
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF0_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_ATTN_BUF_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_GMAX_UB_TO_L1_BUF0_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_GMAX_UB_TO_L1_BUF1_FLAG);
    }

    __aicore__ inline void FreeEventID() {
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF0_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_VEC1_RES_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_ATTN_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_GMAX_UB_TO_L1_BUF0_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_GMAX_UB_TO_L1_BUF1_FLAG);
    }

    // V2 attentionOut GM初始化
    __aicore__ inline void InitAttentionOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize,
        uint32_t qSeqSize, uint32_t headDim, const SeqLensTool<LAYOUT_Q, SEQLEN_T> &qSeqLensTool,
        FaGmTensor<OUT_T, OUT_FORMAT, SEQLEN_T, OUT_IS_TND> &outGmTensor, __gm__ uint8_t *attentionOut) {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool.cuSeqLensParser);
        }
    }

    // buffer id 获取
    __aicore__ inline uint32_t GetBufferIdx(const uint32_t loop) {
        uint32_t vecNum = 2;
        uint32_t doubleBuffer = 2;
        return loop / vecNum % doubleBuffer;
    }

    __aicore__ inline LocalTensor<half> GetLocalGlobalMaxUbByCurIdx(uint32_t spaceIdx) {
        return localGlobalMaxUB[spaceIdx * L1_SINGLE_GLOBAL_MAX_SIZE];
    }

    __aicore__ inline LocalTensor<half> GetLocalGrpMaxUbByLoopIdx(uint32_t loop) {
        uint32_t spaceIdx = loop / 2 * 2 % 20;
        return localGroupMaxUB[spaceIdx * SINGLE_GROUP_MAX_SPACE_SIZE];
    }

    __aicore__ inline LocalTensor<float> GetUpdateScaleByCurIdx(uint32_t spaceIdx) {
        return updateScaleUB[spaceIdx * 128];
    }

    __aicore__ inline void GetPScaleParams(const RunInfo &runInfo, uint16_t &firstLoopStart, uint16_t &firstLoop,
        uint16_t &secondLoopStart, uint16_t &secondLoop) {
        uint32_t subBlockIdx = constInfo.subBlockIdx;
        uint16_t isLoopFirstTaskVecCore = (subBlockIdx == runInfo.s2FirstStartVecCore);
        uint32_t loop = runInfo.loop;
        uint32_t GROUP_MAX_SPACE_LEN = 20;
        uint32_t WHOLE_PROCESS_LOOP = 16;

        uint32_t curS2LoopIdx = runInfo.curS2LoopIdx;
        uint32_t groupMaxEndIdx =
            loop / 2 * 2 % GROUP_MAX_SPACE_LEN + 1; // 这一轮pscale的任务loop在groupMax空间下发的groupMaxEndIdx
        uint16_t toProcessGroupMaxLoopLen =
            (curS2LoopIdx + 1) - (curS2LoopIdx / WHOLE_PROCESS_LOOP) * WHOLE_PROCESS_LOOP + isLoopFirstTaskVecCore;
        toProcessGroupMaxLoopLen = toProcessGroupMaxLoopLen / 2 * 2;
        toProcessGroupMaxLoopLen =
            toProcessGroupMaxLoopLen < WHOLE_PROCESS_LOOP ? toProcessGroupMaxLoopLen : WHOLE_PROCESS_LOOP;
        firstLoop = (toProcessGroupMaxLoopLen / 2 + 1) / 2 * 2;
        secondLoop = toProcessGroupMaxLoopLen - firstLoop;
        int32_t firstLoopStartNotU = groupMaxEndIdx - toProcessGroupMaxLoopLen + 1;
        if (firstLoopStartNotU < 0) {
            firstLoopStart = firstLoopStartNotU + GROUP_MAX_SPACE_LEN;
            firstLoop = GROUP_MAX_SPACE_LEN - firstLoopStart;
            secondLoop = toProcessGroupMaxLoopLen - firstLoop;
            secondLoopStart = 0;
        } else {
            firstLoopStart = firstLoopStartNotU;
            secondLoopStart = firstLoopStart + firstLoop; // 向上对齐到2
            // secondLoopEnd = groupMaxEndIdx;
        }
        // 两片groupmax空间对应其中的一个softmax任务
        firstLoop /= 2;
        secondLoop /= 2;
    }
};
}
#endif // QUANT_FLASH_ATTN_BLOCK_VECTOR_DN_H_
