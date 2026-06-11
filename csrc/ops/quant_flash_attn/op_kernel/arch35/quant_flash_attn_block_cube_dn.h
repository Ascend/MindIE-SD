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
 * \file quant_flash_atten_block_cube_dn.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_NOQUANT_BLOCK_CUBE_DN_H_
#define QUANT_FLASH_ATTN_NOQUANT_BLOCK_CUBE_DN_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_flash_attn_template_tiling_key.h"
#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/memcopy/copy_gm_to_l1.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;

namespace QFA_KERNEL {

template <QFA_LAYOUT LAYOUT_Q> __aicore__ inline constexpr GmFormat GetQueryGmFormat() {
    static_assert((LAYOUT_Q == QFA_LAYOUT::BSND) || (LAYOUT_Q == QFA_LAYOUT::BNSD) || (LAYOUT_Q == QFA_LAYOUT::TND),
        "Get Query GmFormat fail, LAYOUT_Q is incorrect");
    if constexpr (LAYOUT_Q == QFA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_Q == QFA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_Q == QFA_LAYOUT::TND) {
        return GmFormat::TNGD;
    }
}

template <QFA_LAYOUT LAYOUT_KV, bool PAGE_ATTENTION> __aicore__ inline constexpr GmFormat GetKVGmFormat() {
    if constexpr (PAGE_ATTENTION) {
        static_assert((LAYOUT_KV == QFA_LAYOUT::BSND) || (LAYOUT_KV == QFA_LAYOUT::BNSD),
            "Get Key or Value GmFormat fail, LAYOUT_KV is incorrect when PageAttention");
        if constexpr (LAYOUT_KV == QFA_LAYOUT::BSND) {
            return GmFormat::PA_BnBsND;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            return GmFormat::PA_BnNBsD;
        }
    } else {
        static_assert(
            (LAYOUT_KV == QFA_LAYOUT::BSND) || (LAYOUT_KV == QFA_LAYOUT::BNSD) || (LAYOUT_KV == QFA_LAYOUT::TND),
            "Get Key or Value GmFormat fail, LAYOUT_KV is incorrect when KV Continuous");
        if constexpr (LAYOUT_KV == QFA_LAYOUT::BSND) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::TND) {
            return GmFormat::TND;
        }
    }
}

template <QFA_LAYOUT LAYOUT_Q> __aicore__ inline constexpr GmFormat GetQueryScaleGmFormat() {
    if constexpr (LAYOUT_Q == QFA_LAYOUT::BNSD || LAYOUT_Q == QFA_LAYOUT::BSND) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_Q == QFA_LAYOUT::TND) {
        return GmFormat::NTGD;
    }
}

template <QFA_LAYOUT LAYOUT_KV, bool PAGE_ATTENTION> __aicore__ inline constexpr GmFormat GetKeyScaleGmFormat() {
    if constexpr (PAGE_ATTENTION) {
        static_assert((LAYOUT_KV == QFA_LAYOUT::BSND) || (LAYOUT_KV == QFA_LAYOUT::BNSD),
            "Get Key DeScale GmFormat fail, LAYOUT_KV is incorrect when PageAttention");
        if constexpr (LAYOUT_KV == QFA_LAYOUT::BSND) {
            return GmFormat::PA_BnBsND;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            return GmFormat::PA_BnNBsD;
        }
    } else {
        static_assert(
            (LAYOUT_KV == QFA_LAYOUT::BSND) || (LAYOUT_KV == QFA_LAYOUT::BNSD) || (LAYOUT_KV == QFA_LAYOUT::TND),
            "Get Key DeScale GmFormat fail, LAYOUT_KV is incorrect when KV Continuous");
        if constexpr (LAYOUT_KV == QFA_LAYOUT::BSND) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::TND) {
            return GmFormat::TND;
        }
    }
}

template <QFA_LAYOUT LAYOUT_KV, bool PAGE_ATTENTION> __aicore__ inline constexpr GmFormat GetValueScaleGmFormat() {
    if constexpr (PAGE_ATTENTION) {
        static_assert((LAYOUT_KV == QFA_LAYOUT::BSND) || (LAYOUT_KV == QFA_LAYOUT::BNSD),
            "Get Key DeScale GmFormat fail, LAYOUT_KV is incorrect when PageAttention");
        if constexpr (LAYOUT_KV == QFA_LAYOUT::BSND) {
            return GmFormat::PA_BnBsND;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            return GmFormat::PA_BnNBsD;
        }
    } else {
        static_assert(
            (LAYOUT_KV == QFA_LAYOUT::BSND) || (LAYOUT_KV == QFA_LAYOUT::BNSD) || (LAYOUT_KV == QFA_LAYOUT::TND),
            "Get Key DeScale GmFormat fail, LAYOUT_KV is incorrect when KV Continuous");
        if constexpr (LAYOUT_KV == QFA_LAYOUT::BSND) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::TND) {
            return GmFormat::TND2;
        }
    }
}

template <QFA_LAYOUT LAYOUT> __aicore__ inline constexpr bool IS_TND() { return (LAYOUT == QFA_LAYOUT::TND); }

template <typename T> __aicore__ inline constexpr uint32_t GetBlockElemCnt() {
    if constexpr (IS_4_BIT_WIDTH<T>()) {
        return AttentionCommon::BYTE_BLOCK * 2;
    } else {
        return AttentionCommon::BYTE_BLOCK / sizeof(T);
    }
}

template <typename QFAT> class QuantFlashAttnBlockCubeDn {
  public:
    using QUANT_T = typename QFAT::quantType;
    using SCALE_T = typename QFAT::scaleType;
    using OUT_T = typename QFAT::outputType;
    using SEQLEN_T = uint32_t;
    static constexpr bool SOFTMAX_DN = true;
    static constexpr bool PAGE_ATTENTION = QFAT::pageAttention;
    static constexpr bool HAS_MASK = QFAT::hasMask;
    static constexpr QFA_LAYOUT LAYOUT_Q = QFAT::qLayout;
    static constexpr QFA_LAYOUT LAYOUT_KV = QFAT::kvLayout;

    static constexpr uint32_t mBaseSize = 128;
    static constexpr uint32_t s2BaseSize = 256;
    static constexpr uint32_t dBaseSize = 128;
    static constexpr uint32_t dVBaseSize = 128;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_Q>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<LAYOUT_KV, PAGE_ATTENTION>();
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<LAYOUT_Q>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKeyScaleGmFormat<LAYOUT_KV, PAGE_ATTENTION>();
    static constexpr GmFormat V_SCALE_FORMAT = GetValueScaleGmFormat<LAYOUT_KV, PAGE_ATTENTION>();

  private:
    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t QK_L0_S2_SPLIT_SIZE = 128;
    static constexpr bool Q_IS_TND = IS_TND<LAYOUT_Q>();
    static constexpr bool KV_IS_TND = IS_TND<LAYOUT_KV>();
    using COMPUTE_T = float;
    using DATA_T = uint8_t;
    static constexpr uint32_t VEC0 = 0;
    static constexpr uint32_t VEC1 = 1;

    const ConstInfo &constInfo;
    const SeqLensTool<LAYOUT_Q, SEQLEN_T> &qSeqLensTool;
    const SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool;

    GlobalTensor<int32_t> blockTableGm;
    FaGmTensor<DATA_T, Q_FORMAT, SEQLEN_T, Q_IS_TND> queryGm;
    FaGmTensor<DATA_T, KV_FORMAT, SEQLEN_T, KV_IS_TND> keyGm;
    FaGmTensor<DATA_T, KV_FORMAT, SEQLEN_T, KV_IS_TND> valueGm;
    FaGmTensor<SCALE_T, Q_SCALE_FORMAT, SEQLEN_T, Q_IS_TND> queryScaleGm;
    FaGmTensor<SCALE_T, K_SCALE_FORMAT, SEQLEN_T, KV_IS_TND> keyScaleGm;
    FaGmTensor<SCALE_T, V_SCALE_FORMAT, SEQLEN_T, KV_IS_TND> valueScaleGm;

    CopyQueryGmToL1<DATA_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<DATA_T, KV_FORMAT> copyKvGmToL1;
    CopyQueryScaleGmToL1<SCALE_T, Q_SCALE_FORMAT> copyQueryScaleGmToL1;
    CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT> copyKeyScaleGmToL1;
    CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT> copyValueScaleGmToL1;

    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_Q_SIZE = 128 * 128 / 2; // 8K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t L1_Q_DESCALE_SIZE = 128 * (128 / 32) * sizeof(SCALE_T); // 0.5K
    static constexpr uint32_t L1_KV_SIZE = 512 * 128 / 2; // 32K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t L1_KV_DESCALE_SIZE = 512 * (128 / 32) * sizeof(SCALE_T); // 2K
    static constexpr uint32_t L1_P_SIZE = 128 * 256 / 2; // 16K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t L1_P_DESCALE_SIZE = 128 * 2 * (256 / 64 + 1) * sizeof(SCALE_T); // 1.25K
    static constexpr uint32_t L1_Q_BUFCNT = 2;
    static constexpr uint32_t L1_KV_BUFCNT = 4;
    static constexpr uint32_t L1_P_BUFCNT = 20;

    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;

    // 静态
    LocalTensor<DATA_T> pL1Tensor;
    LocalTensor<SCALE_T> pDescaleL1Tensor;
    LocalTensor<DATA_T> qL1Tensor;
    LocalTensor<SCALE_T> qDescaleL1Tensor;
    LocalTensor<DATA_T> kvL1Tensor;
    LocalTensor<SCALE_T> kvDescaleL1Tensor;
    LocalTensor<half> localGlobalMaxL1;

    // =================================L0 Buffer=================================
    static constexpr uint32_t QK_L0A_SIZE = 128 * 128 / 2; // 8K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t QK_L0B_SIZE = 128 * 128 / 2; // 8K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t QK_L0C_SIZE = 128 * 128 * sizeof(COMPUTE_T); // 64K
    static constexpr uint32_t QK_L0AB_BUFCNT = 2;
    static constexpr uint32_t QK_L0C_BUFCNT = 2;
    static constexpr uint32_t PV_L0A_SIZE = 256 * (128 + 16) / 2; // 18K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t PV_L0B_SIZE = 256 * (128 + 16) / 2; // 18K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t PV_L0C_SIZE = (128 + 16) * 128 * sizeof(COMPUTE_T); // 72K
    static constexpr uint32_t V_SCALE_L0A_SIZE = (256 / 64) * ((128 + 16) / 16) * 32;
    static constexpr uint32_t PV_L0AB_BUFCNT = 2;
    static constexpr uint32_t PV_L0C_BUFCNT = 1;

    // 静态
    // L0A
    LocalTensor<DATA_T> qkL0ATensor;
    LocalTensor<DATA_T> pvL0ATensor;

    // L0B
    LocalTensor<DATA_T> qkL0BTensor;
    LocalTensor<DATA_T> pvL0BTensor;

    // L0C
    LocalTensor<float> qkL0CTensor;
    LocalTensor<float> pvL0CTensor;

    // UB
    LocalTensor<half> mm1ResUB[2];
    LocalTensor<float> mm2ResUB;
    LocalTensor<half> peerGlobalMaxUB;

    // =================================Event&Buffer ID===========================
    // mte2 <> mte1 EventID
    static constexpr uint32_t Q_EVENT0 = EVENT_ID2;
    static constexpr uint32_t Q_EVENT1 = EVENT_ID3;
    uint32_t qBufId = 0;
    static constexpr uint32_t KV_EVENT0 = EVENT_ID4;
    static constexpr uint32_t KV_EVENT1 = EVENT_ID5;
    static constexpr uint32_t KV_EVENT2 = EVENT_ID6;
    static constexpr uint32_t KV_EVENT3 = EVENT_ID7;
    uint32_t kvBufId = 0;

    // mte1 <> mmad EventID
    static constexpr uint32_t QK_L0AB_EVENT0 = EVENT_ID2;
    static constexpr uint32_t QK_L0AB_EVENT1 = EVENT_ID3;
    static constexpr uint32_t PV_L0AB_EVENT0 = EVENT_ID4;
    static constexpr uint32_t PV_L0AB_EVENT1 = EVENT_ID5;
    uint32_t qkL0abBufId = 0;
    uint32_t pvL0abBufId = 0;
    // mmad <> fixpipe EventID
    static constexpr uint32_t QK_L0C_EVENT0 = EVENT_ID2;
    static constexpr uint32_t QK_L0C_EVENT1 = EVENT_ID3;
    static constexpr uint32_t PV_L0C_EVENT0 = EVENT_ID4;
    uint32_t qkL0cBufId = 0;
    uint32_t pvL0cBufId = 0;

    uint32_t headDimInt8 = 0;

  public:
    __aicore__ inline QuantFlashAttnBlockCubeDn(ConstInfo &constInfo, SeqLensTool<LAYOUT_Q, SEQLEN_T> &qSeqLensTool,
        SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo(constInfo), qSeqLensTool(qSeqLensTool), kvSeqLensTool(kvSeqLensTool){};

    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue,
        __gm__ uint8_t *blockTable) {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        // constInfo.dSize表示QUANT_T类型的元素数, 这里转换为DATA_T类型的元素数
        headDimInt8 = (constInfo.dSize / GetBlockElemCnt<QUANT_T>()) * GetBlockElemCnt<DATA_T>();
        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, headDimInt8, qSeqLensTool,
            queryGm, query);
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize, headDimInt8,
            kvSeqLensTool, keyGm, key);
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize, headDimInt8,
            kvSeqLensTool, valueGm, value);
        InitQScaleBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
            constInfo.dSize / MXFP_GROUP_SIZE, qSeqLensTool, queryScaleGm, dequantScaleQuery);
        InitKScaleBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize,
            constInfo.dSize / MXFP_GROUP_SIZE, kvSeqLensTool, keyScaleGm, dequantScaleKey);
        InitVScaleBuffer(constInfo.bSize, constInfo.s2Size / (2 * MXFP_GROUP_SIZE), constInfo.n2Size,
            constInfo.blockSize, 2 * constInfo.dSize, kvSeqLensTool, valueScaleGm, dequantScaleValue);
    }

    __aicore__ inline void InitTensors() {
        AllocEventID();

        // L1
        uint32_t addrL1Start = 0;
        pL1Tensor = LocalTensor<DATA_T>(TPosition::A1, addrL1Start, L1_P_SIZE * L1_P_BUFCNT); // 16K * 20 = 320K

        addrL1Start += L1_P_SIZE * L1_P_BUFCNT;
        pDescaleL1Tensor =
            LocalTensor<SCALE_T>(TPosition::A1, addrL1Start, L1_P_DESCALE_SIZE * L1_P_BUFCNT); // 1.25K * 20 = 25K

        addrL1Start += L1_P_DESCALE_SIZE * L1_P_BUFCNT;
        qL1Tensor = LocalTensor<DATA_T>(TPosition::A1, addrL1Start, L1_Q_SIZE * L1_Q_BUFCNT); // 8K * 2 = 16K

        addrL1Start += L1_Q_SIZE * L1_Q_BUFCNT;
        qDescaleL1Tensor =
            LocalTensor<SCALE_T>(TPosition::A1, addrL1Start, L1_Q_DESCALE_SIZE * L1_Q_BUFCNT); // 0.5K * 2 = 1K

        addrL1Start += L1_Q_DESCALE_SIZE * L1_Q_BUFCNT;
        kvL1Tensor = LocalTensor<DATA_T>(TPosition::A1, addrL1Start, L1_KV_SIZE * L1_KV_BUFCNT); // 32K * 4 = 128K

        addrL1Start += L1_KV_SIZE * L1_KV_BUFCNT;
        kvDescaleL1Tensor =
            LocalTensor<SCALE_T>(TPosition::A1, addrL1Start, L1_KV_DESCALE_SIZE * L1_KV_BUFCNT); // 2K * 4 = 8K

        addrL1Start += L1_KV_DESCALE_SIZE * L1_KV_BUFCNT;
        localGlobalMaxL1 = LocalTensor<half>(TPosition::A1, addrL1Start, 256); // 512b

        // L0A
        uint32_t addrL0AStart = 0;
        qkL0ATensor = LocalTensor<DATA_T>(TPosition::A2, addrL0AStart, QK_L0A_SIZE * QK_L0AB_BUFCNT); // 8K * 2 = 16K

        addrL0AStart += QK_L0A_SIZE * QK_L0AB_BUFCNT;
        pvL0ATensor = LocalTensor<DATA_T>(TPosition::A2, addrL0AStart, PV_L0A_SIZE * PV_L0AB_BUFCNT); // 18K * 2 = 36K

        // L0B
        uint32_t addrL0BStart = 0;
        qkL0BTensor = LocalTensor<DATA_T>(TPosition::B2, addrL0BStart, QK_L0B_SIZE * QK_L0AB_BUFCNT); // 8K * 2 = 16K

        addrL0BStart += QK_L0B_SIZE * QK_L0AB_BUFCNT;
        pvL0BTensor = LocalTensor<DATA_T>(TPosition::B2, addrL0BStart, PV_L0B_SIZE * PV_L0AB_BUFCNT); // 8K * 2 = 16K

        // L0C
        uint32_t addrL0CStart = 0;
        qkL0CTensor = LocalTensor<float>(
            TPosition::CO1, addrL0CStart, QK_L0C_SIZE * QK_L0C_BUFCNT / sizeof(float)); // 64K * 2 = 128K

        addrL0CStart += QK_L0C_SIZE * QK_L0C_BUFCNT;
        pvL0CTensor = LocalTensor<float>(
            TPosition::CO1, addrL0CStart, PV_L0C_SIZE * PV_L0C_BUFCNT / sizeof(float)); // 72K * 1 = 72K

        // 刷计算reducesum的L0A buffer
        InitL0BufferForReduceSum();

        // Vector UB
        mm1ResUB[0] = LocalTensor<half>(TPosition::VECCALC, 0, 256 * 128 * 2);
        mm1ResUB[1] = LocalTensor<half>(TPosition::VECCALC, 256, 256 * 128 * 2 - 128);
        mm2ResUB = LocalTensor<float>(TPosition::VECCALC, 128 * 1024, 128 * 64);
        peerGlobalMaxUB = LocalTensor<half>(TPosition::VECCALC, 229888, 128);
    }

    __aicore__ inline void ReleaseTensors() { FreeEventID(); }

    __aicore__ inline void ComputeMm1(const RunInfo &info) {
        if (unlikely(info.isFirstS2Loop)) {
            WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId);
            CopyQGmToL1(info); // 单次最大128行
            CopyQScaleGmToL1(info);
            SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
            WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
        }

        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
        CopyKGmToL1(info); // 单次最大256行
        CopyKScaleGmToL1(info);
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);
        {
            uint32_t loopCnt = CeilDiv((uint32_t)info.actSingleLoopS2Size, (uint32_t)QK_L0_S2_SPLIT_SIZE);
            uint32_t actS2Size = QK_L0_S2_SPLIT_SIZE;
            uint32_t actS2SizeAlign = QK_L0_S2_SPLIT_SIZE;
            for (uint32_t loop = 0; loop < loopCnt; ++loop) {
                if (loop + 1 == loopCnt) {
                    actS2Size = info.actSingleLoopS2Size - loop * QK_L0_S2_SPLIT_SIZE;
                    actS2SizeAlign = info.actSingleLoopS2SizeAlign - loop * QK_L0_S2_SPLIT_SIZE;
                }

                WaitFlag<HardEvent::FIX_M>(QK_L0C_EVENT0 + qkL0cBufId);
                {
                    WaitFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0 + qkL0abBufId);
                    LoadQToL0(info); // 128 * 128
                    LoadKToL0(info, loop, actS2SizeAlign); // 128 * 128
                    SetFlag<HardEvent::MTE1_M>(QK_L0AB_EVENT0 + qkL0abBufId);
                    WaitFlag<HardEvent::MTE1_M>(QK_L0AB_EVENT0 + qkL0abBufId);
                    MatmulQK(info, actS2Size); // 128 * 128 * 128
                    SetFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0 + qkL0abBufId);
                    qkL0abBufId = (qkL0abBufId + 1) % 2;
                }
                SetFlag<HardEvent::M_FIX>(QK_L0C_EVENT0 + qkL0cBufId);
                WaitFlag<HardEvent::M_FIX>(QK_L0C_EVENT0 + qkL0cBufId);
                FixpipeMm1(info, loop, actS2Size); // 128 * 256
                SetFlag<HardEvent::FIX_M>(QK_L0C_EVENT0 + qkL0cBufId);
                qkL0cBufId = (qkL0cBufId + 1) % 2;
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
        kvBufId = (kvBufId + 1) % L1_KV_BUFCNT;

        if (unlikely(info.isLastS2Loop)) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId);
            qBufId = (qBufId + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void ComputeMm2(const RunInfo &info) {
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
        if (info.actSingleLoopS2SizeAlign != info.actSingleLoopS2SizeAlign64) {
            InitConstValueParams<uint16_t> PL1InitParams(1, static_cast<uint16_t>(info.actMSize / 2 / 2), 0, 0x0000);
            uint32_t POffset = info.actSingleLoopS2SizeAlign * info.actMSize / 2 / 2;
            Fill(pL1Tensor[POffset].template ReinterpretCast<uint16_t>(), PL1InitParams);
            Fill(pL1Tensor[POffset * 2 + 32 * info.actMSize / 2 / 2].template ReinterpretCast<uint16_t>(),
                PL1InitParams);
        }
        CopyVGmToL1(info);
        CopyVScaleGmToL1(info);
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);
        {
            if (info.isC2Sync) {
                WaitFlag<HardEvent::FIX_M>(PV_L0C_EVENT0 + pvL0cBufId);
            }
            {
                WaitFlag<HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
                LoadPToL0(info);
                LoadVToL0(info);
                SetFlag<HardEvent::MTE1_M>(PV_L0AB_EVENT0 + pvL0abBufId);
                WaitFlag<HardEvent::MTE1_M>(PV_L0AB_EVENT0 + pvL0abBufId);
                MatmulPV(info);
                SetFlag<HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
                pvL0abBufId = (pvL0abBufId + 1) % 2;
            }
            if (info.isUpdatePScale) {
                SetFlag<HardEvent::M_FIX>(PV_L0C_EVENT0 + pvL0cBufId);
                WaitFlag<HardEvent::M_FIX>(PV_L0C_EVENT0 + pvL0cBufId);
                FixpipeMm2(info);
                SetFlag<HardEvent::FIX_M>(PV_L0C_EVENT0 + pvL0cBufId);
                // pvL0cBufId = (pvL0cBufId + 1) % 1;
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
        kvBufId = (kvBufId + 1) % L1_KV_BUFCNT;
    }

    __aicore__ inline void CopyGMaxL1ToUb(const RunInfo &runInfo) {
        LocalTensor<half> localGlobalMax0 = localGlobalMaxL1;
        LocalTensor<half> localGlobalMax1 = localGlobalMaxL1[L1_SINGLE_GLOBAL_MAX_SIZE];

        __cbuf__ half *localGlobalMax0Cbuf = (__cbuf__ half *)localGlobalMax0.GetPhyAddr();
        __cbuf__ half *localGlobalMax1Cbuf = (__cbuf__ half *)localGlobalMax1.GetPhyAddr();
        __ubuf__ half *peerGlobalMaxUBUbuf = (__ubuf__ half *)peerGlobalMaxUB.GetPhyAddr();

        DataCopyL1ToUB(peerGlobalMaxUBUbuf, localGlobalMax0Cbuf, VEC0, 1, 8, 0, 0);
        DataCopyL1ToUB(peerGlobalMaxUBUbuf, localGlobalMax1Cbuf, VEC1, 1, 8, 0, 0);
    }

  private:
    __aicore__ inline void AllocEventID() {
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT2);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT3);

        SetFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        SetFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        SetFlag<HardEvent::M_MTE1>(PV_L0AB_EVENT0);
        SetFlag<HardEvent::M_MTE1>(PV_L0AB_EVENT1);

        SetFlag<HardEvent::FIX_M>(QK_L0C_EVENT0);
        SetFlag<HardEvent::FIX_M>(QK_L0C_EVENT1);
        SetFlag<HardEvent::FIX_M>(PV_L0C_EVENT0);
    }

    __aicore__ inline void FreeEventID() {
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT2);
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT3);

        WaitFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        WaitFlag<HardEvent::M_MTE1>(PV_L0AB_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(PV_L0AB_EVENT1);

        WaitFlag<HardEvent::FIX_M>(QK_L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(QK_L0C_EVENT1);
        WaitFlag<HardEvent::FIX_M>(PV_L0C_EVENT0);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
        uint32_t headDim, const SeqLensTool<LAYOUT_Q, SEQLEN_T> &qSeqLensTool,
        FaGmTensor<DATA_T, Q_FORMAT, SEQLEN_T, Q_IS_TND> &qGmTensor, __gm__ uint8_t *gm) {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ DATA_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
        uint32_t headDim, const SeqLensTool<LAYOUT_Q, SEQLEN_T> &qSeqLensTool,
        FaGmTensor<SCALE_T, Q_SCALE_FORMAT, SEQLEN_T, Q_IS_TND> &qScaleGmTensor, __gm__ uint8_t *gm) {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qScaleGmTensor.offsetCalculator.Init(
                batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
        uint32_t kvCacheBlockSize, uint32_t headDim, const SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool,
        FaGmTensor<DATA_T, KV_FORMAT, SEQLEN_T, KV_IS_TND> &kvGmTensor, __gm__ uint8_t *gm) {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ DATA_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(
                n2Size, kvCacheBlockSize, headDim, blockTableGm, constInfo.maxBlockNumPerBatch);
            // } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            //     uint32_t d0 = 32 / sizeof(DATA_T);
            //     uint32_t d1 = headDim / d0;
            //     kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
            //                                      constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, kvSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqLensTool.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
        uint32_t kvCacheBlockSize, uint32_t headDim, const SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool,
        FaGmTensor<SCALE_T, K_SCALE_FORMAT, SEQLEN_T, KV_IS_TND> &kScaleGmTensor, __gm__ uint8_t *gm) {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kScaleGmTensor.offsetCalculator.Init(
                n2Size, kvCacheBlockSize, headDim, blockTableGm, constInfo.maxBlockNumPerBatch);
            // } else if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            //     uint32_t d0 = 32 / sizeof(SCALE_T);
            //     uint32_t d1 = headDim / d0;
            //     kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
            //                                      constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, kvSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqLensTool.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
        uint32_t kvCacheBlockSize, uint32_t headDim, const SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool,
        FaGmTensor<SCALE_T, V_SCALE_FORMAT, SEQLEN_T, KV_IS_TND> &vScaleGmTensor, __gm__ uint8_t *gm) {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            vScaleGmTensor.offsetCalculator.Init(
                n2Size, kvCacheBlockSize, headDim, blockTableGm, constInfo.maxBlockNumPerBatch);
            // } else if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            //     uint32_t d0 = 32 / sizeof(SCALE_T);
            //     uint32_t d1 = headDim / d0;
            //     vScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
            //                                      constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            vScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, kvSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqLensTool.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitL0BufferForReduceSum() {
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);

        // InitConstValueParams<uint16_t> vL1InitParams (
        //     .repeatTimes = 1,
        //     .blockNum = static_cast<uint16_t>(PV_L0A_SIZE / 32),
        //     .dstGap = 0,
        //     .initValue = 0x7777
        // );
        InitConstValueParams<uint16_t> vL1InitParams(1, static_cast<uint16_t>(PV_L0A_SIZE / 32), 0, 0x2222);
        uint64_t l1BaseOffset = kvBufId * (L1_KV_SIZE / sizeof(DATA_T));
        Fill(kvL1Tensor[l1BaseOffset].template ReinterpretCast<uint16_t>(), vL1InitParams);

        PipeBarrier<PIPE_MTE2>(); // initvalue会被刷到同一个寄存器，两个Fill需要顺序执行
        // InitConstValueParams<uint16_t> vScaleL1InitParams (
        //     .repeatTimes = 1,
        //     .blockNum = static_cast<uint16_t>(V_SCALE_L0A_SIZE / 32),
        //     .dstGap = 0,
        //     .initValue = 0x7f7f
        // );
        InitConstValueParams<uint16_t> vScaleL1InitParams(1, static_cast<uint16_t>(V_SCALE_L0A_SIZE / 32), 0, 0x7f7f);
        uint32_t vScaleL1Offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        Fill(kvDescaleL1Tensor[vScaleL1Offset].template ReinterpretCast<uint16_t>(), vScaleL1InitParams);

        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId);

        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = (128 + 16) / 16;
        loadData2DParamsA.kStep = 256 / 64;
        loadData2DParamsA.srcStride = loadData2DParamsA.mStep;
        loadData2DParamsA.dstStride = loadData2DParamsA.mStep;
        loadData2DParamsA.ifTranspose = false;

        LoadData2DMxParams loadData2DMXParamsA;
        loadData2DMXParamsA.xStartPosition = 0;
        loadData2DMXParamsA.yStartPosition = 0;
        loadData2DMXParamsA.xStep = (128 + 16) / 16;
        loadData2DMXParamsA.yStep = 256 / 64;
        loadData2DMXParamsA.srcStride = loadData2DMXParamsA.yStep;
        loadData2DMXParamsA.dstStride = loadData2DMXParamsA.yStep;

        for (uint32_t i = 0; i < 2; i++) {
            WaitFlag<HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
            uint32_t pvL0AOffset = pvL0abBufId * (PV_L0A_SIZE / sizeof(DATA_T));
            LoadData(pvL0ATensor[pvL0AOffset].ReinterpretCast<QUANT_T>(),
                kvL1Tensor[l1BaseOffset].ReinterpretCast<QUANT_T>(), kvDescaleL1Tensor[vScaleL1Offset],
                loadData2DParamsA, loadData2DMXParamsA);
            SetFlag<HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
            pvL0abBufId = (pvL0abBufId + 1) % 2;
        }
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId);
        kvBufId = (kvBufId + 1) % L1_KV_BUFCNT;
    }

    __aicore__ inline void CopyQGmToL1(const RunInfo &info) {
        uint64_t queryL1BaseOffset = qBufId * (L1_Q_SIZE / sizeof(DATA_T));
        uint32_t dstStride = (info.actMSize + 31) >> 5 << 5;
        // dSize是QUANT_T的元素数, 先转换为字节数, 再转成DATA_T的元素数
        FaL1Tensor<DATA_T, L1Format::NZ> l1Tensor{.tensor = qL1Tensor[queryL1BaseOffset], .rowCount = dstStride};

        GmCoord gmCoord{.bIdx = info.bIdx,
            .n2Idx = info.n2Idx,
            .gS1Idx = info.gS1Idx,
            .dIdx = 0,
            .gS1DealSize = info.actMSize,
            .dDealSize = headDimInt8};
        copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
    }

    __aicore__ inline void CopyKGmToL1(const RunInfo &info) {
        uint64_t l1BaseOffset = kvBufId * (L1_KV_SIZE / sizeof(DATA_T));
        uint32_t dstStride = info.actSingleLoopS2SizeAlign;
        FaL1Tensor<DATA_T, L1Format::NZ> l1Tensor{.tensor = kvL1Tensor[l1BaseOffset], .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = info.bIdx,
            .n2Idx = info.n2Idx,
            .s2Idx = info.s2Idx,
            .dIdx = 0,
            .s2DealSize = info.actSingleLoopS2Size,
            .dDealSize = headDimInt8};
        copyKvGmToL1(l1Tensor, keyGm, gmCoord);
    }

    __aicore__ inline void CopyVGmToL1(const RunInfo &info) {
        uint64_t l1BaseOffset = kvBufId * (L1_KV_SIZE / sizeof(DATA_T));
        uint32_t dstStride = info.actSingleLoopS2SizeAlign64;
        FaL1Tensor<DATA_T, L1Format::NZ> l1Tensor{.tensor = kvL1Tensor[l1BaseOffset], .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = info.bIdx,
            .n2Idx = info.n2Idx,
            .s2Idx = info.s2Idx,
            .dIdx = 0,
            .s2DealSize = info.actSingleLoopS2SizeAlign64,
            .dDealSize = headDimInt8};
        copyKvGmToL1(l1Tensor, valueGm, gmCoord);
    }

    // copy query scale with full s1g
    __aicore__ inline void CopyQScaleGmToL1(const RunInfo &info) {
        uint32_t offset = qBufId * (L1_Q_DESCALE_SIZE / sizeof(SCALE_T));
        uint32_t dstStride = (info.actMSize + 31) >> 5 << 5;
        uint32_t dDealSize = constInfo.dSize / MXFP_GROUP_SIZE;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = qDescaleL1Tensor[offset], .rowCount = dstStride};

        GmCoord gmCoord{.bIdx = info.bIdx,
            .n2Idx = info.n2Idx,
            .gS1Idx = info.gS1Idx,
            .dIdx = 0,
            .gS1DealSize = info.actMSize,
            .dDealSize = dDealSize};
        copyQueryScaleGmToL1(l1Tensor, queryScaleGm, gmCoord);
    }

    // copy key scale with full s2
    __aicore__ inline void CopyKScaleGmToL1(const RunInfo &info) {
        uint32_t offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        uint32_t dstStride = info.actSingleLoopS2SizeAlign;
        uint32_t dDealSize = constInfo.dSize / MXFP_GROUP_SIZE;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = kvDescaleL1Tensor[offset], .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = info.bIdx,
            .n2Idx = info.n2Idx,
            .s2Idx = info.s2Idx,
            .dIdx = 0,
            .s2DealSize = info.actSingleLoopS2Size,
            .dDealSize = dDealSize};
        copyKeyScaleGmToL1(l1Tensor, keyScaleGm, gmCoord);
    }

    __aicore__ inline void LoadQToL0(const RunInfo &info) {
        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = ((info.actMSize + 31) >> 5 << 5) / 16;
        loadData2DParamsA.kStep = constInfo.dSize / GetBlockElemCnt<QUANT_T>();
        loadData2DParamsA.srcStride = loadData2DParamsA.mStep;
        loadData2DParamsA.dstStride = loadData2DParamsA.mStep;
        loadData2DParamsA.ifTranspose = false;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = 0;
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = ((info.actMSize + 31) >> 5 << 5) / 16; // 128 为 M 轴
        loadData2DMxParamsA.yStep = loadData2DParamsA.kStep;
        loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
        loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;

        uint32_t qkL0Offset = qkL0abBufId * (QK_L0A_SIZE / sizeof(DATA_T));
        uint32_t qL1Offset = qBufId * (L1_Q_SIZE / sizeof(DATA_T));
        uint32_t qScaleL1Offset = qBufId * (L1_Q_DESCALE_SIZE / sizeof(SCALE_T));
        LoadData(qkL0BTensor[qkL0Offset].ReinterpretCast<QUANT_T>(), qL1Tensor[qL1Offset].ReinterpretCast<QUANT_T>(),
            qDescaleL1Tensor[qScaleL1Offset], loadData2DParamsA, loadData2DMxParamsA);
    }

    __aicore__ inline void LoadKToL0(const RunInfo &info, uint32_t subLoop, uint32_t actS2SizeAlign) {
        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = subLoop * (QK_L0_S2_SPLIT_SIZE / 16);
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = actS2SizeAlign / 16;
        loadData2DParamsA.kStep = constInfo.dSize / GetBlockElemCnt<QUANT_T>();
        loadData2DParamsA.srcStride = info.actSingleLoopS2SizeAlign / 16;
        loadData2DParamsA.dstStride = actS2SizeAlign / 16;
        loadData2DParamsA.ifTranspose = false;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = subLoop * (QK_L0_S2_SPLIT_SIZE / 16);
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = actS2SizeAlign / 16;
        loadData2DMxParamsA.yStep = loadData2DParamsA.kStep;
        loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
        loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;

        uint32_t qkL0Offset = qkL0abBufId * (QK_L0B_SIZE / sizeof(DATA_T));
        uint32_t kvL1Offset = kvBufId * (L1_KV_SIZE / sizeof(DATA_T));
        uint32_t kvScaleL1Offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        LoadData(qkL0ATensor[qkL0Offset].ReinterpretCast<QUANT_T>(), kvL1Tensor[kvL1Offset].ReinterpretCast<QUANT_T>(),
            kvDescaleL1Tensor[kvScaleL1Offset], loadData2DParamsA, loadData2DMxParamsA);
    }

    __aicore__ inline void MatmulQK(const RunInfo &info, uint32_t actS2Size) {
        MmadParams mmadParams;
        mmadParams.m = actS2Size;
        mmadParams.n = info.actMSize;
        mmadParams.k = constInfo.dSize;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.disableGemv = true;
        uint32_t qkL0AOffset = qkL0abBufId * (QK_L0A_SIZE / sizeof(DATA_T));
        uint32_t qkL0BOffset = qkL0abBufId * (QK_L0B_SIZE / sizeof(DATA_T));
        uint32_t qkL0COffset = qkL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        Mmad(qkL0CTensor[qkL0COffset], qkL0ATensor[qkL0AOffset].ReinterpretCast<QUANT_T>(),
            qkL0BTensor[qkL0BOffset].ReinterpretCast<QUANT_T>(), mmadParams);
    }

    __aicore__ inline void FixpipeMm1(const RunInfo &info, uint32_t subLoop, uint32_t actS2Size) {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (info.actMSize + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (actS2Size + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((actS2Size + 15) / 16) * 16;
        fixpipeParams.dstStride = 512 / sizeof(half); // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.quantPre = QuantMode_t::F322F16;
        fixpipeParams.dualDstCtl = 0; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.subBlockId = info.loop % 2;

        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = (actS2Size + 1) >> 1 << 1;
        fixpipeParams.params.dstNdStride = 512 / sizeof(half);

        uint32_t qkL0COffset = qkL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        uint32_t ubOffset = subLoop * QK_L0_S2_SPLIT_SIZE * fixpipeParams.dstStride;
        Fixpipe<half, COMPUTE_T, CFG_ROW_MAJOR_UB>(
            mm1ResUB[info.loop / 2 % 2][ubOffset], qkL0CTensor[qkL0COffset], fixpipeParams);
    }

    __aicore__ inline void CopyVScaleGmToL1(const RunInfo &info) {
        uint32_t offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{
            .tensor = kvDescaleL1Tensor[offset], .rowCount = info.actSingleLoopS2SizeAlign64 / 64};

        GmKvCoord gmCoord{.bIdx = info.bIdx,
            .n2Idx = info.n2Idx,
            .s2Idx = info.s2Idx / 64,
            .dIdx = 0,
            .s2DealSize = info.actSingleLoopS2SizeAlign64 / 64,
            .dDealSize = 2 * constInfo.dSize};

        copyValueScaleGmToL1(l1Tensor, valueScaleGm, gmCoord);
    }

    __aicore__ inline void LoadPToL0(const RunInfo &info) {
        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.mStep = (info.actSingleLoopS2SizeAlign64 + 15) / 16;
        loadData2DParamsB.kStep = info.actMSize / GetBlockElemCnt<QUANT_T>();

        loadData2DParamsB.srcStride = loadData2DParamsB.mStep;
        loadData2DParamsB.dstStride =
            AttentionCommon::Align((uint32_t)constInfo.dSize, (uint32_t)GetBlockElemCnt<QUANT_T>()) / 16;
        loadData2DParamsB.ifTranspose = true;

        LoadData2DMxParams load2DMxParamsB;
        load2DMxParamsB.xStartPosition = 0;
        load2DMxParamsB.yStartPosition = 0;
        load2DMxParamsB.xStep = constInfo.dSize / 16;
        load2DMxParamsB.yStep = (info.actSingleLoopS2SizeAlign64 + 63) / 64;
        load2DMxParamsB.srcStride =
            5; // S2BaseSIze=256，5由s2BaseSize / 64 + 1得到，1为解UB Bank冲突预留的一个Block，一次全拷过来
        load2DMxParamsB.dstStride = load2DMxParamsB.yStep;

        uint32_t pvL0BOffset = pvL0abBufId * (PV_L0B_SIZE / sizeof(DATA_T));
        uint32_t pL1Offset = (info.loop % 20) * (L1_P_SIZE / sizeof(DATA_T));
        uint32_t pScaleL1Offset = (info.loop % 20) * (L1_P_DESCALE_SIZE / sizeof(SCALE_T));

        LoadData(pvL0BTensor[pvL0BOffset].ReinterpretCast<QUANT_T>(), pL1Tensor[pL1Offset].ReinterpretCast<QUANT_T>(),
            pDescaleL1Tensor[pScaleL1Offset], loadData2DParamsB, load2DMxParamsB);
    }

    __aicore__ inline void LoadVToL0(const RunInfo &info) {
        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = ((info.actSingleLoopS2SizeAlign64 + 15) / 16 + 3) / 4 * 4;
        loadData2DParamsA.kStep = constInfo.dSize / GetBlockElemCnt<QUANT_T>();

        loadData2DParamsA.srcStride = (info.actSingleLoopS2SizeAlign64 + 15) / 16;
        loadData2DParamsA.dstStride = (constInfo.dSize + 15) / 16 + 1;
        loadData2DParamsA.ifTranspose = true;

        LoadData2DMxParams load2DMxParamsA;
        load2DMxParamsA.xStartPosition = 0;
        load2DMxParamsA.yStartPosition = 0;
        load2DMxParamsA.xStep = (constInfo.dSize + 15) / 16;
        load2DMxParamsA.yStep = (info.actSingleLoopS2SizeAlign64 + GetBlockElemCnt<QUANT_T>() - 1) /
            GetBlockElemCnt<QUANT_T>(); // 2Byte分形的行数，64个V在S2方向上的元素一组对应一个2Byte分形
        load2DMxParamsA.srcStride = load2DMxParamsA.yStep;
        load2DMxParamsA.dstStride = load2DMxParamsA.yStep;

        uint32_t pvL0AOffset = pvL0abBufId * (PV_L0A_SIZE / sizeof(DATA_T));
        uint32_t kvL1Offset = kvBufId * (L1_KV_SIZE / sizeof(DATA_T));
        uint32_t vScaleL1Offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        LoadData(pvL0ATensor[pvL0AOffset].ReinterpretCast<QUANT_T>(), kvL1Tensor[kvL1Offset].ReinterpretCast<QUANT_T>(),
            kvDescaleL1Tensor[vScaleL1Offset], loadData2DParamsA, load2DMxParamsA);
    }

    __aicore__ inline void MatmulPV(const RunInfo &info) {
        MmadParams mmadParams;
        mmadParams.m = constInfo.dSize + 16;
        mmadParams.n = info.actMSize;
        mmadParams.k = info.actSingleLoopS2SizeAlign64;
        mmadParams.cmatrixInitVal = info.isC2Sync;
        mmadParams.cmatrixSource = false;
        mmadParams.disableGemv = true;
        uint32_t pvL0AOffset = pvL0abBufId * (PV_L0A_SIZE / sizeof(DATA_T));
        uint32_t pvL0BOffset = pvL0abBufId * (PV_L0B_SIZE / sizeof(DATA_T));
        uint32_t pvL0COffset = pvL0cBufId * (PV_L0C_SIZE / sizeof(COMPUTE_T));
        Mmad(pvL0CTensor[pvL0COffset], pvL0ATensor[pvL0AOffset].ReinterpretCast<QUANT_T>(),
            pvL0BTensor[pvL0BOffset].ReinterpretCast<QUANT_T>(), mmadParams);
    }

    __aicore__ inline void FixpipeMm2(const RunInfo &info) {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (info.actMSize + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可
        fixpipeParams.mSize = constInfo.dSize + 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = constInfo.dSize + 16;
        fixpipeParams.dstStride = 64; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        fixpipeParams.dualDstCtl = 2; // 双目标模式，按N维度拆分

        uint32_t pvL0COffset = pvL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        Fixpipe<COMPUTE_T, COMPUTE_T, CFG_ROW_MAJOR_UB>(mm2ResUB, pvL0CTensor[pvL0COffset], fixpipeParams);
    }
};

}
#endif
