/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_QMODE2
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_QMODE2

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <class ElementO_, class ElementOTmp_, class ElementS_, class TileCopy_, class OTmpSrcPos_>
class BlockEpilogue<EpilogueAtlasA5BsaQMode2RescaleO, ElementO_, ElementOTmp_, ElementS_, TileCopy_, OTmpSrcPos_> {
  public:
    using DispatchPolicy = EpilogueAtlasA5BsaQMode2RescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using SMDtype = ElementS_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;
    using SElementType = half;

    static constexpr uint32_t UB_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t MX_BLOCK_SIZE = 32;
    static constexpr uint32_t EVEN = 2;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FP4_BLOCK_SIZE = 64;
    static constexpr uint32_t VL_INT8 = 256;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64 * 2;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = SM_COL_MAX_ELEM_NUM * SM_ROW_MAX_ELEM_NUM;
    static constexpr uint32_t MAX_P_L1_BUF_STAGES = 4;
    static constexpr uint32_t S_UB_SLOT_STAGES = 3;
    static constexpr uint32_t S_UB_SLOT_BYTES = MAX_UB_S_ELEM_NUM * sizeof(SElementType);
    static constexpr uint32_t P_UB_DATA_BYTES = MAX_UB_S_ELEM_NUM / 2;
    static constexpr uint32_t P_UB_SCALE_BYTES = MAX_UB_S_ELEM_NUM / MX_BLOCK_SIZE;
    static constexpr uint32_t P_UB_PAYLOAD_BYTES = P_UB_DATA_BYTES + P_UB_SCALE_BYTES;
    static constexpr uint32_t P_UB_GROUP_MAX_BYTES = MAX_UB_S_ELEM_NUM / MX_BLOCK_SIZE * sizeof(SElementType);
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;

    AscendC::LocalTensor<float> vLv0ScaleTensor;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, uint32_t embed_ = 128) {
        using ElementInput = half;
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t HEAD_DIM = 128, SUM_DIM = 8;
        constexpr uint32_t O_CURR_UB_TENSOR_OFFSET = S_UB_SLOT_BYTES * S_UB_SLOT_STAGES; // 32 * 3
        constexpr uint32_t O_HIS_UB_TENSOR_OFFSET = O_CURR_UB_TENSOR_OFFSET +
            SM_ROW_MAX_ELEM_NUM * (HEAD_DIM + SUM_DIM) * UB_OTMP_BUF_STAGES * sizeof(float); // 34 * 2
        constexpr uint32_t P_PAYLOAD_UB_TENSOR_OFFSET =
            O_HIS_UB_TENSOR_OFFSET + SM_ROW_MAX_ELEM_NUM * HEAD_DIM * sizeof(float); // 32
        constexpr uint32_t GROUP_MAX_UB_TENSOR_OFFSET =
            P_PAYLOAD_UB_TENSOR_OFFSET + P_UB_PAYLOAD_BYTES * S_UB_SLOT_STAGES; // 8.5 * 3
        constexpr uint32_t GATHER_INDEX_OFFSET = GROUP_MAX_UB_TENSOR_OFFSET + P_UB_GROUP_MAX_BYTES; // 2
        constexpr uint32_t LM_UB_TENSOR_OFFSET = GATHER_INDEX_OFFSET + VL_INT8; // gather index 256B
        constexpr uint32_t GM_UB_TENSOR_OFFSET =
            LM_UB_TENSOR_OFFSET + SM_ROW_MAX_ELEM_NUM * EVEN * sizeof(ElementInput); // local_max 256B
        constexpr uint32_t VSCALE_UB_TENSOR_OFFSET =
            GM_UB_TENSOR_OFFSET + SM_ROW_MAX_ELEM_NUM * EVEN * sizeof(ElementInput); // global_max 256B
        constexpr uint32_t GL_UB_TENSOR_OFFSET = VSCALE_UB_TENSOR_OFFSET + HEAD_DIM * DB * sizeof(float); // VSCALE 1kb
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GL_UB_TENSOR_OFFSET + SM_ROW_MAX_ELEM_NUM * sizeof(float); // sum 256B
        constexpr uint32_t TMP_UB_TENSOR_OFFSET =
            DM_UB_TENSOR_OFFSET + MAX_P_L1_BUF_STAGES * DM_UB_GLOBAL_ELEM_NUM * sizeof(half);

        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                O_CURR_UB_TENSOR_OFFSET + i * (SM_ROW_MAX_ELEM_NUM * (HEAD_DIM + SUM_DIM)) * sizeof(ElementOTmp));
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(O_HIS_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(O_HIS_UB_TENSOR_OFFSET);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor16 = resource.ubBuf.template GetBufferByByte<half>(DM_UB_TENSOR_OFFSET);
        vLv0ScaleTensor = resource.ubBuf.template GetBufferByByte<float>(VSCALE_UB_TENSOR_OFFSET);
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <uint32_t MODE, pipe_t PIPE> __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag) {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE> __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag) {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorDst>
    __aicore__ inline void SubCoreCompute(const TensorDst &gOTensor, uint32_t curTileMod, uint32_t ubOTmpBufId,
        bool isFirstRescaleO, bool isLastDiv, uint32_t colNumOri, Arch::CrossCoreFlag mm2ToReFlag) {
        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *)goUbTensor32.GetPhyAddr();
        __ubuf__ ElementO *go16Ub = (__ubuf__ ElementO *)goUbTensor16.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *)loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = (__ubuf__ ElementOTmp *)glUbTensor32.GetPhyAddr();
        __ubuf__ half *dmUb = (__ubuf__ half *)dmUbTensor16[curTileMod * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementOTmp *scaleUb = (__ubuf__ ElementOTmp *)vLv0ScaleTensor.GetPhyAddr();

        WaitCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        UpdateSumAndScale(loUb, glUb, dmUb, isLastDiv ? 0 : 1); // scale when O scale
        AscendC::PipeBarrier<PIPE_V>();
        if (isLastDiv) {
            AddDivDeqCastLast(go16Ub, goUb, loUb, glUb, scaleUb, rowNumSub, isFirstRescaleO ? 0 : 1);
        } else {
            AddRescale(goUb, loUb, dmUb, isFirstRescaleO ? 0 : 1);
        }

        // release lo buf
        SetCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        if (isLastDiv) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(gOTensor[rowOffsetSub * colNumOri], goUbTensor16, rowNumSub * colNumOri);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    template <typename T = float, uint32_t headDim = 128>
    static __simd_vf__ inline void UpdateSumAndScale(
        __ubuf__ T *loUb, __ubuf__ T *glUb, __ubuf__ half *dmUb, uint16_t doRescale = 0) { // addRescale
        using namespace AscendC::MicroAPI;
        RegTensor<T> llVreg0, llVreg1;
        RegTensor<T> glVreg0, glVreg1;
        RegTensor<T> dmVreg0, dmVreg1;
        RegTensor<half> vTmp0, vTmp1;

        MaskReg pregFull = CreateMask<T, MaskPattern::ALL>();
        MaskReg pregFull16 = CreateMask<half, MaskPattern::ALL>();
        MaskReg preg8 = CreateMask<T, MaskPattern::VL8>();

        constexpr uint32_t colNum = headDim + 8;
        constexpr static AscendC::MicroAPI::CastTrait castTraitZero{AscendC::MicroAPI::RegLayout::ZERO,
            AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND};

        for (uint16_t i = 0; i < 4; i++) {
            LoadAlign<T, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                llVreg0, loUb + colNum * (i * 8) + headDim, colNum / 8, pregFull);
            LoadAlign<T, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                llVreg1, loUb + colNum * (i * 8 + 32) + headDim, colNum / 8, pregFull);
            LoadAlign<T, LoadDist::DIST_E2B_B32>(glVreg0, glUb + i * 8);
            LoadAlign<T, LoadDist::DIST_E2B_B32>(glVreg1, glUb + i * 8 + 32);
            Add(glVreg0, glVreg0, llVreg0, pregFull);
            Add(glVreg1, glVreg1, llVreg1, pregFull);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(glVreg0, glVreg0, pregFull);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(glVreg1, glVreg1, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(glUb + i * 8, glVreg0, preg8);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(glUb + i * 8 + 32, glVreg1, preg8);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign<T, LoadDist::DIST_NORM>(glVreg0, glUb);
        for (uint16_t j = 0; j < doRescale; j++) {
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(vTmp0, dmUb);
            Cast<int32_t, half, castTraitZero>((RegTensor<int32_t> &)dmVreg0, vTmp0, pregFull16);
            Adds((RegTensor<int32_t> &)dmVreg0, (RegTensor<int32_t> &)dmVreg0, (int32_t)127, pregFull);
            Maxs((RegTensor<int32_t> &)dmVreg0, (RegTensor<int32_t> &)dmVreg0, (int32_t)1, pregFull);
            ShiftLefts((RegTensor<int32_t> &)dmVreg0, (RegTensor<int32_t> &)dmVreg0, (int16_t)23, pregFull);
            StoreAlign((__ubuf__ T *&)dmUb, dmVreg0, pregFull);
            Mul(glVreg0, glVreg0, dmVreg0, pregFull);
        }
        if constexpr (VshrinkExp != 1) {
            uint16_t isLastDiv = doRescale ^ 1;
            for (uint16_t j = 0; j < isLastDiv; j++) {
                Muls(glVreg0, glVreg0, SCALE_SHRINK, pregFull);
            }
        }
        StoreAlign<T, StoreDist::DIST_NORM_B32>(glUb, glVreg0, pregFull);
    }

    template <typename T = float, uint32_t headDim = 128>
    static __simd_vf__ inline void AddRescale(
        __ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ half *dmUb, uint16_t nonFirst) {
        using namespace AscendC::MicroAPI;
        __ubuf__ T *dmUb32 = (__ubuf__ T *)dmUb;
        RegTensor<float> dmVreg0, dmVreg1, scaleVreg0, scaleVreg1;
        RegTensor<float> goVreg0, goVreg0_2, goVreg1, goVreg1_2;
        RegTensor<float> loVreg0, loVreg0_2, loVreg1, loVreg1_2;
        RegTensor<float> mulVreg0, mulVreg0_2, mulVreg1, mulVreg1_2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregFull16 = CreateMask<half, MaskPattern::ALL>();
        constexpr uint32_t vlElemNum = 64, colNum = headDim + 8;
        uint32_t halfRow = 32;

        for (uint16_t i = 0; i < halfRow; i++) {
            uint32_t row1_idx = i + halfRow;
            uint32_t baseOffset0 = i * colNum;
            uint32_t secondOffset0 = baseOffset0 + vlElemNum;
            uint32_t baseOffset1 = row1_idx * colNum;
            uint32_t secondOffset1 = baseOffset1 + vlElemNum;

            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg0, (dmUb32) + i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg1, (dmUb32) + row1_idx);

            LoadAlign<T, LoadDist::DIST_NORM>(loVreg0, loUb + baseOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg0_2, loUb + secondOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg1, loUb + baseOffset1);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg1_2, loUb + secondOffset1);

            for (uint16_t j = 0; j < nonFirst; j++) {
                LoadAlign<T, LoadDist::DIST_NORM>(goVreg0, goUb + i * headDim);
                LoadAlign<T, LoadDist::DIST_NORM>(goVreg0_2, goUb + i * headDim + vlElemNum);
                Add(loVreg0, goVreg0, loVreg0, pregFull);
                Add(loVreg0_2, goVreg0_2, loVreg0_2, pregFull);

                LoadAlign<T, LoadDist::DIST_NORM>(goVreg1, goUb + (i + halfRow) * headDim);
                LoadAlign<T, LoadDist::DIST_NORM>(goVreg1_2, goUb + (i + halfRow) * headDim + vlElemNum);
                Add(loVreg1, goVreg1, loVreg1, pregFull);
                Add(loVreg1_2, goVreg1_2, loVreg1_2, pregFull);
            }

            Mul(loVreg0, loVreg0, dmVreg0, pregFull);
            Mul(loVreg0_2, loVreg0_2, dmVreg0, pregFull);
            Mul(loVreg1, loVreg1, dmVreg1, pregFull);
            Mul(loVreg1_2, loVreg1_2, dmVreg1, pregFull);

            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * headDim, loVreg0, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + i * headDim + vlElemNum, loVreg0_2, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + (i + halfRow) * headDim, loVreg1, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + (i + halfRow) * headDim + vlElemNum, loVreg1_2, pregFull);
        }
    }

    template <typename OutputType, typename T = float, uint32_t headDim = 128>
    static __simd_vf__ inline void AddDivDeqCastLast(__ubuf__ OutputType *go16Ub, __ubuf__ T *goUb, __ubuf__ T *loUb,
        __ubuf__ T *glUb, __ubuf__ T *scaleUb, uint32_t row, uint16_t nonFirst) {
        using namespace AscendC::MicroAPI;
        RegTensor<float> goVreg0, goVreg0_2, goVreg1, goVreg1_2;
        RegTensor<float> glVreg0, glVreg1;
        RegTensor<float> scaleVreg0, scaleVreg1; // vmax
        RegTensor<OutputType> divVreg0, divVreg0_2, divVreg1, divVreg1_2;
        RegTensor<T> loCurVreg0, loCurVreg0_2, loCurVreg1, loCurVreg1_2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();

        constexpr uint32_t vlElemNum = 64;
        constexpr uint32_t colNum = headDim + 8;

        constexpr static AscendC::MicroAPI::CastTrait castTraitZero{AscendC::MicroAPI::RegLayout::ZERO,
            AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
            std::is_same<OutputType, bfloat16_t>::value ? AscendC::RoundMode::CAST_RINT
                                                        : AscendC::RoundMode::CAST_ROUND};

        if constexpr (headDim == 128) {
            LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg0, scaleUb);
            LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg1, scaleUb + vlElemNum);
            for (uint16_t i = 0; i < row; i++) {
                uint32_t baseOffset0 = i * colNum;
                uint32_t secondOffset0 = baseOffset0 + vlElemNum;
                LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg0, glUb + i);
                LoadAlign<T, LoadDist::DIST_NORM>(loCurVreg0, loUb + baseOffset0);
                LoadAlign<T, LoadDist::DIST_NORM>(loCurVreg0_2, loUb + secondOffset0);

                for (uint16_t j = 0; j < nonFirst; j++) {
                    LoadAlign<T, LoadDist::DIST_NORM>(goVreg0, goUb + i * headDim);
                    LoadAlign<T, LoadDist::DIST_NORM>(goVreg0_2, goUb + i * headDim + vlElemNum);
                    Add(loCurVreg0, goVreg0, loCurVreg0, pregFull);
                    Add(loCurVreg0_2, goVreg0_2, loCurVreg0_2, pregFull);
                }
                Mul(loCurVreg0, loCurVreg0, scaleVreg0, pregFull);
                Mul(loCurVreg0_2, loCurVreg0_2, scaleVreg1, pregFull);
                Div(loCurVreg0, loCurVreg0, glVreg0, pregFull);
                Div(loCurVreg0_2, loCurVreg0_2, glVreg0, pregFull);
                Cast<OutputType, T, castTraitZero>(divVreg0, loCurVreg0, pregFull);
                Cast<OutputType, T, castTraitZero>(divVreg0_2, loCurVreg0_2, pregFull);
                StoreAlign<OutputType, StoreDist::DIST_PACK_B32>(go16Ub + i * headDim, divVreg0, pregFull);
                StoreAlign<OutputType, StoreDist::DIST_PACK_B32>(
                    go16Ub + i * headDim + vlElemNum, divVreg0_2, pregFull);
            }
        }
    }

    template <class TensorDst>
    __aicore__ inline void operator()(const TensorDst &gOTensor, GemmCoord actualOriShape, uint32_t curTileMod,
        uint32_t ubOTmpBufId, bool isFirstRescaleO, bool isLastDiv, Arch::CrossCoreFlag mm2ToReFlag) {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        constexpr uint32_t FP4_BLOCK_SIZE = 64;
        if (rowNumOri <= FP4_BLOCK_SIZE) {
            rowNumSub = (subBlockIdx == 0) ? rowNumOri : 0;
        } else {
            rowNumSub = subBlockIdx == 0 ? RESCALE_ROW_MAX_ELEM_NUM : rowNumOri - RESCALE_ROW_MAX_ELEM_NUM;
        }
        rowOffsetSub = subBlockIdx == 0 ? 0 : rowNumOri - rowNumSub;

        if (rowNumSub > 0) {
            SubCoreCompute(gOTensor, curTileMod, ubOTmpBufId, isFirstRescaleO, isLastDiv, colNumOri, mm2ToReFlag);
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
    }

  private:
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<half> dmUbTensor16;
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;

    uint32_t subBlockNum = AscendC::GetSubBlockNum();
    uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t rowOffsetSub = 0;
    uint32_t rowNumSub = 0;

    static constexpr float get_pow2_k(int k) {
        float f = 1.0f;
        if (k > 0) {
            for (int i = 0; i < k; ++i) {
                f *= 2.0f;
            }
        } else {
            for (int i = 0; i < -k; ++i) {
                f /= 2.0f;
            }
        }
        return f;
    }

    static constexpr uint32_t VshrinkExp = 1;
    static constexpr float SCALE_SHRINK = get_pow2_k(VshrinkExp);
};
}
#endif
