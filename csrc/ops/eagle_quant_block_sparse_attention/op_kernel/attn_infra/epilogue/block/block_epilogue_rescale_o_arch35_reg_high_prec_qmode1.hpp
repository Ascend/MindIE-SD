/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <LseMode Mode, class ElementS> struct QMode1RescaleOLseBuffers {};

template <class ElementS> struct QMode1RescaleOLseBuffers<LseMode::OUT_ONLY, ElementS> {
    AscendC::LocalTensor<float> lseUbTensor32;
    AscendC::LocalTensor<ElementS> gmUbTensor16;
};

template <class ElementO_, class ElementOTmp_, class ElementS_, class TileCopy_, class OTmpSrcPos_, LseMode LSE_MODE_,
    LseFormat LSE_FORMAT_, bool Q_IS_TND_>
class BlockEpilogue<EpilogueAtlasA5BsaQMode1RescaleO<LSE_MODE_, LSE_FORMAT_, Q_IS_TND_>, ElementO_, ElementOTmp_,
    ElementS_, TileCopy_, OTmpSrcPos_> : private QMode1RescaleOLseBuffers<LSE_MODE_, ElementS_> {
  public:
    using DispatchPolicy = EpilogueAtlasA5BsaQMode1RescaleO<LSE_MODE_, LSE_FORMAT_, Q_IS_TND_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementLse = float;
    using SMDtype = ElementS_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    static constexpr uint32_t UB_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64 * 2; //! 2* for 11 22 ... 32,32
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t RESCALE_VREG_SIZE = 256 / sizeof(ElementOTmp);
    static constexpr bool DEQ = true;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, uint32_t embed_ = 128, uint32_t qHeads = 1) {
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 128 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 128 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 128 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET + 128 * sizeof(float);
        constexpr uint32_t SCALE_UB_TENSOR_OFFSET = GL_UB_TENSOR_OFFSET + 128 * sizeof(float);
        constexpr uint32_t TMP_UB_TENSOR_OFFSET = SCALE_UB_TENSOR_OFFSET + 8 * 1024;
        constexpr uint32_t VALUE_SCALE_UB_TENSOR_OFFSET = TMP_UB_TENSOR_OFFSET + 1024;

        if constexpr (DispatchPolicy::Q_IS_TND) {
            strideO = qHeads * embed_;
        }
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
            constexpr uint32_t LSE_UB_TENSOR_OFFSET = VALUE_SCALE_UB_TENSOR_OFFSET + 2 * 128 * sizeof(float);
            constexpr uint32_t LSE_UB_ROW_STRIDE = DispatchPolicy::LSE_FORMAT == LseFormat::BNS1 ? 1 : 8;
            static_assert(LSE_UB_TENSOR_OFFSET + RESCALE_ROW_MAX_ELEM_NUM * LSE_UB_ROW_STRIDE * sizeof(float) <=
                ArchTag::UB_SIZE);
            this->lseUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);
            this->gmUbTensor16 = resource.ubBuf.template GetBufferByByte<SMDtype>(GM_UB_TENSOR_OFFSET);
        }
        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] =
                resource.ubBuf.template GetBufferByByte<ElementOTmp>(LO_UB_TENSOR_OFFSET + i * UB_UINT8_BLOCK_SIZE);
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(GO_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(GO_UB_TENSOR_OFFSET);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        scaleTensor =
            resource.ubBuf.template GetBufferByByte<float>(VALUE_SCALE_UB_TENSOR_OFFSET + embed_ * sizeof(float));
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorO>
    __aicore__ inline void CopyUbToGmOutput(
        const TensorO &gOTensor, uint32_t rows, uint32_t rowOffset, uint32_t colNum) {
        if constexpr (DispatchPolicy::Q_IS_TND) {
            AscendC::DataCopyExtParams repeatParams{};
            repeatParams.blockCount = rows;
            repeatParams.blockLen = colNum * sizeof(ElementO);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = (strideO - colNum) * sizeof(ElementO);
            AscendC::DataCopyPad(gOTensor[rowOffset * strideO], goUbTensor16, repeatParams);
        } else {
            DataCopy(gOTensor[rowOffset * colNum], goUbTensor16, rows * colNum);
        }
    }

    template <class TensorO>
    __aicore__ inline void WriteEmptyO(const TensorO &gOTensor, uint32_t rowNumOri, uint32_t colNumOri) {
        constexpr uint32_t FP8_BLOCK_SIZE = 32;
        uint32_t rowNumSplit = rowNumOri <= FP8_BLOCK_SIZE
            ? rowNumOri
            : RoundUp((rowNumOri + subBlockNum - 1) / subBlockNum, FP8_BLOCK_SIZE);
        uint32_t rows = subBlockIdx == 0 ? rowNumSplit : rowNumOri - rowNumSplit;
        uint32_t rowOffset = subBlockIdx == 0 ? 0 : rowNumSplit;
        if (rows == 0) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        auto goUbTensorU16 = goUbTensor16.template ReinterpretCast<uint16_t>();
        AscendC::Duplicate(goUbTensorU16, uint16_t{0}, rows * colNumOri);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        CopyUbToGmOutput(gOTensor, rows, rowOffset, colNumOri);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    template <class TensorLseGm, class TensorLseUb>
    __aicore__ inline void CopyUbToGmLse(TensorLseGm const &gLseTensorTlaTile, TensorLseUb const &ubLseTensorTla) {
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
            AscendC::DataCopyExtParams repeatParams{};
            if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
                repeatParams.blockCount = tla::get<0>(ubLseTensorTla.shape());
                repeatParams.blockLen = sizeof(float);
                repeatParams.srcStride = 0;
                repeatParams.dstStride = (tla::get<0>(gLseTensorTlaTile.stride()) - 1) * sizeof(float);
            } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
                repeatParams.blockCount = 1;
                repeatParams.blockLen = tla::get<0>(ubLseTensorTla.shape()) * sizeof(float);
                repeatParams.srcStride = 0;
                repeatParams.dstStride = 0;
            }
            auto dstOffset = gLseTensorTlaTile.layout()(gLseTensorTlaTile.coord());
            auto srcOffset = ubLseTensorTla.layout()(ubLseTensorTla.coord());
            AscendC::DataCopyPad(gLseTensorTlaTile.data()[dstOffset], ubLseTensorTla.data()[srcOffset], repeatParams);
        }
    }

    template <class TensorLse> __aicore__ inline void WriteEmptyLse(TensorLse const &gLseTensor, uint32_t rowNumOri) {
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
            constexpr uint32_t FP8_BLOCK_SIZE = 32;
            uint32_t rowNumSplit = rowNumOri <= FP8_BLOCK_SIZE
                ? rowNumOri
                : RoundUp((rowNumOri + subBlockNum - 1) / subBlockNum, FP8_BLOCK_SIZE);
            uint32_t rowNumCurSubCore = subBlockIdx == 0 ? rowNumSplit : rowNumOri - rowNumSplit;
            uint32_t rowOffsetCurSubCore = subBlockIdx == 0 ? 0 : rowNumSplit;
            if (rowNumCurSubCore == 0) {
                return;
            }
            uint32_t colNumLseUb = 0;
            uint32_t colStrideLseUb = 0;
            if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
                colNumLseUb = 8;
                colStrideLseUb = 8;
            } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
                colNumLseUb = 1;
                colStrideLseUb = 1;
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::Duplicate(this->lseUbTensor32, -__builtin_inff(), rowNumCurSubCore * colStrideLseUb);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            auto gLseTensorTlaTile =
                GetTile(gLseTensor, tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, 1));
            auto ubLseLayoutTla = tla::MakeLayout(
                tla::MakeShape(rowNumCurSubCore, colNumLseUb), tla::MakeStride(colStrideLseUb, tla::Int<1>{}));
            auto ubLseTensorTla = tla::MakeTensor(this->lseUbTensor32, ubLseLayoutTla, Arch::PositionUB{});
            CopyUbToGmLse(gLseTensorTlaTile, ubLseTensorTla);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        }
    }

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

    template <class TensorO, class TensorLse>
    __aicore__ inline void SubCoreCompute(const TensorO &gOTensor, const TensorLse &gLseTensorTlaTile,
        uint32_t curTileMod, uint32_t ubOTmpBufId, bool isFirstKvSTile, bool isLastKvSTile, uint32_t colNumOri,
        Arch::CrossCoreFlag mm2ToReFlag) {
        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *)goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *)loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = (__ubuf__ ElementOTmp *)glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *dmUb =
            (__ubuf__ ElementOTmp *)dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementOTmp *scaleUb = (__ubuf__ ElementOTmp *)scaleTensor.GetPhyAddr();

        WaitCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        // Also retires the previous LSE DMA before reusing its single UB buffer.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);

        if (isFirstKvSTile) {
            if (!isLastKvSTile) {
                AscendC::DataCopy(goUbTensor32, loUbTensor[ubOTmpBufId], rowNumCurSubCore * colNumOri);
                AscendC::PipeBarrier<PIPE_V>();
            } else { // go = lo div sum
                if (colNumOri == 64) {
                    RescaleFunc64<true, true>(goUb, loUb, dmUb, glUb, scaleUb, rowNumCurSubCore);
                } else {
                    DivFuncLastAndFirst<ElementOTmp, 128, DEQ>(goUb, loUb, glUb, scaleUb, rowNumCurSubCore);
                }
            }
        } else if (!isLastKvSTile) {
            if (colNumOri == 64) {
                RescaleFunc64<false, false>(goUb, loUb, dmUb, glUb, scaleUb, rowNumCurSubCore);
            } else {
                RescaleFunc<ElementOTmp, 128, DEQ>(goUb, loUb, dmUb, scaleUb, rowNumCurSubCore);
            }
        } else {
            if (colNumOri == 64) {
                RescaleFunc64<false, true>(goUb, loUb, dmUb, glUb, scaleUb, rowNumCurSubCore);
            } else {
                RescaleFuncLastNotFirst<ElementOTmp, 128, DEQ>(goUb, loUb, dmUb, glUb, scaleUb, rowNumCurSubCore);
            }
        }

        // release lo buf
        SetCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        if (isLastKvSTile) {
            if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
                __ubuf__ SMDtype *gmUb = (__ubuf__ SMDtype *)this->gmUbTensor16.GetPhyAddr();
                __ubuf__ float *lseUb = (__ubuf__ float *)this->lseUbTensor32.GetPhyAddr();
                uint32_t colNumLseUb = 0;
                uint32_t colStrideLseUb = 0;
                if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                    (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
                    LogSumExpFuncQSAxisIncontinuous(gmUb, glUb, lseUb, rowNumCurSubCore);
                    colNumLseUb = 8;
                    colStrideLseUb = 8;
                } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
                    LogSumExpFuncQSAxisContinuous(gmUb, glUb, lseUb, rowNumCurSubCore);
                    colNumLseUb = 1;
                    colStrideLseUb = 1;
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                auto ubLseLayoutTla = tla::MakeLayout(
                    tla::MakeShape(rowNumCurSubCore, colNumLseUb), tla::MakeStride(colStrideLseUb, tla::Int<1>{}));
                auto ubLseTensorTla = tla::MakeTensor(this->lseUbTensor32, ubLseLayoutTla, Arch::PositionUB{});
                CopyUbToGmLse(gLseTensorTlaTile, ubLseTensorTla);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(goUbTensor16, goUbTensor32, AscendC::RoundMode::CAST_RINT, rowNumCurSubCore * colNumOri);
            } else {
                AscendC::Cast(goUbTensor16, goUbTensor32, AscendC::RoundMode::CAST_NONE, rowNumCurSubCore * colNumOri);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            CopyUbToGmOutput(gOTensor, rowNumCurSubCore, rowOffsetCurSubCore, colNumOri);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    // D=64 has one FP32 vector per row; keep the existing D=128 VFs unchanged.
    template <bool IS_FIRST, bool IS_LAST>
    static __simd_vf__ inline void RescaleFunc64(__ubuf__ float *goUb, __ubuf__ float *loUb, __ubuf__ float *dmUb,
        __ubuf__ float *glUb, __ubuf__ float *scaleUb, uint32_t row) {
        using namespace AscendC::Reg;
        RegTensor<float> goVreg, loVreg, dmVreg, glVreg, scaleVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        if constexpr (IS_LAST) {
            LoadAlign<float>(scaleVreg, scaleUb);
        }
        for (uint16_t i = 0; i < row; ++i) {
            LoadAlign<float>(loVreg, loUb + i * 64);
            if constexpr (!IS_FIRST) {
                LoadAlign<float>(goVreg, goUb + i * 64);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + 2 * i);
                Mul(goVreg, goVreg, dmVreg, pregFull);
                Add(loVreg, goVreg, loVreg, pregFull);
            }
            if constexpr (IS_LAST) {
                Mul(loVreg, loVreg, scaleVreg, pregFull);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(glVreg, glUb + 2 * i);
                Div(loVreg, loVreg, glVreg, pregFull);
            }
            StoreAlign<float>(goUb + i * 64, loVreg, pregFull);
        }
    }

    template <typename T = float, uint32_t colStride = 128, bool DEQ = true>
    static __simd_vf__ inline void RescaleFunc(
        __ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *dmUb, __ubuf__ T *scaleUb, uint32_t row) {
        using namespace AscendC::Reg;

        RegTensor<float> dmVreg0, dmVreg1;
        RegTensor<float> goPreVreg0, goPreVreg0_2, goPreVreg1, goPreVreg1_2;
        RegTensor<float> loVreg0, loVreg0_2, loVreg1, loVreg1_2;
        RegTensor<float> mulVreg0, mulVreg0_2, mulVreg1, mulVreg1_2;
        RegTensor<float> goCurVreg0, goCurVreg0_2, goCurVreg1, goCurVreg1_2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t vlElemNum = 64;

        constexpr uint16_t halfRow = 32;
        __ubuf__ T *ptr_go0 = goUb;
        __ubuf__ T *ptr_go1 = goUb + halfRow * colStride;
        __ubuf__ T *ptr_lo0 = loUb;
        __ubuf__ T *ptr_lo1 = loUb + halfRow * colStride;
        __ubuf__ T *out_go0 = goUb;
        __ubuf__ T *out_go1 = goUb + halfRow * colStride;

        for (uint16_t i = 0; i < halfRow; i++) {
            uint32_t row1_idx = i + halfRow;
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg0, dmUb + 2 * i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg1, dmUb + 2 * row1_idx);

            // Load Row i
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goPreVreg0, ptr_go0, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goPreVreg0_2, ptr_go0, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(loVreg0, ptr_lo0, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(loVreg0_2, ptr_lo0, vlElemNum);

            // Load Row i+32
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goPreVreg1, ptr_go1, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goPreVreg1_2, ptr_go1, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(loVreg1, ptr_lo1, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(loVreg1_2, ptr_lo1, vlElemNum);

            // Multiply (4-issue)
            Mul(mulVreg0, goPreVreg0, dmVreg0, pregFull);
            Mul(mulVreg0_2, goPreVreg0_2, dmVreg0, pregFull);
            Mul(mulVreg1, goPreVreg1, dmVreg1, pregFull);
            Mul(mulVreg1_2, goPreVreg1_2, dmVreg1, pregFull);

            // Always accumulate, scale is deferred to the last iteration
            Add(loVreg0, mulVreg0, loVreg0, pregFull);
            Add(loVreg0_2, mulVreg0_2, loVreg0_2, pregFull);
            Add(loVreg1, mulVreg1, loVreg1, pregFull);
            Add(loVreg1_2, mulVreg1_2, loVreg1_2, pregFull);

            // Store (4-issue)
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go0, loVreg0, vlElemNum, pregFull);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go0, loVreg0_2, vlElemNum, pregFull);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go1, loVreg1, vlElemNum, pregFull);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go1, loVreg1_2, vlElemNum, pregFull);
        }
    }

    template <typename T = float, uint32_t colStride = 128, bool DEQ = true>
    static __simd_vf__ inline void RescaleFuncLastNotFirst(
        __ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *dmUb, __ubuf__ T *glUb, __ubuf__ T *scaleUb, uint32_t row) {
        using namespace AscendC::Reg;

        RegTensor<float> dmVreg0, glVreg0, dmVreg1, glVreg1;
        RegTensor<float> scaleVreg0, scaleVreg1; // 增加 scale 寄存器
        RegTensor<float> goPreVreg0, goPreVreg0_2, goPreVreg1, goPreVreg1_2;
        RegTensor<float> loVreg0, loVreg0_2, loVreg1, loVreg1_2;
        RegTensor<float> mulVreg0, mulVreg0_2, mulVreg1, mulVreg1_2;
        RegTensor<float> goCurVreg0, goCurVreg0_2, goCurVreg1, goCurVreg1_2;
        RegTensor<float> divVreg0, divVreg0_2, divVreg1, divVreg1_2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t vlElemNum = 64;
        constexpr uint16_t halfRow = 32;

        // 若开启反量化，在循环外一次性加载 per-channel scale
        if constexpr (DEQ) {
            LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg0, scaleUb); // col0
            LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg1, scaleUb + vlElemNum); // col1
        }

        __ubuf__ T *ptr_go0 = goUb;
        __ubuf__ T *ptr_go1 = goUb + halfRow * colStride;
        __ubuf__ T *ptr_lo0 = loUb;
        __ubuf__ T *ptr_lo1 = loUb + halfRow * colStride;
        __ubuf__ T *out_go0 = goUb;
        __ubuf__ T *out_go1 = goUb + halfRow * colStride;

        for (uint16_t i = 0; i < halfRow; i++) {
            uint32_t row1_idx = i + halfRow;

            // Load scalars (i and i+32)
            //! 2* for 11 22 ... 32,32
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg0, dmUb + 2 * i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg0, glUb + 2 * i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg1, dmUb + 2 * row1_idx);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg1, glUb + 2 * row1_idx);

            // Load Data
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goPreVreg0, ptr_go0, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goPreVreg0_2, ptr_go0, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goPreVreg1, ptr_go1, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goPreVreg1_2, ptr_go1, vlElemNum);

            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(loVreg0, ptr_lo0, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(loVreg0_2, ptr_lo0, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(loVreg1, ptr_lo1, vlElemNum);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(loVreg1_2, ptr_lo1, vlElemNum);

            // Muls: goPre * dm
            Mul(mulVreg0, goPreVreg0, dmVreg0, pregFull);
            Mul(mulVreg0_2, goPreVreg0_2, dmVreg0, pregFull);
            Mul(mulVreg1, goPreVreg1, dmVreg1, pregFull);
            Mul(mulVreg1_2, goPreVreg1_2, dmVreg1, pregFull);

            // Accumulate in the quantized domain, then apply the deferred scale.
            if constexpr (DEQ) {
                Add(loVreg0, mulVreg0, loVreg0, pregFull);
                Add(loVreg0_2, mulVreg0_2, loVreg0_2, pregFull);
                Add(loVreg1, mulVreg1, loVreg1, pregFull);
                Add(loVreg1_2, mulVreg1_2, loVreg1_2, pregFull);

                Mul(loVreg0, loVreg0, scaleVreg0, pregFull);
                Mul(loVreg0_2, loVreg0_2, scaleVreg1, pregFull);
                Mul(loVreg1, loVreg1, scaleVreg0, pregFull);
                Mul(loVreg1_2, loVreg1_2, scaleVreg1, pregFull);
            } /* else {
                Add(loVreg0,   mulVreg0,   loVreg0,   pregFull);
                Add(loVreg0_2, mulVreg0_2, loVreg0_2, pregFull);
                Add(loVreg1,   mulVreg1,   loVreg1,   pregFull);
                Add(loVreg1_2, mulVreg1_2, loVreg1_2, pregFull);
            } */

            Div(divVreg0, loVreg0, glVreg0, pregFull);
            Div(divVreg0_2, loVreg0_2, glVreg0, pregFull);
            Div(divVreg1, loVreg1, glVreg1, pregFull);
            Div(divVreg1_2, loVreg1_2, glVreg1, pregFull);

            // Stores
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go0, divVreg0, vlElemNum, pregFull);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go0, divVreg0_2, vlElemNum, pregFull);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go1, divVreg1, vlElemNum, pregFull);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go1, divVreg1_2, vlElemNum, pregFull);
        }
    }

    template <typename T = float, uint32_t colStride = 128, bool DEQ = true>
    static __simd_vf__ inline void DivFuncLastAndFirst(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *glUb,
        __ubuf__ T *scaleUb, // 保持接口一致性
        uint32_t row) {
        using namespace AscendC::Reg;

        RegTensor<float> goCurVreg0, goCurVreg0_2, goCurVreg1, goCurVreg1_2;
        RegTensor<float> glVreg0, glVreg1;
        RegTensor<float> scaleVreg0, scaleVreg1; // 仅做声明备用
        RegTensor<float> divVreg0, divVreg0_2, divVreg1, divVreg1_2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t vlElemNum = 64;

        if constexpr (colStride == 128) {
            constexpr uint16_t halfRow = 32;
            if constexpr (DEQ) {
                LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg0, scaleUb);
                LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg1, scaleUb + vlElemNum);
            }
            __ubuf__ T *ptr_go0 = goUb;
            __ubuf__ T *ptr_go1 = goUb + halfRow * colStride;
            __ubuf__ T *ptr_lo0 = loUb;
            __ubuf__ T *ptr_lo1 = loUb + halfRow * colStride;
            __ubuf__ T *out_go0 = goUb;
            __ubuf__ T *out_go1 = goUb + halfRow * colStride;

            for (uint16_t i = 0; i < halfRow; i++) {
                uint32_t row1_idx = i + halfRow;

                //! 2* for 11 22 ... 32,32
                LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg0, glUb + 2 * i);
                LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg1, glUb + 2 * row1_idx);

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goCurVreg0, ptr_lo0, vlElemNum);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goCurVreg0_2, ptr_lo0, vlElemNum);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goCurVreg1, ptr_lo1, vlElemNum);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(goCurVreg1_2, ptr_lo1, vlElemNum);

                if constexpr (DEQ) {
                    Mul(goCurVreg0, goCurVreg0, scaleVreg0, pregFull);
                    Mul(goCurVreg0_2, goCurVreg0_2, scaleVreg1, pregFull);
                    Mul(goCurVreg1, goCurVreg1, scaleVreg0, pregFull);
                    Mul(goCurVreg1_2, goCurVreg1_2, scaleVreg1, pregFull);
                }

                Div(divVreg0, goCurVreg0, glVreg0, pregFull);
                Div(divVreg0_2, goCurVreg0_2, glVreg0, pregFull);
                Div(divVreg1, goCurVreg1, glVreg1, pregFull);
                Div(divVreg1_2, goCurVreg1_2, glVreg1, pregFull);

                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go0, divVreg0, vlElemNum, pregFull);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go0, divVreg0_2, vlElemNum, pregFull);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go1, divVreg1, vlElemNum, pregFull);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(out_go1, divVreg1_2, vlElemNum, pregFull);
            }
        }
    }

    // BNS1: gm has four FP16 copies per Query row, gl has two FP32 copies.
    static __simd_vf__ inline void LogSumExpFuncQSAxisContinuous(
        __ubuf__ SMDtype *gmUb, __ubuf__ float *glUb, __ubuf__ float *lseUb, uint32_t row) {
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY && DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
            using namespace AscendC::Reg;
            RegTensor<SMDtype> gmVreg16, gmOddVreg16;
            RegTensor<float> gmVreg, glVreg, glOddVreg, logGlVreg, lseVreg;
            MaskReg pregRows = UpdateMask<float>(row);
            static constexpr CastTrait castTrait = {
                RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
            static constexpr LnSpecificMode mode = {MaskMergeMode::ZEROING, AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};
            // DINTLV keeps gm[2*j] in gmVreg16. The ZERO cast then selects gm[4*i].
            LoadAlign<SMDtype, LoadDist::DIST_DINTLV_B16>(gmVreg16, gmOddVreg16, gmUb);
            Cast<float, SMDtype, castTrait>(gmVreg, gmVreg16, pregRows);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(glVreg, glOddVreg, glUb);
            Ln<float, &mode>(logGlVreg, glVreg, pregRows);
            Add(lseVreg, gmVreg, logGlVreg, pregRows);
            StoreAlign<float>(lseUb, lseVreg, pregRows);
        }
    }

    // Broadcast each LSE into a 32-byte UB slot for strided Query-axis copies.
    // Only the first float of each slot is copied to GM.
    static __simd_vf__ inline void LogSumExpFuncQSAxisIncontinuous(
        __ubuf__ SMDtype *gmUb, __ubuf__ float *glUb, __ubuf__ float *lseUb, uint32_t row) {
        if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
            using namespace AscendC::Reg;
            RegTensor<SMDtype> gmVreg16;
            RegTensor<float> gmVreg, glVreg, logGlVreg, lseVreg;
            constexpr uint32_t postUpdateStride = 8;
            UnalignReg rowwiseUreg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregHalf = CreateMask<SMDtype, MaskPattern::ALL>();
            static constexpr CastTrait castTrait = {
                RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
            static constexpr LnSpecificMode mode = {MaskMergeMode::ZEROING, AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};
            for (uint16_t i = 0; i < row; ++i) {
                LoadAlign<SMDtype, LoadDist::DIST_BRC_B16>(gmVreg16, gmUb + 4 * i);
                Cast<float, SMDtype, castTrait>(gmVreg, gmVreg16, pregHalf);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(glVreg, glUb + 2 * i);
                Ln<float, &mode>(logGlVreg, glVreg, pregFull);
                Add(lseVreg, gmVreg, logGlVreg, pregFull);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(lseUb, lseVreg, rowwiseUreg, postUpdateStride);
            }
            StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(lseUb, rowwiseUreg, postUpdateStride);
        }
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline void operator()(const TensorO &gOTensor, const TensorLse &gLseTensor, GemmCoord actualOriShape,
        uint32_t curTileMod, uint32_t gatheredKvSTileIdx, bool isFirstKvSTile, bool isLastKvSTile,
        Arch::CrossCoreFlag mm2ToReFlag) {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        constexpr uint32_t FP8_BLOCK_SIZE = 32;
        if (rowNumOri <= FP8_BLOCK_SIZE) {
            rowNumCurSubCore = (subBlockIdx == 0) ? rowNumOri : 0;
        } else {
            uint32_t mhalf = (rowNumOri + subBlockNum - 1) / subBlockNum;
            uint32_t mAlign = (mhalf > FP8_BLOCK_SIZE) ? RoundUp(mhalf, FP8_BLOCK_SIZE) : FP8_BLOCK_SIZE;
            rowNumCurSubCore = (subBlockIdx == 0) ? mAlign : (rowNumOri - mAlign);
        }
        rowOffsetCurSubCore = subBlockIdx == 0 ? 0 : rowNumOri - rowNumCurSubCore;
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;

        if (rowNumCurSubCore > 0) {
            if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
                auto gLseTensorTlaTile =
                    GetTile(gLseTensor, tla::MakeCoord(rowOffsetCurSubCore, 0), tla::MakeShape(rowNumCurSubCore, 1));
                SubCoreCompute(gOTensor, gLseTensorTlaTile, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile,
                    colNumOri, mm2ToReFlag);
            } else {
                SubCoreCompute(
                    gOTensor, nullptr, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile, colNumOri, mm2ToReFlag);
            }
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
    }

    // Preserve the original O-only call without constructing an unused LSE GM tensor.
    template <class TensorO>
    __aicore__ inline void operator()(const TensorO &gOTensor, GemmCoord actualOriShape, uint32_t curTileMod,
        uint32_t gatheredKvSTileIdx, bool isFirstKvSTile, bool isLastKvSTile, Arch::CrossCoreFlag mm2ToReFlag) {
        static_assert(DispatchPolicy::LSE_MODE == LseMode::NONE);
        (*this)(gOTensor, nullptr, actualOriShape, curTileMod, gatheredKvSTileIdx, isFirstKvSTile, isLastKvSTile,
            mm2ToReFlag);
    }

  private:
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<SMDtype> dmUbTensor16;
    AscendC::LocalTensor<SMDtype> glUbTensor16;
    AscendC::LocalTensor<float> dmUbTensor32;
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<float> scaleTensor;
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;

    uint32_t subBlockNum = AscendC::GetSubBlockNum();
    uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t rowOffsetCurSubCore = 0;
    uint32_t rowNumCurSubCore = 0;
    uint32_t strideO = 0;
};
}
#endif
