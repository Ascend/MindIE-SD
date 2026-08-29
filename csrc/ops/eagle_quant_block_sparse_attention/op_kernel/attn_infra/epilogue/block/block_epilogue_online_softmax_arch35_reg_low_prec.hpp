/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"
using AscendC::printf;
#define VF_MIN(a, b) ((b) + (uint16_t((a) - (b)) & -(uint16_t((a) - (b)) >> 15)))

namespace NpuArch::Epilogue::Block {

template <typename DefaultType>
struct ElementPTmpTypeLookup {
    using type = DefaultType;
};

template <> 
struct ElementPTmpTypeLookup<float8_e4m3_t> {
    using type = half;
};

enum class KvBaseTileRegSplitStages {
    ONE,
    TWO
};

template <
    class OutputType_,
    class LayoutS_>
class BlockEpilogue<
    EpilogueOnlineSoftmaxBsa,
    OutputType_,
    Gemm::GemmType<half, LayoutS_>>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementOutputTmp = typename ElementPTmpTypeLookup<ElementOutput>::type;


    using ElementInput = half;
    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FP8_BLOCK_SIZE = 32;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 32768;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64 * 2; //! 2* for 11 22 ... 32,32
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    static constexpr ElementInput MIN_VALUE = -65504.0f;
    static constexpr uint32_t FP8_REP_SIZE = 128 *2;
    static constexpr uint32_t HALF_REP_SIZE = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 512;
    static constexpr uint32_t SM_VREG_SIZE = 256 / sizeof(ElementInput);
    static constexpr uint32_t QUANT_MODE1_SCALE_BYTES = 1024;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_, uint32_t blockSizeY_ = 128)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = LS_UB_TENSOR_OFFSET;

        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 128 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 128 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 128 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  128 * sizeof(float);
        constexpr uint32_t SCALE_UB_TENSOR_OFFSET = GL_UB_TENSOR_OFFSET + 128 * sizeof(float);

        subBlockIdx_ = AscendC::GetSubBlockIdx();

        scaleValue = scaleValue_;
        blockSizeK = blockSizeY_;
        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<uint16_t>(LP_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(GM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LL_UB_TENSOR_OFFSET);
        scaleTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(SCALE_UB_TENSOR_OFFSET);
        tmpTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(LM_UB_TENSOR_OFFSET + 4096U);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline
    void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t m, uint32_t nRound, uint32_t actRowsCube)
    {
        constexpr uint32_t BlockElements = BLOCK_SIZE_IN_BYTE / sizeof(ElementOutput);
        if constexpr (sizeof(ElementOutput) == 2) {
            const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
            const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
            const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());
            constexpr int32_t C0_SIZE = BLOCK_SIZE_IN_BYTE / sizeof(typename TensorDst::Element);
            AscendC::DataCopyParams repeatParams;

            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = m;
            repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / C0_SIZE - m;
            repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / C0_SIZE - m;

            auto dstOffset = dstTensor.layout()(dstTensor.coord());
            auto srcOffset = srcTensor.layout()(srcTensor.coord());

            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
        } else if constexpr (sizeof(ElementOutput) == 1) { // dn 2 Zn
            auto mRound = RoundUp(m, BlockElements);
            auto dstOffset = subBlockIdx_ == 0 ? 0 : (actRowsCube - m) * nRound;
            AscendC::DataCopy(dstTensor.data()[dstOffset],  srcTensor.data(), mRound * nRound);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag){
        
        struct EmptyTensor {};
        EmptyTensor dummyQS;
        EmptyTensor dummyKS;
        operator()<0>(l1PTensorTla, dummyQS, dummyKS, 
                        actualBlockShape,
                        isFirstKvSTile, ubSBufId, l1PBufId,
                        mm1ToSmFlag, smToMm2Flag, AscendC::GlobalTensor<int32_t>());
    }
    
    template <int quant_mode, class TensorP, class TensorQS, class TensorKS, typename ElementIndex>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, TensorQS &gmQSTensorTla, TensorKS &gmKSTensorTla,
         GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag,
        const AscendC::GlobalTensor<ElementIndex>& sparseIndex)
    {
        static_assert(quant_mode == 0 || quant_mode == 1);       
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        uint32_t m;
        uint16_t mRound;
        uint32_t mCopyOffset = 0;
        const uint32_t mTot = actualBlockShape.m();
        if constexpr (quant_mode == 0) {
            mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
            m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
            m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
            mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        } else {
            if (mTot <= FP8_BLOCK_SIZE) {
                m = (subBlockIdx_ == 0) ? mTot : 0;
            } else {
                uint32_t mhalf = (mTot + 1) / 2;
                uint32_t mAlign = (mhalf > FP8_BLOCK_SIZE) ? RoundUp(mhalf, FP8_BLOCK_SIZE) : FP8_BLOCK_SIZE; 
                m = (subBlockIdx_ == 0) ? mAlign : (mTot - mAlign);
                mCopyOffset = (subBlockIdx_ == 0) ? 0 : mAlign;  
            }
            mRound = RoundUp(m, FP8_BLOCK_SIZE);
        }

        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }
        
        uint32_t n = actualBlockShape.n();
        uint16_t nRound = RoundUp(n, 16);
        auto paddingSize = (nRound - n) * FP8_BLOCK_SIZE;

        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        constexpr uint32_t QUANT_BLOCK_SIZE = 64;
        if constexpr (quant_mode == 1){
            auto qlayout = gmQSTensorTla.layout();
            auto klayout = gmKSTensorTla.layout();
            uint32_t blockNum = AscendC::CeilDivision(n, blockSizeK); 
            uint32_t subNLoops = AscendC::CeilDivision(blockSizeK, QUANT_BLOCK_SIZE);
            
            auto qOffset = qlayout(tla::MakeCoord(mTot <= QUANT_BLOCK_SIZE ? 0 : subBlockIdx_, static_cast<uint32_t>(0)));
            float qscale = gmQSTensorTla.data().GetValue(qOffset);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for(uint32_t i = 0; i < blockNum; i++){
                uint32_t index = sparseIndex.GetValue(i);
                auto jloops = n - i * blockSizeK <= QUANT_BLOCK_SIZE ? 1 : subNLoops;
                for(uint32_t j = 0; j < jloops; j++) {
                    float ks = gmKSTensorTla.data().GetValue(index * subNLoops + j);
                    auto s = ks * qscale; // scaleValue on fixpipe
                    scaleTensor.SetValue(i*subNLoops + j, static_cast<ElementInput>(s));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        }
        
        ElementInput minValue = -60000.0f;
        if (isFirstKvSTile) {
            AscendC::Duplicate(gmUbTensor, minValue, 256); // max
            AscendC::Duplicate(glUbTensor, 0.0f, 128); // sum
        }

        if (unlikely(n < nRound)) {
            AscendC::Duplicate(lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM + n * FP8_BLOCK_SIZE], minValue, paddingSize);
            if (mRound == 64) {AscendC::Duplicate(lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM + (n + nRound) * FP8_BLOCK_SIZE], minValue, paddingSize);}
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);// alloc P 
        auto lpUbOutTensor = lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].template ReinterpretCast<float8_e4m3_t>();
        DnSoftmaxQuantBlock<float8_e4m3_t, half, FP8_BLOCK_SIZE, true>(lpUbOutTensor, lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM], 
                        gmUbTensor, gmUbTensor, 
                        dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM],
                        glUbTensor, glUbTensor, scaleTensor,
                        tmpTensor, nRound);

        if (mRound == 64) {
            DnSoftmaxQuantBlock<float8_e4m3_t, half, FP8_BLOCK_SIZE, true>(lpUbOutTensor[FP8_BLOCK_SIZE * nRound], lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM + FP8_BLOCK_SIZE * nRound], 
                        gmUbTensor[128], gmUbTensor[128], 
                        dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM + 64],
                        glUbTensor[64], glUbTensor[64], scaleTensor,
                        tmpTensor, nRound);
        }
        if constexpr (quant_mode == 1) { 
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        }
        
        if (unlikely(n < nRound)) {
            AscendC::PipeBarrier<PIPE_V>();
            auto lpPad = lpUbOutTensor[n * FP8_BLOCK_SIZE].template ReinterpretCast<int8_t>();
            AscendC::Duplicate(lpPad, (int8_t)0, paddingSize);
            if (mRound == 64) {AscendC::Duplicate(lpPad[nRound * FP8_BLOCK_SIZE], (int8_t)0, paddingSize);}
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId); // P enque
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId); // P deque

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);

        auto ubPTensorTla = tla::MakeTensor(lpUbOutTensor,  ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla, tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m, nRound, actualBlockShape.m());
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2); //Free P
        SetCrossCoreSync<4, PIPE_MTE3>(mm1ToSmFlag);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
    }

private:
    float scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<uint16_t> lpUbTensor;
    AscendC::LocalTensor<ElementInput> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    AscendC::LocalTensor<ElementInput> scaleTensor;

    AscendC::LocalTensor<uint8_t> tmpTensor;
    uint32_t subBlockIdx_, blockSizeK;

    template <typename T2, typename T, uint16_t m = 32, bool DEQ = true, uint16_t MAX_COLS = 256, uint16_t DEQ_BLK = 64>
    __aicore__ inline void DnSoftmaxQuantBlock(
        const AscendC::LocalTensor<T2>& dstTensor,       
        const AscendC::LocalTensor<T>& srcDnTensor, 
        const AscendC::LocalTensor<T>& maxTensor,     
        const AscendC::LocalTensor<T>& inMaxTensor,       
        const AscendC::LocalTensor<float>& expMaxTensor,
        const AscendC::LocalTensor<float>& expSumTensor,  
        const AscendC::LocalTensor<float>& inSumTensor, 
        const AscendC::LocalTensor<T>& scaleTensor,
        const AscendC::LocalTensor<uint8_t>& tmpTensor, 
        const uint32_t n)
    {
        using namespace AscendC::MicroAPI;
        
        constexpr static CastTrait castTraitFp32Zero = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitFp32One = {
            RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN,
        };

        __ubuf__ T* srcUb = (__ubuf__ T*)srcDnTensor.GetPhyAddr();
        __ubuf__ T2* dstUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();  

        __ubuf__ T* inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();      
        __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();    

        __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr(); 
        __ubuf__ float* inSumUb = (__ubuf__ float*)inSumTensor.GetPhyAddr();
        __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();

        __ubuf__ float* tmpMaxUbStart = (__ubuf__ float*)tmpTensor.GetPhyAddr();
        __ubuf__ T* scaleUb = (__ubuf__ T*) scaleTensor.GetPhyAddr();
        
        T minValue = -65504.0f;
        constexpr uint16_t VL = 256;
        constexpr uint16_t UNROLL = 4;
        constexpr uint16_t VL_ELE_B16 = 128;
        constexpr uint16_t R = VL / sizeof(T) / m; // R = 4

        __VEC_SCOPE__
        {
            RegTensor<T> vregMax0, vregMax1, vregMax2, vregMax3;
            RegTensor<T> vregAcc0, vregAcc1, vregAcc2, vregAcc3; 
            RegTensor<T> vregSrc0, vregSrc1, vregSrc2, vregSrc3;
            RegTensor<T> vregScale;
            RegTensor<int8_t> vregZero0, vregZero1, vregFp8High0, vregFp8High1;
            RegTensor<T2> vregDst0, vregDst1;
            
            RegTensor<T> vregInMax, vregExpMax16;
            RegTensor<T> vregGlobalMax;
            RegTensor<float> vregExpMax32;
            RegTensor<float> vregInExpSum;  
            RegTensor<float> vregExpSum32; 
            RegTensor<T> vregTmp0, vregTmp1;

            RegTensor<T> vregLocalReduce; 

            MaskReg pregAll = CreateMask<T, MaskPattern::ALL>();
            MaskReg pregAll32 = CreateMask<float, MaskPattern::ALL>(); 
            uint32_t tmpM = m;

            MaskReg pregM = UpdateMask<T>(tmpM);
            MaskReg preg64 = CreateMask<T, MaskPattern::VL64>();
            MaskReg preg32 = CreateMask<T, MaskPattern::VL32>();
            MaskReg pregAllB8 = CreateMask<T2, MaskPattern::ALL>();
            MaskReg pregHalfFp8 = CreateMask<T2, MaskPattern::VL128>();

            Duplicate<T, T>(vregMax0, minValue);
            Duplicate<T, T>(vregMax1, minValue);
            Duplicate<T, T>(vregMax2, minValue);
            Duplicate<T, T>(vregMax3, minValue);

            DataCopy<T, LoadDist::DIST_NORM>(vregInMax, inMaxUb); // 1111 2222 3333 ... 32,32,32,32
            DataCopy<float, LoadDist::DIST_NORM>(vregInExpSum, inSumUb); // 11 22 ... 32,32

            
            for (uint16_t k = 0; k < (uint16_t)((n + DEQ_BLK - 1) / DEQ_BLK); ++k) {
                uint16_t n2 = VF_MIN(n - k * DEQ_BLK, DEQ_BLK);
                DataCopy<T, LoadDist::DIST_BRC_B16>(vregScale, scaleUb + k);
                
                for (uint16_t j = 0; j < (uint16_t)(n2 / R / UNROLL) ; j++) {
                    uint32_t offset = (k * DEQ_BLK * m) + (j * VL_ELE_B16 * UNROLL);
                    DataCopy<T, LoadDist::DIST_NORM>(vregSrc0, srcUb + offset + 0);
                    DataCopy<T, LoadDist::DIST_NORM>(vregSrc1, srcUb + offset + VL_ELE_B16);
                    DataCopy<T, LoadDist::DIST_NORM>(vregSrc2, srcUb + offset + VL_ELE_B16 *2);
                    DataCopy<T, LoadDist::DIST_NORM>(vregSrc3, srcUb + offset + VL_ELE_B16 *3);

                    Mul(vregSrc0, vregSrc0, vregScale, pregAll);
                    Mul(vregSrc1, vregSrc1, vregScale, pregAll);
                    Mul(vregSrc2, vregSrc2, vregScale, pregAll);
                    Mul(vregSrc3, vregSrc3, vregScale, pregAll);

                    Max(vregMax0, vregMax0, vregSrc0, pregAll);
                    StoreAlign(srcUb + offset + VL_ELE_B16*0 ,vregSrc0, pregAll);
                    Max(vregMax1, vregMax1, vregSrc1, pregAll);
                    StoreAlign(srcUb + offset + VL_ELE_B16*1 ,vregSrc1, pregAll);
                    Max(vregMax2, vregMax2, vregSrc2, pregAll);
                    StoreAlign(srcUb + offset + VL_ELE_B16*2 ,vregSrc2, pregAll);
                    Max(vregMax3, vregMax3, vregSrc3, pregAll);
                    StoreAlign(srcUb + offset + VL_ELE_B16*3 ,vregSrc3, pregAll);
                }
            }
            Max(vregMax0, vregMax0, vregMax1, pregAll);
            Max(vregMax2, vregMax2, vregMax3, pregAll);
            Max(vregMax0, vregMax0, vregMax2, pregAll);
            
            Interleave(vregTmp0, vregTmp1, vregMax0, vregMax0);
            Max(vregMax1, vregTmp0, vregTmp1, pregAll); // 11 22, ...., 64,64
            Interleave(vregTmp0, vregTmp1, vregMax1, vregMax1);
            Max(vregLocalReduce, vregTmp0, vregTmp1, pregAll); // 1111 2222 3333 ... 32,32,32,32
            
            // vregLocalReduce is Local max
            Max(vregGlobalMax, vregLocalReduce, vregInMax, pregAll);
            Sub(vregExpMax16, vregInMax, vregGlobalMax, pregAll);
            Exp(vregExpMax16, vregExpMax16, pregAll);
            Cast<float, T, castTraitFp32Zero>(vregExpMax32, vregExpMax16, pregAll); // 11 22 33 .... 32,32

            DeInterleave(vregTmp0, vregTmp1, vregGlobalMax, vregGlobalMax);
            DeInterleave(vregInMax, vregTmp1, vregTmp0, vregTmp0);

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            Duplicate<T, T>(vregAcc0, (T)0.0f);
            Duplicate<T, T>(vregAcc1, (T)0.0f);
            Duplicate<T, T>(vregAcc2, (T)0.0f);
            Duplicate<T, T>(vregAcc3, (T)0.0f);

            for (uint16_t i = 0; i < (uint16_t)(n >> 4); ++i) {
                uint32_t offset = i << 9; // 128 * 4
                DataCopy<T, LoadDist::DIST_NORM>(vregSrc0, srcUb + offset + 0);
                DataCopy<T, LoadDist::DIST_NORM>(vregSrc1, srcUb + offset + VL_ELE_B16);
                DataCopy<T, LoadDist::DIST_NORM>(vregSrc2, srcUb + offset + VL_ELE_B16 *2);
                DataCopy<T, LoadDist::DIST_NORM>(vregSrc3, srcUb + offset + VL_ELE_B16 *3);

                Sub(vregSrc0, vregSrc0, vregInMax, pregAll);
                Sub(vregSrc1, vregSrc1, vregInMax, pregAll);
                Sub(vregSrc2, vregSrc2, vregInMax, pregAll);
                Sub(vregSrc3, vregSrc3, vregInMax, pregAll);
                
                Exp(vregSrc0, vregSrc0, pregAll);
                Exp(vregSrc1, vregSrc1, pregAll);
                Exp(vregSrc2, vregSrc2, pregAll);
                Exp(vregSrc3, vregSrc3, pregAll);

                Add(vregAcc0, vregAcc0, vregSrc0, pregAll);
                Add(vregAcc1, vregAcc1, vregSrc1, pregAll);
                Add(vregAcc2, vregAcc2, vregSrc2, pregAll);
                Add(vregAcc3, vregAcc3, vregSrc3, pregAll);

                Muls(vregSrc0, vregSrc0, (T)448.0f, pregAll);
                Muls(vregSrc1, vregSrc1, (T)448.0f, pregAll);
                Muls(vregSrc2, vregSrc2, (T)448.0f, pregAll);
                Muls(vregSrc3, vregSrc3, (T)448.0f, pregAll);

                // rna
                Maxs((RegTensor<int16_t>&) vregSrc0, (RegTensor<int16_t>&) vregSrc0, (int16_t)(8128+128), pregAll);
                Maxs((RegTensor<int16_t>&) vregSrc1, (RegTensor<int16_t>&) vregSrc1, (int16_t)(8128+128), pregAll);
                Maxs((RegTensor<int16_t>&) vregSrc2, (RegTensor<int16_t>&) vregSrc2, (int16_t)(8128+128), pregAll);
                Maxs((RegTensor<int16_t>&) vregSrc3, (RegTensor<int16_t>&) vregSrc3, (int16_t)(8128+128), pregAll);

                Adds((RegTensor<int16_t>&) vregSrc0, (RegTensor<int16_t>&) vregSrc0, (int16_t)(-8128), pregAll);
                Adds((RegTensor<int16_t>&) vregSrc1, (RegTensor<int16_t>&) vregSrc1, (int16_t)(-8128), pregAll);
                Adds((RegTensor<int16_t>&) vregSrc2, (RegTensor<int16_t>&) vregSrc2, (int16_t)(-8128), pregAll);
                Adds((RegTensor<int16_t>&) vregSrc3, (RegTensor<int16_t>&) vregSrc3, (int16_t)(-8128), pregAll);

                ShiftRights((RegTensor<int16_t>&) vregSrc0, (RegTensor<int16_t>&) vregSrc0, (int16_t)7, pregAll);
                ShiftRights((RegTensor<int16_t>&) vregSrc1, (RegTensor<int16_t>&) vregSrc1, (int16_t)7, pregAll);
                ShiftRights((RegTensor<int16_t>&) vregSrc2, (RegTensor<int16_t>&) vregSrc2, (int16_t)7, pregAll);
                ShiftRights((RegTensor<int16_t>&) vregSrc3, (RegTensor<int16_t>&) vregSrc3, (int16_t)7, pregAll);

                StoreAlign<T2, StoreDist::DIST_PACK_B16>(dstUb + offset + VL_ELE_B16 * 0, (RegTensor<T2>&) vregSrc0, pregAll);
                StoreAlign<T2, StoreDist::DIST_PACK_B16>(dstUb + offset + VL_ELE_B16 * 1, (RegTensor<T2>&) vregSrc1, pregAll);
                StoreAlign<T2, StoreDist::DIST_PACK_B16>(dstUb + offset + VL_ELE_B16 * 2, (RegTensor<T2>&) vregSrc2, pregAll);
                StoreAlign<T2, StoreDist::DIST_PACK_B16>(dstUb + offset + VL_ELE_B16 * 3, (RegTensor<T2>&) vregSrc3, pregAll); // B16 mask reg need
            }
            Add(vregAcc0, vregAcc0, vregAcc1, pregAll);
            Add(vregAcc2, vregAcc2, vregAcc3, pregAll);
            Add(vregAcc0, vregAcc0, vregAcc2, pregAll);

            Interleave(vregTmp0, vregTmp1, vregAcc0, vregAcc0);
            Add(vregAcc1, vregTmp0, vregTmp1, pregAll);
            Interleave(vregTmp0, vregTmp1, vregAcc1, vregAcc1);
            Add(vregLocalReduce, vregTmp0, vregTmp1, pregAll); // 1111 2222 .... 32,32,32,32
            Cast<float, T, castTraitFp32Zero>(vregExpSum32, vregLocalReduce, pregAll); // 11 22 ... 32,32

            // x_sum = sum(exp_max * in_sum + x_sum)
            Mul<float, MaskMergeMode::ZEROING>(vregInExpSum, vregExpMax32, vregInExpSum, pregAll32);
            Add<float, MaskMergeMode::ZEROING>(vregInExpSum, vregInExpSum, vregExpSum32, pregAll32);

            DataCopy<float, StoreDist::DIST_NORM>(expSumUb, vregInExpSum, pregAll32);
            DataCopy<float, StoreDist::DIST_NORM>(expMaxUb, vregExpMax32, pregAll32);
            DataCopy<T, StoreDist::DIST_NORM>(maxUb, vregGlobalMax, pregAll);
        }
    }
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_BSA_LOW_PREC_HPP
