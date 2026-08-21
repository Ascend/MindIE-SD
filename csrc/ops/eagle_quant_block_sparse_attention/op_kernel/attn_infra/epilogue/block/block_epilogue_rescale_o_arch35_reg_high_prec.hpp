/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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

template <
    class ElementO_,
    class ElementOTmp_,
    class ElementS_,
    class TileCopy_,
    class OTmpSrcPos_>
class BlockEpilogue<
    EpilogueAtlasA5BsaRescaleO,
    ElementO_,
    ElementOTmp_,
    ElementS_,
    TileCopy_,
    OTmpSrcPos_>
{
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using SMDtype = ElementS_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    static constexpr uint32_t UB_OTMP_BUF_STAGES = 2;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64 * 2;  //! 2* for 11 22 ... 32,32
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t RESCALE_VREG_SIZE = 256 / sizeof(ElementOTmp);
    static constexpr bool DEQ = true;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, uint32_t embed_ = 128) 
    {
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 4 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 128 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 128 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 128 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  128 * sizeof(float);

        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                LO_UB_TENSOR_OFFSET + i * UB_UINT8_BLOCK_SIZE);
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(GO_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(GO_UB_TENSOR_OFFSET);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        scaleTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET + 4096 * 2 + embed_ * sizeof(float));
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
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

    template <class TensorDst>
    __aicore__ inline
    void SubCoreCompute(const TensorDst &gOTensor,
                        uint32_t curTileMod,
                        uint32_t ubOTmpBufId,
                        bool isFirstKvSTile,
                        bool isLastKvSTile,
                        uint32_t colNumOri,
                        Arch::CrossCoreFlag mm2ToReFlag)
    {
        __ubuf__ ElementOTmp *goUb = (__ubuf__ ElementOTmp *) goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *loUb = (__ubuf__ ElementOTmp *) loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp *glUb = ( __ubuf__ ElementOTmp *) glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp *dmUb = (__ubuf__ ElementOTmp *) dmUbTensor32[curTileMod * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementOTmp *scaleUb = (__ubuf__ ElementOTmp *)scaleTensor.GetPhyAddr();
        
        WaitCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);

        if (isFirstKvSTile) {
            if (!isLastKvSTile) {
                if constexpr (DEQ) {
                    DeqLocalO<ElementOTmp, 128, DEQ>(goUb, loUb, scaleUb, rowNumCurSubCore); AscendC::PipeBarrier<PIPE_V>();
                } else {
                    AscendC::DataCopy(goUbTensor32, loUbTensor[ubOTmpBufId], rowNumCurSubCore * colNumOri); AscendC::PipeBarrier<PIPE_V>();
                }
            } else { // go = lo div sum
                DivFuncLastAndFirst<ElementOTmp, 128, DEQ>(goUb, loUb, glUb, scaleUb, rowNumCurSubCore); 
            }
        } else if (!isLastKvSTile) {
            RescaleFunc<ElementOTmp, 128, DEQ>(goUb, loUb, dmUb, scaleUb, rowNumCurSubCore);  
        } else {
            RescaleFuncLastNotFirst<ElementOTmp, 128, DEQ>(goUb, loUb, dmUb, glUb, scaleUb, rowNumCurSubCore);
        }
        
        // release lo buf
        SetCrossCoreSync<4, PIPE_V>(mm2ToReFlag);
        if (isLastKvSTile) {
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_RINT,
                    rowNumCurSubCore * colNumOri);
            } else {
                AscendC::Cast(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_NONE,
                    rowNumCurSubCore * colNumOri);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(gOTensor[rowOffsetCurSubCore * colNumOri], goUbTensor16, rowNumCurSubCore * colNumOri);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }
    
    template <typename T=float, uint32_t colStride=128, bool DEQ=false>
    __simd_vf__ inline void RescaleFunc(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *dmUb, __ubuf__ T *scaleUb, 
                                        uint32_t row)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> dmVreg0, dmVreg1, scaleVreg0, scaleVreg1;
        RegTensor<float> goPreVreg0, goPreVreg0_2, goPreVreg1, goPreVreg1_2;
        RegTensor<float> loVreg0, loVreg0_2, loVreg1, loVreg1_2;
        RegTensor<float> mulVreg0, mulVreg0_2, mulVreg1, mulVreg1_2;
        RegTensor<float> goCurVreg0, goCurVreg0_2, goCurVreg1, goCurVreg1_2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t vlElemNum = 64;
        
        uint32_t halfRow = (row + 1) / 2; // = 32
        
        // 循环 32 次，每次处理 row_i 和 row_{i+32}
        if constexpr (DEQ) {
            LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg0, scaleUb); // col0
            LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg1, scaleUb + vlElemNum); //col1
        }

        for (uint16_t i = 0; i < halfRow; i++) {
            uint32_t row1_idx = i + halfRow;

            uint32_t baseOffset0 = i * colStride;
            uint32_t secondOffset0 = baseOffset0 + vlElemNum;
            uint32_t baseOffset1 = row1_idx * colStride;
            uint32_t secondOffset1 = baseOffset1 + vlElemNum;

            // Load scalars and broadcast
            //! 2* for 11 22 ... 32,32
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg0, dmUb + 2*i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg1, dmUb + 2*row1_idx);

            // Load Row i
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg0,   goUb + baseOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg0_2, goUb + secondOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg0,      loUb + baseOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg0_2,    loUb + secondOffset0);

            // Load Row i+32
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg1,   goUb + baseOffset1);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg1_2, goUb + secondOffset1);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg1,      loUb + baseOffset1);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg1_2,    loUb + secondOffset1);

            // Multiply (4-issue)
            Mul(mulVreg0,   goPreVreg0,   dmVreg0, pregFull);
            Mul(mulVreg0_2, goPreVreg0_2, dmVreg0, pregFull);
            Mul(mulVreg1,   goPreVreg1,   dmVreg1, pregFull);
            Mul(mulVreg1_2, goPreVreg1_2, dmVreg1, pregFull);

            if constexpr (DEQ) {
                MulDstAdd(loVreg0,   scaleVreg0, mulVreg0,   pregFull);
                MulDstAdd(loVreg0_2, scaleVreg1, mulVreg0_2, pregFull);
                MulDstAdd(loVreg1,   scaleVreg0, mulVreg1,   pregFull);
                MulDstAdd(loVreg1_2, scaleVreg1, mulVreg1_2, pregFull);
            } else {
                Add(loVreg0,   mulVreg0,   loVreg0,   pregFull); 
                Add(loVreg0_2, mulVreg0_2, loVreg0_2, pregFull);
                Add(loVreg1,   mulVreg1,   loVreg1,   pregFull); 
                Add(loVreg1_2, mulVreg1_2, loVreg1_2, pregFull);
            }

            // Store (4-issue)
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + baseOffset0,   loVreg0,   pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + secondOffset0, loVreg0_2, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + baseOffset1,   loVreg1,   pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + secondOffset1, loVreg1_2, pregFull);
        }
    }

    template <typename T=float, uint32_t colStride=128, bool DEQ=false>
    __simd_vf__ inline void RescaleFuncLastNotFirst(__ubuf__ T *goUb, __ubuf__ T *loUb,
                                                    __ubuf__ T *dmUb, __ubuf__ T *glUb,
                                                    __ubuf__ T *scaleUb, 
                                                    uint32_t row)
    {
        using namespace AscendC::MicroAPI;
        
        RegTensor<float> dmVreg0, glVreg0, dmVreg1, glVreg1;
        RegTensor<float> scaleVreg0, scaleVreg1; // 增加 scale 寄存器
        RegTensor<float> goPreVreg0, goPreVreg0_2, goPreVreg1, goPreVreg1_2;
        RegTensor<float> loVreg0, loVreg0_2, loVreg1, loVreg1_2;
        RegTensor<float> mulVreg0, mulVreg0_2, mulVreg1, mulVreg1_2;
        RegTensor<float> goCurVreg0, goCurVreg0_2, goCurVreg1, goCurVreg1_2;
        RegTensor<float> divVreg0, divVreg0_2, divVreg1, divVreg1_2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t vlElemNum = 64;
        
        uint32_t halfRow = (row + 1) / 2; // = 32

        // 若开启反量化，在循环外一次性加载 per-channel scale
        if constexpr (DEQ) {
            LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg0, scaleUb);               // col0
            LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg1, scaleUb + vlElemNum);   // col1
        }
        
        for (uint16_t i = 0; i < halfRow; i++) {
            uint32_t row1_idx = i + halfRow;

            uint32_t baseOffset0 = i * colStride;
            uint32_t secondOffset0 = baseOffset0 + vlElemNum;
            uint32_t baseOffset1 = row1_idx * colStride;
            uint32_t secondOffset1 = baseOffset1 + vlElemNum;

            // Load scalars (i and i+32)
            //! 2* for 11 22 ... 32,32
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg0, dmUb + 2*i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg0, glUb + 2*i);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(dmVreg1, dmUb + 2*row1_idx);
            LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg1, glUb + 2*row1_idx);

            // Load Data
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg0,   goUb + baseOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg0_2, goUb + secondOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg1,   goUb + baseOffset1);
            LoadAlign<T, LoadDist::DIST_NORM>(goPreVreg1_2, goUb + secondOffset1);

            LoadAlign<T, LoadDist::DIST_NORM>(loVreg0,      loUb + baseOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg0_2,    loUb + secondOffset0);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg1,      loUb + baseOffset1);
            LoadAlign<T, LoadDist::DIST_NORM>(loVreg1_2,    loUb + secondOffset1);

            // Muls: goPre * dm
            Mul(mulVreg0,   goPreVreg0,   dmVreg0, pregFull);
            Mul(mulVreg0_2, goPreVreg0_2, dmVreg0, pregFull);
            Mul(mulVreg1,   goPreVreg1,   dmVreg1, pregFull);
            Mul(mulVreg1_2, goPreVreg1_2, dmVreg1, pregFull);

            // Dequantization: Local * scale + mulvreg
            // dstReg与srcReg0相乘后与srcReg1相加
            if constexpr (DEQ) {
                MulDstAdd(loVreg0,   scaleVreg0, mulVreg0,   pregFull);
                MulDstAdd(loVreg0_2, scaleVreg1, mulVreg0_2, pregFull);
                MulDstAdd(loVreg1,   scaleVreg0, mulVreg1,   pregFull);
                MulDstAdd(loVreg1_2, scaleVreg1, mulVreg1_2, pregFull);
            } else {
                Add(loVreg0,   mulVreg0,   loVreg0,   pregFull);
                Add(loVreg0_2, mulVreg0_2, loVreg0_2, pregFull);
                Add(loVreg1,   mulVreg1,   loVreg1,   pregFull);
                Add(loVreg1_2, mulVreg1_2, loVreg1_2, pregFull);
            }

            Div(divVreg0,   loVreg0,   glVreg0, pregFull);
            Div(divVreg0_2, loVreg0_2, glVreg0, pregFull);
            Div(divVreg1,   loVreg1,   glVreg1, pregFull);
            Div(divVreg1_2, loVreg1_2, glVreg1, pregFull);

            // Stores
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + baseOffset0,   divVreg0,   pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + secondOffset0, divVreg0_2, pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + baseOffset1,   divVreg1,   pregFull);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + secondOffset1, divVreg1_2, pregFull);
        }
    }

    template <typename T=float, uint32_t colStride=128, bool DEQ=false>
    __simd_vf__ inline void DivFuncLastAndFirst(__ubuf__ T *goUb, __ubuf__ T *loUb, __ubuf__ T *glUb, 
                                                __ubuf__ T *scaleUb, // 保持接口一致性
                                                uint32_t row)
    {
        using namespace AscendC::MicroAPI;
        
        RegTensor<float> goCurVreg0, goCurVreg0_2, goCurVreg1, goCurVreg1_2;
        RegTensor<float> glVreg0, glVreg1;
        RegTensor<float> scaleVreg0, scaleVreg1; // 仅做声明备用
        RegTensor<float> divVreg0, divVreg0_2, divVreg1, divVreg1_2;
        
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t vlElemNum = 64;
        
        if constexpr (colStride==128) {
            uint32_t halfRow = (row + 1) / 2; // = 32
            
            if constexpr (DEQ) {
                LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg0, scaleUb);
                LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg1, scaleUb + vlElemNum);
            }
            for (uint16_t i = 0; i < halfRow; i++) {
                uint32_t row1_idx = i + halfRow;

                uint32_t baseOffset0 = i * colStride;
                uint32_t secondOffset0 = baseOffset0 + vlElemNum;
                uint32_t baseOffset1 = row1_idx * colStride;
                uint32_t secondOffset1 = baseOffset1 + vlElemNum;
                //! 2* for 11 22 ... 32,32
                LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg0, glUb + 2*i);
                LoadAlign<T, LoadDist::DIST_BRC_B32>(glVreg1, glUb + 2*row1_idx);

                LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg0,   loUb + baseOffset0);
                LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg0_2, loUb + secondOffset0);
                LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg1,   loUb + baseOffset1);
                LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg1_2, loUb + secondOffset1);

                if constexpr (DEQ) {
                    Mul(goCurVreg0,   goCurVreg0,   scaleVreg0, pregFull);
                    Mul(goCurVreg0_2, goCurVreg0_2, scaleVreg1, pregFull);
                    Mul(goCurVreg1,   goCurVreg1,   scaleVreg0, pregFull);
                    Mul(goCurVreg1_2, goCurVreg1_2, scaleVreg1, pregFull);
                }

                Div(divVreg0,   goCurVreg0,   glVreg0, pregFull);
                Div(divVreg0_2, goCurVreg0_2, glVreg0, pregFull);
                Div(divVreg1,   goCurVreg1,   glVreg1, pregFull);
                Div(divVreg1_2, goCurVreg1_2, glVreg1, pregFull);

                StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + baseOffset0,   divVreg0,   pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + secondOffset0, divVreg0_2, pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + baseOffset1,   divVreg1,   pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + secondOffset1, divVreg1_2, pregFull);
            }
        }
    }

    template <typename T=float, uint32_t colStride=128, bool DEQ=false>
    __simd_vf__ inline void DeqLocalO(__ubuf__ T *goUb, __ubuf__ T *loUb,  
                                                __ubuf__ T *scaleUb,  
                                                uint32_t row)
    {
        using namespace AscendC::MicroAPI;
        
        RegTensor<float> goCurVreg0, goCurVreg0_2, goCurVreg1, goCurVreg1_2;
        RegTensor<float> scaleVreg0, scaleVreg1;  
        
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t vlElemNum = 64;
        
        if constexpr (colStride==128) {
            uint32_t halfRow = (row + 1) / 2; // = 32
            
            if constexpr (DEQ) {
                LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg0, scaleUb);
                LoadAlign<T, LoadDist::DIST_NORM>(scaleVreg1, scaleUb + vlElemNum);
            }
            for (uint16_t i = 0; i < halfRow; i++) {
                uint32_t row1_idx = i + halfRow;

                uint32_t baseOffset0 = i * colStride;
                uint32_t secondOffset0 = baseOffset0 + vlElemNum;
                uint32_t baseOffset1 = row1_idx * colStride;
                uint32_t secondOffset1 = baseOffset1 + vlElemNum;

                LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg0,   loUb + baseOffset0);
                LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg0_2, loUb + secondOffset0);
                LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg1,   loUb + baseOffset1);
                LoadAlign<T, LoadDist::DIST_NORM>(goCurVreg1_2, loUb + secondOffset1);

                if constexpr (DEQ) {
                    Mul(goCurVreg0,   goCurVreg0,   scaleVreg0, pregFull);
                    Mul(goCurVreg0_2, goCurVreg0_2, scaleVreg1, pregFull);
                    Mul(goCurVreg1,   goCurVreg1,   scaleVreg0, pregFull);
                    Mul(goCurVreg1_2, goCurVreg1_2, scaleVreg1, pregFull);
                }

                StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + baseOffset0,   goCurVreg0,   pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + secondOffset0, goCurVreg0_2, pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + baseOffset1,   goCurVreg1,   pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(goUb + secondOffset1, goCurVreg1_2, pregFull);
            }
        }
    }

    template <class TensorDst>
    __aicore__ inline
    void operator()(const TensorDst &gOTensor,
                    GemmCoord actualOriShape,
                    uint32_t curTileMod,
                    uint32_t gatheredKvSTileIdx,
                    bool isFirstKvSTile,
                    bool isLastKvSTile,
                    Arch::CrossCoreFlag mm2ToReFlag)
    {
        uint32_t rowNumOri = actualOriShape[0];
        uint32_t colNumOri = actualOriShape[1];
        constexpr uint32_t FP8_BLOCK_SIZE = 32;
        if (rowNumOri <= FP8_BLOCK_SIZE) {
            rowNumCurSubCore = (subBlockIdx == 0) ? rowNumOri : 0;
        } else {
            uint32_t mhalf = (rowNumOri + subBlockNum -1) / subBlockNum;
            uint32_t mAlign = (mhalf > FP8_BLOCK_SIZE) ? RoundUp(mhalf, FP8_BLOCK_SIZE) : FP8_BLOCK_SIZE; 
            rowNumCurSubCore = (subBlockIdx == 0) ? mAlign : (rowNumOri - mAlign);
        }
        rowOffsetCurSubCore = subBlockIdx == 0 ? 0: rowNumOri - rowNumCurSubCore;
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;

        if (rowNumCurSubCore > 0) {
            SubCoreCompute(
                gOTensor,
                curTileMod,
                ubOTmpBufId,
                isFirstKvSTile,
                isLastKvSTile,
                colNumOri,
                mm2ToReFlag);
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
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
};
}
#endif