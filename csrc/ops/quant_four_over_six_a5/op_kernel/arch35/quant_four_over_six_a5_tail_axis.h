/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_tail_axis.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_TAIL_AXIS_H
#define DYNAMIC_MX_QUANT_TAIL_AXIS_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "kernel_operator.h"
#include "quant_four_over_six_a5_common.h"
// #define MX_DBG_A46   // uncomment to dump ComputeAdaptive46 Σe²/scale via PRINTF (see docs/kernel_debug_dump_print.md)
// dst_type_max=4 adaptive — single-pass fusion: dual-candidate Se^2 + Select y, bf16 only.
#define MX_FUSE_A46_Y
namespace QuantFourOverSixA5 {
using namespace AscendC;

template <typename T, typename U, int64_t SCALE_ALG>
class QuantFourOverSixA5TailAxis {
public:
    __aicore__ inline QuantFourOverSixA5TailAxis()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const QuantFourOverSixA5TailAxisTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const QuantFourOverSixA5TailAxisTilingData* tilingData);
    __aicore__ inline void GetGmParams();
    __aicore__ inline void GetUbParams();

    __aicore__ inline void CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum);

    __aicore__ inline void ComputeMaxExp(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF);
    __aicore__ inline void ComputeMaxExpDynamicDtypeRange(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF);
    __aicore__ inline void ComputeMaxExpcuBLAS(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF);
    __aicore__ inline void ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB);
    __aicore__ inline void ComputeScaleDynamicDtypeRange(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB);
    __aicore__ inline void ComputeScalecuBLASFP4(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNumHalfVF, uint32_t totalScaleInUB);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeData(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeDataOptimize(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF);
    __aicore__ inline void FP16Convert(Reg::RegTensor<half>& output, Reg::RegTensor<half>& input, Reg::MaskReg& mask);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg);
    // dst_type_max=4 触发的 per-block 4/6 自适应量化（var(x-dequant) 判据）。实现见下方定义，
    // reduce 部分需编译环境验证 Reg::ReduceSumWithDataBlock。
    __aicore__ inline void ComputeAdaptive46(const LocalTensor<T>& xLocal, const LocalTensor<uint16_t>& maxExpLocal,
        __ubuf__ uint16_t* mxScaleLocalAddr,
        __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF, uint16_t LoopNum1VF,
        uint32_t totalScaleInUB);
    // 4/6 自适应 helper（dst_type_max=4 专用）
    // 由 maxExp(|x| 含尾数)与 addValueBit 派生 E8M0 scale + recipScale,复用 DynamicDtypeRange 位运算
    // （比 cuBLAS 浮点快）。D=6→addValue=0x3f, D=4→addValue=0x7f。
    __aicore__ inline void ComputeScaleDdr46(__ubuf__ uint16_t* maxExpAddr,
        __ubuf__ uint16_t* scale46Out, __ubuf__ uint16_t* recip6Out, __ubuf__ uint16_t* recip4Out,
        uint16_t LoopNum1VF, uint32_t totalScaleInUB);
    // ComputeCandStore46 (bf16 only): 单 pass 同时算 D6/D4 候选 + Σe², 选 winner scale → mask relay。bf16 用 qU(硬件 Cast pack)。
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeCandStore46(
        __ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleAddr, __ubuf__ uint16_t* recip4Addr,
        __ubuf__ uint16_t* y46Addr, __ubuf__ half* seTmpAddr,
        uint16_t LoopNum2VF);

    // QuantVarCand46: 单候选(4 或 6)量化(v→qU, DeInterleave 还原 Cast<U> 打包序, qU 保留供 ③ y Select)
    //   + Σe²(half reduce)。D6/D4 同构仅 recip 不同 → 抽此函数复用。bf16 路径专用。
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void QuantVarCand46(
        Reg::RegTensor<uint16_t>& recip, Reg::RegTensor<T>& vx0, Reg::RegTensor<T>& vx1,
        Reg::RegTensor<U>& qU0, Reg::RegTensor<U>& qU1, Reg::RegTensor<half>& ssOut,
        Reg::MaskReg pregAll16);

    // StoreScaleSelect: DINTLV_B16 load scale46+seTmp → MSE(scale²) → Compare → PACK_B16 mxScale + mask
    __aicore__ inline void StoreScaleSelect(
        __ubuf__ uint16_t* scale46Addr, __ubuf__ half* seTmpAddr,
        __ubuf__ uint16_t* mxScaleAddr, __ubuf__ uint16_t* maskOutAddr,
        uint16_t LoopNum1VF, uint32_t totalScaleInUB);
    // SelectY46: E2B_B16 mask + DINTLV_B16 y46 → Select y + PACK_B16 output
    __aicore__ inline void SelectY46(
        __ubuf__ uint16_t* maskAddr, __ubuf__ uint16_t* y46Addr,
        __ubuf__ uint16_t* yOutAddr, uint16_t LoopNum2VF, uint32_t totalScaleInUB);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<uint8_t> yGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    GlobalTensor<uint8_t> scaleGm_;

    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> recipScaleBuffer_;
    // dst_type_max=4 自适应(4/6)专用 buffer
    //   recipScaleBuffer_: D4 复用为 recip6→winner→mask(前半), 不额外分配
    //   recip4Buf_:        D4 候选 recip4, E2B_B16 广播
    //   scale46Buf_:       scale6∥scale4 DIST_INTLV_B16 (uint16)
    //   seTmp_:            ss6∥ss4 half DIST_INTLV_B16 (替代原 seTmp_)
    //   y46Buf_:           y6∥y4, 2FP4/uint16 (替代原 y46Buf_+y46Buf_)
    TBuf<QuePosition::VECCALC> recip4Buf_;        // D4 only: recip4 (E2B_B16)
    TBuf<QuePosition::VECCALC> scale46Buf_;       // D4 only: scale6∥scale4 (INTLV_B16)
    TBuf<QuePosition::VECCALC> seTmp_;            // D4 only: ss6∥ss4 half (INTLV_B16)
    TBuf<QuePosition::VECCALC> y46Buf_;          // D4 only: y6|y4, 2FP4/uint16, 64B/block
    int64_t roundMode_ = 0;
    int64_t blockSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;

    int64_t rowTileNum_ = 0;                // row 方向上的切核数
    int64_t colTileNum_ = 0;                // col 方向上的切核数
    int64_t rowNum_ = 1;                    // 合轴之后 -2 轴大小
    int64_t colNum_ = 1;                    // 合轴之后 -1 轴大小
    int64_t colNormalBlockNum_ = 0;         // 列方向头核处理的块数 (1 x 256)
    int64_t colTailLen_ = 0;                // 列方向尾块长度 
    int64_t rowNormalBlockNum_ = 0;         // 行方向头核处理的块数 (1 行)
    int64_t rowTailLen_ = 0;                // 行方向尾块长度
    int64_t maxUbBlockNum_ = 0;             // UB最大能放下的处理块数 (1 x 32) (8 的倍数)
    float dstTypeMax_ = 0.0;
    float invDstTypeMax_ = 0.0;

    // GM Params
    int64_t coreIdx_ = 0;                       // Core ID
    int64_t coreColIdx_ = 0;                    // Core 处理的 GM 数据块列方向的核ID数
    int64_t coreRowIdx_ = 0;                    // Core 处理的 GM 数据块行方向的核ID数
    int64_t xGmOffset_ = 0;                     // Core 处理的数据块在 GM 的起始地址偏移
    int64_t scaleGmOffet_ = 0;                  // Core 处理的数据块得到的Scale在 GM 的地址偏移
    int64_t scaleColNum_ = 0;                   // 合轴后数据块在尾轴方向上的 scale 数量大小（偶数对齐）

    // UB Params
    int64_t ubFactorColNum_ = 0;                // Core 处理的 GM 数据块列方向元素个数
    int64_t ubFactorCol32BlockNum_ = 0;         // Core 处理的 GM 数据块列方向 32 长度块个数
    int64_t ubFactorRow1BlockNum_ = 0;          // Core 处理的 GM 数据块行方向 1 长度块个数
    int64_t ubFactorColLoopNum_ = 0;            // Core 处理的 GM 数据块列方向循环次数
    int64_t ubFactorRowLoopNum_ = 0;            // Core 处理的 GM 数据块行方向循环次数
    int64_t ubFactorColNormalBlockNum_ = 0;     // Core 处理的 GM 数据块搬入 UB 列方向 Normal 32 长度数据块
    int64_t ubFactorColTailBlockNum_ = 0;       // Core 处理的 GM 数据块搬入 UB 列方向 Tail 32 长度数据块
    int64_t ubFactorColNormalBlockLen_ = 0;     // Core 处理的 GM 数据块搬入 UB 列方向 Normal 数据块元素个数
    int64_t ubFactorColTailBlockLen_ = 0;       // Core 处理的 GM 数据块搬入 UB 列方向 Tail 数据块元素个数
    int64_t ubFactorRowNormalBlockNum_ = 0;     // Core 处理的 GM 数据块搬入 UB 行方向 Normal 1 长度数据块
    int64_t ubFactorRowTailBlockNum_ = 0;       // Core 处理的 GM 数据块搬入 UB 行方向 Tail 1 长度数据块

    uint16_t f4Emax_ = 0;
    uint16_t addValueBit = 0;
};

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const QuantFourOverSixA5TailAxisTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);                // 获取TilingData数据

    GetGmParams();                              // 计算核间GM地址偏移

    GetUbParams();                              // 计算核内切分参数

    int64_t queueDepth = DB_BUFFER;
    pipe_.InitBuffer(inQueue_, queueDepth, maxUbBlockNum_ * blockSize_ * sizeof(T));
    pipe_.InitBuffer(outQueue_, queueDepth, maxUbBlockNum_ * blockSize_ * sizeof(uint8_t) / DIGIT_TWO);
    pipe_.InitBuffer(mxScaleQueue_, queueDepth, maxUbBlockNum_ * sizeof(uint8_t));
    pipe_.InitBuffer(maxExpBuffer_, maxUbBlockNum_ * sizeof(uint16_t));
    pipe_.InitBuffer(recipScaleBuffer_, maxUbBlockNum_ * sizeof(uint16_t));
    // dst_type_max=4 自适应(4/6)buffer: 仅 dstTypeMax==4 时分配
    // (其他入参 6/7 不启用这些 buffer; tiling 也仅 dstTypeMax==4 时计入 UB 开销)
    if (dstTypeMax_ == DIGIT_FOUR_FLOAT) {
        pipe_.InitBuffer(recip4Buf_,   maxUbBlockNum_ * sizeof(uint16_t));     // recip4
        pipe_.InitBuffer(scale46Buf_, 2 * maxUbBlockNum_ * sizeof(uint16_t));  // scale6∥scale4
        pipe_.InitBuffer(seTmp_,      2 * maxUbBlockNum_ * sizeof(half));      // ss6∥ss4
        pipe_.InitBuffer(y46Buf_, 2 * maxUbBlockNum_ * blockSize_ * sizeof(uint8_t));        // y6|y4, 64B/block
    }

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset_);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xGmOffset_ / DIGIT_TWO);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + scaleGmOffet_);

    if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
        f4Emax_ = FP4_E2M1_BF16_MAX_EXP;
    } else {
        f4Emax_ = FP4_E1M2_MAX_EXP;
    }

    if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT) {
        addValueBit = ADD_VALUE_FOR_BF16_MAN1;
    } else if (dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
        addValueBit = ADD_VALUE_FOR_BF16_MAN2;
    }
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::GetGmParams(){
    coreIdx_ = GetBlockIdx();
    coreColIdx_ = coreIdx_ % colTileNum_;
    coreRowIdx_ = coreIdx_ / colTileNum_;
    xGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * colNum_ + coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT * blockSize_;
    scaleColNum_ = ops::CeilDiv(ops::CeilDiv(colNum_, blockSize_), DIGIT_TWO) * DIGIT_TWO;
    scaleGmOffet_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * scaleColNum_ + coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::GetUbParams(){
    if (coreColIdx_ == colTileNum_ - 1) {
        ubFactorCol32BlockNum_ =  ops::CeilDiv(colTailLen_, blockSize_);
        ubFactorColNum_ = colTailLen_;
    } else {
        ubFactorCol32BlockNum_ =  colNormalBlockNum_ * DIGIT_EIGHT;
        ubFactorColNum_ = ubFactorCol32BlockNum_ * blockSize_;
    }

    if (coreRowIdx_ == rowTileNum_ - 1) {
        ubFactorRow1BlockNum_ = ops::CeilDiv(rowTailLen_, DIGIT_ONE);
    } else {
        ubFactorRow1BlockNum_ = rowNormalBlockNum_ * DIGIT_ONE;
    }

    ubFactorColLoopNum_ = ops::CeilDiv(ubFactorCol32BlockNum_, maxUbBlockNum_);                                             // 列方向上UB循环次数（列方向上的切 UB 块数）
    ubFactorColNormalBlockNum_ = ops::CeilDiv(ubFactorCol32BlockNum_, ubFactorColLoopNum_);                                 // 列方向上头块的Block数量（均衡处理）          用于VF内处理
    ubFactorColNormalBlockNum_ = ops::CeilDiv(ubFactorColNormalBlockNum_, DIGIT_TWO) * DIGIT_TWO;                           // 列方向上头块的Block数量补成偶数，只有尾块有可能需要补Pad
    ubFactorColTailBlockNum_ = ubFactorCol32BlockNum_ - (ubFactorColLoopNum_ - DIGIT_ONE) * ubFactorColNormalBlockNum_;     // 列方向上尾块的Block数量（只有一个尾块）      用于VF内处理
    ubFactorColNormalBlockLen_ = ubFactorColNormalBlockNum_ * blockSize_;                                                   // 列方向上头块的元素数量（均衡处理）           用于数据搬运
    ubFactorColTailBlockLen_ = ubFactorColNum_ - (ubFactorColLoopNum_ - DIGIT_ONE) * ubFactorColNormalBlockLen_;            // 列方向上尾块的元素数量（只有一个尾块）       用于数据搬运

    ubFactorRowNormalBlockNum_ = ops::FloorDiv(maxUbBlockNum_, ubFactorColNormalBlockNum_);                                 // UB 一次至多载入的行数
    ubFactorRowLoopNum_ = ops::CeilDiv(ubFactorRow1BlockNum_, ubFactorRowNormalBlockNum_);                                  // 行方向上 UB 循环次数（行方向上的切 UB 块数）
    ubFactorRowNormalBlockNum_ = ops::CeilDiv(ubFactorRow1BlockNum_, ubFactorRowLoopNum_);                                  // 行方向上的均衡处理                           用于VF内处理
    ubFactorRowTailBlockNum_ = ubFactorRow1BlockNum_ - (ubFactorRowLoopNum_ - DIGIT_ONE) * ubFactorRowNormalBlockNum_;      // 行方向上尾块的行数（只有一个尾块）            用于VF内处理
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ParseTilingData(const QuantFourOverSixA5TailAxisTilingData* tilingData)
{
    roundMode_ = tilingData->roundMode;
    blockSize_ = tilingData->blockSize;
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    rowTileNum_ = tilingData->rowTileNum;
    colTileNum_ = tilingData->colTileNum;
    rowNum_ = tilingData->rowNum;
    colNum_ = tilingData->colNum;
    colNormalBlockNum_ = tilingData->colNormalBlockNum;
    colTailLen_ = tilingData->colTailLen;
    rowNormalBlockNum_ = tilingData->rowNormalBlockNum;
    rowTailLen_ = tilingData->rowTailLen;
    maxUbBlockNum_ = tilingData->maxUbBlockNum;
    dstTypeMax_ = tilingData->dstTypeMax;
    invDstTypeMax_ = tilingData->invDstTypeMax;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::Process()
{
    if (coreIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t dim0LoopIdx = 0;
    int64_t dim1LoopIdx = 0;
    for (dim0LoopIdx = 0; dim0LoopIdx < ubFactorRowLoopNum_ - 1; dim0LoopIdx++) {
        for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
            CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_, ubFactorColNormalBlockNum_);
            Compute(ubFactorRowNormalBlockNum_, ubFactorColNormalBlockNum_);
            CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_, ubFactorColNormalBlockNum_);
        }
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
        Compute(ubFactorRowNormalBlockNum_, ubFactorColTailBlockNum_);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    }
    for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_, ubFactorColNormalBlockNum_);
        Compute(ubFactorRowTailBlockNum_, ubFactorColNormalBlockNum_);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_, ubFactorColNormalBlockNum_);
    }
    CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    Compute(ubFactorRowTailBlockNum_, ubFactorColTailBlockNum_);
    CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum)
{
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
    if (scaleIsNotOdd != 0) {                               // 当前 UB 块一行的scale数不是偶数时，需要将X的Buffer空间预先填充为0
        // 非偶数scale时，提前填充0
        Duplicate<T>(xLocal, static_cast<T>(0), maxUbBlockNum_ * blockSize_);
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }

    DataCopyExtParams copyInParamX = {
        static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(T)),
        static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(T)), static_cast<uint32_t>(scaleIsNotOdd * DIGIT_TWO), static_cast<uint32_t>(0)}; // 一个BlockSize=32，输入x为2Bytes，搬入UB需要跳64Bytes，即2个数据块

    // 搬运X，补Pad示例
    // Eg1: dataLen = 1  => leftPad = [0 * sizeof(T)] Bytes , rightPad = [16 * sizeof(T)] Bytes  , dummy = [15 * sizeof(T) + 2 * 16 * sizeof(T)] Bytes
    // Eg2: dataLen = 17 => leftPad = [0 * sizeof(T)] Bytes , rightPad = [15 * sizeof(T)] Bytes  , dummy = [ 0 * sizeof(T) + 2 * 16 * sizeof(T)] Bytes
    // Eg3: dataLen = 32 => leftPad = [0 * sizeof(T)] Bytes , rightPad = [ 0 * sizeof(T)] Bytes  , dummy = [ 0 * sizeof(T) + 2 * 16 * sizeof(T)] Bytes
    // Eg4: dataLen = 35 => leftPad = [0 * sizeof(T)] Bytes , rightPad = [16 * sizeof(T)] Bytes  , dummy = [13 * sizeof(T) + 0 * 16 * sizeof(T)] Bytes
    // 当补的数(rightPad)是16时，一定能补到UB的下一个DataBlock，实现补到32个数对齐的功能
    int64_t padRightNumTmp = (ubFactorColNum % 32 > 16) ? (32 - ubFactorColNum % 32) : 16;
    int64_t padRightNum = (ubFactorColNum % 32 == 0) ? 0 : padRightNumTmp;
    DataCopyPadExtParams<T> padParams_ = {true, static_cast<uint8_t>(0), static_cast<uint8_t>(padRightNum), static_cast<T>(0)};
    int64_t offset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_;

    DataCopyPad(xLocal, xGm_[offset], copyInParamX, padParams_);
    inQueue_.EnQue(xLocal);
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum)
{
    LocalTensor<uint8_t> scaleLocal = mxScaleQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> yLocal = outQueue_.DeQue<uint8_t>();
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    DataCopyExtParams copyOutParamY = {
        static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum / DIGIT_TWO), static_cast<uint32_t>(0), // 一个 BlockSize = 32，64个数对齐，输出 y 为 0.5 Bytes，占用 32 Bytes，srcStride = 0
        static_cast<uint32_t>((colNum_ - ubFactorColNum) / DIGIT_TWO), static_cast<uint32_t>(0)};

    int64_t offset = (dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_) / DIGIT_TWO;
    DataCopyPad(yGm_[offset], yLocal, copyOutParamY);

    DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
    copyOutParamScale.blockCount = ubFactorRowNum;
    copyOutParamScale.blockLen = ubFactorColBlockNum + scaleIsNotOdd;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = scaleColNum_ - copyOutParamScale.blockLen;

    int64_t scaleOffset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * scaleColNum_  + dim1LoopIdx * ubFactorColNormalBlockNum_;
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[scaleOffset], scaleLocal, copyOutParamScale);

    outQueue_.FreeTensor(yLocal);
    mxScaleQueue_.FreeTensor(scaleLocal);
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeMaxExp(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;
        Reg::RegTensor<uint16_t> vdExpExtract0;
        Reg::RegTensor<uint16_t> vdExpExtract1;
        Reg::RegTensor<uint16_t> vdExpSelect0;
        Reg::RegTensor<uint16_t> vdExpSelect1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> invalidmaskfp16;
        Reg::Duplicate(invalidmaskfp16, INVALID_FLOAT16);
        
        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;
        Reg::UnalignReg ureg;

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, xLocalAddr, vfLen16Double);

            if constexpr (IsSame<T, half>::value) {
                Reg::And(
                    vdExpSelect0, (Reg::RegTensor<uint16_t>&)vdExp0, invalidmaskfp16, Mask);
                Reg::And(
                    vdExpSelect1, (Reg::RegTensor<uint16_t>&)vdExp1, invalidmaskfp16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidmaskfp16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidmaskfp16, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, Mask);
                Reg::And(
                    vdExpExtract0, (Reg::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, Mask);
                Reg::And(
                    vdExpExtract1, (Reg::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, Mask);
                Reg::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                Reg::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                Reg::And(
                    vdExpExtract0, (Reg::RegTensor<uint16_t>&)vdExp0, expMaskBF16, Mask);
                Reg::And(
                    vdExpExtract1, (Reg::RegTensor<uint16_t>&)vdExp1, expMaskBF16, Mask);
            }

            Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, ureg, elementAfterReduce_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeMaxExpDynamicDtypeRange(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF){
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;
        Reg::RegTensor<uint16_t> vdExpSelect0;
        Reg::RegTensor<uint16_t> vdExpSelect1;
        Reg::RegTensor<uint16_t> vdExpExtract0;
        Reg::RegTensor<uint16_t> vdExpExtract1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> invalidMaskFP16;
        Reg::Duplicate(invalidMaskFP16, INVALID_FLOAT16);
        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Reg::UnalignReg ureg;
        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            if constexpr (ops::IsSame<T, half>::value) {
                Reg::And(vdExpSelect0, (Reg::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16, Mask);
                Reg::And(vdExpSelect1, (Reg::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidMaskFP16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidMaskFP16, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, Mask);
                Reg::And(vdExpExtract0, (Reg::RegTensor<uint16_t>&)vdExp0BF16, absMask16Bit, Mask);
                Reg::And(vdExpExtract1, (Reg::RegTensor<uint16_t>&)vdExp1BF16, absMask16Bit, Mask);
                Reg::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                Reg::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                Reg::And(vdExpExtract0, (Reg::RegTensor<uint16_t>&)vdExp0, absMask16Bit, Mask);
                Reg::And(vdExpExtract1, (Reg::RegTensor<uint16_t>&)vdExp1, absMask16Bit, Mask);
            }

            Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, ureg, elementAfterReduce_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeMaxExpcuBLAS(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF){
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        Reg::UnalignReg ureg;
        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            Reg::And(
                (Reg::RegTensor<uint16_t>&)vdExp0, (Reg::RegTensor<uint16_t>&)vdExp0,
                absMask16Bit, Mask);
            Reg::And(
                (Reg::RegTensor<uint16_t>&)vdExp1, (Reg::RegTensor<uint16_t>&)vdExp1,
                absMask16Bit, Mask);
            Reg::Max(
                vdMaxExp, (Reg::RegTensor<uint16_t>&)vdExp0,
                (Reg::RegTensor<uint16_t>&)vdExp1, Mask);
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, ureg, elementAfterReduce_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, f4Emax_);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, EXP_BF16_BIAS);
        Reg::RegTensor<uint16_t> fp8NanRegTensor;
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        Reg::RegTensor<uint16_t> zeroRegTensor;
        Reg::Duplicate(zeroRegTensor, 0);
        Reg::RegTensor<uint16_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        Reg::RegTensor<uint16_t> specialExpRegTensor;
        Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg preMaskScale;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        for (uint16_t i = 0; i < LoopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vfLen16);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vfLen32, preMaskScale);           // 128 个scale，占用 128 * 1 Btyes = vfLen32 * sizeof(uint16_t)

            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroRegTensor, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                recipScaleLocalAddr, halfScale, vfLen16, preMaskScale);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeScaleDynamicDtypeRange(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB){
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;
        Reg::RegTensor<uint16_t> vdMaxExpAdd;
        Reg::RegTensor<uint16_t> vdMaxExpOnly;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> addValue;
        Reg::Duplicate(addValue, addValueBit);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, FP4_E2M1_BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, EXP_BF16_BIAS);
        Reg::RegTensor<uint16_t> fp8NanRegTensor;
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        Reg::RegTensor<uint16_t> zeroRegTensor;
        Reg::Duplicate(zeroRegTensor, 0);
        Reg::RegTensor<uint16_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        Reg::RegTensor<uint16_t> specialExpRegTensor;
        Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;
        Reg::MaskReg preMaskScale;

        for (uint16_t i = 0; i < LoopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vfLen16);
            Reg::And(vdMaxExpOnly, vdMaxExp, expMask, preMaskScale);  // 提取指数位
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExpOnly, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, vdMaxExpOnly, maxExpValue, preMaskScale);
            
            Reg::Add(vdMaxExpAdd, vdMaxExp, addValue, preMaskScale);     // 进位后的结果
            Reg::And(vdMaxExpAdd, vdMaxExpAdd, expMask, preMaskScale);      // 提取进位结果的指数位
            Reg::Select<uint16_t>(vdMaxExpAdd, maxExpValue, vdMaxExpAdd, invalidDataMask);
            Reg::Sub(sharedExp, vdMaxExpAdd, maxExpValue, preMaskScale);

            Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vfLen32, preMaskScale);

            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroRegTensor, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::StoreDist::DIST_NORM>(recipScaleLocalAddr, halfScale, vfLen16, preMaskScale);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeScalecuBLASFP4(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* scaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNumHalfVF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> max16;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> normalExp32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint16_t> expOut;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> recExpOut;

        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        Reg::RegTensor<uint32_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_FP32);
        Reg::RegTensor<uint32_t> zeroRegTensor32;
        Reg::Duplicate(zeroRegTensor32, 0);
        Reg::RegTensor<uint32_t> scaleBias;
        Reg::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        Reg::RegTensor<uint32_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        Reg::RegTensor<uint32_t> fp4NanRegTensor;
        Reg::Duplicate(fp4NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::RegTensor<float> invMax;
        Reg::Duplicate(invMax, invDstTypeMax_);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        Reg::MaskReg preMaskScale;
        uint32_t SixtyFour = 64;
        Reg::MaskReg dataMaskB16Half = Reg::UpdateMask<uint16_t>(SixtyFour);

        static constexpr Reg::CastTrait castTraitHalf2Float = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        for (uint16_t i = 0; i < LoopNumHalfVF; i++) {
            preMaskScale = Reg::UpdateMask<uint32_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, vfLen32);

            Reg::Cast<float, T, castTraitHalf2Float>(
                (Reg::RegTensor<float>&)max32, (Reg::RegTensor<T>&)max16, preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);
         
            Reg::Mul((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32,
                invMax, preMaskScale);
            Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            Reg::And(man32, max32, manMaskFP32, preMaskScale);

            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, NUMBER_ZERO, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, NUMBER_ZERO, preMaskScale);
            Reg::MaskAnd(p0, p0, p1, preMaskScale);
            Reg::MaskAnd(p0, p0, p2, preMaskScale);

            Reg::Adds(expAddOne32, exp32, 1, preMaskScale);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp4NanRegTensor, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expOut, extractExp);

            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(
                scaleLocalAddr + i * 32, expOut, dataMaskB16Half);

            Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            Reg::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            Reg::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

            Reg::StoreAlign<uint16_t>(
                recipScaleLocalAddr + i * vfLen32, recExpOut, dataMaskB16Half);
        }
    }
    return;
}

// ============================================================
// dst_type_max=4: per-block 4/6 自适应量化（var(x-dequant) 判据）—— 实现蓝图
// ============================================================
// 对每个 32 元素 block（D6=6, D4=4, n=32）：
//   1) M = max|x|（复用 ComputeMaxExpcuBLAS，结果已在 maxExpAddr）
//   2) scale_6 = 2^floor(log2(M/6))；scale_4 = 2^floor(log2(M/4))
//      注：scale_4 ∈ {scale_6, 2*scale_6}；当 floor(log2(M/4))==floor(log2(M/6))
//      时二者相等，无需比较。D=4 避免大值饱和、D=6 小值精度高，二者权衡非退化（见
//      架构文档 §14.3：M=5 时 D=4 误差 1.0 < D=6 误差 2.0）。
//   3) 对 s ∈ {scale_6, scale_4}（先 6 后 4 时分复用寄存器）：
//        q       = cast_fp4(x / s)        // 复用 ComputeFP4FromHalf
//        dequant = q * s                  // FP32：q->fp32 后乘 2^exp 构造
//        e       = x - dequant
//        Σe, Σe² = ReduceSumWithDataBlock(e), ReduceSumWithDataBlock(e*e)
//        Σe²_s    = Σe²            // 判据: 统一算 Σe² (n=32 常数, 等价 MSE), 非方差 var=Σe²/n-(Σe/n)²
//   4) s* = argmin(Σe²_s)；输出 mxscale=s*(E8M0)、recipScale=1/s*。
//      最终 y 由调用方 Compute() 末尾 ComputeDataOptimize<roundMode> 用选定 recipScale 写出。
//
// ⚠️ 待编译环境验证/实现：
//   - per-block reduce-sum：Reg::ReduceSumWithDataBlock（与已用的 Reg::ReduceMaxWithDataBlock
//     假设成对存在）。若不存在，回退：MicroAPI::ReduceSum（全 reduce）后按
//     elementAfterReduce_=8 手动聚合 32 元素块。
//   - scale_4 派生：在 ComputeScalecuBLASFP4 的 exp32 基础上，D=4 相比 D=6 多减
//     log2(1.5)≈0.585 的尾数偏移（对应 addValueBit 差异，参考架构文档 §12.3/§13.4）。
//   - 特判：全零块（M=0）直接 y=0、mxscale=E8M0 最小值；幂边界（scale_6==scale_4）跳过比较；
//     var 平局（|Δvar|<ε）改用 mse=Σe² 决断，覆盖 DC/饱和块盲点。
// ============================================================
// 当前占位实现：退化为 D=6 单候选（等价 scale_alg=2 默认 ceil 行为），保证
// dst_type_max=4 路径不崩溃；真正的 4/6 自适应逻辑需在编译环境按上述蓝图实现。
// ============================================================
// ============================================================
// helper 1 (优化版): 由 maxExp(|x| BF16 含尾数)与 addValueBit 派生 E8M0 scale + recipScale
// 复用 ComputeScaleDynamicDtypeRange 的位运算:addValueBit 尾数进位实现 ceil(log2(max/D)),
// 替代 cuBLAS 的 FP32 amax×(1/D) 浮点路径 —— 纯位运算(Add+And),更快,数值结果等价。
//   D=6 → addValue=ADD_VALUE_FOR_BF16_MAN1(0x3f); D=4 → addValue=ADD_VALUE_FOR_BF16_D4(0x7f)
// 要求 maxExp 为"含尾数的 |x| BF16"(由 ComputeMaxExpDynamicDtypeRange 产生),addValueBit 才能
// 在尾数域进位。scale 以 uint16 存储(E8M0 在低字节)。
// ============================================================
// ============================================================
// helper 1 (1-bit 决策版): 由 maxExp 一次算 D6 候选(scale6/recip6)全流程,
// 再由 BF16 尾数 m 的 1-bit 决策(decision = m<8 OR m≥54)派生 D4:
//   scaleE8M0_4 = scaleE8M0_6 + decision,  recip_4 = recip_6 − (decision<<7)
// 数学见 docs/dst_type_max_4_scale_1bit_decision.md(scale4∈{scale6,2·scale6}, 半数 M 无理→无 tie)。
// 省掉 D4 的 addValue/subOffset/shift/halfScale 全套(原两次调用合一)。
// ============================================================
template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeScaleDdr46(
    __ubuf__ uint16_t* maxExpAddr,
    __ubuf__ uint16_t* scale46Out, __ubuf__ uint16_t* recip6Out, __ubuf__ uint16_t* recip4Out,
    uint16_t LoopNum1VF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vdMaxExp, vdMaxExpOnly, vdMaxExpAdd, sharedExp, scaleValue, halfScale;
        Reg::RegTensor<uint16_t> mantReg, decVal, shiftedDec, scaleValue4, halfScale4;
        Reg::RegTensor<uint16_t> expMask;    Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> addVal6;    Reg::Duplicate(addVal6, ADD_VALUE_FOR_BF16_D6_ROUND);
        Reg::RegTensor<uint16_t> subOff6;    Reg::Duplicate(subOff6, SUB_OFFSET_FOR_D6_ROUND);
        Reg::RegTensor<uint16_t> maxExpVal;  Reg::Duplicate(maxExpVal, FP4_E2M1_BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> scaleBias;  Reg::Duplicate(scaleBias, EXP_BF16_BIAS);
        Reg::RegTensor<uint16_t> fp8NanReg;  Reg::Duplicate(fp8NanReg, MAX_EXP_FOR_FP8);
        Reg::RegTensor<uint16_t> zeroReg;    Reg::Duplicate(zeroReg, 0);
        Reg::RegTensor<uint16_t> oneReg;     Reg::Duplicate(oneReg, static_cast<uint16_t>(1));
        Reg::RegTensor<uint16_t> nanReg;     Reg::Duplicate(nanReg, NAN_CUSTOMIZATION);
        Reg::RegTensor<uint16_t> specialExp; Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);
        Reg::RegTensor<uint16_t> mantMask;   Reg::Duplicate(mantMask, static_cast<uint16_t>(0x007f));
        Reg::MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask, preMaskScale;
        Reg::MaskReg decLt8, decGe54, decision;

        for (uint16_t i = 0; i < LoopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            // ---- 公共: maxExp + 指数掩码 + Inf/NaN + invalid + 尾数 ----
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr, vfLen16);
            Reg::And(vdMaxExpOnly, vdMaxExp, expMask, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExpOnly, expMask, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, vdMaxExpOnly, maxExpVal, preMaskScale);
            Reg::And(mantReg, vdMaxExp, mantMask, preMaskScale);                       // m = BF16 尾数(bits 0-6)

            // ---- D6 完整流水线(addValue 0x78 / subOffset 0x180) ----
            Reg::Add(vdMaxExpAdd, vdMaxExp, addVal6, preMaskScale);                    // 尾数进位(round)
            Reg::And(vdMaxExpAdd, vdMaxExpAdd, expMask, preMaskScale);
            Reg::Select<uint16_t>(vdMaxExpAdd, subOff6, vdMaxExpAdd, invalidDataMask); // clamp
            Reg::Sub(sharedExp, vdMaxExpAdd, subOff6, preMaskScale);
            Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanReg, cmpResult);       // Inf/NaN → 0xff
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroReg, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);                   // recip6 = 0x7f00 - sharedExp
            Reg::Select<uint16_t>(halfScale, halfScale, nanReg, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroReg, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExp, halfScale, specialDataMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM>(
                recip6Out, halfScale, vfLen16, preMaskScale);

            // ---- 1-bit 决策: decision = (m<8) OR (m≥54); 边界(invalid/cmp)强制 0 ----
            Reg::CompareScalar<uint16_t, CMPMODE::LT>(decLt8, mantReg, MANT_THR_D6_CARRY, preMaskScale);  // m<8
            Reg::CompareScalar<uint16_t, CMPMODE::GE>(decGe54, mantReg, MANT_THR_D4_CARRY, preMaskScale); // m≥54
            Reg::MaskOr(decision, decLt8, decGe54, preMaskScale);                          // decision = m<8 OR m≥54
            Reg::Select<uint16_t>(decVal, oneReg, zeroReg, decision);                      // 物化 0/1
            Reg::Select<uint16_t>(decVal, zeroReg, decVal, invalidDataMask);               // invalid → 0
            Reg::Select<uint16_t>(decVal, decVal, zeroReg, cmpResult);                     // cmpResult TRUE=normal→保留decVal, FALSE=Inf/NaN→0(原 zeroReg,decVal 参数反, 误清 normal decVal)
            Reg::ShiftLefts(shiftedDec, decVal, SHR_NUM_FOR_BF16, preMaskScale);           // = decision<<7 (0/0x80)

            // ---- 派生 D4: scale4 = scale6 + decVal;  recip4 = recip6 − shiftedDec ----
            Reg::Add(scaleValue4, scaleValue, decVal, preMaskScale);
            // scale46: DIST_INTLV_B16 (even=scale6, odd=scale4) — 双源无 POST, 用 AddrReg 偏移推进 (i×vfLen16Double)
            Reg::AddrReg aReg46 = Reg::CreateAddrReg<uint16_t>(i, vfLen16Double);
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                scale46Out, scaleValue, scaleValue4, aReg46, preMaskScale);
            Reg::Sub(halfScale4, halfScale, shiftedDec, preMaskScale);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM>(
                recip4Out, halfScale4, vfLen16, preMaskScale);
        }
    }
    return;
}


// ============================================================
// helper 2: ComputeCandStore46 (bf16, 单 pass: 双候选 Se^2 + qU 注册 + mask relay)
//   单 pass: D6/D4 候选各【一次量化】得 qU(硬件 Cast pack 后 FP4, 寄存器驻留)→ qU unpack 得 qg(误差源)→ Σe²;
//   同源 Compare(ss4<ss6) 的 b32 mask 经 MaskDeInterleave 转 b16 mask: ② Select winner scale, ③ Select winner qU(寄存器内 fp4x2 整选, 不落 UB → 避开 PACK4 scramble);
//   y 之后写出 mxScale(PACK_B16)。详 docs/dst_type_max_4_fused_path_redesign.md。
//   根因修复: path-C pack-both-then-bitwise-select 踩 PACK4_B32 非线性字节序; 本版 qU 始终在寄存器, Select 不经 UB。
// 数据流(每 LoopNum2VF 迭代 = vfLen16Double 元素 = 8 block):
//   x(DINTLV)→vx0,vx1 ; recip6/recip4(E2B_B16)→bf16 ; scale6/scale4(NORM)→uint16
//   ① 方差 + qU(保留):
//     T=bf16: v=Mul(vx,recip)→Interleave→Cast<U,bf16>(qU)[★保留]→Cast<bf16,U>(qg)[误差源]
//             e=v−qg(bf16)→Cast<float>ZERO/ONE×(v0,v1)→e²→a+b+c+d→ReduceSumWithDataBlock(ss)
//     T=half: 4 sub 各 Cast<float>ZERO/ONE→Mul(recipF)→ComputeFP4FromHalf→clamp→Sub(e=x−qg)→Mul(e²)→Add(combined)
//             →ReduceSumWithDataBlock(ss); qg(float)保留供 ③ pack
//   ② 选 scale: Compare<float,LT>(winner_b32,ss4,ss6)→Pack(pk1)→Select<uint16>(scaleWin)
//   ③ y: MaskInterleave×1(exp1)→Select<uint64>(yFP4, qU4, qU6, exp1) [2-call vsel 覆盖 16 uint32]
//        T=bf16: 寄存器内 fp4x2 整选 qU4 vs qU6
//        T=half: Select<float>(qgWin, qg4, qg6, exp1)→Cast<bf16,float>→Pack/Interleave→Cast<U,bf16>(yFP4)
//        →StoreAlign<PACK4_B32>(yLocalAddr, yFP4_*) [×2]
//   ②' mxScale(y 后): StoreAlign<PACK_B16>(mxScaleLocalAddr, scaleWin)
// 注: qU6/qU4(寄存器驻留)跨 ③ 决策 Select —— 无 UB 落盘 → 不踩 PACK4 scramble。
// ============================================================
// ============================================================
// ============================================================
// StoreScaleSelect: mask+scale NORM 直载→Select→PACK_B16 写 mxScale (独立 VF, 16-block 粒度)
//   免 E2B_B16 广播+ReduceMax 往返, 免 DeInterleave, 用 PACK_B16 直接写 E8M0 字节。
//   maskAddr 指向 seTmp_(每 block 1 uint16_t: 0x0001=D4, 0x0000=D6)。
//   scale6/scale4/mxScale 指针不 POST-advance(值传递), 调用后由 ComputeAdaptive46 re-get。
// ============================================================
// 对齐 ComputeScaleDynamicDtypeRange: CeilDiv 驱动循环, UpdateMask(totalScaleInUB) 隐式处理尾块
template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::StoreScaleSelect(
        __ubuf__ uint16_t* scale46Addr, __ubuf__ half* seTmpAddr,
        __ubuf__ uint16_t* mxScaleAddr, __ubuf__ uint16_t* maskOutAddr,
        uint16_t LoopNum1VF, uint32_t totalScaleInUB)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint16_t> scale6, scale4, k6_e, k4_e;
            Reg::RegTensor<half> ss6, ss4;
            Reg::RegTensor<bfloat16_t> ss6_bf, ss4_bf;
            Reg::RegTensor<uint16_t> ss6_adj, ss4_adj, win;
            Reg::MaskReg maskCmp, preMaskScale;
            Reg::RegTensor<uint16_t> maskBits, oneReg, zeroReg;
            Reg::Duplicate(oneReg, static_cast<uint16_t>(1));
            Reg::Duplicate(zeroReg, static_cast<uint16_t>(0));
            static constexpr Reg::CastTrait castHalf2Bf16 = {
                Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};


            for (uint16_t i = 0; i < LoopNum1VF; i++) {
                preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);

                // 1) DINTLV_B16 load scale46 + seTmp (256 elem per iter, covers 128 blocks)
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                    Reg::LoadDist::DIST_DINTLV_B16>(scale6, scale4, scale46Addr, vfLen16Double);
                Reg::LoadAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                    Reg::LoadDist::DIST_DINTLV_B16>(ss6, ss4, seTmpAddr, vfLen16Double);

                // 2) MSE: half->BF16 + scale^2 (BF16 exponent: k<<8)
                Reg::Cast<bfloat16_t, half, castHalf2Bf16>(ss6_bf, (Reg::RegTensor<half>&)ss6, preMaskScale);
                Reg::ShiftLefts(k6_e, scale6, SHR_NUM_FOR_BF16_EXP, preMaskScale);
                Reg::Add(ss6_adj, (Reg::RegTensor<uint16_t>&)ss6_bf, k6_e, preMaskScale);

                Reg::Cast<bfloat16_t, half, castHalf2Bf16>(ss4_bf, (Reg::RegTensor<half>&)ss4, preMaskScale);
                Reg::ShiftLefts(k4_e, scale4, SHR_NUM_FOR_BF16_EXP, preMaskScale);
                Reg::Add(ss4_adj, (Reg::RegTensor<uint16_t>&)ss4_bf, k4_e, preMaskScale);

                // 3) Compare + Select winner scale -> PACK_B16 mxScale
                Reg::Compare<uint16_t, CMPMODE::LT>(maskCmp, ss4_adj, ss6_adj, preMaskScale);
                Reg::Select<uint16_t>(win, scale4, scale6, maskCmp);
                Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                    Reg::StoreDist::DIST_PACK_B16>(
                    mxScaleAddr, win, vfLen32, preMaskScale);

                // 4) mask -> maskOutAddr (复用 recipScaleAddr)
                Reg::Select<uint16_t>(maskBits, oneReg, zeroReg, maskCmp);
                Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM>(
                    maskOutAddr, maskBits, vfLen16, preMaskScale);
            }
        }
    return;
}

// ============================================================
// QuantVarCand46: 单候选(4/6)量化 + Σe²。D6/D4 同构, 仅 recip 不同, ComputeCandStore46 各调一次。
//   v = x*recip → Interleave 复原原序 → Cast<U> 量化 pack(非原序)→ DeInterleave(uint32=2FP4 奇偶拆分)
//   还原打包序 → unpack qg(与 v 原序对齐)→ e=v−qg → e→half → 配对平方和 → half reduce。
//   qU0/qU1(DeInterleave 后)出参供 ③ y Select; ssOut 出参供 ② Compare。
// ============================================================
template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::QuantVarCand46(
    Reg::RegTensor<uint16_t>& recip, Reg::RegTensor<T>& vx0, Reg::RegTensor<T>& vx1,
    Reg::RegTensor<U>& qU0, Reg::RegTensor<U>& qU1, Reg::RegTensor<half>& ssOut,
    Reg::MaskReg pregAll16)
{
    __VEC_SCOPE__
    {
        static constexpr Reg::CastTrait castTraitU = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitbf162half = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Reg::RegTensor<T> v0bf, v1bf;
        Reg::RegTensor<bfloat16_t> qg0, qg1;
        Reg::RegTensor<T> e0, e1;
        Reg::RegTensor<half> e0h, e1h, combined;

        Reg::Mul(v0bf, vx0, (Reg::RegTensor<T>&)recip, pregAll16);
        Reg::Mul(v1bf, vx1, (Reg::RegTensor<T>&)recip, pregAll16);
        Reg::Interleave(v0bf, v1bf, v0bf, v1bf);                  // 复原原序
        Reg::Cast<U, bfloat16_t, castTraitU>(qU0, v0bf, pregAll16);  // 量化 pack(Cast<U> 打包非原序)
        Reg::Cast<U, bfloat16_t, castTraitU>(qU1, v1bf, pregAll16);
        Reg::Cast<bfloat16_t, U, castTraitU>(qg0, qU0, pregAll16);   // unpack → qg(bf16, 与 v0bf 同 Cast<U> 打包序)
        Reg::Cast<bfloat16_t, U, castTraitU>(qg1, qU1, pregAll16);
        Reg::Sub(e0, v0bf, qg0, pregAll16);   // e = v − qg(bf16, v0bf 与 qg0 同序 → 对齐; 不在 qU 上 DeInterleave 以免错位)
        Reg::Sub(e1, v1bf, qg1, pregAll16);
        Reg::Cast<half, bfloat16_t, castTraitbf162half>(e0h, e0, pregAll16);   // e → half
        Reg::Cast<half, bfloat16_t, castTraitbf162half>(e1h, e1, pregAll16);
        // DeInterleave on half e(Sub 之后, 不破坏 v0bf/qg0 对齐; 无 uint32, 直接 half 粒度)
        Reg::DeInterleave(e0h, e1h, e0h, e1h);
        Reg::Mul(combined, e0h, e0h, pregAll16);                  // combined = e0h²
        Reg::MulAddDst(combined, e1h, e1h, pregAll16);            // += e1h²(配对平方和)
        Reg::ReduceSumWithDataBlock<half>(ssOut, combined, pregAll16);
    }
    return;
}

    // ============================================================
    // SelectY46: VF2 — mask broadcast + y46 DINTLV_B16 + Select + PACK_B16
    // ============================================================
    template <typename T, typename U, int64_t SCALE_ALG>
    __aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::SelectY46(

    __ubuf__ uint16_t* maskAddr, __ubuf__ uint16_t* y46Addr,
    __ubuf__ uint16_t* yOutAddr, uint16_t LoopNum2VF, uint32_t totalScaleInUB)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint16_t> y6, y4, vdMask, yFP4;
            Reg::RegTensor<uint16_t> zeroReg;
            Reg::MaskReg winnerMask;
            Reg::MaskReg yMask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
            Reg::Duplicate(zeroReg, static_cast<uint16_t>(0));

            for (uint16_t i = 0; i < LoopNum2VF; i++) {
    
                // E2B_B16 mask broadcast
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                    Reg::LoadDist::DIST_E2B_B16>(vdMask, maskAddr, elementAfterReduce_);
                Reg::Compare<uint16_t, CMPMODE::NE>(winnerMask, vdMask, zeroReg, yMask);
    
                // DINTLV_B16 load y46
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                    Reg::LoadDist::DIST_DINTLV_B16>(y6, y4, y46Addr, OUT_ELE_NUM_TWO_BLK_X2);
    
                Reg::Select<uint16_t>(yFP4, y4, y6, winnerMask);
                Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                    Reg::StoreDist::DIST_PACK_B16>(yOutAddr,
                    yFP4, OUT_ELE_NUM_ONE_BLK, yMask);
            }
        }
        return;
    }
       

// ============================================================
// ComputeCandStore46Bf16: bf16 专用全流程(方差 + mask relay + y 输出; scale 选定由 StoreScaleSelect 集中)。
//   bf16 路径一次硬件 Cast 往返得 qU(packed FP4 寄存器驻留)→ unpack 得 qg(误差源)→ Σe²;
//   qU6/qU4 寄存器内 Select 跨 ③ 直写 y(不落 UB → 避开 PACK4 scramble)。
// 注: T 独立部分(② 选 scale / mask relay / ②' mxScale)在此重复以保持自包含。
// ============================================================
template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeCandStore46(
    __ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleAddr, __ubuf__ uint16_t* recip4Addr,
    __ubuf__ uint16_t* y46Addr, __ubuf__ half* seTmpAddr,
    uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vx0, vx1;
        Reg::RegTensor<uint16_t> recip6, recip4;
        Reg::RegTensor<half> ss6, ss4;                       // 两候选 per-block Σe²(half reduce)
        Reg::MaskReg pregAll16 = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        Reg::MaskReg preMaskHalf;
        uint32_t elemAfterReduce = elementAfterReduce_;

        // ---- 候选 qU(DeInterleave 后, 供 ③ y Select) + y 输出(候选内部寄存器/trait 已移入 QuantVarCand46)----
        Reg::RegTensor<U> qU6_0, qU6_1, qU4_0, qU4_1;

        // ---- ②/③ 决策寄存器 ----
        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            // ---- load x(DINTLV 双搬) + recip6/4(E2B_B16 per-block 广播) ----
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_DINTLV_B16>(vx0, vx1, xLocalAddr, vfLen16Double);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_E2B_B16>(recip6, recipScaleAddr, elementAfterReduce_);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_E2B_B16>(recip4, recip4Addr, elementAfterReduce_);
            // ============ 两候选(同构, 仅 recip 不同 → 复用 QuantVarCand46)============
            QuantVarCand46<toBf16RoundMode, roundMode>(recip6, vx0, vx1, qU6_0, qU6_1, ss6, pregAll16);
            QuantVarCand46<toBf16RoundMode, roundMode>(recip4, vx0, vx1, qU4_0, qU4_1, ss4, pregAll16);

            // ss6∥ss4 half INTLV_B16 存 seTmp_ (F2: UpdateMask 要 uint32&;  F3: 双源 StoreAlign 用 AddrReg)
            preMaskHalf = Reg::UpdateMask<uint16_t>(elemAfterReduce);
            Reg::AddrReg aRegSe = Reg::CreateAddrReg<half>(i, elementAfterReduceDouble_);
            Reg::StoreAlign<half, Reg::StoreDist::DIST_INTLV_B16>(
                seTmpAddr, ss6, ss4, aRegSe, preMaskHalf);
            // DeInterleave qU6/qU4 -> INTLV_B16 store y46
            Reg::DeInterleave((Reg::RegTensor<uint16_t>&)qU6_0, (Reg::RegTensor<uint16_t>&)qU6_1,
                              (Reg::RegTensor<uint16_t>&)qU6_0, (Reg::RegTensor<uint16_t>&)qU6_1);
            Reg::DeInterleave((Reg::RegTensor<uint16_t>&)qU4_0, (Reg::RegTensor<uint16_t>&)qU4_1,
                              (Reg::RegTensor<uint16_t>&)qU4_0, (Reg::RegTensor<uint16_t>&)qU4_1);
            Reg::AddrReg aRegY46 = Reg::CreateAddrReg<uint16_t>(i, OUT_ELE_NUM_TWO_BLK_X2);
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                y46Addr,   // y46 已是 uint16*(2FP4/uint16, DINTLV_B16)
                (Reg::RegTensor<uint16_t>&)qU6_0, (Reg::RegTensor<uint16_t>&)qU4_0,
                aRegY46, pregAll16);

        }
    }
    return;
}



// ============================================================
// 主: dst_type_max=4 per-block 4/6 自适应（Σe²(x-dequant) 判据, 非方差）
// ⚠️ 需编译环境验证的 MicroAPI 细节见各 helper 标注：
//   Cast<float,U>(FP4→fp32)、per-block recip/scale 广播、ReduceSumWithDataBlock 签名/粒度
// ============================================================
template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeAdaptive46(
    const LocalTensor<T>& xLocal, const LocalTensor<uint16_t>& maxExpLocal,
    __ubuf__ uint16_t* mxScaleLocalAddr,
    __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF, uint16_t LoopNum1VF,
    uint32_t totalScaleInUB)
{
    auto xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    // scale46: scale6∥scale4 DIST_INTLV_B16
    auto scale46Addr   = reinterpret_cast<__ubuf__ uint16_t*>(scale46Buf_.Get<uint16_t>().GetPhyAddr());
    // recip6: 基线 recipScaleBuffer_, D4 复用为 winner→mask
    auto recipScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleBuffer_.Get<uint16_t>().GetPhyAddr());
    auto recip4Addr    = reinterpret_cast<__ubuf__ uint16_t*>(recip4Buf_.Get<uint16_t>().GetPhyAddr());
    // seTmp_: ss6∥ss4 half INTLV_B16
    auto seTmpAddr     = reinterpret_cast<__ubuf__ half*>(seTmp_.Get<half>().GetPhyAddr());
    auto y46Addr       = reinterpret_cast<__ubuf__ uint16_t*>(y46Buf_.Get<uint16_t>().GetPhyAddr());
    // 1) M = max|BF16(x)| (含尾数)。各 POST_UPDATE 调用后, 复用指针一律 re-get 基址(镜像 Compute():1168)
    ComputeMaxExpDynamicDtypeRange(xLocalAddr, maxExpAddr, LoopNum2VF);
    xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    // 2) D6/D4 候选 scale/recip 一次算出(1-bit 决策: D6 全套 + 派生 D4, 见 docs/dst_type_max_4_scale_1bit_decision.md)
    ComputeScaleDdr46(maxExpAddr, scale46Addr, recipScaleAddr, recip4Addr, LoopNum1VF, totalScaleInUB);
    // 候选 buffer 被 ComputeScaleDdr46 内 POST_UPDATE 推进, 读前 re-get
    scale46Addr   = reinterpret_cast<__ubuf__ uint16_t*>(scale46Buf_.Get<uint16_t>().GetPhyAddr());
    recipScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleBuffer_.Get<uint16_t>().GetPhyAddr());
    recip4Addr    = reinterpret_cast<__ubuf__ uint16_t*>(recip4Buf_.Get<uint16_t>().GetPhyAddr());
    xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    // mask 复用 recipScaleBuffer_ (recip6 消耗后覆盖)
    auto maskOutAddr = recipScaleAddr;
    y46Addr = reinterpret_cast<__ubuf__ uint16_t*>(y46Buf_.Get<uint16_t>().GetPhyAddr());   // re-get(F4: 去 auto)
    // 3) Σe² + y 直写 + mask relay (单 pass, 双候选共享 x 加载)
    if (roundMode_ == MODE_RINT) {
        ComputeCandStore46<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(
            xLocalAddr, recipScaleAddr, recip4Addr,
            y46Addr, seTmpAddr, LoopNum2VF);
    } else if (roundMode_ == MODE_ROUND) {
        ComputeCandStore46<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(
            xLocalAddr, recipScaleAddr, recip4Addr,
            y46Addr, seTmpAddr, LoopNum2VF);
    } else if (roundMode_ == MODE_FLOOR) {
        ComputeCandStore46<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(
            xLocalAddr, recipScaleAddr, recip4Addr,
            y46Addr, seTmpAddr, LoopNum2VF);
    }
    // StoreScaleSelect: CeilDiv+UpdateMask 隐式尾块
    seTmpAddr      = reinterpret_cast<__ubuf__ half*>(seTmp_.Get<half>().GetPhyAddr());
    y46Addr        = reinterpret_cast<__ubuf__ uint16_t*>(y46Buf_.Get<uint16_t>().GetPhyAddr());
    maskOutAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleBuffer_.Get<uint16_t>().GetPhyAddr());
    scale46Addr = reinterpret_cast<__ubuf__ uint16_t*>(scale46Buf_.Get<uint16_t>().GetPhyAddr());
    StoreScaleSelect(scale46Addr, seTmpAddr, mxScaleLocalAddr, maskOutAddr,
                     LoopNum1VF, totalScaleInUB);
    maskOutAddr   = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleBuffer_.Get<uint16_t>().GetPhyAddr());
    y46Addr       = reinterpret_cast<__ubuf__ uint16_t*>(y46Buf_.Get<uint16_t>().GetPhyAddr());
    SelectY46(maskOutAddr, y46Addr,  reinterpret_cast<__ubuf__ uint16_t*>(yLocalAddr), LoopNum2VF, totalScaleInUB);
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeData(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<T> vdExp0Convert;
        Reg::RegTensor<T> vdExp1Convert;
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;
        Reg::RegTensor<bfloat16_t> vdBF16Exp0FP4;
        Reg::RegTensor<bfloat16_t> vdBF16Exp1FP4;
        Reg::RegTensor<U> vdExp0FP4;
        Reg::RegTensor<U> vdExp1FP4;

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTrait = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, toBf16RoundMode};

        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<
                T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            Reg::LoadAlign<
                uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, elementAfterReduce_);

            if constexpr (IsSame<T, half>::value) {
                if constexpr (roundMode == RoundMode::CAST_RINT) {
                    FP16Convert(vdExp0, vdExp0, Mask);
                    FP16Convert(vdExp1, vdExp1, Mask);
                }
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, Mask);
                Reg::Mul(
                    vdExp0BF16, vdExp0BF16, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, Mask);
                Reg::Mul(
                    vdExp1BF16, vdExp1BF16, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, Mask);
                Reg::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, Mask);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, Mask);
            } else {
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T>&)halfScaleForMul, Mask);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T>&)halfScaleForMul, Mask);
                Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, Mask);
                Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, Mask);
            }

            Reg::StoreAlign<
                int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, Mask);
            Reg::StoreAlign<
                int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, Mask);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeDataOptimize(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<T> vdExp0Convert;
        Reg::RegTensor<T> vdExp1Convert;
        Reg::RegTensor<float> halfScaleForMulFP32;
        Reg::RegTensor<float> vdExp0ZeroFP32;
        Reg::RegTensor<float> vdExp0OneFP32;
        Reg::RegTensor<float> vdExp1ZeroFP32;
        Reg::RegTensor<float> vdExp1OneFP32;
        Reg::RegTensor<bfloat16_t> vdExp0ZeroBF16;
        Reg::RegTensor<bfloat16_t> vdExp0OneBF16;
        Reg::RegTensor<bfloat16_t> vdExp1ZeroBF16;
        Reg::RegTensor<bfloat16_t> vdExp1OneBF16;

        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;

        Reg::RegTensor<U> vdExp0FP4;
        Reg::RegTensor<U> vdExp1FP4;

        Reg::RegTensor<bfloat16_t> vdBF16Exp0FP4;
        Reg::RegTensor<bfloat16_t> vdBF16Exp1FP4;

        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();

        static constexpr Reg::CastTrait castTrait = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        static constexpr Reg::CastTrait castTraitF16toFp32Zero = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
            RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitF16toFp32One = {
            Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toBF16 = {
            Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, roundMode};

        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<
                T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            Reg::LoadAlign<
                uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, elementAfterReduce_);

            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, bfloat16_t, castTraitF16toFp32Zero>(
                    halfScaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);

                // vdExp0
                Reg::Cast<float, T, castTraitF16toFp32Zero>(vdExp0ZeroFP32, vdExp0, dataMaskB16);
                Reg::Cast<float, T, castTraitF16toFp32One>(vdExp0OneFP32, vdExp0, dataMaskB16);

                Reg::Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                Reg::Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp0ZeroFP32);
                ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp0OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    vdExp0ZeroBF16, vdExp0ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    vdExp0OneBF16, vdExp0OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)vdExp0ZeroBF16, (Reg::RegTensor<uint32_t>&)vdExp0ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)vdExp0OneBF16, (Reg::RegTensor<uint32_t>&)vdExp0OneBF16);
                Reg::Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);

                // vdExp1
                Reg::Cast<float, T, castTraitF16toFp32Zero>(vdExp1ZeroFP32, vdExp1, dataMaskB16);
                Reg::Cast<float, T, castTraitF16toFp32One>(vdExp1OneFP32, vdExp1, dataMaskB16);

                Reg::Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                Reg::Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp1ZeroFP32);
                ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp1OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    vdExp1ZeroBF16, vdExp1ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    vdExp1OneBF16, vdExp1OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)vdExp1ZeroBF16, (Reg::RegTensor<uint32_t>&)vdExp1ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)vdExp1OneBF16, (Reg::RegTensor<uint32_t>&)vdExp1OneBF16);
                Reg::Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);

                Reg::Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0ZeroBF16, dataMaskB16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1ZeroBF16, dataMaskB16);
            } else {
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T>&)halfScaleForMul, dataMaskB16);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T>&)halfScaleForMul, dataMaskB16);
                Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMaskB16);
                Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMaskB16);
            }

            Reg::StoreAlign<
                int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB16);
            Reg::StoreAlign<
                int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB16);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum)
{
    ubFactorColBlockNum = ops::CeilDiv(ubFactorColBlockNum, DIGIT_TWO) * DIGIT_TWO;
    uint32_t totalBlockNum = ubFactorRowBlockNum * ubFactorColBlockNum;             // 当前Ub内总的Block数量 = 当前Ub内总的计算出Scale数量

    uint16_t LoopNum2VF = ops::CeilDiv(static_cast<uint16_t>(totalBlockNum), elementAfterReduce_);                          // 当前Ub内总的Block数量向上整除8，BlockSize=32，即得到双搬256个X的循环次数
    uint16_t LoopNum1VF = ops::CeilDiv(static_cast<uint16_t>(totalBlockNum), static_cast<uint16_t>(vfLen16));               // 当前Ub内总的Block数量向上整除128，即得到单搬128个max的循环次数
    uint16_t LoopNumHalfVF = ops::CeilDiv(static_cast<uint16_t>(totalBlockNum), static_cast<uint16_t>(vfLen32));            // 当前Ub内总的Block数量向上整除64，即得到单搬64个max的循环次数

    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    LocalTensor<uint16_t> scaleLocal = mxScaleQueue_.AllocTensor<uint16_t>();
    LocalTensor<int8_t> yLocal = outQueue_.AllocTensor<int8_t>();
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
    LocalTensor<uint16_t> recipScaleLocal = recipScaleBuffer_.Get<uint16_t>();

    auto xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    auto scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(yLocal.GetPhyAddr());
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    auto recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());

    if constexpr (ops::IsSame<U, fp4x2_e1m2_t>::value || (ops::IsSame<U, fp4x2_e2m1_t>::value && SCALE_ALG == DIGIT_ZERO)) {
        ComputeMaxExp(xLocalAddr, maxExpLocalAddr, LoopNum2VF);
        maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        ComputeScale(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, LoopNum1VF, totalBlockNum);
        xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
        scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
        recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());
        if (roundMode_ == MODE_RINT) {
            ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeData<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
        }
    } else if constexpr (ops::IsSame<U, fp4x2_e2m1_t>::value && SCALE_ALG == DIGIT_TWO) {
        if (dstTypeMax_ == DIGIT_FOUR_FLOAT) {
            // dst_type_max=4 重载为 per-block 4/6 自适应：对每个 block 同时计算
            // D=6 (scale_6=2^floor(log2(M/6))) 与 D=4 (scale_4=2^floor(log2(M/4))) 两种量化，
            // 按 Σe²(x-dequant) 选误差小者输出(非方差)。详见 ComputeAdaptive46。
            ComputeAdaptive46(xLocal, maxExpLocal, scaleLocalAddr,
                              yLocalAddr, LoopNum2VF, LoopNum1VF, totalBlockNum);
            xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
            scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
// fusion path: y already written by SelectY46 inside ComputeAdaptive46
        } else if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT || dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
            // D=6/7 基线(ceil, 不改);round 仅作用于 dst_type_max=4 自适应(ComputeScaleDdr46)
            ComputeMaxExpDynamicDtypeRange(xLocalAddr, maxExpLocalAddr, LoopNum2VF);
            maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
            ComputeScaleDynamicDtypeRange(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, LoopNum1VF, totalBlockNum);
            xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
            scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
            recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());
            if (roundMode_ == MODE_RINT) {
                ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            } else if (roundMode_ == MODE_ROUND) {
                ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            } else if (roundMode_ == MODE_FLOOR) {
                ComputeData<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            }
        } else {
            ComputeMaxExpcuBLAS(xLocalAddr, maxExpLocalAddr, LoopNum2VF);
            maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
            ComputeScalecuBLASFP4(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, LoopNumHalfVF, totalBlockNum);
            xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
            scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
            recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());
            if (roundMode_ == MODE_RINT) {
                ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            } else if (roundMode_ == MODE_ROUND) {
                ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            } else if (roundMode_ == MODE_FLOOR) {
                ComputeData<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            }
        }
    }
    
    inQueue_.FreeTensor(xLocal);
    mxScaleQueue_.EnQue(scaleLocal);
    outQueue_.EnQue(yLocal);
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::FP16Convert(
    Reg::RegTensor<half>& output, Reg::RegTensor<half>& input,
    Reg::MaskReg& mask)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> specialValueTensor;
        Reg::RegTensor<uint16_t> newMantissa;
        Reg::RegTensor<uint16_t> andResult;
        Reg::RegTensor<uint16_t> newValue;
        Reg::MaskReg specialMask;
        Reg::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        Reg::Duplicate(specialValueTensor, specialValue);
        Reg::Duplicate(newMantissa, NEW_MANTISSA);
        Reg::And(andResult, (Reg::RegTensor<uint16_t>&)input, specialValueTensor, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        Reg::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        Reg::Or(newValue, (Reg::RegTensor<uint16_t>&)input, newMantissa, mask);
        Reg::Select<uint16_t>(
            (Reg::RegTensor<uint16_t>&)output, newValue, (Reg::RegTensor<uint16_t>&)input,
            specialMask);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void QuantFourOverSixA5TailAxis<T, U, SCALE_ALG>::ComputeFP4FromHalf(Reg::RegTensor<float>& Reg)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZero;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZero, NEG_ZERO);
    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        Reg::Muls(Reg, Reg, FOUR, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        Reg::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>(
        (Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}

} // namespace QuantFourOverSixA5
#endif // DYNAMIC_MX_QUANT_TAIL_AXIS_H
