/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
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
 * \file norm_rope_concat.h
 * \brief NormRopeConcat kernel: forward Norm + Rope + Concat fused operator
 */

#ifndef _NORM_ROPE_CONCAT_H_
#define _NORM_ROPE_CONCAT_H_

#include "norm_rope_concat_base.h"

namespace nrc {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    using namespace AscendC::Reg;
#endif
template <bool isTraining>
class NormOperationForward : public NormOperation {
public:
    __aicore__ inline NormOperationForward(TPipe *pipe, float eps, float scale, uint32_t normDim, uint32_t normNum,
                                           uint32_t alignedNormDim, uint32_t alignedNormNum)
        : NormOperation(eps, scale, normDim, normNum, alignedNormDim, alignedNormNum)
    {
        pipe->InitBuffer(inQue_, DOUBLE_BUFFER, this->alignedNormDim_ * normNum * sizeof(DTYPE_QUERY));
        inParams_.blockLen = normDim * sizeof(DTYPE_QUERY);
        padParams_.rightPadding = this->alignedNormDim_ - normDim;
        isAligned64_ = normDim % B32_DATA_NUM_PER_REPEAT == 0;
        rptTimes_ = normDim / B32_DATA_NUM_PER_REPEAT;

        if constexpr (isTraining) {
            pipe->InitBuffer(meanQue_, DOUBLE_BUFFER, this->alignedNormNum_ * sizeof(float));
            pipe->InitBuffer(rstdQue_, DOUBLE_BUFFER, this->alignedNormNum_ * sizeof(float));
        }
        uint32_t bufSize = this->alignedNormNum_ + normNum * this->alignedNormDim_;
        pipe->InitBuffer(normQue_, SINGLE_BUFFER, NUM_TWO * this->alignedNormDim_ * sizeof(DTYPE_QUERY));
        bufSize += 2 * this->alignedNormDim_;
        pipe->InitBuffer(buf_, bufSize * sizeof(float));
        weight_ = buf_.GetWithOffset<float>(NUM_TWO * alignedNormDim,
                                            (this->alignedNormNum_ + normNum * this->alignedNormDim_) * sizeof(float));
        bias_ = weight_[this->alignedNormDim_];
    }

    template <NormType normType>
    __aicore__ inline void Process(const LocalTensor<float> &x, int64_t inOffset, int64_t normOffset, uint32_t heads);

    template <NormType normType>
    __aicore__ inline void Prepare(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR rstd);

private:
    __aicore__ inline void CopyIn(int64_t inOffset, uint32_t heads);

    template <NormType normType>
    __aicore__ inline void Compute(const LocalTensor<float> &x, uint32_t heads);

    __aicore__ inline void CopyOutMeanAndRstd(int64_t normOffset, uint32_t heads);

    __aicore__ inline void DoSum(const LocalTensor<float> &out, const LocalTensor<float> &in,
                                 const LocalTensor<float> &buf, uint32_t heads);
    template <NormType normType>
    __aicore__ inline void DoLayerNorm(const LocalTensor<float> &x, uint32_t heads);

    template <NormType normType>
    __aicore__ inline void DoRMSNorm(const LocalTensor<float> &x, uint32_t heads);

    __aicore__ inline void DoMulAdd(const LocalTensor<float> &x, uint32_t heads);

    __aicore__ inline void DoMul(const LocalTensor<float> &x, uint32_t heads);

    #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __aicore__ inline void DoMulAddVfCall(const LocalTensor<float> &x, uint32_t heads);
        __simd_vf__ static inline void DoMulAddVf(__ubuf__ float* xBuf, __ubuf__ float* weightBuf,
                                                  __ubuf__ float* biasBuf, uint32_t heads, uint32_t alignedNormDim)
        {
            RegTensor<float> xRegTensor;
            RegTensor<float> weightRegTensor;
            RegTensor<float> biasRegTensor;

            uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
            uint32_t repeatTimes = (alignedNormDim + oneRepeatSize - 1) / oneRepeatSize;

            for (uint32_t i = 0; i < heads; ++i) {
                uint32_t len = alignedNormDim;
                __ubuf__ float* tempXBuf = xBuf + i * len;
                __ubuf__ float* tempWeightBuf = weightBuf;
                __ubuf__ float* tempBiasBuf = biasBuf;
                for (uint32_t j = 0; j < repeatTimes; ++j) {
                    MaskReg maskReg = UpdateMask<float>(len);
                    LoadAlign(xRegTensor, tempXBuf + j * oneRepeatSize);
                    LoadAlign(weightRegTensor, tempWeightBuf + j * oneRepeatSize);
                    LoadAlign(biasRegTensor, tempBiasBuf + j * oneRepeatSize);
                    MulDstAdd(xRegTensor, weightRegTensor, biasRegTensor, maskReg);
                    StoreAlign(tempXBuf + j * oneRepeatSize, xRegTensor, maskReg);
                }
            }
        }
        __aicore__ inline void DoMulVfCall(const LocalTensor<float> &x, uint32_t heads);
        __simd_vf__ static inline void DoMulVf(__ubuf__ float* xBuf, __ubuf__ float* weightBuf, uint32_t heads,
                                               uint32_t alignedNormDim)
        {
            RegTensor<float> xRegTensor;
            RegTensor<float> weightRegTensor;

            uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
            uint32_t repeatTimes = (alignedNormDim + oneRepeatSize - 1) / oneRepeatSize;

            for (uint32_t i = 0; i < heads; ++i) {
                uint32_t len = alignedNormDim;
                __ubuf__ float* tempXBuf = xBuf + i * len;
                __ubuf__ float* tempWeightBuf = weightBuf;
                for (uint32_t j = 0; j < repeatTimes; ++j) {
                    MaskReg maskReg = UpdateMask<float>(len);
                    LoadAlign(xRegTensor, tempXBuf + j * oneRepeatSize);
                    LoadAlign(weightRegTensor, tempWeightBuf + j * oneRepeatSize);
                    Mul(xRegTensor, xRegTensor, weightRegTensor, maskReg);
                    StoreAlign(tempXBuf + j * oneRepeatSize, xRegTensor, maskReg);
                }
            }
        }
    #endif
private:
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQue_;
    TQue<QuePosition::VECIN, SINGLE_BUFFER> normQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> meanQue_, rstdQue_;
    LocalTensor<float> weight_, bias_;
    TBuf<TPosition::VECCALC> buf_;
    DataCopyExtParams inParams_{0, 0, 0, 0, 0}, outParams_{1, 0, 0, 0, 0};
    DataCopyPadExtParams<DTYPE_QUERY> padParams_{true, 0, 0, 0};
    bool isAligned64_{false};
    uint32_t rptTimes_{0};
};

template <bool isTraining>
__aicore__ inline void NormOperationForward<isTraining>::DoSum(const LocalTensor<float> &out,
                                                               const LocalTensor<float> &in,
                                                               const LocalTensor<float> &buf, uint32_t heads)
{
    if (isAligned64_) {
        Duplicate(buf, 0.f, B32_DATA_NUM_PER_REPEAT * heads);
        for (uint32_t i = 0; i < rptTimes_; ++i) {
            Add(buf, in[i * B32_DATA_NUM_PER_REPEAT], buf, B32_DATA_NUM_PER_REPEAT, heads,
                {1, 1, 1, 8, static_cast<uint8_t>(this->normDim_ / 8), 8});
        }
        WholeReduceSum(out, buf, B32_DATA_NUM_PER_REPEAT, heads, 1, 1, 8);
    } else {
        SumParams sumParams{heads, this->alignedNormDim_, this->normDim_};
        Sum(out, in, buf.ReinterpretCast<uint8_t>(), sumParams);
    }
}

template <bool isTraining>
__aicore__ inline void NormOperationForward<isTraining>::DoMulAdd(const LocalTensor<float> &x, uint32_t heads)
{
    if (isAligned64_) {
        for (uint32_t i = 0; i < rptTimes_; ++i) {
            FusedMulAdd(x[i * B32_DATA_NUM_PER_REPEAT], weight_[i * B32_DATA_NUM_PER_REPEAT],
                        bias_[i * B32_DATA_NUM_PER_REPEAT], B32_DATA_NUM_PER_REPEAT, heads,
                        {1, 1, 1, static_cast<uint8_t>(this->normDim_ / 8), 0, 0});
        }
    } else {
        for (uint32_t i = 0; i < heads; ++i) {
            FusedMulAdd(x[i * this->alignedNormDim_], weight_, bias_, this->alignedNormDim_);
        }
    }
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    template <bool isTraining>
    __aicore__ inline void NormOperationForward<isTraining>::DoMulAddVfCall(const LocalTensor<float> &x, uint32_t heads)
    {
        __ubuf__ float* xBuf = (__ubuf__ float*)x.GetPhyAddr();
        __ubuf__ float* weightBuf = (__ubuf__ float*)weight_.GetPhyAddr();
        __ubuf__ float* biasBuf = (__ubuf__ float*)bias_.GetPhyAddr();
        DoMulAddVf(xBuf, weightBuf, biasBuf, heads, this->alignedNormDim_);
    }

    template <bool isTraining>
    __aicore__ inline void NormOperationForward<isTraining>::DoMulVfCall(const LocalTensor<float> &x, uint32_t heads)
    {
        __ubuf__ float* xBuf = (__ubuf__ float*)x.GetPhyAddr();
        __ubuf__ float* weightBuf = (__ubuf__ float*)weight_.GetPhyAddr();
        DoMulVf(xBuf, weightBuf, heads, this->alignedNormDim_);
    }
#endif

template <bool isTraining>
__aicore__ inline void NormOperationForward<isTraining>::DoMul(const LocalTensor<float> &x, uint32_t heads)
{
    if (isAligned64_) {
        for (uint32_t i = 0; i < rptTimes_; ++i) {
            Mul(x[i * B32_DATA_NUM_PER_REPEAT], x[i * B32_DATA_NUM_PER_REPEAT], weight_[i * B32_DATA_NUM_PER_REPEAT],
                B32_DATA_NUM_PER_REPEAT, heads, {1, 1, 1, static_cast<uint8_t>(this->normDim_ / 8),
                static_cast<uint8_t>(this->normDim_ / 8), 0});
        }
    } else {
        for (uint32_t i = 0; i < heads; ++i) {
            Mul(x[i * this->alignedNormDim_], x[i * this->alignedNormDim_], weight_, this->alignedNormDim_);
        }
    }
}

template <bool isTraining>
template <NormType normType>
__aicore__ inline void NormOperationForward<isTraining>::Prepare(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                                           GM_ADDR mean, GM_ADDR rstd)
{
    this->xGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)x);
    this->weightGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)weight);
    this->biasGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)bias);
    this->meanGm_.SetGlobalBuffer((__gm__ float *)mean);
    this->rstdGm_.SetGlobalBuffer((__gm__ float *)rstd);
    if constexpr (normType == NormType::NONE) {
        return;
    }
    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        LocalTensor<DTYPE_QUERY> norm = normQue_.AllocTensor<DTYPE_QUERY>();
        DataCopy(norm, this->weightGm_, this->alignedNormDim_);
        DataCopy(norm[this->alignedNormDim_], this->biasGm_, this->alignedNormDim_);
        normQue_.EnQue(norm);
        LocalTensor<DTYPE_QUERY> deQue = normQue_.DeQue<DTYPE_QUERY>();
        if constexpr (sizeof(DTYPE_QUERY) == sizeof(float)) {
            Adds(weight_, deQue.ReinterpretCast<float>(), 0.f, 2 * this->alignedNormDim_);
        } else {
            Cast(weight_, deQue, RoundMode::CAST_NONE, 2 * this->alignedNormDim_);
        }
        normQue_.FreeTensor(deQue);
    }

    if constexpr (normType == NormType::RMS_NORM_AFFINE) {
        LocalTensor<DTYPE_QUERY> norm = normQue_.AllocTensor<DTYPE_QUERY>();
        DataCopy(norm, this->weightGm_, this->alignedNormDim_);
        normQue_.EnQue(norm);
        LocalTensor<DTYPE_QUERY> deQue = normQue_.DeQue<DTYPE_QUERY>();
        if constexpr (sizeof(DTYPE_QUERY) == sizeof(float)) {
            Adds(weight_, deQue.ReinterpretCast<float>(), 0.f, this->alignedNormDim_);
        } else {
            Cast(weight_, deQue, RoundMode::CAST_NONE, this->alignedNormDim_);
        }
        normQue_.FreeTensor(deQue);
    }
}

template <bool isTraining>
template <NormType normType>
__aicore__ inline void NormOperationForward<isTraining>::Process(const LocalTensor<float> &x,
                                                                           int64_t inOffset, int64_t normOffset,
                                                                           uint32_t heads)
{
    CopyIn(inOffset, heads);
    Compute<normType>(x, heads);
    if constexpr (IsLayerNorm(normType)) {
        CopyOutMeanAndRstd(normOffset, heads);
    }
}

template <bool isTraining>
__aicore__ inline void NormOperationForward<isTraining>::CopyIn(int64_t inOffset, uint32_t heads)
{
    LocalTensor<DTYPE_QUERY> buf = inQue_.AllocTensor<DTYPE_QUERY>();
    inParams_.blockCount = heads;
    DataCopyPad(buf, this->xGm_[inOffset], inParams_, padParams_);
    inQue_.EnQue(buf);
}

template <bool isTraining>
template <NormType normType>
__aicore__ inline void NormOperationForward<isTraining>::DoLayerNorm(const LocalTensor<float> &x, uint32_t heads)
{
    uint32_t size = heads * this->alignedNormDim_;
    LocalTensor<float> reducedX = buf_.Get<float>();
    LocalTensor<float> tmp0 = reducedX[this->alignedNormNum_];
    LocalTensor<float> tmp1 = x[this->normNum_ * this->alignedNormDim_];
    uint32_t brcdShape[2] = {heads, this->alignedNormDim_};
    DoSum(reducedX, x, tmp0, heads);
    PipeBarrier<PIPE_V>();
    if constexpr (isTraining) {
        LocalTensor<float> mean = meanQue_.AllocTensor<float>();
        Muls(mean, reducedX, this->scale_, this->alignedNormNum_);
        meanQue_.EnQue(mean);
    }

    uint32_t srcShape[2] = {heads, 1};
    LocalTensor<uint8_t> shareBuf = tmp1.ReinterpretCast<uint8_t>();
    BroadCast<float, 2, 1>(tmp0, reducedX, brcdShape, srcShape, shareBuf);
    PipeBarrier<PIPE_V>();
    Axpy(x, tmp0, -this->scale_, size);
    PipeBarrier<PIPE_V>();
    Mul(tmp1, x, x, size);
    PipeBarrier<PIPE_V>();
    DoSum(reducedX, tmp1, tmp0, heads);
    PipeBarrier<PIPE_V>();
    Muls(reducedX, reducedX, this->scale_, this->alignedNormNum_);
    PipeBarrier<PIPE_V>();
    Adds(reducedX, reducedX, this->eps_, this->alignedNormNum_);
    PipeBarrier<PIPE_V>();
    Sqrt(reducedX, reducedX, this->alignedNormNum_);
    PipeBarrier<PIPE_V>();
    if constexpr (isTraining) {
        LocalTensor<float> rstd = rstdQue_.AllocTensor<float>();
        Duplicate(rstd, 1.f, this->alignedNormNum_);
        PipeBarrier<PIPE_V>();
        Div(rstd, rstd, reducedX, this->alignedNormNum_);
        rstdQue_.EnQue(rstd);
    }
    BroadCast<float, 2, 1>(tmp0, reducedX, brcdShape, srcShape, shareBuf);
    PipeBarrier<PIPE_V>();
    Div(x, x, tmp0, size);
    PipeBarrier<PIPE_V>();
    if constexpr (normType == NormType::LAYER_NORM_AFFINE || normType == NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) {
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            DoMulAddVfCall(x, heads);
        #else
            DoMulAdd(x, heads);
        #endif
    }
    PipeBarrier<PIPE_V>();
}

template <bool isTraining>
template <NormType normType>
__aicore__ inline void NormOperationForward<isTraining>::DoRMSNorm(const LocalTensor<float> &x, uint32_t heads)
{
    uint32_t size = heads * this->alignedNormDim_;
    LocalTensor<float> reducedX = buf_.Get<float>();
    LocalTensor<float> tmp0 = reducedX[this->alignedNormNum_];
    LocalTensor<float> tmp1 = x[this->normNum_ * this->alignedNormDim_];

    Mul(tmp0, x, x, size);
    PipeBarrier<PIPE_V>();
    DoSum(reducedX, tmp0, tmp1, heads);
    PipeBarrier<PIPE_V>();
    Muls(reducedX, reducedX, this->scale_, this->alignedNormNum_);
    PipeBarrier<PIPE_V>();
    Adds(reducedX, reducedX, this->eps_, this->alignedNormNum_);
    PipeBarrier<PIPE_V>();
    Sqrt(reducedX, reducedX, this->alignedNormNum_);
    PipeBarrier<PIPE_V>();
    uint32_t brcdShape[2] = {heads, this->alignedNormDim_};
    uint32_t srcShape[2] = {heads, 1};
    LocalTensor<uint8_t> shareBuf = tmp1.ReinterpretCast<uint8_t>();
    BroadCast<float, 2, 1>(tmp0, reducedX, brcdShape, srcShape, shareBuf);
    PipeBarrier<PIPE_V>();
    Div(x, x, tmp0, size);
    PipeBarrier<PIPE_V>();
    if constexpr (normType == NormType::RMS_NORM_AFFINE) {
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            DoMulVfCall(x, heads);
        #else
            DoMul(x, heads);
        #endif
    }
    PipeBarrier<PIPE_V>();
}

template <bool isTraining>
template <NormType normType>
__aicore__ inline void NormOperationForward<isTraining>::Compute(const LocalTensor<float> &x, uint32_t heads)
{
    LocalTensor<DTYPE_QUERY> input = inQue_.DeQue<DTYPE_QUERY>();
    uint32_t size = heads * this->alignedNormDim_;
    if constexpr (sizeof(DTYPE_QUERY) < sizeof(float)) {
        Cast(x, input, RoundMode::CAST_NONE, size);
    } else {
        Adds(x, input.ReinterpretCast<float>(), 0.f, size);
    }
    inQue_.FreeTensor(input);
    PipeBarrier<PIPE_V>();

    if constexpr (IsLayerNorm(normType)) {
        DoLayerNorm<normType>(x, heads);
    }

    if constexpr (IsRMSNorm(normType)) {
        DoRMSNorm<normType>(x, heads);
    }
}

template <bool isTraining>
__aicore__ inline void NormOperationForward<isTraining>::CopyOutMeanAndRstd(int64_t normOffset, uint32_t heads)
{
    if constexpr (!isTraining) {
        return;
    }
    LocalTensor<float> mean = meanQue_.DeQue<float>();
    LocalTensor<float> rstd = rstdQue_.DeQue<float>();
    outParams_.blockLen = heads * sizeof(float);
    DataCopyPad(this->meanGm_[normOffset], mean, outParams_);
    DataCopyPad(this->rstdGm_[normOffset], rstd, outParams_);
    meanQue_.FreeTensor(mean);
    rstdQue_.FreeTensor(rstd);
}


template <RopeType ropeType>
class RopeOperationForward : public RopeOperation<ropeType> {
public:
    __aicore__ inline RopeOperationForward(TPipe *pipe, GM_ADDR ropeSin, GM_ADDR ropeCos, uint32_t actualSeq,
                                           uint32_t ropeDim, uint32_t ropeNum, uint32_t alignedRopeDim)
        : RopeOperation<ropeType>(pipe, ropeSin, ropeCos, actualSeq, ropeDim, ropeNum, alignedRopeDim)
    {
        pipe->InitBuffer(outQue_, DOUBLE_BUFFER, this->alignedRopeDim_ * ropeNum * sizeof(DTYPE_QUERY));
        outParams_.blockLen = this->ropeDim_ * sizeof(DTYPE_QUERY);
    }
    template <RopeType actualRopeType>
    __aicore__ inline void Process(const LocalTensor<float> &x, int64_t outOffset, uint32_t heads);

    template <RopeType actualRopeType>
    __aicore__ inline void Prepare(GM_ADDR y, uint32_t curSeq, uint32_t totalSeq)
    {
        outParams_.dstStride = (totalSeq - 1) * outParams_.blockLen;
        this->yGm_.SetGlobalBuffer((__gm__ DTYPE_QUERY *)y);
        this->isActive_ = curSeq < this->actualSeq_;
    }

private:
    template <RopeType actualRopeType>
    __aicore__ inline void Compute(const LocalTensor<float> &x, uint32_t heads);

    __aicore__ inline void CopyOut(int64_t outOffset, uint32_t heads);

private:
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQue_;
    DataCopyExtParams outParams_{0, 0, 0, 0, 0};
};

template <RopeType ropeType>
template <RopeType actualRopeType>
__aicore__ inline void RopeOperationForward<ropeType>::Process(const LocalTensor<float> &x,
                                                                               int64_t outOffset, uint32_t heads)
{
    Compute<actualRopeType>(x, heads);
    CopyOut(outOffset, heads);
}

template <RopeType ropeType>
template <RopeType actualRopeType>
__aicore__ inline void RopeOperationForward<ropeType>::Compute(const LocalTensor<float> &x,
                                                                               uint32_t heads)
{
    LocalTensor<DTYPE_QUERY> output = outQue_.AllocTensor<DTYPE_QUERY>();
    uint32_t size = this->alignedRopeDim_ * heads;
    if constexpr (actualRopeType != RopeType::NONE) {
        if (this->isActive_) {
            this->Rotate(x);
            PipeBarrier<PIPE_V>();
            this->Collect(x);
            PipeBarrier<PIPE_V>();
        }
    }
    if (sizeof(DTYPE_QUERY) == sizeof(float)) {
        Adds(output.ReinterpretCast<float>(), x, 0.f, size);
    } else {
        Cast(output, x, RoundMode::CAST_ROUND, size);
    }
    PipeBarrier<PIPE_V>();
    outQue_.EnQue(output);
}

template <RopeType ropeType>
__aicore__ inline void RopeOperationForward<ropeType>::CopyOut(int64_t outOffset, uint32_t heads)
{
    LocalTensor<DTYPE_QUERY> output = outQue_.DeQue<DTYPE_QUERY>();
    outParams_.blockCount = heads;
    DataCopyPad(this->yGm_[outOffset], output, outParams_);
    outQue_.FreeTensor(output);
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder, bool isTraining>
class NormRopeConcat {
public:
    __aicore__ inline NormRopeConcat() = default;

    __aicore__ inline NormRopeConcat(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR encoder_query,
                                     GM_ADDR encoder_key, GM_ADDR encoder_value, GM_ADDR norm_query_weight,
                                     GM_ADDR norm_query_bias, GM_ADDR norm_key_weight, GM_ADDR norm_key_bias,
                                     GM_ADDR norm_added_query_weight, GM_ADDR norm_added_query_bias,
                                     GM_ADDR norm_added_key_weight, GM_ADDR norm_added_key_bias, GM_ADDR rope_sin,
                                     GM_ADDR rope_cos, GM_ADDR query_output, GM_ADDR key_output, GM_ADDR value_output,
                                     GM_ADDR norm_query_mean, GM_ADDR norm_query_rstd, GM_ADDR norm_key_mean,
                                     GM_ADDR norm_key_rstd, GM_ADDR norm_added_query_mean,
                                     GM_ADDR norm_added_query_rstd, GM_ADDR norm_added_key_mean,
                                     GM_ADDR norm_added_key_rstd)
        : query_(query), key_(key), value_(value), encoderQuery_(encoder_query), encoderKey_(encoder_key),
          encoderValue_(encoder_value), normQueryWeight_(norm_query_weight), normQueryBias_(norm_query_bias),
          normKeyWeight_(norm_key_weight), normKeyBias_(norm_key_bias), normAddedQueryWeight_(norm_added_query_weight),
          normAddedQueryBias_(norm_added_query_bias), normAddedKeyWeight_(norm_added_key_weight),
          normAddedKeyBias_(norm_added_key_bias), ropeSin_(rope_sin), ropeCos_(rope_cos), queryOutput_(query_output),
          keyOutput_(key_output), valueOutput_(value_output), normQueryMean_(norm_query_mean),
          normQueryRstd_(norm_query_rstd), normKeyMean_(norm_key_mean), normKeyRstd_(norm_key_rstd),
          normAddedQueryMean_(norm_added_query_mean), normAddedQueryRstd_(norm_added_query_rstd),
          normAddedKeyMean_(norm_added_key_mean), normAddedKeyRstd_(norm_added_key_rstd)
    {
    }

    __aicore__ inline void Init(TPipe *pipe, const NormRopeConcatTilingData &tilingData);
    __aicore__ inline void Process();

private:
    template <NormType actualNormType, RopeType actualRopeType>
    __aicore__ inline void ProcessOne(RopeOperationForward<ropeType> &ropeOp, NormOperationForward<isTraining> &normOp,
                                      GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR rstd, GM_ADDR y,
                                      uint32_t xSeq, uint32_t totalSeq, uint32_t outputSeq);
    __aicore__ inline void GetSeqRange(uint32_t &startSeq, uint32_t &endSeq, uint32_t totalSeq, uint32_t usedCore);

private:
    TPipe *pipe_;
    uint32_t blockIdx_;
    uint32_t querySeq_, keySeq_, valueSeq_;
    uint32_t encoderQuerySeq_, encoderKeySeq_, encoderValueSeq_;
    uint32_t totalQuerySeq_, totalKeySeq_, totalValueSeq_;
    uint32_t ropeActualSeq_;
    uint32_t batch_, headNum_, headDim_;
    uint32_t usedCore_;
    uint32_t splitHeadNum_, avgHeads_, tailHeads_;
    uint32_t normDim_, alignedNormDim_, alignedNormNum_;
    uint32_t ropeDim_, alignedRopeDim_;
    float eps_, scale_;

    GM_ADDR query_;
    GM_ADDR key_;
    GM_ADDR value_;
    GM_ADDR encoderQuery_;
    GM_ADDR encoderKey_;
    GM_ADDR encoderValue_;
    GM_ADDR normQueryWeight_;
    GM_ADDR normQueryBias_;
    GM_ADDR normKeyWeight_;
    GM_ADDR normKeyBias_;
    GM_ADDR normAddedQueryWeight_;
    GM_ADDR normAddedQueryBias_;
    GM_ADDR normAddedKeyWeight_;
    GM_ADDR normAddedKeyBias_;
    GM_ADDR ropeSin_;
    GM_ADDR ropeCos_;
    GM_ADDR queryOutput_;
    GM_ADDR keyOutput_;
    GM_ADDR valueOutput_;
    GM_ADDR normQueryMean_;
    GM_ADDR normQueryRstd_;
    GM_ADDR normKeyMean_;
    GM_ADDR normKeyRstd_;
    GM_ADDR normAddedQueryMean_;
    GM_ADDR normAddedQueryRstd_;
    GM_ADDR normAddedKeyMean_;
    GM_ADDR normAddedKeyRstd_;

    TBuf<TPosition::VECCALC> buf_;
};

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder, bool isTraining>
__aicore__ inline void NormRopeConcat<normType, addedNormType, ropeType, concatOrder, isTraining>::Init(
    TPipe *pipe, const NormRopeConcatTilingData &tilingData)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    querySeq_ = tilingData.querySeq;
    keySeq_ = tilingData.keySeq;
    valueSeq_ = tilingData.valueSeq;
    encoderQuerySeq_ = tilingData.encoderQuerySeq;
    encoderKeySeq_ = tilingData.encoderKeySeq;
    encoderValueSeq_ = tilingData.encoderValueSeq;
    totalQuerySeq_ = tilingData.totalQuerySeq;
    totalKeySeq_ = tilingData.totalKeySeq;
    totalValueSeq_ = tilingData.totalValueSeq;
    batch_ = tilingData.batch;
    headNum_ = tilingData.headNum;
    headDim_ = tilingData.headDim;
    usedCore_ = tilingData.usedCore;
    splitHeadNum_ = tilingData.splitHeadNum;
    avgHeads_ = tilingData.avgHeads;
    tailHeads_ = tilingData.tailHeads;
    normDim_ = tilingData.normDim;
    ropeDim_ = tilingData.ropeDim;
    ropeActualSeq_ = tilingData.ropeActualSeq;
    eps_ = tilingData.eps;
    scale_ = tilingData.scale;

    alignedNormDim_ = CeilAlign(normDim_, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(DTYPE_QUERY)));
    alignedNormNum_ = CeilAlign(avgHeads_, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(float)));
    alignedRopeDim_ = CeilAlign(ropeDim_, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(DTYPE_QUERY)));

    uint32_t bufSize = NUM_TWO * avgHeads_ * alignedNormDim_ * sizeof(float) + MIN_SHARE_BUFFER;
    pipe_->InitBuffer(buf_, bufSize);
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder, bool isTraining>
__aicore__ inline void NormRopeConcat<normType, addedNormType, ropeType, concatOrder, isTraining>::GetSeqRange(
    uint32_t &startSeq, uint32_t &endSeq, uint32_t totalSeq, uint32_t usedCore)
{
    uint32_t eachCoreSeq = totalSeq / usedCore;
    uint32_t tailCoreSeq = totalSeq % usedCore;
    if (blockIdx_ < tailCoreSeq) {
        startSeq = blockIdx_ * (eachCoreSeq + 1);
        endSeq = startSeq + eachCoreSeq + 1;
    } else {
        startSeq = blockIdx_ * eachCoreSeq + tailCoreSeq;
        endSeq = startSeq + eachCoreSeq;
    }
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder, bool isTraining>
__aicore__ inline void NormRopeConcat<normType, addedNormType, ropeType, concatOrder, isTraining>::Process()
{
    RopeOperationForward<ropeType> ropeOp(pipe_, ropeSin_, ropeCos_, ropeActualSeq_, ropeDim_, avgHeads_,
                                          alignedRopeDim_);
    NormOperationForward<isTraining> normOp(pipe_, eps_, scale_, normDim_, avgHeads_, alignedNormDim_, alignedNormNum_);
    uint32_t outputSeq = 0;
    uint32_t outputEncoderSeq = querySeq_;
    if constexpr (concatOrder == ConcatOrder::AFTER_ENCODER) {
        outputSeq = encoderQuerySeq_;
        outputEncoderSeq = 0;
    }
    ProcessOne<normType, ropeType>(ropeOp, normOp, query_, normQueryWeight_, normQueryBias_, normQueryMean_,
                                   normQueryRstd_, queryOutput_, querySeq_, totalQuerySeq_, outputSeq);
    ProcessOne<addedNormType, ropeType>(ropeOp, normOp, encoderQuery_, normAddedQueryWeight_, normAddedQueryBias_,
                                        normAddedQueryMean_, normAddedQueryRstd_, queryOutput_, encoderQuerySeq_,
                                        totalQuerySeq_, outputEncoderSeq);
    outputSeq = 0;
    outputEncoderSeq = keySeq_;
    if constexpr (concatOrder == ConcatOrder::AFTER_ENCODER) {
        outputSeq = encoderKeySeq_;
        outputEncoderSeq = 0;
    }
    ProcessOne<normType, ropeType>(ropeOp, normOp, key_, normKeyWeight_, normKeyBias_, normKeyMean_, normKeyRstd_,
                                   keyOutput_, keySeq_, totalKeySeq_, outputSeq);
    ProcessOne<addedNormType, ropeType>(ropeOp, normOp, encoderKey_, normAddedKeyWeight_, normAddedKeyBias_,
                                        normAddedKeyMean_, normAddedKeyRstd_, keyOutput_, encoderKeySeq_, totalKeySeq_,
                                        outputEncoderSeq);
    outputSeq = 0;
    outputEncoderSeq = valueSeq_;
    if constexpr (concatOrder == ConcatOrder::AFTER_ENCODER) {
        outputSeq = encoderValueSeq_;
        outputEncoderSeq = 0;
    }
    ProcessOne<NormType::NONE, RopeType::NONE>(ropeOp, normOp, value_, nullptr, nullptr, nullptr, nullptr, valueOutput_,
                                               valueSeq_, totalValueSeq_, outputSeq);
    ProcessOne<NormType::NONE, RopeType::NONE>(ropeOp, normOp, encoderValue_, nullptr, nullptr, nullptr, nullptr,
                                               valueOutput_, encoderValueSeq_, totalValueSeq_, outputEncoderSeq);
}

template <NormType normType, NormType addedNormType, RopeType ropeType, ConcatOrder concatOrder, bool isTraining>
template <NormType actualNormType, RopeType actualRopeType>
__aicore__ inline void NormRopeConcat<normType, addedNormType, ropeType, concatOrder, isTraining>::ProcessOne(
    RopeOperationForward<ropeType> &ropeOp, NormOperationForward<isTraining> &normOp, GM_ADDR x, GM_ADDR weight,
    GM_ADDR bias, GM_ADDR mean, GM_ADDR rstd, GM_ADDR y, uint32_t xSeq, uint32_t totalSeq, uint32_t outputSeq)
{
    uint32_t startSeq{0}, endSeq{0};
    GetSeqRange(startSeq, endSeq, xSeq, usedCore_);
    if (startSeq < endSeq) {
        ropeOp.template Prepare<actualRopeType>(y, startSeq, totalSeq);
        normOp.template Prepare<actualNormType>(x, weight, bias, mean, rstd);
        for (uint32_t s = startSeq; s < endSeq; ++s) {
            ropeOp.template PreProcess<actualRopeType>(outputSeq + s);
            for (uint32_t b = 0; b < batch_; ++b) {
                uint64_t inOffset = static_cast<uint64_t>(b * xSeq + s) * headNum_ * headDim_;
                uint64_t normOffset = static_cast<uint64_t>(b * xSeq + s) * headNum_;
                uint64_t ropeOffset = static_cast<uint64_t>(b) * headNum_ * totalSeq * headDim_ +
                                      static_cast<uint64_t>(outputSeq + s) * headDim_;
                for (uint32_t n = 0; n < splitHeadNum_ - 1; ++n) {
                    LocalTensor<float> buf = buf_.Get<float>();
                    normOp.template Process<actualNormType>(buf, inOffset, normOffset, avgHeads_);
                    ropeOp.template Process<actualRopeType>(buf, ropeOffset, avgHeads_);
                    inOffset += static_cast<uint64_t>(avgHeads_) * headDim_;
                    normOffset += avgHeads_;
                    ropeOffset += static_cast<uint64_t>(avgHeads_) * totalSeq * headDim_;
                }
                // tail
                LocalTensor<float> buf = buf_.Get<float>();
                normOp.template Process<actualNormType>(buf, inOffset, normOffset, tailHeads_);
                ropeOp.template Process<actualRopeType>(buf, ropeOffset, tailHeads_);
                inOffset += static_cast<uint64_t>(tailHeads_) * headDim_;
                normOffset += tailHeads_;
            }
        }
    }
}
} // namespace nrc
#endif // _NORM_ROPE_CONCAT_H_

