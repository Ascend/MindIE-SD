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
 * \file vf_mxfp4_attenout_dn.h
 * \brief
 */

#ifndef ATTENOUT_DN_MXFP4_H_
#define ATTENOUT_DN_MXFP4_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false

template <uint16_t S1Base = 128>
__simd_vf__ void processOutVF(__ubuf__ float *attenOutMD, __ubuf__ float *attenOutDM, __ubuf__ float *rsum) {
    MaskReg preg_all32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    RegTensor<float> vreg_rrsum, vreg_ones;
    Duplicate(vreg_ones, (float)1.0);
    LoadAlign(vreg_rrsum, (__ubuf__ float *)rsum);
    Div(vreg_rrsum, vreg_ones, vreg_rrsum, preg_all32);

    for (uint16_t i = 0; i < S1Base / 16; i++) {
        RegTensor<float> vreg_src1, vreg_src2, vreg_src3, vreg_src4, vreg_src5, vreg_src6, vreg_src7, vreg_src8;
        RegTensor<float> vreg_src9, vreg_src10, vreg_src11, vreg_src12, vreg_src13, vreg_src14, vreg_src15, vreg_src16;
        RegTensor<bfloat16_t> vreg_bf16_1, vreg_bf16_2;

        LoadAlign(vreg_src1, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 0);
        LoadAlign(vreg_src2, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 1);
        LoadAlign(vreg_src3, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 2);
        LoadAlign(vreg_src4, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 3);
        LoadAlign(vreg_src5, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 4);
        LoadAlign(vreg_src6, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 5);
        LoadAlign(vreg_src7, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 6);
        LoadAlign(vreg_src8, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 7);
        LoadAlign(vreg_src9, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 8);
        LoadAlign(vreg_src10, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 9);
        LoadAlign(vreg_src11, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 10);
        LoadAlign(vreg_src12, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 11);
        LoadAlign(vreg_src13, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 12);
        LoadAlign(vreg_src14, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 13);
        LoadAlign(vreg_src15, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 14);
        LoadAlign(vreg_src16, (__ubuf__ float *)attenOutDM + i * 64 * 16 + 64 * 15);

        Mul(vreg_src1, vreg_src1, vreg_rrsum, preg_all32);
        Mul(vreg_src2, vreg_src2, vreg_rrsum, preg_all32);
        Mul(vreg_src3, vreg_src3, vreg_rrsum, preg_all32);
        Mul(vreg_src4, vreg_src4, vreg_rrsum, preg_all32);
        Mul(vreg_src5, vreg_src5, vreg_rrsum, preg_all32);
        Mul(vreg_src6, vreg_src6, vreg_rrsum, preg_all32);
        Mul(vreg_src7, vreg_src7, vreg_rrsum, preg_all32);
        Mul(vreg_src8, vreg_src8, vreg_rrsum, preg_all32);
        Mul(vreg_src9, vreg_src9, vreg_rrsum, preg_all32);
        Mul(vreg_src10, vreg_src10, vreg_rrsum, preg_all32);
        Mul(vreg_src11, vreg_src11, vreg_rrsum, preg_all32);
        Mul(vreg_src12, vreg_src12, vreg_rrsum, preg_all32);
        Mul(vreg_src13, vreg_src13, vreg_rrsum, preg_all32);
        Mul(vreg_src14, vreg_src14, vreg_rrsum, preg_all32);
        Mul(vreg_src15, vreg_src15, vreg_rrsum, preg_all32);
        Mul(vreg_src16, vreg_src16, vreg_rrsum, preg_all32);

        Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src1, preg_all32);
        Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src2, preg_all32);
        Or((RegTensor<uint16_t> &)vreg_src1, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
            preg_all16);

        Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src3, preg_all32);
        Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src4, preg_all32);
        Or((RegTensor<uint16_t> &)vreg_src2, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
            preg_all16);

        Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src5, preg_all32);
        Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src6, preg_all32);
        Or((RegTensor<uint16_t> &)vreg_src3, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
            preg_all16);

        Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src7, preg_all32);
        Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src8, preg_all32);
        Or((RegTensor<uint16_t> &)vreg_src4, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
            preg_all16);

        Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src9, preg_all32);
        Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src10, preg_all32);
        Or((RegTensor<uint16_t> &)vreg_src5, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
            preg_all16);

        Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src11, preg_all32);
        Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src12, preg_all32);
        Or((RegTensor<uint16_t> &)vreg_src6, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
            preg_all16);

        Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src13, preg_all32);
        Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src14, preg_all32);
        Or((RegTensor<uint16_t> &)vreg_src7, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
            preg_all16);

        Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src15, preg_all32);
        Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src16, preg_all32);
        Or((RegTensor<uint16_t> &)vreg_src8, (RegTensor<uint16_t> &)vreg_bf16_1, (RegTensor<uint16_t> &)vreg_bf16_2,
            preg_all16);

        Interleave(vreg_src1, vreg_src5, vreg_src1, vreg_src5);
        Interleave(vreg_src2, vreg_src6, vreg_src2, vreg_src6);
        Interleave(vreg_src3, vreg_src7, vreg_src3, vreg_src7);
        Interleave(vreg_src4, vreg_src8, vreg_src4, vreg_src8);

        Interleave(vreg_src1, vreg_src3, vreg_src1, vreg_src3);
        Interleave(vreg_src5, vreg_src7, vreg_src5, vreg_src7);
        Interleave(vreg_src2, vreg_src4, vreg_src2, vreg_src4);
        Interleave(vreg_src6, vreg_src8, vreg_src6, vreg_src8);

        Interleave(vreg_src1, vreg_src2, vreg_src1, vreg_src2);
        Interleave(vreg_src3, vreg_src4, vreg_src3, vreg_src4);
        Interleave(vreg_src5, vreg_src6, vreg_src5, vreg_src6);
        Interleave(vreg_src7, vreg_src8, vreg_src7, vreg_src8);

        StoreAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float *)attenOutMD + 8 * i + (8 + 1) * 8 * 8 * 0, (RegTensor<float> &)vreg_src1, 8 + 1,
            preg_all32);
        StoreAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float *)attenOutMD + 8 * i + (8 + 1) * 8 * 8 * 1, (RegTensor<float> &)vreg_src2, 8 + 1,
            preg_all32);
        StoreAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float *)attenOutMD + 8 * i + (8 + 1) * 8 * 8 * 2, (RegTensor<float> &)vreg_src3, 8 + 1,
            preg_all32);
        StoreAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float *)attenOutMD + 8 * i + (8 + 1) * 8 * 8 * 3, (RegTensor<float> &)vreg_src4, 8 + 1,
            preg_all32);
        StoreAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float *)attenOutMD + 8 * i + (8 + 1) * 8 * 8 * 4, (RegTensor<float> &)vreg_src5, 8 + 1,
            preg_all32);
        StoreAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float *)attenOutMD + 8 * i + (8 + 1) * 8 * 8 * 5, (RegTensor<float> &)vreg_src6, 8 + 1,
            preg_all32);
        StoreAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float *)attenOutMD + 8 * i + (8 + 1) * 8 * 8 * 6, (RegTensor<float> &)vreg_src7, 8 + 1,
            preg_all32);
        StoreAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            (__ubuf__ float *)attenOutMD + 8 * i + (8 + 1) * 8 * 8 * 7, (RegTensor<float> &)vreg_src8, 8 + 1,
            preg_all32);
    }
}

template <uint16_t S1Base = 128>
__aicore__ inline void processOut(
    const LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor, const LocalTensor<float> &rSum) {
    __ubuf__ float *attenOutMD = (__ubuf__ float *)dstTensor.GetPhyAddr();
    __ubuf__ float *attenOutDM = (__ubuf__ float *)srcTensor.GetPhyAddr();
    __ubuf__ float *rsum = (__ubuf__ float *)rSum.GetPhyAddr();

    processOutVF<S1Base>(attenOutMD, attenOutDM, rsum);
}

} // namespace Mxfp4Api
#endif // ATTENOUT_DN_MXFP4_H_
