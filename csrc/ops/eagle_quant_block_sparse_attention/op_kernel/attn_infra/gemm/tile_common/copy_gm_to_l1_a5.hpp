/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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

#ifndef GEMM_TILE_COPY_GM_TO_L1_A5_HPP
#define GEMM_TILE_COPY_GM_TO_L1_A5_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/arch.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy_tla.hpp"
#include "../../../attn_infra/numeric_size.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Tile {

/// Partial specialization for CopyGmToL1, AtlasA5, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0) {
        static_assert(tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, while TensorDst must be L1 and zN");

        const uint32_t nValue = tla::get<0>(srcTensor.shape());
        const uint32_t dValue = tla::get<1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
        }
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value &&
        tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor) {
        static_assert(tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and zN, while TensorDst must be L1 and zN");

        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0) {
        static_assert(tla::detail::isColumnMajor<typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and ColumnMajor, "
            "while TensorDst must be L1 and nZ");

        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<0>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideCol = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
        }
        intriParams.dstNzC0Stride = dstOuterStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, nZ in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value &&
        tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor) {
        static_assert(tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and nZ, "
            "while TensorDst must be L1 and nZ");

        const uint32_t blockCount = tla::get<0, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<0, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

#if (__CCE_AICORE__ == 310)
/// Copy MX scale A from row-major GM to the zZ layout required by LoadData MX.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isMxScaleForRowMajorA<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>> {
    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor) {
        static_assert(tla::detail::isMxScaleForRowMajorA<float8_e8m0_t, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleForzZ<float8_e8m0_t, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "TensorSrc must be E8M0 GM row-major scale A and TensorDst must be E8M0 L1 zZ");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Dn2NzParams intriParams;
        intriParams.dnNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(cols);
        intriParams.dValue = rows;
        intriParams.srcDnMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideRow / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();
        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Copy MX scale A from column-major GM to zZ L1.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isMxScaleForColumnMajorA<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleForzZ<float8_e8m0_t, LayoutDst>::value>> {
    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor) {
        static_assert(tla::detail::isMxScaleForColumnMajorA<float8_e8m0_t, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleForzZ<float8_e8m0_t, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "TensorSrc must be E8M0 GM column-major scale A and TensorDst must be E8M0 L1 zZ");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;
        intriParams.ndNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(cols);
        intriParams.dValue = rows;
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideRow / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();
        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Copy MX scale B from row-major GM to nN L1.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isMxScaleForRowMajorB<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>> {
    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor) {
        static_assert(tla::detail::isMxScaleForRowMajorB<float8_e8m0_t, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleFornN<float8_e8m0_t, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "TensorSrc must be E8M0 GM row-major scale B and TensorDst must be E8M0 L1 nN");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;
        intriParams.ndNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(rows);
        intriParams.dValue = cols;
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideCol / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();
        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Copy MX scale B from column-major GM to the nN layout required by LoadData MX.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
    tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isMxScaleForColumnMajorB<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScaleFornN<float8_e8m0_t, LayoutDst>::value>> {
    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor) {
        static_assert(tla::detail::isMxScaleForColumnMajorB<float8_e8m0_t, typename TensorSrc::Layout>::value &&
                tla::detail::isMxScaleFornN<float8_e8m0_t, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "TensorSrc must be E8M0 GM column-major scale B and TensorDst must be E8M0 L1 nN");

        const uint32_t rows = tla::get<0>(srcTensor.originShape());
        const uint32_t cols = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Dn2NzParams intriParams;
        intriParams.dnNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(rows);
        intriParams.dValue = cols;
        intriParams.srcDnMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideCol / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();
        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};
#endif

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_GM_TO_L1_A5_HPP
