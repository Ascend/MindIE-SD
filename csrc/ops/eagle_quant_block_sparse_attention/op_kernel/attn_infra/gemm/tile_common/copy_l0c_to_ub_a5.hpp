/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_L0C_TO_UB_A5_HPP
#define GEMM_TILE_COPY_L0C_TO_UB_A5_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/arch.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l0c_to_dst.hpp"
#include "../../../tla/tensor.hpp"

#if (__CCE_AICORE__ == 310)
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};
#endif

namespace NpuArch::Gemm::Tile {

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
    
    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, bool subBlockId, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity::PER_TENSOR,
    ReluEnable_,
    std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0f;

        __aicore__ inline
        Params() = default;

        __aicore__ inline
        Params(float scalar)
        {
            scale = scalar;
        }
    };
    Params params;

    __aicore__ inline
    CopyL0CToUBTla() = default;

    __aicore__ inline
    CopyL0CToUBTla(Params const &params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and zN"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams;

        intriParams.nSize = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        intriParams.mSize = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        intriParams.dstStride = intriParams.mSize * (BYTE_PER_C0 / sizeof(ElementDst));

        intriParams.quantPre = quantPre;
        intriParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(
        TensorDst const &dstTensor, TensorSrc const &srcTensor, bool subBlockId, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and zN"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams;

        intriParams.nSize = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        intriParams.mSize = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        intriParams.dstStride = intriParams.mSize * (BYTE_PER_C0 / sizeof(ElementDst));

        intriParams.quantPre = quantPre;
        intriParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and zN"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams;

        //shape = ((16, ceil_div(rows, 16)), (16, ceil_div(cols, 16)))
        //stride = ((16, 256), (1, round_up(rows, 16) * 16))
        // zN/NZ Fixpipe consumes the physical fractal extent stored in TLA nested shape.
        // 源NZ矩阵在N方向上的大小
        intriParams.nSize = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        // 源NZ矩阵在M方向上的大小
        intriParams.mSize = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        // 源NZ矩阵中的相邻Z排布的起始地址偏移，单位是C0_SIZE
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        // 目的NZ矩阵中相邻Z排布的起始地址偏移，单位是元素
        intriParams.dstStride = intriParams.mSize * (BYTE_PER_C0 / sizeof(ElementDst));

        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(
        TensorDst const &dstTensor, TensorSrc const &srcTensor, bool subBlockId, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and zN"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams;

        intriParams.nSize = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        intriParams.mSize = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        intriParams.dstStride = intriParams.mSize * (BYTE_PER_C0 / sizeof(ElementDst));

        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity::PER_TENSOR,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0f;

        __aicore__ inline
        Params() = default;

        __aicore__ inline
        Params(float scalar)
        {
            scale = scalar;
        }
    };
    Params params;

    __aicore__ inline
    CopyL0CToUBTla() = default;

    __aicore__ inline
    CopyL0CToUBTla(Params const &params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, bool subBlockId, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockId;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_M,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = RoundUp(tla::get<0>(dstTensor.shape()), 2); // m must be even when spilt m
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 1;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_M,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and zN"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams;

        intriParams.nSize = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        intriParams.mSize = RoundUp(tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape()), 2);
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        intriParams.dstStride = intriParams.mSize * (BYTE_PER_C0 / sizeof(ElementDst));

        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 1;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

// NOTE: SPLIT_M + PER_CHANNEL is NOT supported by single Fixpipe with dualDstCtl=1,
// because per-channel dequant (scale tensor) conflicts with dual-core split mode.
// Workaround: manually decompose into two independent Fixpipe calls (sub0 + sub1),
// each with a separate sub-tile and the same full scale tensor.
// This keeps the dequantization correct while achieving dual-core output.
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_M,
    ScaleGranularity::PER_CHANNEL,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    __aicore__ inline
    CopyL0CToUBTla() = default;

    __aicore__ inline
    CopyL0CToUBTla(Params const &params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor,
        AscendC::LocalTensor<uint64_t> const &scale, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        uint32_t mSize = tla::get<0>(dstTensor.shape());
        uint32_t nSize = tla::get<1>(dstTensor.shape());
        uint32_t mPerSubCore = mSize / 2;

        // --- prepare sub-tiles for dual sub-core fixpipe ---
        // CAUTION: L0C srcTensor has a nested fractal shape (tuple<tuple<C<16>, uint>, tuple<C<16>, uint>>).
        // Do NOT use tla::get<1>(srcTensor.shape()) as MakeShape argument here,
        // because it returns a nested tuple fragment, not a flat uint.
        // Always pass flat uint values (nSize) to MakeShape; MakeLayoutTile handles the fractal mapping internally.
        auto dstSub0 = tla::GetTile(dstTensor, tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, nSize));
        auto srcSub0 = tla::GetTile(srcTensor, tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, nSize));

        auto dstSub1 = tla::GetTile(dstTensor, tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, nSize));
        auto srcSub1 = tla::GetTile(srcTensor, tla::MakeCoord(mPerSubCore, 0), tla::MakeShape(mPerSubCore, nSize));

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams0;
        intriParams0.nSize = tla::get<1>(dstSub0.shape());
        intriParams0.mSize = tla::get<0>(dstSub0.shape());
        intriParams0.srcStride = tla::get<1, 1>(srcSub0.stride()) / tla::get<0, 0>(srcSub0.stride());
        intriParams0.dstStride = tla::get<0>(dstSub0.stride());
        intriParams0.quantPre = quantPre;
        intriParams0.reluEn = reluEn;
        intriParams0.unitFlag = unitFlag;
        intriParams0.dualDstCtl = 0;
        intriParams0.subBlockId = 0;

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams1;
        intriParams1.nSize = tla::get<1>(dstSub1.shape());
        intriParams1.mSize = tla::get<0>(dstSub1.shape());
        intriParams1.srcStride = tla::get<1, 1>(srcSub1.stride()) / tla::get<0, 0>(srcSub1.stride());
        intriParams1.dstStride = tla::get<0>(dstSub1.stride());
        intriParams1.quantPre = quantPre;
        intriParams1.reluEn = reluEn;
        intriParams1.unitFlag = unitFlag;
        intriParams1.dualDstCtl = 0;
        intriParams1.subBlockId = 1;

        // --- execute dual fixpipe for sub-core 0 and sub-core 1 ---
        auto dstOffset0 = dstSub0.layout()(dstSub0.coord());
        auto srcOffset0 = srcSub0.layout()(srcSub0.coord());
        auto dstOffset1 = dstSub1.layout()(dstSub1.coord());
        auto srcOffset1 = srcSub1.layout()(srcSub1.coord());
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstSub0.data()[dstOffset0], srcSub0.data()[srcOffset0], scale, intriParams0);
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstSub1.data()[dstOffset1], srcSub1.data()[srcOffset1], scale, intriParams1);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_M,
    ScaleGranularity::PER_CHANNEL,
    ReluEnable_,
    std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    __aicore__ inline
    CopyL0CToUBTla() = default;

    __aicore__ inline
    CopyL0CToUBTla(Params const &params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor,
        AscendC::LocalTensor<uint64_t> const &scale, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and zN"
        );

        uint32_t mSize = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        uint32_t nSize = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        uint32_t mPerSubCore = mSize / 2;

        auto dstSub0 = tla::GetTile(dstTensor, tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, nSize));
        auto srcSub0 = tla::GetTile(srcTensor, tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, nSize));

        auto dstSub1 = tla::GetTile(dstTensor, tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, nSize));
        auto srcSub1 = tla::GetTile(srcTensor, tla::MakeCoord(mPerSubCore, 0), tla::MakeShape(mPerSubCore, nSize));

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams0;
        intriParams0.nSize = tla::get<1, 0>(dstSub0.shape()) * tla::get<1, 1>(dstSub0.shape());
        intriParams0.mSize = tla::get<0, 0>(dstSub0.shape()) * tla::get<0, 1>(dstSub0.shape());
        intriParams0.srcStride = tla::get<1, 1>(srcSub0.stride()) / tla::get<1, 0>(srcSub0.shape());
        intriParams0.dstStride = intriParams0.mSize * (BYTE_PER_C0 / sizeof(ElementDst));
        intriParams0.quantPre = quantPre;
        intriParams0.reluEn = reluEn;
        intriParams0.unitFlag = unitFlag;
        intriParams0.dualDstCtl = 0;
        intriParams0.subBlockId = 0;

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> intriParams1;
        intriParams1.nSize = tla::get<1, 0>(dstSub1.shape()) * tla::get<1, 1>(dstSub1.shape());
        intriParams1.mSize = tla::get<0, 0>(dstSub1.shape()) * tla::get<0, 1>(dstSub1.shape());
        intriParams1.srcStride = tla::get<1, 1>(srcSub1.stride()) / tla::get<1, 0>(srcSub1.shape());
        intriParams1.dstStride = intriParams1.mSize * (BYTE_PER_C0 / sizeof(ElementDst));
        intriParams1.quantPre = quantPre;
        intriParams1.reluEn = reluEn;
        intriParams1.unitFlag = unitFlag;
        intriParams1.dualDstCtl = 0;
        intriParams1.subBlockId = 1;

        auto dstOffset0 = dstSub0.layout()(dstSub0.coord());
        auto srcOffset0 = srcSub0.layout()(srcSub0.coord());
        auto dstOffset1 = dstSub1.layout()(dstSub1.coord());
        auto srcOffset1 = srcSub1.layout()(srcSub1.coord());
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_UB>(
            dstSub0.data()[dstOffset0], srcSub0.data()[srcOffset0], scale, intriParams0);
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_NZ_UB>(
            dstSub1.data()[dstOffset1], srcSub1.data()[srcOffset1], scale, intriParams1);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_N,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = RoundUp(tla::get<1>(dstTensor.shape()), 32);
        intriParams.mSize = tla::get<0>(dstTensor.shape()); // m must be even when spilt m
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 2;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity::PER_CHANNEL,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    __aicore__ inline
    CopyL0CToUBTla() = default;

    __aicore__ inline
    CopyL0CToUBTla(Params const &params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor,
        AscendC::LocalTensor<uint64_t> const &scale, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe with scale tensor for per-channel quant
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], scale, intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L0C_TO_UB_A5_HPP
