/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

#ifndef TLA_LAYOUT_HPP
#define TLA_LAYOUT_HPP

#include "../attn_infra/base_defs.hpp"
#include "../attn_infra/numeric_size.hpp"
#include "../tla/numeric/integral_constant.hpp"
#include "../tla/tuple.hpp"
#include "../tla/int_tuple.hpp"
#include "../attn_infra/layout/layout.hpp"

namespace tla {

// Aliases

template <class... Shapes> using Shape = tla::tuple<Shapes...>;

template <class... Strides> using Stride = tla::tuple<Strides...>;

template <class... Coords> using Coord = tla::tuple<Coords...>;

template <class... Ts> HOST_DEVICE constexpr Shape<Ts...> MakeShape(Ts const &...t) { return {t...}; }
template <class... Ts> HOST_DEVICE constexpr Stride<Ts...> MakeStride(Ts const &...t) { return {t...}; }
template <class... Ts> HOST_DEVICE constexpr Coord<Ts...> MakeCoord(Ts const &...t) { return {t...}; }

//
// Layout
//

namespace detail {

template <int Rank, class Sequence = void> struct MakeOriginShapeTypeImpl;

template <int Rank, size_t... Is> struct MakeOriginShapeTypeImpl<Rank, tla::index_sequence<Is...>> {
    template <size_t> using repeat_type = uint32_t;
    using type = Shape<repeat_type<Is>...>;
};

template <class Stride>
using MakeOriginShapeType =
    typename MakeOriginShapeTypeImpl<rank_v<Stride>, tla::make_index_sequence<rank_v<Stride>>>::type;

struct UnpackedMakeOriginShapeU32 {
    template <class... T> HOST_DEVICE constexpr auto operator()(T const &...a) const {
        return MakeShape(static_cast<uint32_t>(a)...);
    }
};

} // namespace detail

template <class Shape, class Stride, class OriginShape = detail::MakeOriginShapeType<Stride>>
struct Layout : private tla::tuple<Shape, Stride, OriginShape> {
    // NOTE: This defaults static Shapes/Strides correctly, but not dynamic
    HOST_DEVICE constexpr Layout(
        Shape const &shape = {}, Stride const &stride = {}, OriginShape const &originShape = {})
        : tla::tuple<Shape, Stride, OriginShape>(shape, stride, originShape) {}

    //
    // Accessors
    //

    static constexpr int rank = rank_v<Stride>;
    static constexpr int depth = depth_v<Stride>;

    template <int... I> HOST_DEVICE constexpr decltype(auto) shape() {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> &>(*this));
    }

    template <int... I> HOST_DEVICE constexpr decltype(auto) shape() const {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const &>(*this));
    }

    template <int... I> HOST_DEVICE constexpr decltype(auto) stride() {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> &>(*this));
    }

    template <int... I> HOST_DEVICE constexpr decltype(auto) stride() const {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const &>(*this));
    }

    template <int... I> HOST_DEVICE constexpr decltype(auto) originShape() {
        return get<2, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> &>(*this));
    }

    template <int... I> HOST_DEVICE constexpr decltype(auto) originShape() const {
        return get<2, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const &>(*this));
    }

    template <class Coord> HOST_DEVICE constexpr auto operator()(Coord const &coord) const {
        return crd2offset(coord, shape(), stride());
    }
};

// Layout construction

template <class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr auto MakeLayout(Shape const &shape, Stride const &stride, OriginShape const &originShape) {
    static_assert(is_tuple<Shape>::value || is_integral<Shape>::value);
    static_assert(is_tuple<Stride>::value || is_integral<Stride>::value);
    static_assert(is_tuple<OriginShape>::value || is_integral<OriginShape>::value);
    return Layout<Shape, Stride, OriginShape>(shape, stride, originShape);
}

template <class Shape, class Stride> HOST_DEVICE constexpr auto MakeLayout(Shape const &shape, Stride const &stride) {
    return MakeLayout(shape, stride, tla::transform_apply(shape, Product{}, detail::UnpackedMakeOriginShapeU32{}));
}

// Convenience tags for common layouts

template <class LayoutTag> HOST_DEVICE constexpr auto MakeLayoutFromTag(LayoutTag const &tag) {
    static_assert(std::is_same_v<LayoutTag, NpuArch::layout::RowMajor> ||
            std::is_same_v<LayoutTag, NpuArch::layout::ColumnMajor> || std::is_same_v<LayoutTag, NpuArch::layout::zN> ||
            std::is_same_v<LayoutTag, NpuArch::layout::nZ>,
        "Unsupported LayoutTag for MakeLayoutFromTag, only support NpuArch::layout::RowMajor or"
        "NpuArch::layout::ColumnMajor or NpuArch::layout::zN or NpuArch::layout::nZ");

    if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::RowMajor>) {
        return MakeLayout(MakeShape(tag.shape(0), tag.shape(1)), MakeStride(tag.stride(0), Int<1>{}));
    } else if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::ColumnMajor>) {
        return MakeLayout(MakeShape(tag.shape(0), tag.shape(1)), MakeStride(Int<1>{}, tag.stride(1)));
    } else { // zN or nZ
        return MakeLayout(MakeShape(MakeShape(tag.shape(0), tag.shape(1)), MakeShape(tag.shape(2), tag.shape(3))),
            MakeStride(MakeStride(tag.stride(0), tag.stride(1)), MakeStride(tag.stride(2), tag.stride(3))));
    }
}

// Return the shape of a mode
template <int... Is, class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride, OriginShape> &layout) {
    return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride, OriginShape> const &layout) {
    return layout.template shape<Is...>();
}

// Return the stride of a mode
template <int... Is, class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr decltype(auto) stride(Layout<Shape, Stride, OriginShape> &layout) {
    return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr decltype(auto) stride(Layout<Shape, Stride, OriginShape> const &layout) {
    return layout.template stride<Is...>();
}

// Return the rank of layout
template <int... Is, class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr decltype(auto) originShape(Layout<Shape, Stride, OriginShape> &layout) {
    return layout.template originShape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr decltype(auto) originShape(Layout<Shape, Stride, OriginShape> const &layout) {
    return layout.template originShape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr auto rank(Layout<Shape, Stride, OriginShape> const &layout) {
    return rank(shape<Is...>(layout));
}

// Return the depth of the layout
template <int... Is, class Shape, class Stride, class OriginShape>
HOST_DEVICE constexpr auto depth(Layout<Shape, Stride, OriginShape> const &layout) {
    return depth(shape<Is...>(layout));
}

// Return the offset of coord
template <class Coord, class Shape, class Stride>
HOST_DEVICE constexpr auto crd2offset(Coord const &coord, Shape const &shape, Stride const &stride);

namespace detail {

template <class Coord, class Shape, class Stride, int... Is>
HOST_DEVICE constexpr auto crd2offset_ttt(Coord const &coord, Shape const &shape, Stride const &stride, seq<Is...>) {
    return (... + crd2offset(get<Is>(coord), get<Is>(shape), get<Is>(stride)));
}

template <class CInt, class STuple, class DTuple, int I0, int... Is>
HOST_DEVICE constexpr auto crd2offset_itt(
    CInt const &coord, STuple const &shape, DTuple const &stride, seq<I0, Is...>) {
    if constexpr (sizeof...(Is) == 0) { // Avoid recursion and mod on single/last iter
        return crd2offset(coord, get<I0>(shape), get<I0>(stride));
    } else if constexpr (is_constant<0, CInt>::value) {
        return crd2offset(_0{}, get<I0>(shape), get<I0>(stride)) +
            (_0{} + ... + crd2offset(_0{}, get<Is>(shape), get<Is>(stride)));
    } else { // General case
        return crd2offset(coord % Product{}(get<I0>(shape)), get<I0>(shape), get<I0>(stride)) +
            crd2offset_itt(coord / Product{}(get<I0>(shape)), shape, stride, seq<Is...>{});
    }
}

} // end namespace detail

template <class Coord, class Shape, class Stride>
HOST_DEVICE constexpr auto crd2offset(Coord const &coord, Shape const &shape, Stride const &stride) {
    if constexpr (is_tuple<Coord>::value) {
        if constexpr (is_tuple<Shape>::value) { // tuple tuple tuple
            static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value, "Mismatched Ranks");
            static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value, "Mismatched Ranks");
            return detail::crd2offset_ttt(coord, shape, stride, tuple_seq<Coord>{});
        } else { // tuple "int" "int"
            static_assert(sizeof(Coord) == 0, "Invalid parameters");
        }
    } else {
        if constexpr (is_tuple<Shape>::value) { // "int" tuple tuple
            static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
            return detail::crd2offset_itt(coord, shape, stride, tuple_seq<Shape>{});
        } else { // "int" "int" "int"
            return coord * stride;
        }
    }
}

template <class Layout> struct is_layout : false_type {};
template <class Shape, class Stride, class OriginShape>
struct is_layout<Layout<Shape, Stride, OriginShape>> : true_type {};

// Layout Check
namespace detail {

template <class Layout, class Enable = void> struct isVector {
    static bool const value = false;
};

template <class Layout> struct isVector<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 1>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Layout, class Enable = void> struct isRowMajor {
    static bool const value = false;
};

template <class Layout> struct isRowMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<1>(Layout{}) == 1);
};

template <class Layout, class Enable = void> struct isColumnMajor {
    static bool const value = false;
};

template <class Layout> struct isColumnMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void> struct iszN {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszN<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        NpuArch::BytesToBits(NpuArch::BYTE_PER_C0) / NpuArch::SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        NpuArch::BytesToBits(NpuArch::BYTE_PER_FRACTAL) / NpuArch::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == NpuArch::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
            stride<1, 0>(Layout{}) == 1 && stride<0, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void> struct iszZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszZ<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        NpuArch::BytesToBits(NpuArch::BYTE_PER_C0) / NpuArch::SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        NpuArch::BytesToBits(NpuArch::BYTE_PER_FRACTAL) / NpuArch::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == NpuArch::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
            stride<1, 0>(Layout{}) == 1 && stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void> struct isnZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct isnZ<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        NpuArch::BytesToBits(NpuArch::BYTE_PER_C0) / NpuArch::SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        NpuArch::BytesToBits(NpuArch::BYTE_PER_FRACTAL) / NpuArch::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == ELE_NUM_PER_C0 && shape<1, 0>(Layout{}) == NpuArch::C0_NUM_PER_FRACTAL &&
            stride<0, 0>(Layout{}) == 1 && stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

#if (__CCE_AICORE__ == 310)
template <class Element, class Layout, class Enable1 = void, class Enable2 = void> struct isMxScaleForRowMajorA {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForRowMajorA<AscendC::fp8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 1 && rank_v<decltype(stride<1>(Layout{}))> == 2 &&
        !is_constant<2, decltype(stride<0>(Layout{}))>::value &&
        ((rank_v<decltype(shape<0>(Layout{}))> == 1 && rank_v<decltype(shape<1>(Layout{}))> == 2) ||
            (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void> struct isMxScaleForColumnMajorA {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForColumnMajorA<AscendC::fp8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 1 && rank_v<decltype(stride<1>(Layout{}))> == 2 &&
        is_constant<2, decltype(stride<0>(Layout{}))>::value &&
        ((rank_v<decltype(shape<0>(Layout{}))> == 1 && rank_v<decltype(shape<1>(Layout{}))> == 2) ||
            (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void, class Enable3 = void>
struct isMxScaleForRowMajorB {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForRowMajorB<AscendC::fp8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 1>,
    std::enable_if_t<!is_constant<2, decltype(stride<0, 1>(Layout{}))>::value &&
        ((rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 1) ||
            (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void, class Enable3 = void>
struct isMxScaleForColumnMajorB {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForColumnMajorB<AscendC::fp8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 1>,
    std::enable_if_t<is_constant<2, decltype(stride<0, 1>(Layout{}))>::value &&
        ((rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 1) ||
            (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void> struct isMxScaleForzZ {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForzZ<AscendC::fp8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
        rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void> struct isMxScaleFornN {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleFornN<AscendC::fp8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
        rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static bool const value = true;
};
#endif

} // end namespace detail

// Advanced Layout constructions
// Make a vector layout.
template <class T> HOST_DEVICE constexpr auto MakeLayout(T const &len) {
    return MakeLayout(MakeShape(len), MakeStride(Int<1>{}), MakeShape(len));
}

// Make a inner layout with Rows and Cols.
template <class Element, class LayoutTag, class T, class U>
HOST_DEVICE constexpr auto MakeLayout(T const &rows, U const &cols) {
    static_assert(std::is_same_v<LayoutTag, NpuArch::layout::RowMajor> ||
            std::is_same_v<LayoutTag, NpuArch::layout::ColumnMajor> ||
            std::is_same_v<LayoutTag, NpuArch::layout::VectorLayout> ||
            std::is_same_v<LayoutTag, NpuArch::layout::zN> || std::is_same_v<LayoutTag, NpuArch::layout::nZ> ||
            std::is_same_v<LayoutTag, NpuArch::layout::zZ>,
        "Unsupported LayoutTag for MakeLayoutFromTag, only support NpuArch::layout::RowMajor or"
        "NpuArch::layout::ColumnMajor or NpuArch::layout::zN or NpuArch::layout::nZ or NpuArch::layout::zZ");

    constexpr uint32_t ELE_NUM_PER_C0 =
        NpuArch::BytesToBits(NpuArch::BYTE_PER_C0) / NpuArch::SizeOfBits<Element>::value;
    constexpr uint32_t ELE_NUM_PER_FRACTAL =
        NpuArch::BytesToBits(NpuArch::BYTE_PER_FRACTAL) / NpuArch::SizeOfBits<Element>::value;

    if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::VectorLayout>) {
        return MakeLayout(MakeShape(cols), MakeStride(Int<1>{}), MakeShape(cols));
    } else if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::RowMajor>) {
#if (__CCE_AICORE__ == 310)
        if constexpr (std::is_same_v<Element, fp4x2_e2m1_t> || std::is_same_v<Element, fp4x2_e1m2_t>) {
            return MakeLayout(
                MakeShape(rows, cols), MakeStride((int64_t)RoundUp(cols, 2), Int<1>{}), MakeShape(rows, cols));
        }
#endif
        return MakeLayout(MakeShape(rows, cols), MakeStride((int64_t)cols, Int<1>{}), MakeShape(rows, cols));
    } else if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::ColumnMajor>) {
#if (__CCE_AICORE__ == 310)
        if constexpr (std::is_same_v<Element, fp4x2_e2m1_t> || std::is_same_v<Element, fp4x2_e1m2_t>) {
            return MakeLayout(
                MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)RoundUp(rows, 2)), MakeShape(rows, cols));
        }
#endif
        return MakeLayout(MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)rows), MakeShape(rows, cols));
    } else if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::zN>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<NpuArch::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<NpuArch::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<NpuArch::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0)),
            MakeShape(rows, cols));
    } else if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::zZ>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<NpuArch::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<NpuArch::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{},
                           RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * NpuArch::C0_NUM_PER_FRACTAL),
                MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else {
        return MakeLayout(
            MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(Int<NpuArch::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<NpuArch::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<1>{}, RoundUp((int64_t)cols, Int<NpuArch::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0),
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    }
}

// Make a MxScale layout with Rows and Cols.
template <class Element, class LayoutTag, bool isMxScaleB, class T, class U>
HOST_DEVICE constexpr auto MakeMxScaleLayout(T const &rows, U const &cols) {
    static_assert(std::is_same_v<Element, AscendC::fp8_e8m0_t> &&
            (std::is_same_v<LayoutTag, NpuArch::layout::RowMajor> ||
                std::is_same_v<LayoutTag, NpuArch::layout::ColumnMajor> ||
                std::is_same_v<LayoutTag, NpuArch::layout::zZ> || std::is_same_v<LayoutTag, NpuArch::layout::nN>),
        "only support RowMajor, ColumnMajor, zZ, nN in fp8_e8m0_t dtype");

    constexpr uint32_t ELE_NUM_PER_C0 = 2;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 32;

    if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::RowMajor>) {
        if constexpr (!isMxScaleB) {
            return MakeLayout(MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                MakeStride(RoundUp(cols, Int<ELE_NUM_PER_C0>{}), MakeStride(Int<1>{}, Int<ELE_NUM_PER_C0>{})),
                MakeShape(rows, cols));
        } else {
            return MakeLayout(MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                MakeStride(MakeStride(Int<1>{}, cols * ELE_NUM_PER_C0), Int<ELE_NUM_PER_C0>{}), MakeShape(rows, cols));
        }
    } else if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::ColumnMajor>) {
        if constexpr (!isMxScaleB) {
            return MakeLayout(MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                MakeStride(Int<ELE_NUM_PER_C0>{}, MakeStride(Int<1>{}, rows * ELE_NUM_PER_C0)), MakeShape(rows, cols));
        } else {
            return MakeLayout(MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                MakeStride(MakeStride(Int<1>{}, Int<ELE_NUM_PER_C0>{}), RoundUp(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(rows, cols));
        }
    } else if constexpr (std::is_same_v<LayoutTag, NpuArch::layout::zZ>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<NpuArch::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<NpuArch::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{},
                           RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * NpuArch::C0_NUM_PER_FRACTAL),
                MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else {
        return MakeLayout(
            MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(Int<NpuArch::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<NpuArch::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(Int<ELE_NUM_PER_C0>{},
                    RoundUp((int64_t)rows, Int<ELE_NUM_PER_C0>{}) * NpuArch::C0_NUM_PER_FRACTAL)),
            MakeShape(rows, cols));
    }
}

namespace detail {

template <class OriginBase, class TileShape, class Coord, int... Is>
HOST_DEVICE constexpr auto CropOriginShape(
    OriginBase const &originBase, TileShape const &tileShape, Coord const &coord, seq<Is...>) {
    return MakeShape(tla::min(static_cast<uint32_t>(get<Is>(tileShape)),
        (static_cast<uint32_t>(get<Is>(coord)) < static_cast<uint32_t>(get<Is>(originBase)))
            ? (static_cast<uint32_t>(get<Is>(originBase)) - static_cast<uint32_t>(get<Is>(coord)))
            : 0u)...);
}

} // namespace detail

template <class Layout, class TileShape, class Coord>
HOST_DEVICE constexpr auto GetTileLayout(Layout const &layout, TileShape const &tileShape, Coord const &coord) {
    static_assert(is_tuple<TileShape>::value && depth_v<TileShape> == 1 && rank_v<TileShape> >= 1);
    static_assert(is_tuple<Coord>::value && depth_v<Coord> == 1 && rank_v<Coord> == rank_v<TileShape>);

    auto tileOriginShape = detail::CropOriginShape(layout.originShape(), tileShape, coord, tuple_seq<TileShape>{});

    if constexpr (Layout::depth == 1) {
        static_assert(Layout::rank == rank_v<TileShape>);
        return MakeLayout(tileShape, layout.stride(), tileOriginShape);
    } else if constexpr (Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 1 &&
        rank_v<decltype(shape<1>(Layout{}))> == 2) {
        const uint32_t rows = get<0>(tileShape);
        const uint32_t cols = get<1>(tileShape);
        constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<1, 0>(layout))::value;
        return MakeLayout(MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            layout.stride(), tileOriginShape);
    } else if constexpr (Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 2 &&
        rank_v<decltype(shape<1>(Layout{}))> == 1) {
        const uint32_t rows = get<0>(tileShape);
        const uint32_t cols = get<1>(tileShape);
        constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<0, 0>(layout))::value;
        return MakeLayout(MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
            layout.stride(), tileOriginShape);
    } else if constexpr (is_static<decltype(shape<0, 0>(layout))>::value &&
        is_static<decltype(shape<1, 0>(layout))>::value) {
        const uint32_t rows = get<0>(tileShape);
        const uint32_t cols = get<1>(tileShape);
        constexpr uint32_t dstInnerShapeRow = decltype(shape<0, 0>(layout))::value;
        constexpr uint32_t dstInnerShapeCol = decltype(shape<1, 0>(layout))::value;
        return MakeLayout(MakeShape(MakeShape(Int<dstInnerShapeRow>{}, CeilDiv<dstInnerShapeRow>(rows)),
                              MakeShape(Int<dstInnerShapeCol>{}, CeilDiv<dstInnerShapeCol>(cols))),
            layout.stride(), tileOriginShape);
    } else {
        const uint32_t rows = get<0>(tileShape);
        const uint32_t cols = get<1>(tileShape);
        const uint32_t dstInnerShapeRow = shape<0, 0>(layout);
        const uint32_t dstInnerShapeCol = shape<1, 0>(layout);
        return MakeLayout(MakeShape(MakeShape(dstInnerShapeRow, CeilDiv(rows, dstInnerShapeRow)),
                              MakeShape(dstInnerShapeCol, CeilDiv(cols, dstInnerShapeCol))),
            layout.stride(), tileOriginShape);
    }
}

template <class T, class U> HOST_DEVICE constexpr auto MakeLayoutL0C(T const &rows, U const &cols) {
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
    return MakeLayout(
        MakeShape(MakeShape(Int<NpuArch::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<NpuArch::C0_NUM_PER_FRACTAL>{})),
            MakeShape(Int<NpuArch::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<NpuArch::C0_NUM_PER_FRACTAL>{}))),
        MakeStride(MakeStride(Int<NpuArch::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
            MakeStride(
                Int<1>{}, RoundUp((int64_t)rows, Int<NpuArch::C0_NUM_PER_FRACTAL>{}) * NpuArch::C0_NUM_PER_FRACTAL)),
        MakeShape(rows, cols));
}

} // end namespace tla

#endif // TLA_LAYOUT_HPP
