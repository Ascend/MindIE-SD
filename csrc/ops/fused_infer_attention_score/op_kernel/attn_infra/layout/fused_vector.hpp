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

#ifndef FUSED_VECTOR_HPP
#define FUSED_VECTOR_HPP

#include "../../attn_infra/fused_base_defs.hpp"
#include "../../attn_infra/fused_coord.hpp"

namespace NpuArch::layout {

struct VectorLayout {
  public:
    /// Logical rank of tensor
    static constexpr int RANK = 1;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Shape vector
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

    /// Logical coordinate
    using TensorCoord = Coord<RANK, Index>;

  public:
    // Methods

    HOST_DEVICE
    VectorLayout(Index size = 0) : shape_(MakeCoord(size)), stride_(MakeCoord(LongIndex(1))) {}

    HOST_DEVICE
    VectorLayout(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element> HOST_DEVICE static VectorLayout MakeLayoutInUb(TensorCoord const &tileShape) {
        return VectorLayout{NpuArch::Detail::Alignment::RoundUp<BYTE_PER_BLK / sizeof(Element)>(tileShape[0])};
    }

    HOST_DEVICE
    LongIndex GetOffset(TensorCoord const &coord) const { return stride_[0] * coord[0]; }

    /// Returns the layout of a tile_common.
    HOST_DEVICE
    VectorLayout GetTileLayout(TensorCoord const &tileShape) const { return VectorLayout(tileShape, stride()); }

    /// Returns the shape of the layout
    HOST_DEVICE
    Shape &shape() { return shape_; }

    /// Returns the shape of the layout
    HOST_DEVICE
    Shape shape() const { return shape_; }

    /// Returns the shape of the layout
    HOST_DEVICE
    typename Shape::Index shape(int index) const { return shape_[index]; }

    /// Returns the shape of the layout
    HOST_DEVICE
    typename Shape::Index &shape(int index) { return shape_[index]; }

    /// Returns the stride of the layout
    HOST_DEVICE
    Stride stride() const { return stride_; }

    /// Returns the stride of the layout
    HOST_DEVICE
    Stride &stride() { return stride_; }

    /// Returns the stride of the layout
    HOST_DEVICE
    typename Stride::Index &stride(int index) { return stride_[index]; }

    /// Returns the stride of the layout
    HOST_DEVICE
    typename Stride::Index stride(int index) const { return stride_[index]; }

  private:
    /// Stride data member
    Shape shape_;
    Stride stride_;
};

} // namespace NpuArch::layout

#endif // LAYOUT_VECTOR_HPP
