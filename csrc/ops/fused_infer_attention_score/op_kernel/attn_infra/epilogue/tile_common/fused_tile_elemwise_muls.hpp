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

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP
#define EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP

#include "../../../attn_infra/gemm/fused_helper.hpp"

namespace NpuArch::Epilogue::Tile {
template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_> struct TileElemWiseMuls {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    __aicore__ inline TileElemWiseMuls() {}

    __aicore__ inline void operator()(AscendC::LocalTensor<ElementCompute> dstLocal,
        AscendC::LocalTensor<ElementCompute> srcTensor, ElementCompute scalar) {
        AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH);
    }
};
}

#endif // EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP
