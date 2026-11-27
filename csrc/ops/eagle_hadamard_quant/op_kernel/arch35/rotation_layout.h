/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 */
#pragma once

#include <cstdint>

namespace RotateQuantAptOpt {
// Keep these values stable: rotationIsNz occupies one byte in the CANN tiling ABI.
enum RotationLayout : uint8_t {
    ROTATION_ND = 0,
    ROTATION_PACKED = 1, // CANN: [B,rowTile,columnTile,row,column].
    ROTATION_L1_NZ = 2, // MindIE: [B,columnTile,rowTile,row,column].
};
} // namespace RotateQuantAptOpt
