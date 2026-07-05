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
 * \file fia_tiling_info.cpp
 * \brief
 */

#include "fia_tiling_info.h"

namespace optiling {

std::string QuantModeToSerialString(FiaQuantMode fiaQuantMode) {
    switch (fiaQuantMode) {
    case FiaQuantMode::NO_QUANT:
        return "NO_QUANT";
    case FiaQuantMode::ANTI_QUANT:
        return "ANTI_QUANT";
    case FiaQuantMode::FULL_QUANT:
        return "FULL_QUANT";
    default:
        return "UNKNOWN";
    }
}

std::string SituationToSerialString(RopeMode ropeMode) {
    if (ropeMode == RopeMode::ROPE_SPLIT) {
        return "qkHeadDim = vHeadDim and rope exist";
    } else {
        return "rope not exist";
    }
}
} // namespace optiling
