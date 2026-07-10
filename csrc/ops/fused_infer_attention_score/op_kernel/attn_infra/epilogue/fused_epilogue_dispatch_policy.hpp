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

#ifndef FUSED_EPILOGUE_DISPATCH_POLICY_HPP
#define FUSED_EPILOGUE_DISPATCH_POLICY_HPP

#include "../../attn_infra/fused_base_defs.hpp"
#include "../../attn_infra/arch/fused_arch.hpp"

namespace NpuArch::Epilogue {

enum class LseMode { NONE = 0, OUT_ONLY = 1 };
enum class SinkMode { DISABLE = 0, ENABLE = 1 };
enum class MaskMode { NO_MASK = 0, MASK_CAUSAL = 1, MASK_SPEC = 2, MASK_SWA = 4 };
// For AtlasA2, FA Infer online Softmax
template <LseMode LSE_MODE_, SinkMode SINK_MODE_, MaskMode MASK_MODE_, typename SM_DTYPE_>
struct EpilogueAtlasA2OnlineSoftmax {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr SinkMode SINK_MODE = SINK_MODE_;
    static constexpr MaskMode MASK_MODE = MASK_MODE_;
};

// For AtlasA2, FA Infer RescaleO
template <LseMode LSE_MODE_, typename SM_DTYPE_> struct EpilogueAtlasA2RescaleO {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
};

// For AtlasA2, FA Infer Deal kv-len=0
template <LseMode LSE_MODE_> struct EpilogueAtlasA2InitOutWhenZero {
    using ArchTag = Arch::AtlasA2;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
};

} // namespace NpuArch::Epilogue

#endif // EPILOGUE_DISPATCH_POLICY_HPP
