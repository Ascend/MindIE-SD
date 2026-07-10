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

#ifndef FUSED_BLOCK_EPILOGUE_HPP
#define FUSED_BLOCK_EPILOGUE_HPP

#include "../../../attn_infra/fused_base_defs.hpp"

namespace NpuArch::Epilogue::Block {

template <class DispatchPolicy, class... Args> class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

} // namespace NpuArch::Epilogue::Block

#include "../../../attn_infra/epilogue/block/fused_block_epilogue_online_softmax.hpp"
#include "../../../attn_infra/epilogue/block/fused_block_epilogue_online_softmax_low_prec.hpp"
#include "../../../attn_infra/epilogue/block/fused_block_epilogue_rescale_o.hpp"
#include "../../../attn_infra/epilogue/block/CombineScale.hpp"
#include "../../../attn_infra/epilogue/block/fused_block_epilogue_rescale_o_low_prec.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_init_outputs.hpp"
#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
