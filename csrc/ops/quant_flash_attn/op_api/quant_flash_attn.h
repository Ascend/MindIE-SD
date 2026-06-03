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

#ifndef OP_API_INC_LEVEL0_FLASH_ATTN_H_
#define OP_API_INC_LEVEL0_FLASH_ATTN_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief FlashAttn level-0 operator。
 *        封装FlashAttn算子的底层调度，完成InferShape与Kernel Launch注册。
 *        该接口为内部接口，仅供aclnn层调用。
 *
 * @param q                   query tensor
 * @param k                   key tensor
 * @param v                   value tensor
 * @param blockTableOptional  分页KV缓存块映射表（可选，INT32）
 * @param cuSeqlensQOptional  query累积序列长度tensor（可选，INT32）
 * @param cuSeqlensKvOptional kv累积序列长度tensor（可选，INT32）
 * @param sequsedQOptional    query各batch实际序列长度tensor（可选，INT32）
 * @param sequsedKvOptional   kv各batch实际序列长度tensor（可选，INT32）
 * @param sinksOptional       可学习sink权重（可选，FLOAT32）
 * @param attnMaskOptional    attnMask参数（可选，INT8）
 * @param metadataOptional    预计算tiling元数据（可选，INT32）
 * @param softmaxScale         softmax缩放系数（float）
 * @param maskMode            掩码模式（int64_t）
 * @param winLeft             左窗口大小（int64_t）
 * @param winRight            右窗口大小（int64_t）
 * @param layoutQ             query布局字符串
 * @param layoutKv            kv布局字符串
 * @param layoutOut           输出布局字符串
 * @param returnSoftmaxLse    是否输出softmax_lse（int64_t）
 * @param deterministic       是否确定性计算（int64_t）
 * @param executor            op执行器
 * @return std::array<const aclTensor*, 2> [attnOut, softmaxLse]
 *         任意元素为nullptr表示对应输出的InferShape或Launch失败。
 */
const std::array<const aclTensor *, 2> FlashAttn(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *blockTableOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, double softmaxScale, int32_t maskMode,
    int32_t winLeft, int32_t winRight, const char *layoutQ, const char *layoutKv, const char *layoutOut,
    int32_t returnSoftmaxLse, int32_t deterministic, aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_FLASH_ATTN_H_
