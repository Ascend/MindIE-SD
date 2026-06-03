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

#ifndef OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_
#define OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantFlashAttn的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train_infer
 *
 * @param q                   [IN]  query tensor。数据类型FLOAT16/BFLOAT16，ND格式。
 *                                  layout由layoutQ决定，支持BSND/BNSD/TND。
 *                                  BSND: shape=(B, S_q, N_q, D)
 *                                  BNSD: shape=(B, N_q, S_q, D)
 *                                  TND:  shape=(T_q, N_q, D)，T_q为总token数
 * @param k                   [IN]  key tensor。数据类型与q一致，layout由layoutKv决定。
 *                                  BSND: shape=(B, S_kv, N_kv, D)
 *                                  TND:  shape=(T_kv, N_kv, D)
 *                                  PA_ND: shape=(NumBlocks, BlockSize, N_kv, D)，分页注意力格式
 *                                  PA_Nz: shape=(NumBlocks, N_kv, BlockSize/16, D, 16)，分页注意力NZ格式
 * @param v                   [IN]  value tensor。数据类型与q一致，layout由layoutKv决定，shape同k。
 * @param blockTableOptional  [IN]  可选。分页KV缓存的块映射表，数据类型INT32。
 *                                  shape=(B, MaxBlocksPerSeq)，与layoutKv=PA_ND/PA_Nz配合使用。
 * @param cuSeqlensQOptional  [IN]  可选。query的累积序列长度（前缀和），数据类型INT32。
 *                                  shape=(B+1,)，cu_seqlens_q[i+1]-cu_seqlens_q[i]为第i个sample的q序列长度。
 *                                  TND layout变长场景时有效。与seqused_q互斥。
 * @param cuSeqlensKvOptional [IN]  可选。kv的累积序列长度（前缀和），数据类型INT32。
 *                                  shape=(B+1,)，与cuSeqlensQOptional配合使用。
 * @param sequsedQOptional    [IN]  可选。各batch中query的实际序列长度，数据类型INT32。
 *                                  shape=(B,)，padded batch模式下有效。与cuSeqlensQOptional互斥。
 * @param sequsedKvOptional   [IN]  可选。各batch中kv的实际序列长度，数据类型INT32。
 *                                  shape=(B,)，与sequsedQOptional配合使用。
 * @param sinksOptional       [IN]  可选。可学习的sink注意力权重，数据类型FLOAT32。
 * @param attnMaskOptional    [IN]  可选。attnmask的参数，数据类型INT8。
 * @param metadataOptional    [IN]  可选。预计算的tiling切分方案，数据类型INT32，由上游算子传入。
 *                                  当该输入不为nullptr时，tiling侧跳过切分方案计算，直接使用该元数据。
 * @param softmaxScale        [IN]  ATTR可选。softmax缩放系数，对应老接口的scaleValue。
 *                                  数据类型DOUBLE。默认值0.0表示使用1/sqrt(D)。
 * @param maskMode            [IN]  ATTR可选。掩码模式，对应老接口的sparseMode。数据类型INT。
 *                                  0: 不使用掩码；1: 因果掩码（下三角）；2: 非因果掩码（上三角）；
 *                                  3: prefix/band掩码；4: 滑动窗口掩码（使用winLeft/winRight）。
 * @param winLeft             [IN]  ATTR可选。左侧注意力窗口大小（maskMode=4时的winLefts）。数据类型INT。
 * @param winRight            [IN]  ATTR可选。右侧注意力窗口大小（maskMode=4时的winRight）。数据类型INT。
 * @param layoutQ             [IN]  ATTR可选。query的数据布局，支持"BSND"/"BNSD"/"TND"（大小写不敏感）。
 * @param layoutKv            [IN]  ATTR可选。key/value的数据布局，支持"BSND"/"TND"/"PA_ND"/"PA_Nz"。
 * @param layoutOut           [IN]  ATTR可选。输出的数据布局，支持"BSND"/"BNSD"/"TND"（大小写不敏感）。
 * @param returnSoftmaxLse    [IN]  ATTR可选。是否输出softmax_lse。INT类型，1表示输出，0表示不输出。
 *                                  训练正向传播时置1，推理时置0。
 * @param deterministic       [IN]  ATTR可选。是否使用确定性计算（影响性能）。INT类型。
 * @param attnOut        [OUT] 必选输出。attention计算结果，数据类型与q一致，layout由layoutOut决定。
 * @param softmaxLseOptional  [OUT] 可选输出。softmax的log-sum-exp值，FLOAT32类型。
 *                                  returnSoftmaxLse=1时有效，shape取决于layout和序列长度：
 *                                  TND: (T_q, N_q) 或 (N_q, T_q)；其他layout: (B, N_q, S_q)。
 * @param workspaceSize       [OUT] workspace大小（字节数）。
 * @param executor            [OUT] op执行器句柄，供第二段接口使用。
 * @return aclnnStatus 执行状态。ACLNN_SUCCESS表示成功。
 */
aclnnStatus aclnnQuantFlashAttnGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qDescale, const aclTensor *kDescale, const aclTensor *vDescale,
    const aclTensor *blockTableOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, int32_t qQuantMode, int32_t kQuantMode,
    int32_t vQuantMode, int32_t quantBlockSizeQs, int32_t quantBlockSizeKs, int32_t quantBlockSizeVs,
    double softmaxScale, int32_t maskMode, int32_t winLeft, int32_t winRight, int32_t maxSeqlenQ, int32_t maxSeqlenKV,
    const char *layoutQ, const char *layoutKv, const char *layoutOut, int32_t softmaxPrecision,
    int32_t returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLseOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnQuantFlashAttn的第二段接口，用于执行计算。
 * @param workspace       [IN] 由第一段接口计算得到的workspace设备内存指针。
 * @param workspaceSize   [IN] workspace大小（字节数）。
 * @param executor        [IN] 第一段接口输出的op执行器句柄。
 * @param stream          [IN] 用于执行计算的acl stream。
 * @return aclnnStatus 执行状态。
 */
aclnnStatus aclnnQuantFlashAttn(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_QUANT_FLASH_ATTN_H_
