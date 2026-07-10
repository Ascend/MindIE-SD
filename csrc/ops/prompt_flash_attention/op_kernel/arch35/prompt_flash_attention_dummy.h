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
 * \file prompt_flash_attention_dummy.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_DUMMY_H
#define PROMPT_FLASH_ATTENTION_DUMMY_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

template <typename T> class PromptFlashAttentionDummy {
  public:
    __aicore__ inline PromptFlashAttentionDummy(){};
    __aicore__ inline void Init(
        __gm__ uint8_t *attentionOut, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling);
    __aicore__ inline void Process();

  protected:
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    GlobalTensor<T> attentionOutGm;
};

template <typename T>
__aicore__ inline void PromptFlashAttentionDummy<T>::Init(
    __gm__ uint8_t *attentionOut, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling) {
    attentionOutGm.SetGlobalBuffer((__gm__ T *)attentionOut);
    tilingData = tiling;
}

template <typename T> __aicore__ inline void PromptFlashAttentionDummy<T>::Process() {
    uint32_t blockIdx = GetBlockIdx();
}
#endif // PROMPT_FLASH_ATTENTION_DUMMY_H
