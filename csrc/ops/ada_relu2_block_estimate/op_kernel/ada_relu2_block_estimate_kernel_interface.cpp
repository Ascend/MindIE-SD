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

#include "ada_relu2_block_estimate_regular.h"
#include "ada_relu2_block_estimate_int8_quant.h"

template <typename ElementQ, typename ElementK, typename ElementS>
__aicore__ inline void AdaRelu2Regular(GM_ADDR query, GM_ADDR key, GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKV,
    GM_ADDR qScale, GM_ADDR kScale, GM_ADDR score, GM_ADDR rowSum, GM_ADDR workspace, GM_ADDR tiling) {
    AdaRelu2Params params{
        query, key, actualSeqLengths, actualSeqLengthsKV, qScale, kScale, score, rowSum, workspace, tiling};
    AdaRelu2RegularKernel<ElementQ, ElementK, ElementS> op;
    op.Init(params);
    op.Process();
}

template <typename ElementQ, typename ElementK, typename ElementS>
__aicore__ inline void AdaRelu2Int8Quant(GM_ADDR query, GM_ADDR key, GM_ADDR actualSeqLengths,
    GM_ADDR actualSeqLengthsKV, GM_ADDR qScale, GM_ADDR kScale, GM_ADDR score, GM_ADDR rowSum, GM_ADDR workspace,
    GM_ADDR tiling) {
    AdaRelu2Params params{
        query, key, actualSeqLengths, actualSeqLengthsKV, qScale, kScale, score, rowSum, workspace, tiling};
    AdaRelu2Int8QuantKernel<ElementQ, ElementK, ElementS> op;
    op.Init(params);
    op.Process();
}
