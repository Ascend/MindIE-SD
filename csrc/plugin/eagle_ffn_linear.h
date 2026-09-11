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

#ifndef EAGLE_FFN_LINEAR_MINDIE_SD_IMPL_H
#define EAGLE_FFN_LINEAR_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <optional>

at::Tensor eagle_ffn_linear_mindie_sd_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight1,
    const at::Tensor &weight2,
    const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &bias2,
    c10::string_view activation,
    int64_t inner_precise);

#endif // EAGLE_FFN_LINEAR_MINDIE_SD_IMPL_H
