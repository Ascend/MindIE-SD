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

#ifndef INDEXED_ADALN_MINDIE_SD_IMPL_H
#define INDEXED_ADALN_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>

at::Tensor indexed_adaln_mindie_sd_impl_npu(const at::Tensor &x, const at::Tensor &scale_table,
    const at::Tensor &shift_table, const at::Tensor &mod_index, const c10::optional<double> &epsilon);

#endif // INDEXED_ADALN_MINDIE_SD_IMPL_H
