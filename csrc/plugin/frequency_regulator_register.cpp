/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <torch/library.h>

#include "frequency_regulator.h"

TORCH_LIBRARY_FRAGMENT(mindiesd, m) { m.def("frequency_regulator(int freq) -> Tensor"); }

TORCH_LIBRARY_IMPL(mindiesd, PrivateUse1, m) { m.impl("frequency_regulator", &frequency_regulator_impl_npu); }

TORCH_LIBRARY_IMPL(mindiesd, BackendSelect, m) { m.impl("frequency_regulator", &frequency_regulator_impl_npu); }
