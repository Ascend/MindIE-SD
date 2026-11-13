/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

#ifndef TLA_NUMERIC_MATH_HPP
#define TLA_NUMERIC_MATH_HPP

#include "../../attn_infra/detail/macros.hpp"
#include "../../tla/type_traits.hpp"

namespace tla {

//
// Common Operations
//

template <class T, class U, TLA_REQUIRES(std::is_arithmetic<T>::value &&std::is_arithmetic<U>::value)>
HOST_DEVICE constexpr auto max(T const &t, U const &u) {
    return t < u ? u : t;
}

template <class T, class U, TLA_REQUIRES(std::is_arithmetic<T>::value &&std::is_arithmetic<U>::value)>
HOST_DEVICE constexpr auto min(T const &t, U const &u) {
    return t < u ? t : u;
}

template <class T, class U, TLA_REQUIRES(std::is_arithmetic<T>::value &&std::is_arithmetic<U>::value)>
HOST_DEVICE constexpr auto add(T const &t, U const &u) {
    return t + u;
}

} // namespace tla

#endif // TLA_NUMERIC_MATH_HPP
