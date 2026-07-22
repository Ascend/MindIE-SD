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
 * \file fa_l1_tensor.h
 * \brief
 */
#ifndef FA_L1_TENSOR_H
#define FA_L1_TENSOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

using AscendC::LocalTensor;

enum class L1Format { NZ = 0 };

enum class ScaleTrans { NO_TRANS = 0, ND2NZ = 1, DN2NZ = 2 };

template <typename Q_T, L1Format FORMAT> struct FaL1Tensor {
    LocalTensor<Q_T> tensor;
    uint32_t rowCount;
};

#endif
