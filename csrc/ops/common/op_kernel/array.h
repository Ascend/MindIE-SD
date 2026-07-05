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
 * \file array.h
 * \brief
 */
#ifndef ARRAY_H
#define ARRAY_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

template <typename T, uint32_t N, uint64_t Stride = 0> class Array {
  private:
    T elems[N];

  public:
    __aicore__ inline T &operator[](uint32_t i) { return elems[i]; }
};

template <typename T, uint32_t N, uint64_t Stride> class Array<LocalTensor<T>, N, Stride> {
  private:
    LocalTensor<T> tensor;

  public:
    __aicore__ inline void Init(const LocalTensor<T> &tensor) { this->tensor = tensor; }
    __aicore__ inline LocalTensor<T> operator[](uint32_t i) { return tensor[i * Stride]; }
};

template <typename T, uint32_t N> class Array<GlobalTensor<T>, N> {
  private:
    GlobalTensor<T> tensor;
    uint64_t stride = 0;

  public:
    __aicore__ inline void Init(const GlobalTensor<T> &tensor, uint64_t splitSize) {
        this->tensor = tensor;
        this->stride = splitSize;
    }
    __aicore__ inline GlobalTensor<T> operator[](uint32_t i) { return tensor[i * stride]; }
};
#endif // ARRAY_H
