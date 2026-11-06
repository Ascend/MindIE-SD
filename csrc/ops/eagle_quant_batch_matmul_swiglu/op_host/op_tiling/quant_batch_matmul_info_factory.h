/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_info_factory.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_INFO_FACTORY_H
#define QUANT_BATCH_MATMUL_INFO_FACTORY_H

#include <pthread.h>
#include <shared_mutex>

#include "eagle_quant_batch_matmul_swiglu_tiling.h"

namespace optiling {
class EagleQbmmInfoFactory {
  public:
    EagleQbmmInfoFactory() = default;
    ~EagleQbmmInfoFactory() = default;

    EagleQbmmInfo *Get() {
        EagleQbmmInfo *ptr = nullptr;
        auto threadId = pthread_self();
        {
            std::shared_lock<std::shared_mutex> read_lock(mutex_);
            auto it = inst_.find(threadId);
            if (it != inst_.end()) {
                return &(it->second);
            }
        }
        // Not found: acquire write lock and double-check
        std::unique_lock<std::shared_mutex> write_lock(mutex_);
        auto it = inst_.find(threadId);
        if (it == inst_.end()) {
            ptr = &(inst_[threadId]);
        } else {
            ptr = &(it->second);
        }
        return ptr;
    }

  private:
    std::map<pthread_t, EagleQbmmInfo> inst_;
    std::shared_mutex mutex_;
};

} // namespace optiling
#endif // QUANT_BATCH_MATMUL_INFO_FACTORY_H
