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
 * \file tiling_cache.h
 * \brief
 */
#pragma once

#include <map>
#include <shared_mutex>

namespace Ops {
namespace EagleQbmm {
constexpr uint32_t kMaxTilingCacheEntryNum = 500;

template <typename HashInput, typename HashItem> class TilingCache {
  public:
    void Add(uint32_t key, [[maybe_unused]] const HashInput &hash_input, const HashItem &value) {
        // hash_input 暂时没有用到，但编译不进行告警
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (size_ >= kMaxTilingCacheEntryNum) {
            return;
        }

        if (map_.find(key) != map_.end()) {
            return;
        }

        map_.emplace(key, value);
        size_++;
        return;
    }

    void Replace(uint32_t key, const HashInput &hash_input, const HashItem &value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (size_ >= kMaxTilingCacheEntryNum) {
            return;
        }

        if (map_.find(key) == map_.end()) {
            size_++;
        }
        map_.erase(key);
        map_.emplace(key, value);
        return;
    }

    bool Get(uint32_t key, const HashInput &hash_input, HashItem &value) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto iter = map_.find(key);
        if (iter == map_.end()) {
            return false;
        }
        if (!(hash_input == iter->second.input())) {
            OP_LOGD("CUBE", "inconsistent input data");
            return false;
        }

        value = iter->second;
        return true;
    }

  private:
    std::map<uint32_t, HashItem> map_;
    uint32_t size_ = 0;
    std::shared_mutex mutex_;
};
} // namespace NN
} // namespace Ops
