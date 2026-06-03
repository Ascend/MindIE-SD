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
 * \file stream_k.h
 * \brief
 */

#ifndef LOAD_BALANCE_COMMON_H
#define LOAD_BALANCE_COMMON_H

#include <cstdint>
#include <array>
#include <functional>
#include <unordered_map>

namespace load_balance {

/******************************** ERROR CODE ********************************/
#define LOAD_BALANCE_SUCCESS 0
#define LOAD_BALANCE_ERROR_INVALID_INPUT 1

/******************************** CLASS DEF  ********************************/

template <class T> using Range = std::pair<T, T>;

template <class T, size_t ROW, size_t COL> using Table = std::array<std::array<T, COL>, ROW>;

using CostFunc = std::function<uint32_t(uint32_t, uint32_t)>; // The cost function of the algo

enum class SocVersion : uint32_t { BUTT };

struct DeviceInfo {
    uint32_t aicCoreMaxNum{0U}; // At most amount of aic core would be turned on, this is not final used number
    uint32_t aicCoreMinNum{0U}; // At least amount of aic core would be turned on, this is not final used number
    uint32_t aivCoreMaxNum{0U}; // At most amount of aiv core would be turned on, this is not final used number
    uint32_t aivCoreMinNum{0U}; // At least amount of aiv core would be turned on, this is not final used number
    uint32_t cvRatio;
    SocVersion version{SocVersion::BUTT};
};

struct GeneralBalanceParam {
    uint32_t mBaseSize{1U}; // At least one
    uint32_t s2BaseSize{1U}; // At least one
    int64_t faToleranceRatio{2U}; // The larger the value, the smaller the tolerance is
    bool fdOn{true}; // Turn on to activate fd
    int64_t fdTolerance{0U}; // no fd + tolerance < fd, then choose no fd
    CostFunc costFunc{nullptr}; // Customize cost func. Set to nullptr to use default cost func
};

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    BUTT,
};

enum class Layout : uint8_t { BSND = 0, BNSD, BSH, NBSD, TND, NTD, PA_NZ, PA_BBND, PA_BNBD, BUTT };

enum class DataType : uint8_t { FP32 = 0, FP16, INT4, INT8, INT32, BUTT };

/******************************** UTIL FUNC ********************************/
template <class T> inline T CeilDiv(T a, T b) {
    if (b == 0) {
        return 0;
    }
    return a / b + (a % b != 0); // avoid overflow
}

template <class T> inline T FloorDiv(T a, T b) {
    if (b == 0) {
        return 0;
    }
    return a / b;
}

template <class T> inline T SafeFloorDiv(T a, T b, T val) {
    static_assert(std::is_integral_v<T>, "must be integer type");
    if (b == 0) {
        return val;
    }
    return a / b;
}

template <typename T> inline T AddOne(T val) {
    static_assert(std::is_integral_v<T>, "must be integer type");
    return val + 1;
}

template <typename T> inline T MinusOne(T val) {
    static_assert(std::is_integral_v<T>, "must be integer type");
    return val - 1;
}

template <typename T> inline T ToOpenInterval(T val) { return AddOne(val); }

template <typename T> inline T ToClosedInterval(T val) { return MinusOne(val); }

template <typename T> inline T IndexToNum(T val) { return AddOne(val); }

template <typename T> inline T NumToIndex(T val) { return MinusOne(val); }

template <typename T> T Clip(T value, T minValue, T maxValue) {
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template <typename T> inline bool IsWithinTolerance(T limit, T tolerance, T value) {
    return limit + tolerance >= value;
}

static Layout ConvertToLayout(const std::string &layoutStr) {
    static std::unordered_map<std::string, Layout> layoutTable{{"BSND", Layout::BSND}, {"BNSD", Layout::BNSD},
        {"BSH", Layout::BSH}, {"NBSD", Layout::NBSD}, {"TND", Layout::TND}, {"NTD", Layout::NTD},
        {"PA_NZ", Layout::PA_NZ}, {"PA_BBND", Layout::PA_BBND}, {"PA_BNBD", Layout::PA_BNBD}};
    if (layoutTable.find(layoutStr) != layoutTable.end()) {
        return layoutTable[layoutStr];
    }
    return Layout::BUTT;
}

static uint32_t GetDataTypeByteSize(DataType type) {
    switch (type) {
    case (DataType::FP32):
    case (DataType::INT32):
        return 4U;
    case (DataType::FP16):
        return 2U;
    case (DataType::INT8):
        return 1U;
    default:
        return 2U;
    }
}

}
#endif
