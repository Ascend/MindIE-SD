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
 * \file ifa_matmul_policy.h
 * \brief
 */
#ifndef IFA_MATMUL_POLICY_H
#define IFA_MATMUL_POLICY_H

#include "ifa_flag_data.h"
#include "ifa_cube_in_buffer.h"
#include "ifa_copy_cube_in.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE, const auto &MM_CFG,
    typename MM_CB>
class IFAMatmulPolicyNormal : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
  public:
    using UserDefDataType = IFAFlagData;
    using CubeInBufferA =
        AscendC::Impl::Detail::IFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInA =
        AscendC::Impl::Detail::IFACopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // IFA_MATMUL_POLICY_H
