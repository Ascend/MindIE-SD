# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# quant_four_over_six_a5（源算子 dynamic_mx_quant 迁移，T2）
# 说明：
#  - 算子发现由 csrc/ops CMakeLists 的 op_add_subdirectory 自动 glob，无需改主 CMakeLists；
#  - op_type 注册名 QuantFourOverSixA5（def.cpp OP_ADD），构建链命名规则（op_build/ascendc_bin_param_build）
#    要求 def 文件名、目录名、kernel 入口函数名均为 op_type 小写（quant_four_over_six_a5），
#    与 MindIE-SD 全仓约定一致（mul_add/quant_flash_attn 等）；
#  - G1: op_host/tiling_base.h 为 ops-nn 仓携带头（CANN 9.1 无此头），include 写法 "op_host/tiling_base.h"
#    依赖 include 根 ${CMAKE_CURRENT_SOURCE_DIR}/..（= 算子根目录）；
#  - G2: atvoss/broadcast/broadcast_tiling.h 在 CANN pkg_inc/op_common 根下；
#  - G3: log/log.h、util/*、op_kernel/* 等 op_common 系头需 include/op_common[ /op_host] 根；
#  - G4: CANN 9.1 将 runtime/infer_shape_context.h 移至 exe_graph/runtime/，补 include/exe_graph 根
#    （源 include 行不改，保持 infershape 逻辑零改动）。

add_ops_compile_options(
        OP_NAME QuantFourOverSixA5
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                # G3(kernel 侧): opc 默认 include 根不含 op_common 系，
                # op_kernel/math_util.h、op_kernel/platform_util.h 需此根
                -I${ASCEND_CANN_PACKAGE_PATH}/include/op_common
                -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common
                -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
                -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common
                -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common/op_host
)

set(quant_four_over_six_a5_depends quant_four_over_six_a5 PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        quant_four_over_six_a5_def.cpp
)

# T9b: 仅保留 46 自适应路径——optimize_tiling（非尾轴 10000~20043）已随清理删除
target_sources(optiling PRIVATE
        arch35/quant_four_over_six_a5_tiling_arch35.cpp
        arch35/quant_four_over_six_a5_tail_axis_tiling_arch35.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        arch35/quant_four_over_six_a5_tiling_arch35.cpp
        arch35/quant_four_over_six_a5_tail_axis_tiling_arch35.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        quant_four_over_six_a5_infershape.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common
        # CANN 9.0.1：util/math_util.h、util/shape_util.h 在 pkg_inc/include 的 op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common/op_host
        # broadcast_tiling 相对包含 pkg_inc/op_common/log/log.h，其 #include "dlog_pub.h"
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/base
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/include/base
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/include/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/op_common/op_host
)

target_include_directories(op_host_aclnn PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
