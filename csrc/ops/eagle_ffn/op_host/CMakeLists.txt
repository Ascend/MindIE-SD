# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

add_ops_compile_options(
        OP_NAME EagleFfn
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

target_sources(op_host_aclnnInner PRIVATE
        eagle_ffn_def.cpp
)

set(FFN_3RD_MM_TILING_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_key.cpp
)
file(GLOB FFN_3RD_COMMON_TILING_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/common/op_host/op_tiling/*.cpp
)

# vendored tiling 单独编译对象库；日志/错误宏经 ffn_3rd_log_compat.h 接仓内轻量日志
# （utils/inc），不依赖 CANN pkg_inc/op_common 私有符号，该 include 顺序仅作用于本库。
add_library(ffn_3rd_tiling OBJECT
        ffn_tiling.cpp
        ffn_arch35_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/common/src/tiling_util.cpp
        ${FFN_3RD_MM_TILING_SRCS}
        ${FFN_3RD_COMMON_TILING_SRCS}
)
set_target_properties(ffn_3rd_tiling PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(ffn_3rd_tiling PRIVATE
        -std=gnu++17
        -fvisibility=hidden
)
target_compile_definitions(ffn_3rd_tiling PRIVATE
        LOG_CPP
        PROCESS_LOG
)
target_link_libraries(ffn_3rd_tiling PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:ops_utils_tiling_headers>
)
target_include_directories(ffn_3rd_tiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/base
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/tiling
)

target_sources(optiling PRIVATE $<TARGET_OBJECTS:ffn_3rd_tiling>)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE $<TARGET_OBJECTS:ffn_3rd_tiling>)
endif ()

target_sources(opsproto PRIVATE
        ffn_infershape.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_eagle_ffn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/ffn.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 共享 install 只拷 op_kernel 顶层，补 3rd 子目录源码
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/3rd/
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/eagle_ffn/3rd
        FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h" PATTERN "*.hpp"
)
