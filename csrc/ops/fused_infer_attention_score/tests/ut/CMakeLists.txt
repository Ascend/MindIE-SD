# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

cmake_minimum_required(VERSION 3.16)

project(mindiesd_fia_arch35_tiling_ut LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(FIA_UT_DIR "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(FIA_OP_DIR "${FIA_UT_DIR}/../.." REALPATH)
get_filename_component(OPS_DIR "${FIA_OP_DIR}/.." REALPATH)

if(CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "${CUSTOM_ASCEND_CANN_PACKAGE_PATH}")
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
elseif(DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/latest")
endif()

message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

set(FIA_HOST_DIR "${FIA_OP_DIR}/op_host")
set(FIA_COMMON_HOST_DIR "${OPS_DIR}/common/op_host")
set(FIA_COMMON_ARCH35_HOST_DIR "${OPS_DIR}/common/op_host/arch35")
set(FIA_COMMON_KERNEL_DIR "${OPS_DIR}/common/op_kernel")
set(FIA_EAGLE_KERNEL_DIR "${FIA_OP_DIR}/op_kernel/eagle_common/op_kernel")
set(FIA_UTILS_INC_DIR "${OPS_DIR}/utils/inc")
set(FIA_PROMPT_HOST_DIR "${OPS_DIR}/prompt_flash_attention/op_host")
set(FIA_PROMPT_KERNEL_DIR "${OPS_DIR}/prompt_flash_attention/op_kernel")
set(FIA_INCRE_HOST_DIR "${OPS_DIR}/incre_flash_attention/op_host")
set(FIA_INCRE_KERNEL_DIR "${OPS_DIR}/incre_flash_attention/op_kernel")

file(GLOB FIA_CHECKER_SOURCES "${FIA_HOST_DIR}/checkers/*checker*.cpp")

set(FIA_TILING_SOURCES
    "${OPS_DIR}/utils/src/fallback_comm.cpp"
    "${FIA_COMMON_HOST_DIR}/fia_tiling_shape.cpp"
    "${FIA_COMMON_HOST_DIR}/fia_tiling_info.cpp"
    "${FIA_COMMON_HOST_DIR}/split_core.cpp"
    "${FIA_COMMON_HOST_DIR}/split_core_v2.cpp"
    "${FIA_COMMON_ARCH35_HOST_DIR}/fia_tiling_nonquant_gqa.cpp"
    "${FIA_COMMON_ARCH35_HOST_DIR}/fia_tiling_fullquant_mx.cpp"
    "${FIA_COMMON_ARCH35_HOST_DIR}/fia_tiling_fullquant_gqa.cpp"
    "${FIA_PROMPT_HOST_DIR}/prompt_flash_attention_tiling.cpp"
    "${FIA_PROMPT_HOST_DIR}/prompt_flash_attention_tiling_v2.cpp"
    "${FIA_INCRE_HOST_DIR}/incre_flash_attention_tiling.cpp"
    "${FIA_INCRE_HOST_DIR}/incre_flash_attention_tiling_v2.cpp"
    "${FIA_INCRE_HOST_DIR}/incre_flash_attention_tiling_check.cpp"
    "${FIA_HOST_DIR}/fused_infer_attention_score_tiling.cpp"
    "${FIA_HOST_DIR}/fused_infer_attention_score_tiling_register.cpp"
    "${FIA_HOST_DIR}/fused_infer_attention_score_tiling_info_parser.cpp"
    "${FIA_HOST_DIR}/arch35/fused_infer_attention_score_tiling_v4.cpp"
    "${FIA_HOST_DIR}/arch35/fused_infer_attention_score_tiling_impl.cpp"
    ${FIA_CHECKER_SOURCES}
)

set(FIA_UT_COMMON_SOURCES
    "${FIA_UT_DIR}/common/fia_arch35_legacy_tiling_stub.cpp"
    "${FIA_UT_DIR}/common/tiling_context_faker.cpp"
    "${FIA_UT_DIR}/common/tiling_case_executor.cpp"
)

set(FIA_UT_CASE_SOURCES
    "${FIA_UT_DIR}/op_host/arch35/test_fused_infer_attention_score_tiling.cpp"
)

set(FIA_INCLUDE_DIRS
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include/external"
    "${ASCEND_CANN_PACKAGE_PATH}/include/experiment"
    "${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef"
    "${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime"
    "${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof"
    "${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/profiling"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/msprof"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/msprof/profiling"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/msprof/toolchain"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/base/context_builder"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling"
    "${FIA_UT_DIR}/common"
    "${FIA_HOST_DIR}"
    "${FIA_HOST_DIR}/checkers"
    "${FIA_COMMON_HOST_DIR}"
    "${FIA_COMMON_ARCH35_HOST_DIR}"
    "${FIA_COMMON_KERNEL_DIR}"
    "${FIA_EAGLE_KERNEL_DIR}"
    "${FIA_UTILS_INC_DIR}"
    "${FIA_PROMPT_HOST_DIR}"
    "${FIA_PROMPT_KERNEL_DIR}"
    "${FIA_INCRE_HOST_DIR}"
    "${FIA_INCRE_KERNEL_DIR}"
)

set(FIA_LINK_DIRS
    "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64/stub"
)

set(FIA_LINK_LIBS
    graph
    graph_base
    exe_graph
    platform
    register
    opp_registry
    rt2_registry
    unified_dlog
    ascendalog
    error_manager
    tiling_api
    metadef
    ops_base
    mmpa
    c_sec
    dl
    pthread
)

add_library(mindiesd_fia_arch35_tiling_ut_obj OBJECT ${FIA_TILING_SOURCES})
target_include_directories(mindiesd_fia_arch35_tiling_ut_obj PRIVATE ${FIA_INCLUDE_DIRS})
target_link_directories(mindiesd_fia_arch35_tiling_ut_obj PUBLIC ${FIA_LINK_DIRS})
target_compile_definitions(mindiesd_fia_arch35_tiling_ut_obj PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    ASCENDC_OP_TEST
    ASCENDC_OP_TEST_UT
    LOG_CPP
    PROCESS_LOG
    OP_TILING_LIB
)
target_compile_options(mindiesd_fia_arch35_tiling_ut_obj PRIVATE
    -fPIC
    -include
    "${FIA_UT_DIR}/common/fia_arch35_legacy_tiling_stub.h"
    -Wno-deprecated-declarations
    -Wno-sign-compare
    -Wno-unused-parameter
    -Wno-unused-variable
)

add_executable(mindiesd_fia_arch35_tiling_ut
    ${FIA_UT_COMMON_SOURCES}
    ${FIA_UT_CASE_SOURCES}
    $<TARGET_OBJECTS:mindiesd_fia_arch35_tiling_ut_obj>
)
target_include_directories(mindiesd_fia_arch35_tiling_ut PRIVATE ${FIA_INCLUDE_DIRS})
target_link_directories(mindiesd_fia_arch35_tiling_ut PRIVATE ${FIA_LINK_DIRS})
target_compile_definitions(mindiesd_fia_arch35_tiling_ut PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    ASCENDC_OP_TEST
    ASCENDC_OP_TEST_UT
    FIA_ARCH35_STANDALONE_UT
)
target_compile_options(mindiesd_fia_arch35_tiling_ut PRIVATE
    -fPIC
    -Wno-deprecated-declarations
    -Wno-sign-compare
    -Wno-unused-parameter
)
target_link_libraries(mindiesd_fia_arch35_tiling_ut PRIVATE
    ${FIA_LINK_LIBS}
)

enable_testing()
add_test(NAME mindiesd_fia_arch35_tiling_ut COMMAND mindiesd_fia_arch35_tiling_ut)
