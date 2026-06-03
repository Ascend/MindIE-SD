# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(OP_NAME ${PARENT_DIR} NAME)

if (NOT BUILD_OPEN_PROJECT)
    return()
endif()

file(GLOB_RECURSE AICPU_JSON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu*.cpp)

if (NOT AICPU_SRC)
    return()
endif()

set(AICPU_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(AICPU_OBJ_DIR ${AICPU_OUT_DIR}/obj)
set(AICPU_SO ${AICPU_OUT_DIR}/libtransformer_aicpu_kernels.so)
set(AICPU_MERGED_JSON ${AICPU_OUT_DIR}/cust_aicpu_kernel.json)
set(AICPU_CXX ${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)

if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64/libaicpu_context.a)
    set(LIBAICPU_CONTEXT_PATH ${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64/libaicpu_context.a)
else()
    set(LIBAICPU_CONTEXT_PATH ${ASCEND_CANN_PACKAGE_PATH}/lib64/libaicpu_context.a)
endif()

if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64/libbase_ascend_protobuf.a)
    set(LIBBASE_ASCEND_PROTOBUF_PATH ${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64/libbase_ascend_protobuf.a)
else()
    set(LIBBASE_ASCEND_PROTOBUF_PATH ${ASCEND_CANN_PACKAGE_PATH}/lib64/libbase_ascend_protobuf.a)
endif()

set(AICPU_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc/log
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_kernel/load_balance
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_kernel/load_balance/section_stream_k
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/cce
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/msprof
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context/common
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context/cpu_proto
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common/context/utils
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aicpu_common/context
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aicpu_common/context/common
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aicpu_common/context/cpu_proto
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aicpu_common/context/utils
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/datagw/aicpu/common
)

set(AICPU_INCLUDE_FLAGS)
foreach(include_dir ${AICPU_INCLUDE_DIRS})
    list(APPEND AICPU_INCLUDE_FLAGS -I${include_dir})
endforeach()

set(AICPU_COMPILE_DEFINITIONS
    -DOPS_UTILS_LOG_SUB_MOD_NAME=\"AICPU\"
    -DOPS_UTILS_LOG_PACKAGE_TYPE=\"[Custom]\"
)

set(AICPU_OBJECTS)
foreach(src ${AICPU_SRC})
    get_filename_component(src_name ${src} NAME_WE)
    set(obj ${AICPU_OBJ_DIR}/${src_name}.o)
    add_custom_command(
        OUTPUT ${obj}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${AICPU_OBJ_DIR}
        COMMAND ${AICPU_CXX}
                -std=c++17
                -fPIC
                -O2
                -D_FORTIFY_SOURCE=2
                -D_GLIBCXX_USE_CXX11_ABI=1
                -Dgoogle=ascend_private
                -DDISABLE_COMPILE_V1
                ${AICPU_COMPILE_DEFINITIONS}
                -fvisibility=hidden
                ${AICPU_INCLUDE_FLAGS}
                -c ${src}
                -o ${obj}
        DEPENDS ${src}
        VERBATIM
    )
    list(APPEND AICPU_OBJECTS ${obj})
endforeach()

add_custom_command(
    OUTPUT ${AICPU_SO}
    COMMAND ${AICPU_CXX}
            -shared ${AICPU_OBJECTS}
            -Wl,--whole-archive
            ${LIBAICPU_CONTEXT_PATH}
            ${LIBBASE_ASCEND_PROTOBUF_PATH}
            -Wl,--no-whole-archive
            -Wl,-Bsymbolic
            -Wl,--exclude-libs=libbase_ascend_protobuf.a
            -s
            -o ${AICPU_SO}
    DEPENDS ${AICPU_OBJECTS}
    VERBATIM
)

add_custom_command(
    OUTPUT ${AICPU_MERGED_JSON}
    COMMAND ${CMAKE_COMMAND} -E copy ${AICPU_JSON_FILES} ${AICPU_MERGED_JSON}
    DEPENDS ${AICPU_JSON_FILES}
    VERBATIM
)

add_custom_target(${OP_NAME}_aicpu ALL
    DEPENDS ${AICPU_SO} ${AICPU_MERGED_JSON}
)
if (TARGET custom_aicpu_kernels)
    add_dependencies(custom_aicpu_kernels ${OP_NAME}_aicpu)
endif()

install(FILES ${AICPU_SO}
    DESTINATION packages/vendors/${VENDOR_NAME}/op_impl/cpu/aicpu_kernel/impl
    OPTIONAL
)

install(FILES ${AICPU_MERGED_JSON}
    DESTINATION packages/vendors/${VENDOR_NAME}/op_impl/cpu/config
    OPTIONAL
)
