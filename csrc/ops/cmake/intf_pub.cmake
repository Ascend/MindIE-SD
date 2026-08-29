# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

add_library(intf_pub INTERFACE)
target_include_directories(intf_pub
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${ASCEND_CANN_PACKAGE_PATH}/include/external
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
            # CANN >= 9.1 (e.g. 9.1.T560) moved prof_api.h into msprof/profiling/ while
            # prof_common.h stays in msprof/toolchain/; add both subdirs so the quoted
            # cross-include resolves. Non-existent dirs only trigger a harmless
            # -Wmissing-include-dirs warning (same as platform/slog on some packages).
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/toolchain
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/profiling
            ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
            ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling
            ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/msprof
            ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/msprof/toolchain
)
target_link_directories(intf_pub
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_compile_options(intf_pub
        INTERFACE
            -fPIC
            -O2
            -Wall -Wundef -Wcast-qual -Wpointer-arith -Wdate-time
            -Wfloat-equal -Wformat=2 -Wshadow
            -Wsign-compare -Wunused-macros -Wvla -Wdisabled-optimization -Wempty-body -Wignored-qualifiers
            -Wimplicit-fallthrough=3 -Wtype-limits -Wshift-negative-value -Wswitch-default
            -Wframe-larger-than=32768 -Woverloaded-virtual
            -Wnon-virtual-dtor -Wshift-overflow=2 -Wshift-count-overflow
            -Wwrite-strings -Wmissing-format-attribute -Wformat-nonliteral
            -Wdelete-non-virtual-dtor -Wduplicated-cond
            -Wtrampolines -Wsized-deallocation -Wlogical-op -Wsuggest-attribute=format
            -Wduplicated-branches
            -Wmissing-include-dirs -Wformat-signedness
            -Wreturn-local-addr -Wextra
            -Wredundant-decls -Wfloat-conversion
            -Wno-write-strings -Wall -Wno-dangling-else -Wno-comment -Wno-conversion-null -Wno-return-type
            -Wno-unknown-pragmas -Wno-sign-compare
            -Wno-error=undef
            -Wno-error=comment
            -Wno-error=conversion-null
            -Wno-error=dangling-else
            -Wno-error=return-type
            -Wno-error=shadow
            -Wno-error=sign-compare
            -Wno-error=unknown-pragmas
            -Wno-error=unused-parameter
            -Wno-error=cast-qual
            -Wno-error=format=
            -Wno-error=maybe-uninitialized
            -Wno-error=missing-field-initializers
            -Wno-error=redundant-decls
            -Wno-error=unused-variable
            $<$<COMPILE_LANGUAGE:C>:-Wnested-externs>
            $<$<CONFIG:Debug>:-g>
            $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
)
target_compile_definitions(intf_pub
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>
            $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
)
target_link_options(intf_pub
        INTERFACE
            $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
            $<$<CONFIG:Release>:-s>
            -Wl,-z,relro
            -Wl,-z,now
            -Wl,-z,noexecstack
)
