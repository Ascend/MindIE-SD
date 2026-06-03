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
        OP_NAME QuantFlashAttn
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(quant_flash_attn_depends common PARENT_SCOPE)
target_sources(op_host_aclnnInner PRIVATE
        quant_flash_attn_def.cpp
)

target_sources(optiling PRIVATE
        quant_flash_attn_tiling.cpp
        quant_flash_attn_tiling_info_parser.cpp
        arch35/quant_flash_attn_tiling_dn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host/fia_tiling_shape.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        quant_flash_attn_tiling.cpp
        quant_flash_attn_tiling_info_parser.cpp
        arch35/quant_flash_attn_tiling_dn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host/fia_tiling_shape.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        quant_flash_attn_infershape.cpp
)

target_sources(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/aclnn_quant_flash_attn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/aclnn_quant_flash_attn_inner.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/quant_flash_attn.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)
