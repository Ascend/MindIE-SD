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
        OP_NAME EagleQuantBlockSparseAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -fpermissive
)

# def.cpp is routed to the "inner" aclnn generator so that opbuild produces
# aclnnInnerEagleQuantBlockSparseAttention and does NOT collide with the
# hand-written public aclnnEagleQuantBlockSparseAttention provided via op_api.
target_sources(op_host_aclnnInner PRIVATE
        eagle_quant_block_sparse_attention_def.cpp
)

target_sources(optiling PRIVATE
        eagle_quant_block_sparse_attention_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        eagle_quant_block_sparse_attention_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        eagle_quant_block_sparse_attention_infershape.cpp
)

target_sources(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/aclnn_eagle_quant_block_sparse_attention.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/eagle_quant_block_sparse_attention.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)
