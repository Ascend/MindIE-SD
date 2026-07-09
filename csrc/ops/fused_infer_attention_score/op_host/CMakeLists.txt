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
        OP_NAME EagleFusedInferAttentionScore
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
                --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
)

set(fused_infer_attention_score_depends common prompt_flash_attention incre_flash_attention PARENT_SCOPE)
set(eagle_fused_infer_attention_score_depends common prompt_flash_attention incre_flash_attention PARENT_SCOPE)

target_sources(op_host_aclnnInner PRIVATE
        fused_infer_attention_score_def.cpp
)

file(GLOB_RECURSE FUSED_INFER_ATTENTION_SCORE_CHECKER_SRC_FILES
        checkers/*checker*.cpp
)

target_sources(optiling PRIVATE
        ../../common/op_host/fia_tiling_shape.cpp
        ../../common/op_host/fia_tiling_info.cpp
        ../../common/op_host/split_core.cpp
        ../../common/op_host/split_core_v2.cpp
        ../../common/op_host/arch35/fia_tiling_nonquant_gqa.cpp
        ../../common/op_host/arch35/fia_tiling_fullquant_mx.cpp
        ../../common/op_host/arch35/fia_tiling_fullquant_gqa.cpp
        ../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp
        ../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
        ../../incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
        ../../incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp
        ../../incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp
        arch35/fused_infer_attention_score_tiling_v4.cpp
        arch35/fused_infer_attention_score_tiling_impl.cpp
        fused_infer_attention_score_tiling.cpp
        fused_infer_attention_score_tiling_info_parser.cpp
        fused_infer_attention_score_tiling_register.cpp
        ${FUSED_INFER_ATTENTION_SCORE_CHECKER_SRC_FILES}
)

target_sources(opsproto PRIVATE
        fused_infer_attention_score_infershape.cpp
)

target_sources(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/aclnn_fused_infer_attention_score_inner.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/aclnn_fused_infer_attention_score_v4.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/aclnn_fused_infer_attention_score_v5.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/checkers
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host/arch35
        ${CMAKE_CURRENT_SOURCE_DIR}/../../prompt_flash_attention/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../../incre_flash_attention/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
        ${CMAKE_BINARY_DIR}/autogen/inner
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/inc
)
