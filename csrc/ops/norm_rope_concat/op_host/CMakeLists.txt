# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#
# MindIE is licensed under Mulan PSL v2.

add_ops_compile_options(
    OP_NAME NormRopeConcat
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -fpermissive
)

target_sources(op_host_aclnn PRIVATE
    norm_rope_concat_def.cpp
)

target_sources(optiling PRIVATE
    norm_rope_concat_tiling.cpp
)

target_sources(opsproto PRIVATE
    norm_rope_concat_proto.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

