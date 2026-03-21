#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

src=${current_script_dir}/../csrc/ops/ascendc
dst=${current_script_dir}/custom_project

function create_empty_custom_project(){
    cd ${current_script_dir}
    rm -rf ${dst}
    ${msopgen} gen -i ir_demo.json -f onnx \
        -c ai_core-ascend910,ai_core-ascend910b,ai_core-ascend910_93 -lan cpp -out ${dst}
    rm ${dst}/framework/onnx_plugin/*.cc
    rm ${dst}/op_host/*.h
    rm ${dst}/op_host/*.cpp
    rm ${dst}/op_kernel/*.cpp
    rm ${dst}/cmake/util/tiling_data_def_build.py
}

function release_framework_onnx(){
    cd ${src}/framework/onnx_plugin
    # 如需控制哪些文件发布，可以按照字母序列举具体文件
    local files=(
        ascend_laser_attention_plugin.cpp
    )
    cp ${files[@]} ${dst}/framework/onnx_plugin
}

function release_op_host(){
    cd ${src}/op_host
    local files=(
        ascend_laser_attention.cpp
        ascend_laser_attention_tiling.h
        ascend_la_preprocess.cpp
        la_preprocess_tiling.h
        block_sparse_attention_proto.cpp
        block_sparse_attention_tiling.cpp
        block_sparse_attention.cpp
        block_sparse_attention_tiling_compile_info.h
        block_sparse_attention_tiling_const.h
        block_sparse_attention_tiling_context.h
        block_sparse_attention_tiling_struct.h
        block_sparse_attention_tiling_v2.h
        block_sparse_attention_tiling.h
        data_copy_transpose_tiling_def.h
        data_copy_transpose_tiling.h
        error_manager.h 
        ops_error.h
        sparse_block_estimate.cpp
        sparse_block_estimate_tiling.h
    )
    cp ${files[@]} ${dst}/op_host
}

function release_op_kernel(){
    cd ${src}/op_kernel
    local files=(
        ascend_laser_attention.cpp
        address_const.h
        AddressMappingForwardOnline.h
        AddressMappingVectorForwardOnline.h
        CubeForward.h
        ppmatmul_const.h
        VectorForward.h
        la_preprocess.cpp
        la_preprocess.h
        block_sparse_attention.cpp 
        block_sparse_attention_base.h
        block_sparse_attention_empty_tensor.h
        block_sparse_attention_s1s2_bns1_x910_base.h
        block_sparse_attention_s1s2_bns1_x910.h
        kernel_data_copy_transpose.h
        sparse_block_estimate.cpp
        sparse_block_estimate.h
    )
    cp ${files[@]} ${dst}/op_kernel
}

function revise_settings(){
    cd ${dst}
    sed -i "s#/usr/local/Ascend/latest#${local_toolkit}#g" CMakePresets.json
    sed -i "s#\"value\": \"customize\"#\"value\": \"aie_ascendc\"#g" CMakePresets.json
    sed -i "s#\"value\": \"True\"#\"value\": \"False\"#g" CMakePresets.json
    local line_num=$(grep -Fn "ENABLE_SOURCE_PACKAGE" CMakePresets.json | cut -d : -f 1)
    local offset_line_num=$((line_num+2))
    sed -i "${offset_line_num}s#\"value\": \"False\"#\"value\": \"True\"#g" CMakePresets.json
    sed -i "6s# <foss@huawei.com>##g" cmake/util/makeself/makeself-header.sh

    if [ ! -f "CMakeLists.txt" ]; then
        echo "Error: Can't find CMakeLists.txt"
        exit 1
    fi

    OPTIONS=(
        "-fno-common"
        "-Wfloat-equal"
        "-Wall"
        "-Wextra"
    )

    cp CMakeLists.txt CMakeLists.txt.bak

    if grep -q "add_compile_options" CMakeLists.txt; then
        for opt in "${OPTIONS[@]}"; do
            if ! grep -q "$opt" CMakeLists.txt; then
                sed -i "0,/add_compile_options/ s/\(add_compile_options.*\))/\1 $opt)/" CMakeLists.txt
                echo "Add: $opt"
            else
                echo "$opt already exists and no need to add again"
            fi
        done
    else
        sed -i '/project(opp)/a add_compile_options('"${OPTIONS[*]}"')' CMakeLists.txt
    fi
}

function build_and_install(){
    cd ${dst}
    bash build.sh
    bash ${dst}/build_out/*.run --install-path=${current_script_dir}
}

function build_ascendc_ops(){
    ori_path=${PWD}

    create_empty_custom_project
    release_framework_onnx
    release_op_host
    release_op_kernel
    revise_settings
    build_and_install
    cd ${ori_path}
}

build_ascendc_ops