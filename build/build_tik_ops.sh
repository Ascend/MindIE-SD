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

src=${current_script_dir}/../csrc/ops/tik
dst=${current_script_dir}/custom_project_tik

function create_empty_custom_project(){
    cd ${current_script_dir}
    rm -rf ${dst}
    ${msopgen} gen -i ir_demo.json -f onnx -c ai_core-ascend310p -out ${dst}
    rm ${dst}/framework/onnx_plugin/*.cc
    rm ${dst}/op_proto/*.h
    rm ${dst}/op_proto/*.cc
    rm ${dst}/tbe/impl/*.py
    rm ${dst}/tbe/op_info_cfg/ai_core/ascend310p/*.ini
}

function release_framework_onnx(){
    cd ${src}/framework/onnx_plugin
    local files=(
        flash_attention_tik_plugin.cpp
    )
    cp ${files[@]} ${dst}/framework/onnx_plugin
}

function release_op_proto(){
    cd ${src}/op_proto
    local files=(
        flash_attention_tik.cpp
        flash_attention_tik.h
    )
    cp ${files[@]} ${dst}/op_proto
}

function release_op_impl(){
    cd ${src}/tbe/impl
    local files=(
        constants.py
        flash_attention_fwd.py
        flash_attention_tik.py
        tik_ops_utils.py
    )
    cp ${files[@]} ${dst}/tbe/impl
}

function release_cfg(){
    cd ${src}/tbe/op_info_cfg/ai_core/ascend310p
    local files=(
        flash_attention_tik.ini
    )
    cp ${files[@]} ${dst}/tbe/op_info_cfg/ai_core/ascend310p
}

function revise_settings(){
    cd ${dst}
    sed -i "43i export ASCEND_TENSOR_COMPILER_INCLUDE=${local_toolkit}/include" build.sh
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
    sed -i '/project(opp)/a set(CMAKE_EXPORT_COMPILE_COMMANDS ON)' CMakeLists.txt
}

function build_and_install(){
    cd ${dst}
    bash build.sh
    chmod +w ${current_script_dir}/vendors
    bash ${dst}/build_out/*.run --install-path=${current_script_dir}/vendors
    chmod -w ${current_script_dir}/vendors
}

function build_tik_ops(){
    ori_path=${PWD}
    create_empty_custom_project
    release_framework_onnx
    release_op_proto
    release_op_impl
    release_cfg
    revise_settings
    build_and_install
    cd ${ori_path}
}

build_tik_ops