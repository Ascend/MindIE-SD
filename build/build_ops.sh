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

# 构建环境使用CANN主线包，容易引入兼容性问题。同时为了更好地控制对外发布内容，我们
# 在构建环境用msopgen工具生成工程，然后将要发布的算子交付件拷贝到新生成的工程构建
set -e

current_script_dir=$(realpath $1)

if [ -n "${ASCEND_TOOLKIT_HOME}" ]; then
    local_toolkit=${ASCEND_TOOLKIT_HOME}
    echo "Using ASCEND_TOOLKIT_HOME: ${local_toolkit}"
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    local_toolkit=/usr/local/Ascend/ascend-toolkit/latest
    echo "Using default toolkit path: ${local_toolkit}"
elif [ -d "/home/slave1/Ascend/ascend-toolkit/latest" ]; then
    local_toolkit=/home/slave1/Ascend/ascend-toolkit/latest
    echo "Using alternative toolkit path: ${local_toolkit}"
else
    echo "Can not find toolkit path, please set ASCEND_TOOLKIT_HOME"
    echo "eg: export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest"
    exit 1
fi

msopgen=${local_toolkit}/python/site-packages/bin/msopgen
if [ ! -f ${msopgen} ]; then
    echo "${msopgen} not exists"
    exit 1
fi

base_ascendc_ops='laser_attention;la_preprocess;ada_block_sparse_attention;sparse_block_estimate'
cann_9_required_ops='quant_flash_attn;quant_flash_attn_metadata'
ascend950_only_ops='fused_infer_attention_score'
ascendc_ops="${base_ascendc_ops}"
cann_version_file="${local_toolkit}/compiler/version.info"

function is_ascend950_only_compute_unit() {
    local compute_units="$1"
    local unit
    if [ -z "${compute_units}" ]; then
        return 1
    fi

    IFS=';' read -ra compute_unit_array <<< "${compute_units}"
    for unit in "${compute_unit_array[@]}"; do
        if [ "${unit}" != "ascend950" ]; then
            return 1
        fi
    done
    return 0
}

function remove_op_from_semicolon_list() {
    local op_list="$1"
    local op_to_remove="$2"
    local filtered_ops=''
    local item
    local skip_group='false'

    IFS=';' read -ra op_array <<< "${op_list}"
    for item in "${op_array[@]}"; do
        if [ "${item}" = "${op_to_remove}" ]; then
            skip_group='true'
            continue
        fi
        if [ "${skip_group}" = 'true' ] && [[ "${item}" =~ ^[0-9]+-[0-9]+$ ]]; then
            skip_group='false'
            continue
        fi
        skip_group='false'
        if [ -z "${filtered_ops}" ]; then
            filtered_ops="${item}"
        else
            filtered_ops="${filtered_ops};${item}"
        fi
    done

    echo "${filtered_ops}"
}

if [ -f "${cann_version_file}" ] && grep -Eq '^Version=([0-8])(\.|$)' "${cann_version_file}"; then
    cann_version=$(grep -E '^Version=' "${cann_version_file}" | head -n 1 | cut -d= -f2)
    ascend_compute_unit=${ASCEND_COMPUTE_UNIT:-'ascend910;ascend910b;ascend910_93'}
    echo "Warning: ops ${cann_9_required_ops};${ascend950_only_ops} require CANN >= 9.0, current CANN is ${cann_version}. Skip compiling them."
else
    ascend_compute_unit=${ASCEND_COMPUTE_UNIT:-'ascend950'}
    ascendc_ops="${ascendc_ops};${cann_9_required_ops}"
    if is_ascend950_only_compute_unit "${ascend_compute_unit}"; then
        ascendc_ops="${ascendc_ops};${ascend950_only_ops}"
    else
        echo "Warning: ops ${ascend950_only_ops} only support ascend950. Current ASCEND_COMPUTE_UNIT=${ascend_compute_unit}. Skip compiling them."
    fi
fi

function sync_aicpu_ops_to_transformer_vendor(){
    src_cpu_dir="${current_script_dir}/vendors/aie_ascendc/op_impl/cpu"
    dst_cpu_dir="${current_script_dir}/vendors/customize_transformer/op_impl/cpu"

    if [ -d "$src_cpu_dir" ]; then
        echo "Syncing AICPU op_impl from aie_ascendc to customize_transformer..."
        mkdir -p "$dst_cpu_dir"
        cp -a "$src_cpu_dir/." "$dst_cpu_dir/"
    fi
}

function build_ops(){
    ori_path=${PWD}
    cd ${current_script_dir}
    rm -rf vendors
    local ascend_op_name=${ASCEND_OP_NAME:-"${ascendc_ops}"}
    if ! is_ascend950_only_compute_unit "${ascend_compute_unit}"; then
        if [ "${ascend_op_name}" = "ALL" ] || [ "${ascend_op_name}" = "all" ]; then
            ascend_op_name="${ascendc_ops}"
            echo "Warning: ASCEND_OP_NAME=ALL includes ${ascend950_only_ops}, which only supports ascend950. Use ASCEND_OP_NAME=${ascend_op_name} instead."
        elif [[ ";${ascend_op_name};" == *";${ascend950_only_ops};"* ]]; then
            ascend_op_name=$(remove_op_from_semicolon_list "${ascend_op_name}" "${ascend950_only_ops}")
            echo "Warning: remove ${ascend950_only_ops} from ASCEND_OP_NAME for ASCEND_COMPUTE_UNIT=${ascend_compute_unit}."
        fi
    fi
    echo "Build AscendC ops: ${ascend_op_name}"
    echo "Build Ascend compute units: ${ascend_compute_unit}"
    source ${current_script_dir}/build_ascendc_ops.sh -n "${ascend_op_name}" -c "${ascend_compute_unit}"
    source ${current_script_dir}/build_tik_ops.sh
    sync_aicpu_ops_to_transformer_vendor
    rm -rf ${current_script_dir}/vendors/aie_ascendc/bin
    rm -rf ${current_script_dir}/vendors/customize/bin
    rm -rf ${current_script_dir}/vendors/customize_transformer/bin
    so_files=$(find "${current_script_dir}/vendors" -name "*.so" -type f 2>/dev/null)
    if [ -n "$so_files" ]; then
        echo "Stripping .so files..."
        for so_file in $so_files; do
            if [[ "$so_file" == *"/op_impl/cpu/aicpu_kernel/impl/"* ]]; then
                echo "Skipping AICPU kernel so: $so_file"
                continue
            fi
            strip "$so_file"
        done
    fi
    cd ${current_script_dir}
}

copy_ops() {
    SRC_DIR="${current_script_dir}/vendors"
    DST_DIR="${current_script_dir}/../mindiesd/ops/vendors"

    echo "Source directory: $SRC_DIR"
    echo "Destination directory: $DST_DIR"

    # Check source directory
    if [ ! -d "$SRC_DIR" ]; then
        echo "Error: source directory $SRC_DIR does not exist!"
        return 1
    fi

    # Create destination directory
    mkdir -p "$DST_DIR"

    # (Optional) Clean the target directory
    echo "Cleaning destination directory..."
    rm -rf "${DST_DIR:?}/"*

    echo "Copying all subdirectories under pkg to mindie/ops..."
    for subdir in "$SRC_DIR"/*; do
        if [ -d "$subdir" ]; then
            echo "Copying directory: $subdir → $DST_DIR"
            cp -a "$subdir" "$DST_DIR/"
        fi
    done
    echo "Copy finished!"
}

build_ops
copy_ops
