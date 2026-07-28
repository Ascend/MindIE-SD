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
ascendc_ops=${ASCEND_OP_NAME:-'laser_attention;la_preprocess;ada_block_sparse_attention;sparse_block_estimate;norm_rope_concat;quant_flash_attn;quant_flash_attn_metadata;fused_infer_attention_score;mul_add'}

# ascend950 backend requires CANN 9.0+; remove ascend950 when CANN < 9.0
default_compute_unit='ascend910;ascend910b;ascend910_93;ascend950'
cann_version_file="${local_toolkit}/compiler/version.info"
if [ -f "${cann_version_file}" ] && grep -Eq '^Version=([0-8])(\.|$)' "${cann_version_file}"; then
    echo "Detected CANN < 9.0 from ${cann_version_file}, disable ascend950 backend."
    default_compute_unit='ascend910;ascend910b;ascend910_93'
fi
ascend_compute_unit=${ASCEND_COMPUTE_UNIT:-${default_compute_unit}}

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
    echo "Build AscendC ops: ${ascendc_ops}"
    echo "Build Ascend compute units: ${ascend_compute_unit}"
    source ${current_script_dir}/build_ascendc_ops.sh -n "${ascendc_ops}" -c "${ascend_compute_unit}"
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

