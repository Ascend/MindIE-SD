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

ascendc_ops=${ASCEND_OP_NAME:-'laser_attention;la_preprocess;ada_block_sparse_attention;sparse_block_estimate;ada_relu2_block_estimate;norm_rope_concat;eagle_split_norm_rope_concat;quant_flash_attn;quant_flash_attn_metadata;fused_infer_attention_score;eagle_quant_block_sparse_attention;eagle_block_sparse_attention;mul_add;eagle_indexed_gate_norm_modulate;eagle_ffn;eagle_hadamard_quant;quant_four_over_six_a5;sigmoid_mul_dynamic_mx_quant;eagle_quant_batch_matmul_swiglu;indexed_ada_layer_norm'}

function remove_ops_from_semicolon_list() {
    local op_list="$1"
    local ops_to_remove="$2"
    local filtered_ops=''
    local item
    local drop
    local keep

    IFS=';' read -ra op_array <<< "${op_list}"
    IFS=';' read -ra drop_array <<< "${ops_to_remove}"
    for item in "${op_array[@]}"; do
        keep='true'
        for drop in "${drop_array[@]}"; do
            if [ "${item}" = "${drop}" ]; then
                keep='false'
                break
            fi
        done
        if [ "${keep}" = 'true' ]; then
            if [ -z "${filtered_ops}" ]; then
                filtered_ops="${item}"
            else
                filtered_ops="${filtered_ops};${item}"
            fi
        fi
    done

    echo "${filtered_ops}"
}

# ascend950 backend requires CANN 9.0+; remove ascend950 when CANN < 9.0
default_compute_unit='ascend910;ascend910b;ascend910_93;ascend950'
# Ops whose OpDef registers ascend950 as its only AICore().AddConfig() soc.
# Exclude their sources entirely on CANN < 9.0, which does not support this backend.
ascend950_only_ops='quant_flash_attn;quant_flash_attn_metadata;fused_infer_attention_score;eagle_ffn;ada_relu2_block_estimate;eagle_quant_block_sparse_attention;eagle_hadamard_quant;quant_four_over_six_a5;sigmoid_mul_dynamic_mx_quant'
# Ops whose sources use CANN 9+ only APIs even when built for ascend910b/ascend910_93:
# eagle_block_sparse_attention pulls in the MXFP4 type fp4x2_e2m1_t (arch35 only, not
# declared by any 8.5.0 header). AddConfig cannot express this, so such ops must leave
# the op list entirely on older toolkits.
cann_9_required_ops='eagle_block_sparse_attention;eagle_quant_batch_matmul_swiglu'
# eagle_quant_block_sparse_attention adds 4bit (MXFP4) input support and therefore
# uses AscendC APIs introduced in CANN 9.1.x, e.g. asc_copy_l12l0b_mx. Those APIs
# are absent from CANN 9.0.x headers, so the op cannot be built on 9.0.x and is
# removed from the op list instead of aborting the whole build.
cann_91_required_ops='eagle_quant_block_sparse_attention'
cann_version_file="${local_toolkit}/compiler/version.info"
if [ -f "${cann_version_file}" ] && grep -Eq '^Version=([0-8])(\.|$)' "${cann_version_file}"; then
    echo "Detected CANN < 9.0 from ${cann_version_file}, disable ascend950 backend."
    default_compute_unit='ascend910;ascend910b;ascend910_93'
    ascendc_ops=$(remove_ops_from_semicolon_list "${ascendc_ops}" "${ascend950_only_ops};${cann_9_required_ops}")
    echo "Skip ascend950-only ops: ${ascend950_only_ops}"
    echo "Skip CANN 9.0+ required ops: ${cann_9_required_ops}"
    echo "Build AscendC ops after filtering: ${ascendc_ops}"
else
    # CANN >= 9.0 (or version.info unavailable): the minor version of CANN 9.x
    # decides whether the 4bit op can be built. 9.1.x+ provides the required APIs.
    cann_version_value=$(grep -m1 '^Version=' "${cann_version_file}" 2>/dev/null | cut -d'=' -f2 | tr -d '[:space:]' || true)
    IFS='.' read -r cann_major cann_minor _ <<< "${cann_version_value}"
    cann_major=${cann_major//[^0-9]/}
    cann_minor=${cann_minor//[^0-9]/}
    if [ "${cann_major}" = '9' ] && [ "${cann_minor}" = '0' ]; then
        echo "Detected CANN 9.0.x (Version=${cann_version_value}) from ${cann_version_file}."
        ascendc_ops=$(remove_ops_from_semicolon_list "${ascendc_ops}" "${cann_91_required_ops}")
        echo "WARNING: Removed ${cann_91_required_ops} from the operator build list."
        echo "    Reason: to support 4bit input, this operator uses APIs that are not available"
        echo "            in this CANN version (e.g. asc_copy_l12l0b_mx, introduced in CANN 9.1.x),"
        echo "            so it cannot be built on CANN 9.0.x."
        echo "    If the 4bit feature of ${cann_91_required_ops} is not needed, roll back to the"
        echo "    previous version and install the operator that supports 8bit input."
        echo "    To use the 4bit feature, upgrade to CANN 9.1.x or later and rebuild."
        echo "Build AscendC ops after filtering: ${ascendc_ops}"
    fi
fi
ascend_compute_unit=${ASCEND_COMPUTE_UNIT:-${default_compute_unit}}
# OpDef files consult this variable while op_build loads the op host .so (see
# IsSocEnabled in norm_rope_concat_def.cpp), so it has to reach op_build's environment
# and not only the CMake cache.
export ASCEND_COMPUTE_UNIT="${ascend_compute_unit}"

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
    echo "SKIP_ALL_OPS_PLUGIN_BUILD=${SKIP_ALL_OPS_PLUGIN_BUILD:-0}"
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
