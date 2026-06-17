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

COMM_TYPE=${COMM_TYPE:-0}

if [[ "${COMM_TYPE}" != "0" && "${COMM_TYPE}" != "1" && "${COMM_TYPE}" != "2" ]]; then
    echo "COMM_TYPE must be 0, 1, or 2." >&2
    exit 1
fi

echo "Wan attention chunk mode: comm_type=${COMM_TYPE}"

torchrun --nproc_per_node=8 generate.py \
              --task t2v-14B \
              --size 1280*720 \
              --ckpt_dir ${model_base} \
              --dit_fsdp \
              --t5_fsdp \
              --sample_steps 50 \
              --ulysses_size 8 \
              --comm_type "${COMM_TYPE}" \
              --vae_parallel \
              --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
              --use_attentioncache \
              --start_step 20 \
              --attentioncache_interval 2 \
              --end_step 47
