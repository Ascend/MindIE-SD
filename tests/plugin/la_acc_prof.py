#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import csv
import sys
import os
import itertools
from multiprocessing import Process, Queue
import torch
import torch_npu

from mindiesd.layers.flash_attn.attention_forward import attention_forward

BATCH_SIZE_MIN = 1         # 1 is min batch_size
BATCH_SIZE_MAX = 2         # 2 is max_batch_size

HEAD_NUMS_MIN = 1          # 1 is min head_nums
HEAD_NUMS_MAX = 2          # 2 is max head_nums

Q_SEQLEN_MIN = 4000           # 4000 is min qseqlen
Q_SEQLEN_MAX = 118889      # 118889 is max qseqlen

HEAD_DIMS_MIN = 128        # 128 is min head_dims
HEAD_DIMS_MAX = 128        # 128 is max head_dims

BATCH_SIZE = 'batch_size'
HEAD_NUM = 'head_num'
Q_SEQLEN = 'q_seqlen'
KV_SEQLEN = 'kv_seqlen'
KV_SEQLEN = 'kv_seqlen'
HEAD_DIM = 'head_dim'
DTYPE = 'dtype'
NPU = 'npu'


def read_configurations(file_path):
    configurations = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dtype = getattr(torch, row[DTYPE], None)

            if dtype is None:
                raise ValueError(f"Unsupported dtype: {row[DTYPE]}")
            configurations.append({
                BATCH_SIZE: int(row[BATCH_SIZE]),
                HEAD_NUM: int(row[HEAD_NUM]),
                Q_SEQLEN: int(row[Q_SEQLEN]),
                KV_SEQLEN: int(row[KV_SEQLEN]),
                HEAD_DIM: int(row[HEAD_DIM]),
                DTYPE: dtype,
            })
    return configurations


def generate_enumerated_configurations(output_file='enumerated_cases.csv'):
    # Define ranges for each parameter
    batch_sizes = range(BATCH_SIZE_MIN, BATCH_SIZE_MAX)
    head_nums = range(HEAD_NUMS_MIN, HEAD_NUMS_MAX)
    q_seqlens = range(Q_SEQLEN_MIN, Q_SEQLEN_MAX)
    kv_seqlens = q_seqlens      # kv_seqlen equals q_seqlen
    head_dims = [128]           # 128 is fixed head_dim

    # Generate all combinations
    configurations = []
    for batch_size, head_num, q_seqlen in itertools.product(batch_sizes, head_nums, q_seqlens):
        configurations.append({
            BATCH_SIZE: batch_size,
            HEAD_NUM: head_num,
            Q_SEQLEN: q_seqlen,
            KV_SEQLEN: q_seqlen,
            HEAD_DIM: head_dims[0],    # 0 is index
            DTYPE: getattr(torch, 'bfloat16'),
        })

    # Save configurations to CSV
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = [
            BATCH_SIZE, HEAD_NUM, Q_SEQLEN, KV_SEQLEN, HEAD_DIM, DTYPE
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for config in configurations:
            config[DTYPE] = config[DTYPE]  # Convert dtype to string for writing
            writer.writerow(config)

    return configurations


def process_configuration(config, result_queue):
    batch_size = config[BATCH_SIZE]
    head_num = config[HEAD_NUM]
    q_seqlen = config[Q_SEQLEN]
    kv_seqlen = config[KV_SEQLEN]
    head_dim = config[HEAD_DIM]
    dtype = config[DTYPE]

    scale_value = head_dim ** -0.5      # 0.5 is used to calculate sacle_value

    try:
        query_raw = torch.randn((batch_size, q_seqlen, head_num, head_dim), device=NPU, dtype=dtype)
        key_raw = torch.randn((batch_size, kv_seqlen, head_num, head_dim), device=NPU, dtype=dtype)
        value_raw = torch.randn(batch_size, kv_seqlen, head_num, head_dim, device=NPU, dtype=dtype)

        # Call the attention forward function
        attention_out = attention_forward(query_raw, key_raw, value_raw, opt_mode="manual",
                                          op_type="ascend_laser_attention", layout="BNSD")
        torch.npu.synchronize()

        fascore = torch_npu.npu_fusion_attention(query_raw, key_raw, value_raw, head_num=head_num,
            input_layout="BSND", scale=scale_value,
            pre_tockens=2147483647,    # 2147483647 is pre_tockens
            next_tockens=2147483647    # 2147483647 is next_tockens
        )[0]

        cosine_sim_vs_fascore = torch.cosine_similarity(
            attention_out.to("cpu").to(dtype=torch.float32).reshape(1, -1),
            fascore.to("cpu").reshape(1, -1)
        )[0].item()

        delta = (attention_out - fascore).abs()
        max_error = delta.max().item()
        mean_error = delta.mean().item()

        # Send results back to the main process
        result_queue.put({BATCH_SIZE: batch_size, HEAD_NUM: head_num, Q_SEQLEN: q_seqlen, KV_SEQLEN: kv_seqlen,
            HEAD_DIM: head_dim, DTYPE: str(dtype), 'cosine_sim_vs_fascore': cosine_sim_vs_fascore,
            'max_error': max_error, 'mean_error': mean_error})

    except Exception as e:
        # If an error occurs, send a failure message to the main process
        result_queue.put({BATCH_SIZE: batch_size, HEAD_NUM: head_num, Q_SEQLEN: q_seqlen, KV_SEQLEN: kv_seqlen,
            HEAD_DIM: head_dim, DTYPE: str(dtype), 'cosine_sim_vs_fascore': None,
            'max_error': None, 'mean_error': None})


def test(test_acc, configurations, output_file='acc_output_results.csv'):
    result_queue = Queue()

    if not (test_acc and output_file):
        return
    
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = [
            BATCH_SIZE, HEAD_NUM, Q_SEQLEN, KV_SEQLEN, HEAD_DIM, DTYPE,
            'cosine_sim_vs_fascore', 'max_error', 'mean_error'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Start processing configurations
        processes = []
        for config in configurations:
            # Start a new process for each configuration
            p = Process(target=process_configuration, args=(config, result_queue))
            processes.append(p)
            p.start()

            # Wait for the current process to finish before starting the next one
            p.join()

            # Check the result from the queue
            while not result_queue.empty():
                result = result_queue.get()
                if test_acc and output_file:
                    writer.writerow(result)
                    csvfile.flush()


if __name__ == "__main__":
    test_acc = True

    # Option 1: Load configurations from a file
    config_file = "./plugin/test_la.csv"
    configurations = read_configurations(config_file)
    test(test_acc, configurations)

    # Option 2: Generate enumerated configurations and save to a file
    configurations = generate_enumerated_configurations(output_file='enumerated_cases.csv')

    test(test_acc, configurations)