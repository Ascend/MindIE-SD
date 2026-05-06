#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import torch_npu
import torch.distributed as dist


class ParallelManager():
    def __init__(self, world_size=1, rank=0, group=None):
        self.sp_size = world_size
        self.sp_group = group
        self.enable_sp = world_size > 1
        self.rank = rank


PARALLEL_MANAGER = ParallelManager()


def set_parallel_manager(world_size, rank, group):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(world_size, rank, group)


def get_sequence_parallel_group():
    return PARALLEL_MANAGER.sp_group


def get_sequence_parallel_size():
    return PARALLEL_MANAGER.sp_size


def get_sequence_parallel_state():
    return PARALLEL_MANAGER.enable_sp


def get_sequence_parallel_rank():
    return PARALLEL_MANAGER.rank


def init_parallel_env(enable_sequence_parallelism):
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend='hccl', init_method='env://',
        world_size=world_size, rank=rank
        )
    if enable_sequence_parallelism:
        set_parallel_manager(world_size, rank, dist.group.WORLD)