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

import os
import multiprocessing
from multiprocessing.managers import BaseManager
from dataclasses import dataclass
import threading
import queue
import time
import random
import argparse

from mindiesd.utils.logs.logging import logger
from mindiesd.utils.exception import ModelExecError
from .task_transfer import UpdateTaskTransfer
from .greedy_algorithm import eplb_greedy

upload_queues = {}
instruction_queues = {}


class ScheduleManager(BaseManager):
    pass


ScheduleManager.register('get_upload_queues', callable=lambda rank: upload_queues[rank])
ScheduleManager.register('get_instruction_queues', callable=lambda rank: instruction_queues[rank])


@dataclass
class SchedulerContext:  # pylint: disable=too-many-instance-attributes
    scheduler_args: argparse.Namespace
    world_size: int
    redundant: int
    experts_set: set
    experts_per_rank: int
    load_report_buffer: dict
    local_expert_buffer: dict
    update_count: int = 0


def get_args():
    parser = argparse.ArgumentParser(description="EPLB scheduler")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--expert_num", type=int, default=32)
    parser.add_argument("--block_num", type=int, default=30)
    parser.add_argument("--max_move", type=int, default=5)
    parser.add_argument("--redundant", type=int, default=0)
    parser.add_argument("--mode", type=str, default="A2A")
    parser.add_argument("--auth_key", type=str, default=os.environ.get("EPLB_AUTH_KEY", "secret_key"))

    return parser.parse_args()


def start_manager_server(addr, auth_key):
    auth_bytes = auth_key.encode('utf-8')
    multiprocessing.current_process().authkey = auth_bytes
    manager = ScheduleManager(address=addr, authkey=auth_bytes)
    try:
        server = manager.get_server()
    except OSError as error:
        logger.error(
            "[MindIE-SD/eplb] EPLB scheduler failed to bind address. "
            "issue=manager server startup failed, scheduler_addr=%s:%s, actual_error=%s. "
            "possible_cause=the scheduler port is already in use or the configured address is unavailable. "
            "Troubleshooting: check the listening process, release the port, or configure another scheduler address.",
            addr[0],
            addr[1],
            error,
        )
        raise
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()


def get_manager_client(addr, auth_key):
    auth_bytes = auth_key.encode('utf-8')
    manager = ScheduleManager(address=addr, authkey=auth_bytes)
    return manager


def _init_scheduler_context(scheduler_args):
    world_size = scheduler_args.world_size
    server_addr = (scheduler_args.host, scheduler_args.port)
    redundant = scheduler_args.redundant
    auth_key = scheduler_args.auth_key
    experts_set = set(range(scheduler_args.expert_num))
    experts_per_rank = scheduler_args.expert_num // world_size

    num_moe_layers = scheduler_args.block_num
    load_report_buffer = {idx: {} for idx in range(num_moe_layers)}
    local_expert_buffer = {idx: {} for idx in range(num_moe_layers)}

    # zmq
    for rank in range(world_size):
        upload_queues[rank] = queue.Queue()
        instruction_queues[rank] = queue.Queue()

    start_manager_server(server_addr, auth_key)

    return SchedulerContext(
        scheduler_args=scheduler_args,
        world_size=world_size,
        redundant=redundant,
        experts_set=experts_set,
        experts_per_rank=experts_per_rank,
        load_report_buffer=load_report_buffer,
        local_expert_buffer=local_expert_buffer,
    )


def _complete_local_expert_list(context, local_expert_list):
    scheduler_args = context.scheduler_args
    if (
        scheduler_args.mode == "EX"
        and context.redundant > 0
        and len(local_expert_list) != (context.experts_per_rank + context.redundant)
    ):
        random_range = list(context.experts_set - set(local_expert_list))
        redundant_expert = random.sample(random_range, context.redundant)  # nosec B311
        return local_expert_list + redundant_expert
    return local_expert_list


def _emit_layer_update(context, layer_idx, transfer):
    scheduler_args = context.scheduler_args
    response = context.load_report_buffer[layer_idx]
    expert_dict = dict(sorted(context.local_expert_buffer[layer_idx].items()))
    context.load_report_buffer[layer_idx] = {}
    context.local_expert_buffer[layer_idx] = {}

    logger.debug(
        "[MindIE-SD/eplb] EPLB greedy compute started. layer_idx=%s, world_size=%s, mode=%s.",
        layer_idx,
        context.world_size,
        scheduler_args.mode,
    )
    result = eplb_greedy(
        response=response,
        algorithm_type=scheduler_args.mode,
        device_to_expert=expert_dict,
        world_size=context.world_size,
        expert_num=scheduler_args.expert_num,
        max_move=scheduler_args.max_move,
        redundant=context.redundant,
    )
    update, device_indices_list, local_expert_indices_list, local_expert_list, expert_trans_tensor = result
    if not update:
        logger.error(
            "[MindIE-SD/eplb] EPLB layout was not updated. "
            "issue=greedy algorithm produced no layout update, layer_idx=%s, update_count=%s. "
            "possible_cause=rank reports are incomplete, the target MoE layer is not covered, "
            "or the reported load does not trigger an update. "
            "Troubleshooting: check world_size rank reports, block_num, load data, and EPLB thresholds.",
            layer_idx,
            context.update_count,
        )
        return

    transfer.update_emit_task(
        device_indices_list,
        local_expert_indices_list,
        local_expert_list,
        expert_trans_tensor,
        context.world_size,
    )
    context.update_count += 1
    logger.debug(
        "[MindIE-SD/eplb] Layer layout computed. layer_idx=%s, update_count=%s.",
        layer_idx,
        context.update_count,
    )


def _process_rank_report(context, rank):
    try:
        report = upload_queues[rank].get_nowait()
    except queue.Empty:
        return True

    try:
        layer_idx = report['moe_layer_idx']
        load_data = report['load']
        local_expert_list = _complete_local_expert_list(context, report['local_expert_list'])

        context.load_report_buffer[layer_idx][rank] = load_data
        context.local_expert_buffer[layer_idx][rank] = local_expert_list
        transfer = UpdateTaskTransfer(instruction_queues, layer_idx)

        if len(context.load_report_buffer[layer_idx]) == context.world_size:
            _emit_layer_update(context, layer_idx, transfer)
        return False
    except Exception as e:
        raise ModelExecError(
            "[MindIE-SD/eplb] EPLB scheduler failed. "
            f"issue=failed to process upload queue, rank={rank}, world_size={context.world_size}, "
            f"mode={context.scheduler_args.mode}, "
            f"actual_error={e}. possible_cause=invalid load report, greedy algorithm failure, or queue state "
            "mismatch. Troubleshooting: inspect worker load report fields, EPLB mode/redundant settings, "
            "and scheduler traceback."
        ) from e


def run_scheduler(scheduler_args):
    context = _init_scheduler_context(scheduler_args)
    logger.debug(
        "[MindIE-SD/eplb] Scheduler monitor started. world_size=%s, host=%s, port=%s, mode=%s.",
        context.world_size,
        scheduler_args.host,
        scheduler_args.port,
        scheduler_args.mode,
    )

    while True:
        all_queues_empty = True
        try:
            for rank in range(context.world_size):
                all_queues_empty = _process_rank_report(context, rank) and all_queues_empty
        except (KeyboardInterrupt, SystemExit):
            logger.debug("[MindIE-SD/eplb] Scheduler received exit signal.")
            break
        if all_queues_empty:
            time.sleep(0.1)

    logger.debug("[MindIE-SD/eplb] Scheduler update count. count=%s.", context.update_count)
    logger.debug("[MindIE-SD/eplb] Scheduler cycle ended.")


if __name__ == '__main__':
    cli_args = get_args()
    run_scheduler(cli_args)
