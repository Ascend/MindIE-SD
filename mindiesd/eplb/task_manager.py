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

import threading
from multiprocessing.connection import AuthenticationError

import torch_npu

from mindiesd.utils.logs.logging import logger
from .eplb_scheduler import get_manager_client
from .task_payload import TaskType, TaskPayload
from .task_transfer import ProfileTaskTransfer
from .task_handler import handle_profile_task, handle_update_layout_task, handle_unknown_task

TASK_DISPATCHER = {
    TaskType.PROFILE: handle_profile_task,
    TaskType.UPDATE_LAYOUT: handle_update_layout_task,
}


def _log_unknown_instruction(instruction):
    logger.error(
        "[MindIE-SD/eplb] Unknown instruction ignored. "
        "issue=instruction is not a TaskPayload, instruction=%s. "
        "possible_cause=an unsupported object was inserted into the EPLB instruction queue. "
        "Troubleshooting: inspect the producer that writes instruction_queue and remove non-TaskPayload "
        "objects; ensure the queue only contains PROFILE or UPDATE_LAYOUT TaskPayload objects.",
        instruction,
    )


def _log_unknown_task_type(instruction):
    logger.error(
        "[MindIE-SD/eplb] Unknown task type. "
        "issue=unsupported EPLB task type, task_type=%s. "
        "possible_cause=an invalid TaskPayload was inserted into the EPLB instruction queue. "
        "Troubleshooting: inspect the TaskPayload producer and ensure task_type only uses PROFILE or "
        "UPDATE_LAYOUT; if a new task type was introduced, add it to TASK_DISPATCHER before enqueueing it.",
        instruction.task_type,
    )


def parse_module(module):
    dispatcher_list = []
    expert_load_collector_list = []
    for _, child in module.named_modules():
        if hasattr(child, 'dispatcher') and hasattr(child, 'expert_load_collector'):
            dispatcher_list.append(child.dispatcher)
            expert_load_collector_list.append(child.expert_load_collector)
    return dispatcher_list, expert_load_collector_list


def expert_info_transfer_pool(module, instruction_queue, upload_queue, device):
    dispatcher_list, expert_load_collector_list = parse_module(module)
    transfer_stream = torch_npu.npu.Stream(device)

    for idx, collector in enumerate(expert_load_collector_list):
        collector.task_transfer = ProfileTaskTransfer(instruction_queue, idx, collector.lb_interval)

    while True:
        instruction = instruction_queue.get()
        if instruction is None or instruction == 'exit':
            logger.debug("[MindIE-SD/eplb] Expert info transfer pool received exit instruction.")
            break
        if isinstance(instruction, TaskPayload):
            handler_function = TASK_DISPATCHER.get(instruction.task_type, handle_unknown_task)
            if handler_function is handle_unknown_task:
                _log_unknown_task_type(instruction)
            handler_function(instruction, upload_queue, expert_load_collector_list, dispatcher_list, transfer_stream)
        else:
            _log_unknown_instruction(instruction)


def connect_to_schedule_manager(rank_in_group, ip, port, auth_key):
    addr = (ip, port)
    manager = get_manager_client(addr, auth_key)
    try:
        manager.connect()
    except AuthenticationError as error:
        logger.error(
            "[MindIE-SD/eplb] EPLB worker authentication failed. "
            "issue=scheduler rejected worker credentials, rank_in_group=%s, manager_addr=%s:%s, actual_error=%s. "
            "possible_cause=the scheduler and worker use different EPLB authentication keys. "
            "Troubleshooting: compare scheduler --auth_key, worker auth_key, and EPLB_AUTH_KEY in both "
            "environments; use the same key on every rank, then restart scheduler and workers.",
            rank_in_group,
            ip,
            port,
            error,
        )
        raise
    except OSError as error:
        logger.error(
            "[MindIE-SD/eplb] EPLB worker failed to connect to scheduler. "
            "issue=scheduler connection unavailable, rank_in_group=%s, manager_addr=%s:%s, actual_error=%s. "
            "possible_cause=the scheduler is not running, the configured address is incorrect, "
            "or the network path is unavailable. "
            "Troubleshooting: if actual_error is Connection refused, check whether the scheduler process is running "
            "and whether manager_addr is listening with ps and ss -ltnp; if the scheduler is listening, compare "
            "manager_addr with scheduler --host/--port and worker ip/port; if the address matches but connection "
            "still fails, run nc -vz or telnet from the worker environment to check routing, firewall, or container "
            "network namespace.",
            rank_in_group,
            ip,
            port,
            error,
        )
        raise
    logger.debug(
        "[MindIE-SD/eplb] Connected to schedule manager. rank_in_group=%s, manager_addr=%s:%s.",
        rank_in_group,
        ip,
        port,
    )
    instruction_queue = manager.get_instruction_queues(rank=rank_in_group)  # pylint: disable=no-member
    upload_queue = manager.get_upload_queues(rank=rank_in_group)  # pylint: disable=no-member
    return instruction_queue, upload_queue


def construct_expert_info_transfer_pool(**kwargs):
    module = kwargs['module']
    rank_in_group = kwargs['rank_in_group']
    device = kwargs['device']
    ip = kwargs['ip']
    port = kwargs['port']
    auth_key = kwargs['auth_key']
    instruction_queue, upload_queue = connect_to_schedule_manager(rank_in_group, ip, port, auth_key)
    if instruction_queue is None or upload_queue is None:
        return None, None
    worker = threading.Thread(
        target=expert_info_transfer_pool, args=(module, instruction_queue, upload_queue, device), daemon=True
    )
    worker.start()
    return worker, instruction_queue
