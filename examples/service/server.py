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


import argparse

import ray
import torch
import torch_npu
from fastapi import FastAPI, HTTPException
from torch_npu.contrib import transfer_to_npu
import uvicorn
from request import GeneratorRequest
from worker import GeneratorWorker

torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False

app = FastAPI()


class Engine:
    def __init__(self, world_size: int, args):
        if not ray.is_initialized():
            ray.init(resources={"NPU": 8})

        num_workers = world_size
        self.workers = [
            GeneratorWorker.remote(args, rank=rank, world_size=world_size)
            for rank in range(num_workers)
        ]

    async def generate(self, request: GeneratorRequest):
        results = ray.get([
            worker.generate.remote(request)
            for worker in self.workers
        ])

        return next(path for path in results if path is not None)


@app.post("/generate")
async def generate_image(request: GeneratorRequest):
    try:
        result = await engine.generate(request)
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    from worker import _parse_args

    args = _parse_args()
    args.world_size = 8

    engine = Engine(
        world_size=args.world_size,
        args=args
    )

    uvicorn.run(app, host="0.0.0.0", port=6000)