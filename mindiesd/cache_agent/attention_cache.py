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

from mindiesd.utils.logs.logging import logger
from .cache import CacheConfig, CacheBase


class AttentionCache(CacheBase):
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache = [None] * self._config.blocks_count
        self._cache_ready = [False] * self._config.blocks_count

    def apply_imp(self, func: callable, *args, **kwargs):
        if (
            self._config.step_start < self._cur_step <= self._config.step_end
            and ((self._cur_step - self._config.step_start) % self._config.step_interval != 0)
            and self._cache_ready[self._cur_block]
        ):
            raw = self._cache[self._cur_block]
            logger.debug("[AttentionCache] step: %s block: %s reuse cache.", self._cur_step, self._cur_block)
            # 命中缓存时仍触发 block 上注册的 forward hooks，避免跳过模型自带 hook；
            # raw 与缓存共享对象，交给 hooks 前需复制，避免原地修改型 hook 污染缓存
            attn = self._call_with_hooks(func, raw, args, kwargs, protect_result=True)
        else:
            try:
                raw, attn, cacheable = self._compute_with_hooks(func, args, kwargs)
            except Exception:
                # 计算失败时缓存不可信：显式失效，避免命中路径复用上一步的陈旧结果
                self._cache[self._cur_block] = None
                self._cache_ready[self._cur_block] = False
                raise
            if self._config.step_start <= self._cur_step < self._config.step_end:  # 当在step_end之前才缓存
                # 缓存未经 forward hooks 处理的原始输出，避免命中时 hook 被重复应用
                self._cache[self._cur_block] = self._clone_result(raw) if cacheable else None
                self._cache_ready[self._cur_block] = cacheable
                logger.debug("[AttentionCache] step: %s block: %s update cache.", self._cur_step, self._cur_block)
        return attn

    def _release(self):
        self._cache = [None] * self._config.blocks_count
        self._cache_ready = [False] * self._config.blocks_count
