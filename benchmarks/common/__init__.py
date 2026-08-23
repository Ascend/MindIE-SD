#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Shared benchmark utilities.

Single home for logic consumed by both the runtime plugin
(xpu-perf-plugin, imported on the NPU box) and the offline report tool
(scripts/benchmark_report.py), so the two can never drift apart:

- env_util: peak_flops/peak_bw resolution from env.json.
- metrics: MFU/MBU accounting formula.
- schema: op slot / seq-axis / series-key tables.
"""
