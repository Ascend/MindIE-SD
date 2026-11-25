# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
import os
from pathlib import Path

import pytest

from tests.ops.sigmoid_mul_dynamic_mx_quant.batch import Batch, prepare_vendor


def pytest_collection_modifyitems(config, items):
    local = [item for item in items if Path(item.path).parent == Path(__file__).parent]
    if not local or config.option.collectonly:
        return
    prepare_vendor()
    plan = {"items": [item.nodeid for item in local], "device": os.environ.get("MINDIE_TEST_DEVICE_ID", "0")}
    batch = Batch("small", plan, output=os.environ.get("MX_SMALL_OUTPUT"), resume=config.getoption("--resume"))
    config._mx_batch = batch
    config._mx_reports = {}
    completed = [item for item in local if batch.complete(item.nodeid)]
    config._mx_all_done = len(completed) == len(items)
    items[:] = [item for item in items if item not in completed]
    config.hook.pytest_deselected(items=completed)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    batch = getattr(item.config, "_mx_batch", None)
    if batch is None or Path(item.path).parent != Path(__file__).parent:
        return
    reports = item.config._mx_reports.setdefault(item.nodeid, [])
    reports.append(report)
    if call.when == "teardown":
        failed = [r for r in reports if r.failed]
        if failed:
            batch.save(item.nodeid, "FAIL", detail=str(failed[0].longrepr))
        elif any(r.skipped for r in reports):
            batch.save(item.nodeid, "SKIP")
        elif any(r.when == "call" and r.passed for r in reports):
            batch.save(item.nodeid, "PASS")


def pytest_sessionfinish(session, exitstatus):
    batch = getattr(session.config, "_mx_batch", None)
    if batch is not None:
        batch.close()
        if session.config._mx_all_done and exitstatus == 5:
            session.exitstatus = 0
