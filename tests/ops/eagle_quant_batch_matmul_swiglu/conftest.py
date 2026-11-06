"""Per-case checkpointing; committed cases are deselected before NPU fixtures."""

import signal
import os
from pathlib import Path
import pytest
from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_state import Batch


def pytest_addoption(parser):
    parser.addoption("--resume", default=None, help="Resume this operator's immutable small-shape batch")


def pytest_collection_modifyitems(session, config, items):
    if config.option.collectonly:
        return
    local = [item for item in items if Path(item.path).parent == Path(__file__).parent]
    if not local:
        return
    device = int(os.environ.get("MINDIE_TEST_DEVICE_ID", "0"))
    if device < 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise pytest.UsageError(
            "QBMM checkpoints require a nonnegative physical MINDIE_TEST_DEVICE_ID and no visibility mask"
        )
    plan = dict(nodeids=[item.nodeid for item in local], device=device)
    batch = Batch("small", plan=plan, resume=config.getoption("--resume"))
    config._qbmm_batch = batch
    config._qbmm_reports = {}
    config._qbmm_already_complete = False
    config._qbmm_signal = None

    def interrupt(signum, frame):
        config._qbmm_signal = signum
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        batch.ensure_no_failures()
    except RuntimeError as exc:
        raise pytest.UsageError(str(exc)) from exc
    batch.history([device])
    pending, complete = [], []
    for item in local:
        (complete if batch.get(item.nodeid) else pending).append(item)
    items[:] = [item for item in items if item not in complete]
    config.hook.pytest_deselected(items=complete)
    config._qbmm_already_complete = not pending
    print(f"QBMM batch={batch.path} complete={len(complete)} pending={len(pending)}", flush=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    config = item.config
    if not hasattr(config, "_qbmm_batch") or Path(item.path).parent != Path(__file__).parent:
        return
    reports = config._qbmm_reports.setdefault(item.nodeid, [])
    reports.append(
        dict(
            when=report.when,
            outcome=report.outcome,
            detail=str(report.longrepr),
            sections=report.sections,
            properties=report.user_properties,
            duration=report.duration,
        )
    )
    if report.when == "teardown":
        skipped = any(r["outcome"] == "skipped" for r in reports)
        if not skipped and (
            not any(r["when"] == "call" for r in reports)
            or any(r["when"] != "call" and r["outcome"] != "passed" for r in reports)
        ):
            return
        if any(r["outcome"] == "failed" for r in reports):
            status = "FAIL"
        elif skipped:
            status = "SKIP"
        else:
            status = "PASS"
        identity_file = config._qbmm_batch.path / "device.json"
        config._qbmm_batch.commit(
            item.nodeid,
            status=status,
            device=config._qbmm_batch.plan["parameters"]["device"],
            reports=reports,
            artifacts=[identity_file] if identity_file.exists() else [],
        )


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not hasattr(config, "_qbmm_batch"):
        return
    batch = config._qbmm_batch
    lines = ["PARTIAL" if exitstatus and not config._qbmm_already_complete else "COMPLETE"]
    for key in batch.plan["parameters"]["nodeids"]:
        record = batch.get(key)
        lines.append(f"{record['status'] if record else 'PARTIAL'} {key}")
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    (batch.path / "final_summary.txt").write_text(text, encoding="utf-8")
    if config._qbmm_already_complete and exitstatus == 5:
        session.exitstatus = 0
    elif config._qbmm_signal is not None:
        session.exitstatus = 128 + config._qbmm_signal
    elif exitstatus == 2:
        session.exitstatus = 130
    batch.close()
