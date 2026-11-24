# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Checkpoint only this directory's real operator cases, after each verdict."""

from pathlib import Path
import signal
import sys
import pytest
from tests.ops.device_support import selected_device_index
from tests.ops.eagle_quant_block_sparse_attention.eqbsa_state import Batch, read, save, small_device_identity

HERE = Path(__file__).resolve().parent


def pytest_addoption(parser):
    # Whole-suite collection already registers --resume in the sibling QBMM
    # plugin. Register it here for an isolated EQBSA invocation, in either cwd.
    paths = []
    skip_value = False
    for arg in sys.argv[1:]:
        if skip_value:
            skip_value = False
            continue
        if arg in ("--resume", "--test-mode", "-k", "-m", "--confcutdir", "--rootdir", "-c"):
            skip_value = True
            continue
        path = Path(arg.split("::", 1)[0])
        if not arg.startswith("-") and path.exists():
            paths.append(path.resolve())
    if any(path != HERE and not path.is_relative_to(HERE) for path in paths):
        return
    if not paths and Path.cwd().resolve() != HERE:
        return
    # Another operator can already own the shared spelling in whole-suite collection.
    try:
        parser.addoption("--resume", default=None, help="Resume one operator's small-shape batch")
    except ValueError as error:
        if "already added" not in str(error):
            raise


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if config.option.collectonly:
        return
    local = [item for item in items if Path(item.path).resolve().parent == HERE]
    if not local:
        return
    device = selected_device_index()
    resume = config.getoption("--resume", default=None)
    nodeids = [item.nodeid for item in local]
    if resume:
        old = read(Path(resume) / "plan.json")["parameters"]
        if config.option.keyword or config.option.markexpr:
            if nodeids != old["nodeids"]:
                raise pytest.UsageError("resume conflicts with the selected tests")
        elif not set(old["nodeids"]).issubset(nodeids):
            raise pytest.UsageError("original cases are missing from this collection")
        else:
            nodeids = old["nodeids"]
        # Fully committed batches skip every fixture. Verify the original board and chip here too.
        device_file = Path(resume) / "device.json"
        if device_file.exists() and read(device_file) != small_device_identity(device):
            raise pytest.UsageError("small-shape resume cannot change physical device or host")
    plan = {
        "nodeids": nodeids,
        "device": device,
        "scenarios": {item.nodeid: item.callspec.params for item in local if item.nodeid in nodeids},
    }
    batch = Batch("small", plan, resume)
    config._eqbsa_batch, config._eqbsa_reports = batch, {}
    config._eqbsa_signal = None
    config._eqbsa_old_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupt(sig, frame):
        config._eqbsa_signal = sig
        raise KeyboardInterrupt

    for sig in config._eqbsa_old_signals:
        signal.signal(sig, interrupt)
    batch.history([device])
    completed = [item for item in local if item.nodeid not in nodeids or batch.get(item.nodeid)]
    items[:] = [item for item in items if item not in completed]
    config.hook.pytest_deselected(items=completed)
    config._eqbsa_pending = [key for key in nodeids if batch.get(key) is None]
    print(f"EQBSA batch={batch.path} committed={len(nodeids) - len(config._eqbsa_pending)}/{len(nodeids)}", flush=True)
    if any(batch.get(key) and batch.get(key)["status"] == "FAIL" for key in nodeids):
        # Keep the saved verdict and let sessionfinish print its original report.
        items[:] = [item for item in items if Path(item.path).resolve().parent != HERE]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    config = item.config
    if not hasattr(config, "_eqbsa_batch") or Path(item.path).resolve().parent != HERE:
        return
    reports = config._eqbsa_reports.setdefault(item.nodeid, [])
    reports.append(
        {
            "when": report.when,
            "outcome": report.outcome,
            "detail": str(report.longrepr),
            "sections": report.sections,
            "metrics": report.user_properties,
            "duration": report.duration,
        }
    )
    if report.when != "teardown":
        return
    if not any(r["when"] == "call" for r in reports) and not any(r["outcome"] == "skipped" for r in reports):
        return  # No numerical verdict after a failed setup.
    if any(r["when"] != "call" and r["outcome"] == "failed" for r in reports):
        return
    status = (
        "FAIL"
        if any(r["outcome"] == "failed" for r in reports)
        else "SKIP"
        if any(r["outcome"] == "skipped" for r in reports)
        else "PASS"
    )
    batch = config._eqbsa_batch
    path = batch.attempt("small") / "case.json"
    save(path, {"nodeid": item.nodeid, "reports": reports})
    device_file = batch.path / "device.json"
    batch.commit(
        item.nodeid,
        status=status,
        device=batch.plan["parameters"]["device"],
        reports=reports,
        artifacts=[path, *([device_file] if device_file.exists() else [])],
    )


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not hasattr(config, "_eqbsa_batch"):
        return
    batch = config._eqbsa_batch
    try:
        records = [(key, batch.get(key)) for key in batch.plan["parameters"]["nodeids"]]
        failed = any(r and r["status"] == "FAIL" for _, r in records)
        complete = all(r and r["status"] == "PASS" for _, r in records)
        lines = ["COMPLETE" if complete else "PARTIAL", "Status  Case  Metrics/diagnostics"]
        if not any(r for _, r in records):
            lines.append("No completed results")
        for key, record in records:
            lines.append(f"{record['status'] if record else 'PARTIAL'} {key}")
            if record:
                lines.extend(str(r["metrics"]) for r in record["reports"] if r["when"] == "call")
        text = "\n".join(lines) + "\n"
        print(text, flush=True)
        (batch.path / "final_summary.txt").write_text(text, encoding="utf-8")
        if config._eqbsa_signal:
            session.exitstatus = 128 + config._eqbsa_signal
        elif failed:
            session.exitstatus = 1
        elif exitstatus == 5 and complete:
            session.exitstatus = 0
        elif exitstatus == 2:
            session.exitstatus = 130
    finally:
        for sig, handler in config._eqbsa_old_signals.items():
            signal.signal(sig, handler)
        batch.close()
