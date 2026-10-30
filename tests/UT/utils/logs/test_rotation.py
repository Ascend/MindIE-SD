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


# pylint: disable=redefined-outer-name

import importlib
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

logs = importlib.import_module("mindiesd.utils.logs.logging")


@pytest.fixture
def handler(tmp_path):
    instance = logs.MindIELogFileHandler(str(tmp_path), 4, 1024, 2, "daily")
    yield instance
    instance.close()


def record(text="message"):
    return logging.LogRecord("rotation-test", logging.INFO, __file__, 1, text, (), None)


def test_real_log_file_open_size_and_time_rotation(handler):
    assert not handler.should_rollover(record())
    handler.emit(record("hello"))
    assert handler.stream is not None
    assert not handler.should_rollover(record())
    handler._max_file_size = 1
    assert handler.should_rollover(record())
    handler._max_file_size = 0
    handler._next_rollover = 0
    assert handler.should_rollover(record())
    old_path = handler._cur_log_file
    with patch.object(handler, "_get_log_name", return_value="rotated.log"):
        handler.do_rollover()
    assert handler._cur_log_file.endswith("rotated.log")
    assert handler.stream is None
    assert (handler._real_log_path + "/rotated.log") != old_path
    handler.emit(record("after rotate"))
    with open(handler._cur_log_file, encoding="utf-8") as stream:
        assert "after rotate" in stream.read()


def test_emit_handles_io_failure(handler):
    with (
        patch.object(handler, "should_rollover", side_effect=OSError("disk")),
        patch.object(handler, "handleError") as report,
    ):
        entry = record()
        handler.emit(entry)
        report.assert_called_once_with(entry)


def test_history_count_and_time_cleanup(handler, tmp_path):
    old = tmp_path / "old.log"
    old.write_text("old")
    handler._history_files = [(str(old), "20000101000000")]
    handler._delete_file_by_time()
    assert not old.exists()
    recent = datetime.now().strftime("%Y%m%d%H%M%S")
    handler._history_files = [(str(tmp_path / "absent"), recent)]
    handler._delete_file_by_time()
    assert len(handler._history_files) == 1
    handler._max_file_num = 1
    handler._delete_file_by_number()
    assert handler._history_files == []
    handler._max_file_num = 0
    handler._delete_file_by_number()


def test_filename_parse_and_nonfile(handler, tmp_path):
    assert handler._get_time_str("invalid") is None
    assert handler._get_time_str(f"mindie-sd_{logs.get_pid()}_20261028000000.log") == "20261028000000"
    directory = tmp_path / "directory"
    directory.mkdir()
    handler._cur_log_file = str(directory)
    assert not handler.should_rollover(record())
    handler._cur_log_file = str(tmp_path / "missing")


@pytest.mark.parametrize(
    "cycle,age,expected",
    [
        ("daily", 1, True),
        ("daily", 3, False),
        ("weekly", 5, True),
        ("weekly", 15, False),
        ("monthly", 5, True),
        ("monthly", 65, False),
        ("yearly", 100, True),
        ("yearly", 800, False),
    ],
)
def test_calendar_rotation(handler, cycle, age, expected):
    handler._rotate_cycle = cycle
    current = datetime(2026, 10, 28)
    assert handler._check_time_rotate(current - timedelta(days=age), current) is expected


def test_calendar_boundary_and_bad_cycle(handler):
    handler._rotate_cycle_num = 1
    assert not handler._check_month(datetime(2025, 12, 31), datetime(2026, 1, 1))
    assert not handler._check_month(datetime(2026, 9, 28), datetime(2026, 10, 28))
    assert handler._check_month(datetime(2026, 9, 29), datetime(2026, 10, 28))
    assert not handler._check_year(datetime(2025, 9, 30), datetime(2026, 10, 1))
    assert not handler._check_year(datetime(2025, 10, 1), datetime(2026, 10, 1))
    assert handler._check_year(datetime(2025, 10, 2), datetime(2026, 10, 1))
    assert handler._check_year(datetime(2025, 11, 1), datetime(2026, 10, 1))
    handler._rotate_cycle = "invalid"
    with pytest.raises(ValueError, match="Unknown rotate"):
        handler._check_time_rotate(datetime.now(), datetime.now())


def test_path_owner_permission_and_mkdir(tmp_path):
    assert logs.check_path("") is False
    assert logs.check_path("x" * logs.MAX_PATH_LEN) is False
    tmp_path.chmod(0o750)
    assert logs.check_path(str(tmp_path), checking_conf=True)
    assert logs.check_owner_permission(str(tmp_path), 0o750)
    assert not logs.check_owner_permission(str(tmp_path), 0o700)
    with patch.object(logs, "get_uid", return_value=-1):
        assert not logs.check_owner_permission(str(tmp_path), 0o750)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    assert not logs.check_path(str(link))
    assert logs.create_directory_with_permissions(str(tmp_path / "nested/path"), 0o750)
    with patch.object(logs.os, "makedirs", side_effect=PermissionError("denied")):
        assert not logs.create_directory_with_permissions(str(tmp_path / "new/path"), 0o750)


def test_formatter_levels_and_init(tmp_path):
    formatter = logs.LoggerFormatter("%(asctime)s %(message)s")
    assert formatter._filter(None) == "log is None!"
    assert formatter._filter("a\n\r\t  b") == "a b"
    assert len(formatter._filter("x" * (logs.MAX_LOG_STRING_LEN + 1))) == logs.MAX_LOG_STRING_LEN
    assert "message" in formatter.format(record())
    for name, value in [("DEBUG", 10), ("info", 20), ("WARN", 30), ("ERROR", 40), ("CRITICAL", 50)]:
        assert logs.str_to_loglevel(name) == value
    with pytest.raises(ValueError):
        logs.str_to_loglevel("invalid")
    config = dict(
        component_log_level="INFO",
        disable_log=False,
        component_log_verbose="true",
        component_log_stdout="true",
        component_log_to_file="true",
        mindie_log_path=str(tmp_path),
        rotate_max_file_num=4,
        rotate_max_file_size=1,
        rotate_cycle_num=1,
        rotate_cycle="daily",
    )
    test_logger = logging.Logger("isolated")
    with patch.object(logs, "logger", test_logger), patch.object(logs, "ENV", SimpleNamespace(**config)):
        assert logs.init_logger() is test_logger
        assert len(test_logger.handlers) == 2
    for output in test_logger.handlers:
        output.close()
    config.update(disable_log=True)
    with patch.object(logs, "logger", logging.Logger("disabled")), patch.object(logs, "ENV", SimpleNamespace(**config)):
        assert logs.init_logger().disabled
    config.update(disable_log=False, component_log_verbose="false", component_log_stdout="false")
    with (
        patch.object(logs, "logger", logging.Logger("bad-path")),
        patch.object(logs, "ENV", SimpleNamespace(**config)),
        patch.object(logs, "check_path", return_value=False),
    ):
        assert not logs.init_logger().handlers
    config.update(mindie_log_path="relative")
    with (
        patch.object(logs, "logger", logging.Logger("failed-directory")),
        patch.object(logs, "ENV", SimpleNamespace(**config)),
        patch.object(logs, "create_directory_with_permissions", return_value=False),
    ):
        assert not logs.init_logger().handlers


@pytest.mark.parametrize("entry_kind", ["outside_symlink", "inside_symlink", "directory"])
def test_history_cleanup_only_removes_owned_regular_entries(tmp_path, entry_kind):
    directory = tmp_path / "debug"
    directory.mkdir()
    sibling = tmp_path / "debug-other"
    sibling.mkdir()
    target = (sibling if entry_kind == "outside_symlink" else directory) / "keep.txt"
    target.write_text("must survive", encoding="utf-8")
    unsafe = directory / f"mindie-sd_{logs.get_pid()}_20000101000000.log"
    if entry_kind == "directory":
        unsafe.mkdir()
    else:
        unsafe.symlink_to(target)
    normal = directory / f"mindie-sd_{logs.get_pid()}_20000102000000.log"
    normal.write_text("expired regular history", encoding="utf-8")
    instance = logs.MindIELogFileHandler(str(directory), 4, 1024, 1, "daily")
    try:
        assert target.read_text(encoding="utf-8") == "must survive"
        assert unsafe.is_dir() if entry_kind == "directory" else unsafe.is_symlink()
        assert not normal.exists(), "Expired regular log history must still be cleaned"
        assert instance._history_files == []
    finally:
        instance.close()
