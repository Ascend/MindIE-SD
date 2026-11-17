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


import io
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mindiesd.utils import file_utils as files
from mindiesd.utils.exception import ParametersInvalid


def secure_file(tmp_path, name="input.txt", text="first\nsecond\n"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    path.chmod(0o640)
    return path


def test_safe_open_modes_and_size(tmp_path):
    path = secure_file(tmp_path)
    with files.safe_open(str(path), "r+", encoding="utf-8") as stream:
        assert stream.readline() == "first\n"
        stream.seek(0)
        stream.write("FIRST")
    with files.safe_open(str(path), "rb") as stream:
        assert stream.read().startswith(b"FIRST")
    with pytest.raises(ParametersInvalid, match="size"):
        files.safe_open(str(path), max_file_size=1)


def test_safe_open_honors_explicit_permission_limit(tmp_path):
    path = secure_file(tmp_path)
    path.chmod(0o600)
    with pytest.raises(PermissionError, match="permission"):
        files.safe_open(str(path), permission_mode=0o400)


def test_path_validation_and_symlink(tmp_path):
    path = secure_file(tmp_path)
    assert files.standardize_path(str(path)) == os.path.realpath(path)
    with pytest.raises(ParametersInvalid, match="None"):
        files.standardize_path(None)
    with pytest.raises(ParametersInvalid, match="length"):
        files.standardize_path("long", max_path_length=3)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ParametersInvalid, match="symbolic"):
        files.standardize_path(str(link))
    assert files.standardize_path(str(link), check_link=False) == str(path)


def test_file_and_directory_safety(tmp_path):
    path = secure_file(tmp_path)
    tmp_path.chmod(0o750)
    files.check_file_safety(str(path))
    files.check_dir_safety(str(tmp_path))
    files.check_file_under_dir(str(tmp_path))
    (tmp_path / "subdir").mkdir(mode=0o750)
    files.check_file_under_dir(str(tmp_path))
    for method in (files.check_file_safety, files.check_dir_safety):
        with pytest.raises(ParametersInvalid, match="exist"):
            method(str(tmp_path / "missing"))
    with pytest.raises(ParametersInvalid, match="not a file"):
        files.check_file_safety(str(tmp_path))
    with pytest.raises(ParametersInvalid, match="not a dir"):
        files.check_dir_safety(str(path))
    with pytest.raises(ParametersInvalid, match="number"):
        files.check_dir_safety(str(tmp_path), max_file_num=0)
    files.check_dir_safety(str(tmp_path), is_check_file_num=False)
    files.check_file_safety(str(path), max_file_size=0, is_check_file_size=False)
    with pytest.raises(PermissionError):
        files.check_max_permission(str(path), 0o400)
    assert set(files.safe_listdir(str(tmp_path))) == {"input.txt", "subdir"}
    with pytest.raises(ParametersInvalid):
        files.safe_listdir(str(tmp_path), max_file_num=0)
    assert files.safe_readlines(io.StringIO("a\nb\n")) == ["a\n", "b\n"]
    with pytest.raises(ParametersInvalid, match="lines"):
        files.safe_readlines(io.StringIO("a\nb\n"), max_line_num=1)


@pytest.mark.parametrize("uid,gid,allowed", [(0, 3, True), (11, 3, True), (12, 22, True), (12, 3, False)])
def test_owner_policy(uid, gid, allowed):
    metadata = SimpleNamespace(st_uid=11, st_gid=22)
    with patch.object(
        files, "os", SimpleNamespace(stat=lambda path: metadata, geteuid=lambda: uid, getgid=lambda: gid)
    ):
        if allowed:
            files.check_owner("/input")
        else:
            with pytest.raises(ParametersInvalid, match="permission"):
                files.check_owner("/input")
