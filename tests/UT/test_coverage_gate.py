# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

import pytest

from tests.scripts.check_file_coverage import check_coverage


def test_missing_source_cannot_be_hidden_by_high_total(tmp_path):
    source = tmp_path / "mindiesd"
    source.mkdir()
    (source / "covered.py").write_text("value = 1\n")
    (source / "unimported.py").write_text("value = 2\n")
    report = {
        "meta": {"branch_coverage": True},
        "files": {
            "mindiesd/covered.py": {
                "summary": {"num_statements": 1, "covered_lines": 1, "num_branches": 0, "covered_branches": 0}
            }
        },
    }
    rows = check_coverage(report, source)
    assert rows[0]["passed"]
    assert not rows[1]["passed"]


def test_branch_misses_count_even_when_all_lines_are_covered(tmp_path):
    source = tmp_path / "mindiesd"
    source.mkdir()
    (source / "conditional.py").write_text("value = 1\n")
    report = {
        "meta": {"branch_coverage": True},
        "files": {
            "mindiesd/conditional.py": {
                "summary": {"num_statements": 4, "covered_lines": 4, "num_branches": 2, "covered_branches": 0}
            }
        },
    }
    row = check_coverage(report, source)[0]
    assert not row["passed"]
    assert row["percent"] == pytest.approx(100 * 4 / 6)
    with pytest.raises(ValueError, match="branch"):
        check_coverage({"meta": {"branch_coverage": False}}, source)
