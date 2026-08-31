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

"""Unit tests for common.env_util.load_peaks."""

import json

import pytest
from common.env_util import load_peaks


def _write_env(tmp_path, data):
    env_file = tmp_path / "env.json"
    env_file.write_text(json.dumps(data), encoding="utf-8")
    return str(env_file)


# --- file form --------------------------------------------------------------
def test_file_form_prefers_device_entry(tmp_path):
    env_file = _write_env(
        tmp_path,
        {"common": {"peak_flops": 100.0}, "DevA": {"peak_flops": 200.0, "peak_bw": 300.0}},
    )
    assert load_peaks(env_file, device_name="DevA") == (200.0, 300.0)


def test_file_form_merges_common_and_device(tmp_path):
    env_file = _write_env(
        tmp_path,
        {"common": {"peak_flops": 100.0, "peak_bw": 400.0}, "DevA": {"peak_flops": 200.0}},
    )
    # device entry overlays common; missing key falls back to common
    assert load_peaks(env_file, device_name="DevA") == (200.0, 400.0)


def test_file_form_falls_back_to_first_non_common(tmp_path):
    env_file = _write_env(
        tmp_path,
        {"common": {}, "DevA": {"peak_flops": 200.0, "peak_bw": 300.0}, "DevB": {}},
    )
    assert load_peaks(env_file) == (200.0, 300.0)


def test_file_form_common_is_default_device(tmp_path):
    # common carries peaks -> default device (e.g. 425/9*8 = 377.78 CUBE flops)
    env_file = _write_env(
        tmp_path,
        {
            "common": {"peak_flops": 377.78},
            "DevA": {"peak_flops": 200.0, "peak_bw": 300.0},
        },
    )
    assert load_peaks(env_file) == (377.78, None)


def test_file_form_unknown_device_falls_back(tmp_path):
    env_file = _write_env(tmp_path, {"DevA": {"peak_flops": 200.0, "peak_bw": 300.0}})
    assert load_peaks(env_file, device_name="Missing") == (200.0, 300.0)


def test_file_form_missing_peaks_yields_none(tmp_path):
    env_file = _write_env(tmp_path, {"DevA": {"other": 1}})
    assert load_peaks(env_file) == (None, None)


# --- inline json form -------------------------------------------------------
def test_inline_json_flat():
    assert load_peaks('{"peak_flops": 560, "peak_bw": 1275}') == (560.0, 1275.0)


def test_inline_json_missing_key_yields_none():
    assert load_peaks('{"peak_flops": 560}') == (560.0, None)


# --- malformed input --------------------------------------------------------
def test_none_input_yields_none():
    assert load_peaks(None) == (None, None)


def test_empty_string_yields_none():
    assert load_peaks("") == (None, None)


def test_invalid_json_yields_none():
    assert load_peaks("{not json") == (None, None)


def test_missing_file_degrades_gracefully():
    # A path-like string that does not exist must not raise; treated as
    # unparsable inline JSON -> (None, None).
    assert load_peaks("/no/such/env.json") == (None, None)


# --- value coercion ---------------------------------------------------------
def test_numeric_strings_are_coerced(tmp_path):
    env_file = _write_env(tmp_path, {"DevA": {"peak_flops": "560", "peak_bw": "1275.5"}})
    assert load_peaks(env_file) == (560.0, 1275.5)


def test_unparsable_numeric_yields_none(tmp_path):
    env_file = _write_env(tmp_path, {"DevA": {"peak_flops": "abc", "peak_bw": 1.0}})
    assert load_peaks(env_file) == (None, 1.0)


def test_non_dict_json_yields_none(tmp_path):
    env_file = tmp_path / "env.json"
    env_file.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_peaks(str(env_file)) == (None, None)


@pytest.mark.parametrize(
    "bad", [None, "not-a-number", object(), {"x": 1}],
)
def test_to_float_invalid_values(bad):
    from common.env_util import _to_float

    assert _to_float(bad) is None
