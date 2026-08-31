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

"""Unit tests for scripts.mindie_bench CLI parsing and case generation.

Covers --op parsing (JSON and lenient bare forms, "default" value, structure
params incl. reserved "func"), --config scan matrix (seqlen/dtype/sparse
only), responsibility-split validation (scan keys rejected in --op, structure
keys rejected in --config), cartesian expansion, precedence, and the
subcommand structure. No NPU / xpu_perf dependency.
"""

import argparse

import mindie_bench as mb
import pytest


# --- parse_dict (lenient bare form) -----------------------------------------
def test_parse_dict_strict_json():
    assert mb.parse_dict('{"fa": "default"}', "--op") == {"fa": "default"}


def test_parse_dict_lenient_bare_form():
    assert mb.parse_dict("{fa: default, mm: {K: 5120, N: 13824}}", "--op") == {
        "fa": "default",
        "mm": {"K": 5120, "N": 13824},
    }


def test_parse_dict_lenient_with_func_dotted_name():
    assert mb.parse_dict("{bsa: {func: torch_npu.npu_fusion_attention}}", "--op") == {
        "bsa": {"func": "torch_npu.npu_fusion_attention"}
    }


def test_parse_dict_lenient_numbers_and_bool():
    assert mb.parse_dict("{fa: {num_heads: 32, causal: false}}", "--op") == {
        "fa": {"num_heads": 32, "causal": False}
    }


def test_parse_dict_lenient_python_style_literals():
    # Python-style True/False/None normalize to JSON literals (True -> true)
    # instead of failing json.loads with a misleading "expecting value" error.
    assert mb.parse_op_spec("{fa: {causal: True}}") == {"fa": {"causal": True}}
    assert mb.parse_dict("{bsa: {mask_type: None}}", "--op") == {
        "bsa": {"mask_type": None}
    }


def test_parse_dict_invalid_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_dict("{fa: ", "--op")


# --- parse_op_spec ----------------------------------------------------------
def test_op_spec_omitted_means_all():
    assert mb.parse_op_spec(None) == {op: {} for op in mb.VALID_OPS}
    assert mb.parse_op_spec("") == {op: {} for op in mb.VALID_OPS}


def test_op_spec_default_value():
    assert mb.parse_op_spec('{"fa": "default"}') == {"fa": {}}


def test_op_spec_empty_object():
    # unified defaults form: {} == "default"
    assert mb.parse_op_spec('{"fa": {}}') == {"fa": {}}
    assert mb.parse_op_spec('{fa: {}, mm: {}}') == {"fa": {}, "mm": {}}
    assert mb.parse_op_spec("{fa: {}, mm: {}, gmm: {}, bsa: {}}") == {
        op: {} for op in ("fa", "mm", "gmm", "bsa")
    }


def test_op_spec_dict_params():
    spec = mb.parse_op_spec('{"fa": {"num_heads": 16}, "bsa": {"func": "f1"}}')
    assert spec == {"fa": {"num_heads": 16}, "bsa": {"func": "f1"}}


def test_op_spec_lenient_form():
    assert mb.parse_op_spec("{fa: default, mm: {K: 5120}}") == {"fa": {}, "mm": {"K": 5120}}


def test_op_spec_unknown_op_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"foo": "default"}')


def test_op_spec_bad_value_type_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"fa": 42}')


def test_op_spec_unknown_string_value_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"fa": "fast"}')


def test_op_spec_scan_key_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"fa": {"q_len": 8192}}')
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"fa": {"dtype": "bf16"}}')
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"fa": {"seqlen": [1024]}}')


def test_op_spec_mm_scan_axis_rejected_case_insensitive():
    # mm's scan axis is "M" (upper); _normalize_key lowercases it, so both
    # spellings must be rejected or seqlen scanning is silently overridden.
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"mm": {"M": 8192}}')
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"mm": {"m": 8192}}')
    # structure params that merely contain "m" in their name still work
    assert mb.parse_op_spec('{"mm": {"K": 5120, "N": 13824}}') == {
        "mm": {"K": 5120, "N": 13824}
    }


def test_op_spec_list_value_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_op_spec('{"fa": {"num_heads": [16, 32]}}')


# --- parse_config -----------------------------------------------------------
def test_config_normalizes_and_allows_scan_keys():
    cfg = mb.parse_config(
        '{"seqlen": [1024, 2048], "dtype": ["BF16", "fp8"], "sparse": [0.8], "quant": ["W8A8"]}'
    )
    assert cfg == {
        "seqlen": [1024, 2048],
        "dtype": ["bf16", "fp8"],
        "sparsity": [0.8],
        "quant_algo": ["W8A8"],
    }


def test_config_passes_through_peaks_seed_timeout():
    # Regression guard for CONFIG_ALLOWED_KEYS: seed/timeout/peak_flops/peak_bw
    # must survive parse and flow into every case template (MFU basis).
    cfg = mb.parse_config(
        '{"peak_flops": 560, "peak_bw": 1275, "seed": 42, "timeout": 300}'
    )
    assert cfg == {"peak_flops": 560, "peak_bw": 1275, "seed": 42, "timeout": 300}
    spec = mb.parse_op_spec('{"fa": "default"}')
    cases = mb.build_inline_cases(spec, cfg)
    case = cases["fa"][0]
    assert case["peak_flops"] == 560
    assert case["peak_bw"] == 1275
    assert case["seed"] == 42
    assert case["timeout"] == 300


def test_config_lenient_form():
    assert mb.parse_config("{seqlen: [1024], dtype: [bf16]}") == {
        "seqlen": [1024],
        "dtype": ["bf16"],
    }


def test_config_scalar_dtype_lowercased():
    assert mb.parse_config('{"dtype": "BF16"}') == {"dtype": "bf16"}


def test_config_rejects_func():
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_config('{"func": ["a"]}')


def test_config_rejects_structure_keys():
    with pytest.raises(argparse.ArgumentTypeError):
        mb.parse_config('{"num_heads": [32]}')


def test_config_none_returns_none():
    assert mb.parse_config(None) is None


# --- build_inline_cases -----------------------------------------------------
def test_build_inline_defaults_fill_omitted_params():
    spec = mb.parse_op_spec('{"fa": "default"}')
    cases = mb.build_inline_cases(spec, {"seqlen": [1024]})
    case = cases["fa"][0]
    assert case["q_len"] == 1024
    assert case["batch_size"] == 1
    assert case["num_heads"] == 32
    assert case["head_dim"] == 128
    assert case["dtype"] == "bf16"


def test_build_inline_config_scan_matrix():
    spec = mb.parse_op_spec('{"fa": "default"}')
    cases = mb.build_inline_cases(spec, {"seqlen": [1024, 2048], "dtype": ["bf16", "mxfp8"]})
    assert len(cases["fa"]) == 4


def test_build_inline_precedence_op_over_config():
    spec = mb.parse_op_spec('{"fa": {"num_heads": 16}}')
    cases = mb.build_inline_cases(spec, {"seqlen": [1024]})
    assert cases["fa"][0]["num_heads"] == 16


def test_build_inline_seqlen_maps_per_op_axis():
    spec = mb.parse_op_spec('{"fa": "default", "mm": "default"}')
    cases = mb.build_inline_cases(spec, {"seqlen": [4096, 8192]})
    assert cases["fa"][0]["q_len"] == 4096
    assert cases["mm"][0]["M"] == 4096


def test_build_inline_func_passthrough():
    spec = mb.parse_op_spec('{"bsa": {"func": "torch_npu.npu_fusion_attention"}}')
    cases = mb.build_inline_cases(spec, {"seqlen": [1024]})
    assert cases["bsa"][0]["func"] == "torch_npu.npu_fusion_attention"


def test_build_inline_sparsity_applies():
    spec = mb.parse_op_spec('{"bsa": "default"}')
    # "sparse" is normalized to "sparsity" by parse_config before expansion
    cfg = mb.parse_config('{"seqlen": [1024], "sparse": [0.6, 0.9]}')
    cases = mb.build_inline_cases(spec, cfg)
    sparsities = {c["sparsity"] for c in cases["bsa"]}
    assert sparsities == {0.6, 0.9}


def test_build_inline_all_ops_no_config():
    spec = mb.parse_op_spec(None)
    cases = mb.build_inline_cases(spec, None)
    assert set(cases) == set(mb.VALID_OPS)
    assert all(len(v) == 1 for v in cases.values())


# --- device auto-detection --------------------------------------------------
_USAGES_SAMPLE = """\
0 21.0
1 60.0
2 5.0
NPU 0 OK 82.6
"""


def test_parse_npu_smi_usage_filters_by_threshold():
    free = mb._parse_npu_smi_usage(_USAGES_SAMPLE, threshold=50.0)
    assert free == [0, 2]  # 1 (60%) excluded; non-numeric tail ignored


def test_parse_npu_smi_usage_bordered_table():
    # npu-smi info -t usages prints a bordered table; HBM-Usage is the 2nd
    # field and trailing columns (AICore-Usage) must not be read as the usage.
    sample = """\
+-----+--------------+----------------+
| NPU | HBM-Usage(%) | AICore-Usage   |
+=====+==============+================+
| 0   | 12.5%        | 1.2%           |
| 1   | 63.2%        | 0.8%           |
| 2   | 4.0%         | 3.0%           |
+-----+--------------+----------------+
"""
    assert mb._parse_npu_smi_usage(sample, threshold=50.0) == [0, 2]


def test_parse_npu_smi_usage_no_free():
    assert mb._parse_npu_smi_usage("0 90.0\n1 88.0\n", threshold=50.0) == []


def test_parse_npu_smi_usage_unparseable():
    assert mb._parse_npu_smi_usage("no devices here\n", threshold=50.0) == []


def test_resolve_devices_explicit():
    ns = argparse.Namespace(devices="0,2")
    assert mb._resolve_devices(ns) == "0,2"


def test_devices_defaults_to_zero():
    ns = mb.build_parser().parse_args(["run"])
    assert ns.devices == "0"
    assert mb._resolve_devices(ns) == "0"


def test_resolve_devices_auto_uses_detected(monkeypatch):
    monkeypatch.setattr(mb, "detect_free_devices", lambda: [0, 1, 3])
    ns = argparse.Namespace(devices="auto")
    assert mb._resolve_devices(ns) == "0,1,3"


def test_resolve_devices_auto_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(mb, "detect_free_devices", lambda: None)
    ns = argparse.Namespace(devices="auto")
    assert mb._resolve_devices(ns) == "0"


class _FakeProps:
    total_memory = 64 * 1024**3  # 64 GiB


class _FakeNpu:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 4

    @staticmethod
    def mem_get_info(i):
        free = {0: 60, 1: 10, 2: 50, 3: 32}[i]
        return free * 1024**3, 64 * 1024**3

    @staticmethod
    def get_device_properties(i):
        return _FakeProps()


def test_detect_free_torch_uses_free_hbm_fraction(monkeypatch):
    fake_torch = type("torch", (), {"npu": _FakeNpu})()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    # ids 0 (60/64), 2 (50/64), 3 (32/64) free >= 0.5; 1 (10/64) busy
    assert mb._detect_free_torch() == [0, 2, 3]


# --- report dir -------------------------------------------------------------
def test_resolve_report_dir_explicit():
    ns = argparse.Namespace(report_dir="reports/r1")
    assert mb._resolve_report_dir(ns) == "reports/r1"


def test_save_run_command_writes_file(tmp_path):
    ns = argparse.Namespace()
    path = mb._save_run_command(
        ns, str(tmp_path), {"fa": {}}, {"seqlen": [1024]}, "0"
    )
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "ops: {\"fa\": {}}" in content
    assert '"seqlen": [1024]' in content


# --- CLI structure ----------------------------------------------------------
def test_parser_has_three_subcommands():
    help_text = mb.build_parser().format_help()
    assert "run" in help_text and "report" in help_text and "compare" in help_text


def test_run_argv_maps_kebab_case():
    ns = mb.build_parser().parse_args(
        [
            "run",
            "--op",
            "{fa: {num_heads: 32}}",  # bare form, shell-split into tokens
            "--config",
            "{seqlen: [1024], dtype: [BF16]}",
            "--devices",
            "0,1",
            "--report-dir",
            "reports/r1",
        ]
    )
    assert ns.cmd == "run"
    # nargs="+" joins shell-split tokens back into the bare JSON string
    assert mb._join_nargs(ns.op) == "{fa: {num_heads: 32}}"
    assert mb._join_nargs(ns.config) == "{seqlen: [1024], dtype: [BF16]}"
    assert ns.devices == "0,1"
    assert ns.report_dir == "reports/r1"
    spec = mb.parse_op_spec(mb._join_nargs(ns.op))
    assert spec == {"fa": {"num_heads": 32}}


def test_join_nargs_plain_string():
    assert mb._join_nargs('{"fa": "default"}') == '{"fa": "default"}'
    assert mb._join_nargs(None) is None


def test_report_argv_defaults():
    ns = mb.build_parser().parse_args(["report"])
    assert ns.baseline_dir == mb._DEFAULT_BASELINE_DIR
    assert ns.no_html is False


def test_compare_argv_threshold():
    ns = mb.build_parser().parse_args(["compare", "--threshold", "0.05"])
    assert ns.threshold == 0.05


def test_cmd_report_passes_env_none(monkeypatch):
    # bridge into benchmark_report: env must be present (None; peaks come from
    # --config per case) so dev's env-file-based cmd_baseline does not crash
    # on a missing attribute.
    import sys
    import types

    fake = types.ModuleType("benchmark_report")
    captured = {}
    fake.cmd_baseline = lambda args: captured.update(args=args)
    fake.cmd_render = lambda args: None
    monkeypatch.setitem(sys.modules, "benchmark_report", fake)
    ns = mb.build_parser().parse_args(["report", "--report-dir", "reports/r", "--no-html"])
    mb.cmd_report(ns)
    assert captured["args"].env is None
    assert captured["args"].report_dir == "reports/r"
    assert captured["args"].no_html is True


def test_cmd_compare_passes_env_none(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("benchmark_report")
    captured = {}
    fake.cmd_compare = lambda args: captured.update(args=args)
    monkeypatch.setitem(sys.modules, "benchmark_report", fake)
    ns = mb.build_parser().parse_args(["compare"])
    mb.cmd_compare(ns)
    assert captured["args"].env is None
    assert captured["args"].threshold == 0.03


def test_help_exit_code():
    with pytest.raises(SystemExit):
        mb.main(["--help"])
