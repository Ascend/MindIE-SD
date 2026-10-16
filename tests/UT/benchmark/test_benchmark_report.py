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

"""Unit tests for scripts.benchmark_report (baseline/compare/render).

Uses synthetic jsonl entries in the exact xpu-perf export format
({"op_name", "arguments", "targets"}) so the offline accounting path is
verified without any NPU or xpu_perf dependency.
"""

import json
import shutil

import benchmark_report as br
import pytest


def _entry(op_name, arguments, targets):
    return {"op_name": op_name, "arguments": arguments, "targets": targets}


def _write_report(tmp_path, entries):
    jsonl = tmp_path / "run" / "NPU" / "op" / "op-NPU.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(
        "".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8"
    )
    return tmp_path / "run"


# --- _slot_for --------------------------------------------------------------
def test_slot_for_fa_omits_default_kv_len():
    # kv_len not set -> q_len default -> omitted
    args = {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0}
    assert br._slot_for("fa", args) == "q_len=8192|dtype=bf16"


def test_slot_for_fa_omits_explicit_equal_kv_len():
    args = {"q_len": 8192, "kv_len": 8192, "dtype": "bf16"}
    assert br._slot_for("fa", args) == "q_len=8192|dtype=bf16"


def test_slot_for_fa_keeps_different_kv_len():
    args = {"q_len": 8192, "kv_len": 16384, "dtype": "bf16"}
    assert br._slot_for("fa", args) == "q_len=8192|kv_len=16384|dtype=bf16"


def test_slot_for_gmm_and_mm():
    assert br._slot_for("gmm", {"num_tokens": 1024, "top_k": 8, "quant_algo": "NO_QUANT"}) == (
        "num_tokens=1024|top_k=8|quant_algo=NO_QUANT"
    )
    assert br._slot_for("mm", {"M": 1024, "K": 5120, "N": 13824, "quant_algo": "W8A8"}) == (
        "M=1024|K=5120|N=13824|quant_algo=W8A8"
    )


# --- parse_slot / relative_drift -------------------------------------------
def test_parse_slot_roundtrip():
    assert br.parse_slot("q_len=8192|kv_len=16384|dtype=bf16") == {
        "q_len": "8192",
        "kv_len": "16384",
        "dtype": "bf16",
    }


def test_parse_slot_empty():
    assert br.parse_slot("") == {}


@pytest.mark.parametrize(
    ("current", "baseline", "expected"),
    [
        (0.5, 0.5, 0.0),
        (0.55, 0.5, 0.1),
        (None, 0.5, None),
        (0.5, None, None),
        (1.0, 0.0, float("inf")),
        (0.0, 0.0, 0.0),
    ],
)
def test_relative_drift(current, baseline, expected):
    actual = br.relative_drift(current, baseline)
    if expected is None or expected == float("inf"):
        assert actual == expected
    else:
        assert actual == pytest.approx(expected)


# --- collect_baseline -------------------------------------------------------
def test_collect_baseline_recomputes_util(tmp_path):
    entries = [
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {"latency(us)": 100.0, "calc_flops_power(tflops)": 280.0, "mem_bw(GB/s)": 637.5},
        ),
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "mxfp8"},
            {"latency(us):": 0},
        ),
    ]
    # second entry has a malformed targets key (latency(us): instead of
    # latency(us)) -> must be skipped without raising.
    grouped = br.collect_baseline(entries)
    slot = "q_len=8192|dtype=bf16"
    assert slot in grouped["fa"]
    assert grouped["fa"][slot]["MFU"] == 0.5
    assert grouped["fa"][slot]["MBU"] == 0.5
    assert grouped["fa"][slot]["latency(us)"] == 100.0


def test_collect_baseline_skips_unknown_op(tmp_path):
    entries = [_entry("unknown_op", {}, {"latency(us)": 1.0})]
    assert br.collect_baseline(entries) == {}


def test_collect_baseline_drops_errored_cases(tmp_path):
    # an errored/crashed case produces an empty summary -> must not appear
    entries = [
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {},
        ),  # no targets at all
        _entry(
            "fa",
            {"q_len": 4096, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {"latency(us)": 42.0, "calc_flops_power(tflops)": 100.0, "mem_bw(GB/s)": 200.0},
        ),
    ]
    grouped = br.collect_baseline(entries)
    assert "q_len=8192|dtype=bf16" not in grouped["fa"]
    assert "q_len=4096|dtype=bf16" in grouped["fa"]


def test_collect_baseline_latency_carried_through(tmp_path):
    entries = [
        _entry(
            "mm",
            {
                "M": 1024, "K": 5120, "N": 13824, "quant_algo": "NO_QUANT",
                "peak_flops": 560.0, "peak_bw": 1275.0,
            },
            {"latency(us)": 42.5, "calc_flops_power(tflops)": 100.0, "mem_bw(GB/s)": 200.0},
        )
    ]
    grouped = br.collect_baseline(entries)
    slot = "M=1024|K=5120|N=13824|quant_algo=NO_QUANT"
    assert grouped["mm"][slot]["latency(us)"] == 42.5


# --- compare ----------------------------------------------------------------
def test_compare_no_violations(tmp_path):
    entries = [
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {"latency(us)": 100.0, "calc_flops_power(tflops)": 280.0, "mem_bw(GB/s)": 637.5},
        )
    ]
    report_dir = _write_report(tmp_path, entries)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    (baseline_dir / "fa.json").write_text(
        json.dumps({"q_len=8192|dtype=bf16": {"MFU": 0.5, "MBU": 0.5, "latency(us)": 100.0}}),
        encoding="utf-8",
    )
    current = br.collect_baseline(br.load_report_entries(report_dir))
    checked, violations = br.compare(current, baseline_dir, threshold=0.03)
    assert checked >= 1
    assert violations == []


def test_compare_detects_drift(tmp_path):
    entries = [
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {"latency(us)": 100.0, "calc_flops_power(tflops)": 300.0, "mem_bw(GB/s)": 637.5},
        )
    ]
    report_dir = _write_report(tmp_path, entries)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    (baseline_dir / "fa.json").write_text(
        json.dumps({"q_len=8192|dtype=bf16": {"MFU": 0.5, "MBU": 0.5, "latency(us)": 100.0}}),
        encoding="utf-8",
    )
    current = br.collect_baseline(br.load_report_entries(report_dir))
    checked, violations = br.compare(current, baseline_dir, threshold=0.03)
    assert violations, "expected at least one drift violation"
    op, slot, metric, detail = violations[0]
    assert op == "fa" and metric == "MFU"


def test_compare_flags_missing_in_baseline(tmp_path):
    entries = [
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {"latency(us)": 1.0},
        )
    ]
    report_dir = _write_report(tmp_path, entries)
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    (baseline_dir / "fa.json").write_text("{}", encoding="utf-8")
    current = br.collect_baseline(br.load_report_entries(report_dir))
    _, violations = br.compare(current, baseline_dir, threshold=0.03)
    assert any(v[2] == "missing_in_baseline" for v in violations)


# --- load_report_entries ----------------------------------------------------
def test_load_report_entries_reads_lines_in_file_order(tmp_path):
    entries = [
        _entry(
            "fa",
            {"q_len": 4096, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {"latency(us)": 1.0},
        ),
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {"latency(us)": 2.0},
        ),
    ]
    report_dir = _write_report(tmp_path, entries)
    loaded = br.load_report_entries(report_dir)
    assert len(loaded) == 2
    assert loaded[0]["arguments"]["q_len"] == 4096


def test_load_report_entries_merges_multiple_runs(tmp_path):
    # Two separate runs (subdirs) under one report dir are merged into one set
    # of entries; a later run overwrites the same slot.
    run1 = _write_report(
        tmp_path,
        [
            _entry(
                "fa",
                {"q_len": 4096, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
                {"latency(us)": 1.0},
            )
        ],
    )
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    target1 = merged_dir / "run1" / "NPU"
    target1.mkdir(parents=True, exist_ok=True)

    for jsonl in (run1 / "NPU").rglob("*.jsonl"):
        shutil.copy2(jsonl, target1 / jsonl.name)
    target2 = merged_dir / "run2" / "NPU"
    target2.mkdir(parents=True, exist_ok=True)
    (target2 / "fa-NPU.jsonl").write_text(
        json.dumps(
            _entry(
                "fa",
                {"q_len": 4096, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
                {"latency(us)": 99.0},
            )
        )
        + "\n"
        + json.dumps(
            _entry(
                "mm",
                {
                    "M": 1024, "K": 5120, "N": 13824, "quant_algo": "NO_QUANT",
                    "peak_flops": 560.0, "peak_bw": 1275.0,
                },
                {"latency(us)": 3.0},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = br.load_report_entries(merged_dir)
    grouped = br.collect_baseline(loaded)
    # fa slot from run2 (newer mtime) wins; mm from run2 present
    assert grouped["fa"]["q_len=4096|dtype=bf16"]["latency(us)"] == 99.0
    assert "M=1024|K=5120|N=13824|quant_algo=NO_QUANT" in grouped["mm"]


def test_render_html_shows_command_and_env(tmp_path):
    data = {
        "generated_at": "2026-01-01T00:00:00+08:00",
        "backend": {"device_name": "Ascend950PR"},
        "runtime": {"device_ids": [0]},
        "env": {"peak_flops": 377.78, "peak_bw": None},
        "command": "python mindie_bench.py run --op {bsa: default} --config {seqlen: [1024]}",
        "ops": {
            "bsa": {
                "q_len=1024|num_heads=32|head_dim=128|sparsity=0.8|dtype=bf16": {
                    "MFU": 0.5, "MBU": 0.6, "latency(us)": 3.0,
                }
            }
        },
    }
    out = tmp_path / "cmd.html"
    br.render_html(data, str(out))
    content = out.read_text(encoding="utf-8")
    assert "Command" in content
    assert "mindie_bench.py run --op {bsa: default}" in content
    assert "Peak config (CUBE flops / bandwidth)" in content
    assert "377.78" in content


# --- build_csv_section / aggregate -----------------------------------------
def test_aggregate_cases_fa_series():
    cases = {
        "q_len=1024|dtype=bf16": {"MFU": 0.1, "MBU": 0.2, "latency(us)": 3.0},
        "q_len=2048|dtype=bf16": {"MFU": 0.2, "MBU": 0.3, "latency(us)": 4.0},
    }
    agg = br._aggregate_cases("fa", cases)
    assert agg["bf16"][1024]["MFU"] == 0.1
    assert agg["bf16"][2048]["latency(us)"] == 4.0


def test_build_csv_section_bsa_has_sparsity_column():
    cases = {
        "q_len=1024|sparsity=0.8|dtype=bf16": {"MFU": 0.1, "MBU": 0.2, "latency(us)": 3.0},
        "q_len=2048|sparsity=0.9|dtype=bf16": {"MFU": 0.2, "MBU": 0.3, "latency(us)": 4.0},
    }
    csv_data = br.build_csv_section({"bsa": cases})
    rows = csv_data["bsa"]["bf16"]
    assert rows[0]["sparsity"] == "0.8"
    assert rows[1]["sparsity"] == "0.9"
    assert "seq_len" in rows[0]


# --- CSV peak update (CSV as data source) -----------------------------------
def test_build_csv_section_carries_peak_columns():
    cases = {
        "q_len=1024|dtype=bf16": {"MFU": 0.1, "MBU": 0.2, "latency(us)": 3.0},
    }
    csv_data = br.build_csv_section(
        {"fa": cases}, peaks={"fa": {"peak_flops": 560.0, "peak_bw": 1275.0}}
    )
    row = csv_data["fa"]["bf16"][0]
    assert row["peak_flops"] == 560.0
    assert row["peak_bw"] == 1275.0


def test_write_csv_files_roundtrips_peaks(tmp_path):
    cases = {
        "q_len=1024|dtype=bf16": {"MFU": 0.5, "MBU": 0.5, "latency(us)": 3.0},
    }
    out = tmp_path / "csvs"
    br.write_csv_files(
        {"fa": cases}, str(out), peaks={"fa": {"peak_flops": 560.0, "peak_bw": 1275.0}}
    )
    assert br.read_peaks_from_csv(str(out), ["fa"]) == {
        "fa": {"peak_flops": 560.0, "peak_bw": 1275.0}
    }


def test_read_peaks_from_csv_missing_file_returns_empty(tmp_path):
    assert br.read_peaks_from_csv(str(tmp_path / "no_such_dir"), ["fa"]) == {}


def test_read_peaks_from_csv_skips_non_numeric(tmp_path):
    cases = {
        "q_len=1024|dtype=bf16": {"MFU": 0.5, "MBU": 0.5, "latency(us)": 3.0},
    }
    out = tmp_path / "csvs"
    br.write_csv_files(
        {"fa": cases}, str(out), peaks={"fa": {"peak_flops": 560.0, "peak_bw": None}}
    )
    csv_path = out / "fa.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("560.0", "not-a-number"), encoding="utf-8"
    )
    assert br.read_peaks_from_csv(str(out), ["fa"]) == {}


def test_op_peaks_skips_early_entries_without_peaks():
    # the earliest run may lack peaks; a later run that carries them must win
    entries = [
        _entry("fa", {"q_len": 1024, "dtype": "bf16"}, {"latency(us)": 1.0}),
        _entry(
            "fa",
            {"q_len": 1024, "dtype": "bf16", "peak_flops": 560.0, "peak_bw": 1275.0},
            {"latency(us)": 1.0},
        ),
        _entry(
            "gmm",
            {"num_tokens": 8, "top_k": 4, "quant_algo": "NO_QUANT"},
            {"latency(us)": 1.0},
        ),
    ]
    peaks = br._op_peaks(entries)
    assert peaks["fa"] == {"peak_flops": 560.0, "peak_bw": 1275.0}
    assert "gmm" not in peaks


def test_cmd_baseline_applies_csv_peak_update(tmp_path, monkeypatch):
    # end-to-end CSV-as-data-source: run report -> edit peak in <op>.csv ->
    # re-run report -> MFU and env peaks are recomputed from the CSV value.
    import types

    entries = [
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0},
            {"latency(us)": 100.0, "calc_flops_power(tflops)": 280.0},
        )
    ]
    report_dir = _write_report(tmp_path, entries)
    out = tmp_path / "reports"
    monkeypatch.setattr(br, "DEFAULT_REPORT_DIR", str(out))
    args = types.SimpleNamespace(
        report_dir=str(report_dir),
        baseline_dir=str(tmp_path / "baselines"),
        no_html=True,
    )

    def _snapshot():
        newest = max(out.glob("benchmark-report_*.json"), key=lambda p: p.name)
        return json.loads(newest.read_text(encoding="utf-8"))

    br.cmd_baseline(args)
    slot = "q_len=8192|dtype=bf16"
    assert _snapshot()["ops"]["fa"][slot]["MFU"] == 0.5
    assert _snapshot()["env"]["peak_flops"] == 560.0

    csv_path = out / "fa.csv"
    assert csv_path.exists()
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("560.0", "700.0"), encoding="utf-8"
    )
    br.cmd_baseline(args)
    assert _snapshot()["ops"]["fa"][slot]["MFU"] == pytest.approx(0.4)
    assert _snapshot()["env"]["peak_flops"] == 700.0
    # user edits the CSV data source (560 -> 700); report must pick the new one
    cases = {
        "q_len=1024|dtype=bf16": {"MFU": 0.5, "MBU": 0.5, "latency(us)": 3.0},
    }
    out = tmp_path / "csvs"
    br.write_csv_files(
        {"fa": cases}, str(out), peaks={"fa": {"peak_flops": 560.0, "peak_bw": None}}
    )
    csv_path = out / "fa.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("560.0", "700.0"), encoding="utf-8"
    )
    assert br.read_peaks_from_csv(str(out), ["fa"]) == {"fa": {"peak_flops": 700.0}}


def test_apply_csv_peak_updates_recomputes_util():
    entries = [
        _entry(
            "fa",
            {"q_len": 8192, "dtype": "bf16", "peak_flops": 560.0},
            {"latency(us)": 100.0, "calc_flops_power(tflops)": 280.0},
        )
    ]
    assert br.collect_baseline(entries)["fa"]["q_len=8192|dtype=bf16"]["MFU"] == 0.5
    updated = br.apply_csv_peak_updates(entries, {"fa": {"peak_flops": 700.0, "peak_bw": None}})
    assert updated == 1
    grouped = br.collect_baseline(entries)
    assert grouped["fa"]["q_len=8192|dtype=bf16"]["MFU"] == pytest.approx(0.4)


# --- render_html ------------------------------------------------------------
def test_render_html_writes_report(tmp_path):
    data = {
        "generated_at": "2026-01-01T00:00:00+08:00",
        "backend": {"device_name": "Ascend950PR"},
        "runtime": {"device_ids": [0]},
        "env": {"peak_flops": 560.0, "peak_bw": 1275.0},
        "ops": {
            "fa": {
                "q_len=1024|dtype=bf16": {"MFU": 0.1, "MBU": 0.2, "latency(us)": 3.0},
                "q_len=2048|dtype=bf16": {"MFU": 0.2, "MBU": 0.3, "latency(us)": 4.0},
                "q_len=1024|dtype=mxfp8": {"MFU": 0.3, "MBU": 0.4, "latency(us)": 2.0},
            }
        },
    }
    out = tmp_path / "report.html"
    br.render_html(data, str(out))
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "MindIE-SD Core Ops Benchmark Report" in content
    assert "Ascend950PR" in content
    # fa (compute-bound) shows MFU only: one MFU chart, no MBU chart/column
    assert "FA · MFU (dtype)" in content
    assert "FA · MBU" not in content
    assert content.count("<svg") >= 1
    # per-series performance tables below the charts (label = series key)
    assert "Data — bf16" in content
    assert "Data — mxfp8" in content
    assert "<th>seq len</th>" in content
    assert "<th>MBU</th>" not in content


def test_render_html_gmm_shows_mfu_and_mbu(tmp_path):
    data = {
        "generated_at": "2026-01-01T00:00:00+08:00",
        "backend": {"device_name": "Ascend950PR"},
        "runtime": {"device_ids": [0]},
        "env": {"peak_flops": 560.0, "peak_bw": 1275.0},
        "ops": {
            "gmm": {
                "num_tokens=1024|top_k=8|quant_algo=NO_QUANT": {
                    "MFU": 0.4, "MBU": 0.5, "latency(us)": 3.0,
                },
                "num_tokens=2048|top_k=8|quant_algo=NO_QUANT": {
                    "MFU": 0.5, "MBU": 0.6, "latency(us)": 4.0,
                },
            }
        },
    }
    out = tmp_path / "gmm.html"
    br.render_html(data, str(out))
    content = out.read_text(encoding="utf-8")
    # gmm additionally shows MBU
    assert "GMM · MFU (quant_algo)" in content
    assert "GMM · MBU (quant_algo)" in content
    assert "<th>MBU</th>" in content
    # utilization rendered as percentage (0.4 -> 40.00%)
    assert "40.00%" in content
    assert "50.00%" in content


def test_render_html_shows_heads_dim_and_func(tmp_path):
    data = {
        "generated_at": "2026-01-01T00:00:00+08:00",
        "backend": {"device_name": "Ascend950PR"},
        "runtime": {"device_ids": [0]},
        "env": {"peak_flops": 560.0, "peak_bw": 1275.0},
        "ops": {
            "fa": {
                "q_len=1024|num_heads=16|head_dim=128|dtype=bf16"
                "|func=torch_npu.npu_fusion_attention": {
                    "MFU": 0.4, "MBU": 0.2, "latency(us)": 3.0,
                },
                "q_len=2048|num_heads=32|head_dim=256|dtype=bf16": {
                    "MFU": 0.5, "MBU": 0.3, "latency(us)": 4.0,
                },
            }
        },
    }
    out = tmp_path / "report2.html"
    br.render_html(data, str(out))
    content = out.read_text(encoding="utf-8")
    # series labels carry head count / dim / function
    assert "bf16 h16 d128 fn=torch_npu.npu_fusion_attention" in content
    assert "bf16 h32 d256" in content
    assert "fn=torch_npu.npu_fusion_attention" in content
