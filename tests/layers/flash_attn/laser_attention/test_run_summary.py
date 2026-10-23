# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Check Event aggregation without torch, CSV timings, or an NPU."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.layers.flash_attn.laser_attention.summarize_laser_attention_layer import (
    format_summary,
    summarize,
    validate_report,
)


class TestRunSummary(unittest.TestCase):
    @staticmethod
    def report(path="current"):
        return {
            "profiled": True,
            "device_id": 3,
            "warmup": 5,
            "iterations": 3,
            "seed": 42,
            "layout": "BSND",
            "dtype": "bfloat16",
            "q_shape": [1],
            "kv_shape": [2],
            "path_order": [path],
            "results": {path: {"event_us": 3400, "samples": 3, "event_samples_us": [4000, 3400, 3600]}},
        }

    def test_invalid_protocol_and_configuration_are_rejected(self):
        reference = self.report()
        validate_report(reference, "current", reference)
        for change in (
            {"profiled": False},
            {"warmup": 10},
            {"device_id": 4},
            {"iterations": 0},
            {"iterations": 2},
            {"path_order": ["legacy"]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_report(dict(reference, **change), "current", reference)

        for profiled in (0, 1, "true", None):
            report = dict(reference, profiled=profiled)
            with self.subTest(profiled=profiled), self.assertRaisesRegex(ValueError, "Inconsistent profiling mode"):
                validate_report(report, "current", report)
        for iterations in (True, False, 1.0, "1", None):
            report = dict(reference, iterations=iterations)
            with self.subTest(iterations=iterations), self.assertRaisesRegex(ValueError, "positive integer iterations"):
                validate_report(report, "current", report)

    def test_minimum_and_sample_count_are_checked(self):
        for change in (
            {"event_us": 3600},
            {"samples": 1},
            {"event_samples_us": []},
            {"event_samples_us": [3400, 4000]},
            {"event_samples_us": [3400, float("nan"), 4000]},
        ):
            report = copy.deepcopy(self.report())
            report["results"]["current"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_report(report, "current", self.report())

    def test_rows_are_grouped_by_path_and_need_no_csv(self):
        def read_report(path, **_kwargs):
            self.assertEqual(path.name, "profiled_timing.json")
            return json.dumps(self.report(path.parent.name))

        with (
            patch.object(Path, "read_text", read_report),
            patch.object(Path, "glob", lambda path, _pattern: [path / "PROF_test"]),
            patch.object(Path, "is_dir", return_value=True),
        ):
            report = summarize(Path("captures"), 10)
        self.assertEqual([row["path"] for row in report["rows"]], ["current"] * 10 + ["legacy"] * 10)
        self.assertEqual([row["run"] for row in report["rows"]], list(range(1, 11)) * 2)
        rendered = format_summary(report)
        self.assertEqual(len(rendered.splitlines()), 28)
        self.assertNotIn("First", rendered)
        self.assertNotIn("LAformal", rendered)
        for row in report["rows"]:
            self.assertEqual(row["event_us"], 3400)
            self.assertEqual(row["event_samples_us"], [4000, 3400, 3600])
            expected = (Path("captures") / f"run_{row['run']:02d}" / row["path"] / "PROF_test").absolute()
            self.assertEqual(row["profile_dir"], str(expected))
            self.assertIn(str(expected), rendered)
        self.assertNotIn("Data is saved in", rendered)

    def test_final_results_use_each_scenarios_minimum(self):
        def read_report(path, **_kwargs):
            report = self.report(path.parent.name)
            index = int(path.parent.parent.name.split("_")[1])
            value = 3000 + index if path.parent.name == "current" else 5000 - index
            report["results"][path.parent.name].update(event_us=value, event_samples_us=[value + 10, value, value + 20])
            return json.dumps(report)

        with (
            patch.object(Path, "read_text", read_report),
            patch.object(Path, "glob", lambda path, _pattern: [path / "PROF_test"]),
            patch.object(Path, "is_dir", return_value=True),
        ):
            report = summarize(Path("captures"), 20)
        self.assertEqual(len(report["rows"]), 40)
        self.assertEqual([row["event_us"] for row in report["best_results"]], [3001, 4980])
        self.assertEqual([row["run"] for row in report["best_results"]], [1, 20])
        rendered = format_summary(report)
        self.assertIn("CastV(BF16->FP16)+LA(BF16-out)", rendered)
        self.assertIn("CastQKV(BF16->FP16)+LA(FP32-out)+CastOut(FP32->FP16->BF16)", rendered)
        self.assertIn("3001.000000", rendered)
        self.assertIn("4980.000000", rendered)
        self.assertIn("observed under the current test conditions", rendered)

    def test_missing_or_ambiguous_capture_directory_is_rejected(self):
        for profiles in ([], [Path("PROF_a"), Path("PROF_b")]):
            with (
                self.subTest(profiles=profiles),
                patch.object(Path, "read_text", return_value=json.dumps(self.report())),
                patch.object(Path, "glob", return_value=profiles),
                patch.object(Path, "is_dir", return_value=True),
                self.assertRaises(ValueError),
            ):
                summarize(Path("captures"), 1)

    def test_direct_mode_needs_only_timing_json_and_rejects_mixed_modes(self):
        def read_report(path, **_kwargs):
            self.assertEqual(path.name, "timing.json")
            return json.dumps(dict(self.report(path.parent.name), profiled=False))

        with (
            patch.object(Path, "read_text", read_report),
            patch.object(Path, "glob", side_effect=AssertionError("No PROF lookup in direct mode")),
        ):
            report = summarize(Path("captures"), 2, profiled=False)
        self.assertFalse(report["profiled"])
        self.assertIn("without msprof", format_summary(report))
        for row in report["rows"]:
            self.assertIsNone(row["profile_dir"])
            self.assertTrue(row["timing_file"].endswith("timing.json"))
            self.assertEqual(row["event_samples_us"], [4000, 3400, 3600])
        with patch.object(Path, "read_text", return_value=json.dumps(self.report())), self.assertRaises(ValueError):
            summarize(Path("captures"), 1, profiled=False)


if __name__ == "__main__":
    unittest.main()
