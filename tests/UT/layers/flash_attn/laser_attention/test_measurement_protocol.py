# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Verify the timing boundary without requiring torch or an NPU."""

import builtins
import csv
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

from tests.UT.layers.flash_attn.laser_attention.profile_laser_attention_layer import (
    main,
    measure_calls,
    parse_args,
    write_event_samples,
)


class TestMeasurementProtocol(unittest.TestCase):
    def test_device_selected_before_layer_import(self):
        calls = []
        torch = SimpleNamespace(npu=SimpleNamespace(set_device=calls.append))
        original_import = builtins.__import__

        class StopAtLayerImport(Exception):
            pass

        def import_module(name, *args, **kwargs):
            if name == "torch":
                return torch
            if name == "torch_npu":
                return SimpleNamespace()
            if name == "tests.UT.layers.flash_attn.laser_attention.laser_attention_layer_common" or name.startswith(
                "mindiesd"
            ):
                self.assertEqual(calls, [4], "Layer imports must run after selecting the requested NPU")
                raise StopAtLayerImport
            return original_import(name, *args, **kwargs)

        with patch(
            "tests.UT.layers.flash_attn.laser_attention.profile_laser_attention_layer.parse_args",
            return_value=SimpleNamespace(device_id=4),
        ):
            with patch.object(sys, "path", sys.path.copy()), patch("builtins.__import__", side_effect=import_module):
                with self.assertRaises(StopAtLayerImport):
                    main()

    def test_fixed_warmup_and_individual_event_minimum(self):
        for iterations in (1, 10):
            with self.subTest(iterations=iterations):
                self.check_event_minimum(iterations)

    def check_event_minimum(self, iterations):
        calls = []
        events = []
        durations = iter(range(iterations, 0, -1))

        class Event:
            def __init__(self, enable_timing):
                self.label = len(events)
                events.append(self)
                calls.append(("create", enable_timing))

            def record(self):
                calls.append(self.label)

            def elapsed_time(self, end):
                self_test.assertEqual(calls.count("layer"), 5 + iterations)
                self_test.assertIs(end, events[self.label + iterations])
                calls.append(("elapsed", self.label))
                return next(durations)

        def forward():
            calls.append("layer")
            return "output"

        npu = SimpleNamespace(Event=Event)
        self_test = self
        output, timing = measure_calls(SimpleNamespace(npu=npu), forward, iterations)
        expected = [("create", True)] * (2 * iterations)
        for index in range(iterations):
            expected += [index, index + iterations]
        expected += ["layer"] * 5
        for index in range(iterations):
            expected += [index, "layer", index + iterations]
        expected += [("elapsed", index) for index in range(iterations)]
        self.assertEqual(calls, expected)
        self.assertEqual(output, "output")
        self.assertEqual(
            timing,
            {
                "event_us": 1000,
                "samples": iterations,
                "event_samples_us": [value * 1000 for value in range(iterations, 0, -1)],
            },
        )

    def test_invalid_event_time_is_rejected(self):
        for duration in (0, -1, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                event = Mock(elapsed_time=Mock(return_value=duration))
                npu = SimpleNamespace(Event=Mock(return_value=event))
                with self.assertRaisesRegex(RuntimeError, "Invalid timing"):
                    measure_calls(SimpleNamespace(npu=npu), Mock(), 10)

    def test_cli_default_iterations_and_invalid_counts(self):
        base = ["profile", "--device-id", "3", "--output", "result.json"]
        with patch.dict("os.environ", {}, clear=True), patch.object(sys, "argv", base):
            self.assertEqual(parse_args().iterations, 10)
        for extra in (["--iterations", "0"], ["--warmup", "10"]):
            with patch.object(sys, "argv", base + extra), patch("sys.stderr"), self.assertRaises(SystemExit):
                parse_args()

    def test_csv_preserves_all_samples_in_call_order(self):
        values = [4000.5 - index for index in range(10)]
        opened = mock_open()
        with patch.object(Path, "open", opened):
            write_event_samples(Path("event_samples.csv"), {"current": {"event_samples_us": values}})
        text = "".join(call.args[0] for call in opened().write.call_args_list)
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual([row["Path"] for row in rows], ["current"] * 10)
        self.assertEqual([int(row["Sample"]) for row in rows], list(range(1, 11)))
        self.assertEqual([float(row["Layer Duration(us)"]) for row in rows], values)


if __name__ == "__main__":
    unittest.main()
