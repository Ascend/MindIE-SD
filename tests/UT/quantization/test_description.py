#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU tests for the quantization description input boundary."""

import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from utils.utils.torch_npu_mock import mock_torch_npu


class TestQuantDescription(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mock_torch_npu()
        cls.quantize_module = importlib.import_module("mindiesd.quantization.quantize")
        cls.parameters_invalid = importlib.import_module("mindiesd.utils").ParametersInvalid
        cls.algorithms = importlib.import_module("mindiesd.quantization.mode").QuantAlgorithm

    def setUp(self):
        self.file_open = self.patch(self.quantize_module.file_utils, "safe_open")
        self.weight_check = self.patch(self.quantize_module.file_utils, "check_file_safety")
        self.weight_open = self.patch(self.quantize_module.safetensors, "safe_open")

    def patch(self, target, attribute):
        patcher = mock.patch.object(target, attribute)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def load(self, text):
        self.file_open.return_value.__enter__.return_value.read.return_value = text
        return self.quantize_module.get_cfg_and_weights("/tmp/quant_model_description_w8a8.json")

    def assert_invalid(self, text, message):
        with mock.patch.object(self.quantize_module.QuantConfig, "parse_from_dict") as config_parse:
            with self.assertRaisesRegex(self.parameters_invalid, message):
                self.load(text)
            config_parse.assert_not_called()
        self.weight_check.assert_not_called()
        self.weight_open.assert_not_called()

    def test_exported_tensor_entries(self):
        description = {
            "model_quant_type": "W8A8",
            "blocks.0.proj.weight": "W8A8",
            "blocks.0.proj.input_scale": "W8A8",
            "blocks.0.proj.quant_bias": "W8A8",
            "blocks.0.proj.bias": "FLOAT",
            "blocks.1.proj.linear.weight": "w8a8_dynamic",
            "blocks.1.attn.fa_q.scale": "FP8_DYNAMIC",
            "blocks.2.attn.quant_type": "MXFP4_DYNAMIC",
        }
        cfg, weights = self.load(json.dumps(description))
        self.assertEqual(cfg.quant_algo, self.algorithms.W8A8)
        self.assertEqual(cfg.exclude_layers, ("blocks.0.proj.bias",))
        self.assertEqual(set(cfg.quantized_layers), set(description) - {"model_quant_type", "blocks.0.proj.bias"})
        self.assertEqual(cfg.quantized_layers["blocks.1.proj.linear.weight"].quant_algo, self.algorithms.W8A8_DYNAMIC)
        self.assertIs(weights, self.weight_open.return_value)
        self.weight_open.assert_called_once_with(
            os.path.realpath("/tmp/quant_model_weight_w8a8.safetensors"), framework="pytorch"
        )

    def test_all_existing_algorithm_values(self):
        for algorithm in self.algorithms:
            with self.subTest(algorithm=algorithm.value):
                cfg, _ = self.load(json.dumps({"model_quant_type": algorithm.value, "layer": algorithm.value}))
                self.assertEqual(cfg.quant_algo, algorithm)
                self.assertEqual(cfg.quantized_layers["layer"].quant_algo, algorithm)

    def test_model_type_is_metadata(self):
        cfg, _ = self.load('{"model_quant_type":"W8A8"}')
        self.assertEqual(cfg.quantized_layers, {})
        self.assertEqual(cfg.exclude_layers, ())

    def test_dynamic_names_remain_supported(self):
        cfg, _ = self.load('{"model_quant_type":"W8A8","layer1":"W8A8","0":"FLOAT"}')
        self.assertIn("layer1", cfg.quantized_layers)
        self.assertEqual(cfg.exclude_layers, ("0",))

    def test_invalid_top_level(self):
        for value in (None, True, False, 1, 1.5, "W8A8", [], ["W8A8"], {}):
            with self.subTest(value=value):
                self.assert_invalid(json.dumps(value), "JSON object|model_quant_type")

    def test_missing_model_type(self):
        self.assert_invalid('{"layer":"W8A8"}', "model_quant_type.*required")

    def test_model_type_requires_string(self):
        for value in (None, True, False, 8, 8.0, [], {}):
            with self.subTest(value=value):
                self.assert_invalid(json.dumps({"model_quant_type": value}), "model_quant_type.*string")

    def test_invalid_model_algorithm(self):
        for value in ("", "TYPO", "FLOAT", "w8a8", " W8A8", "W8A8 "):
            with self.subTest(value=value):
                self.assert_invalid(json.dumps({"model_quant_type": value}), "model_quant_type.*QuantAlgorithm")

    def test_layer_type_requires_string(self):
        for value in (None, True, False, 8, 8.0, [], {}, {"algorithm": "W8A8"}):
            with self.subTest(value=value):
                self.assert_invalid(json.dumps({"model_quant_type": "W8A8", "layer": value}), "layer.*string")

    def test_invalid_layer_algorithm(self):
        for value in ("", "TYPO", "float", " W8A8", "W8A8 "):
            with self.subTest(value=value):
                self.assert_invalid(json.dumps({"model_quant_type": "W8A8", "layer": value}), "layer.*QuantAlgorithm")

    def test_invalid_field_names(self):
        for name in ("", " ", "layer\n", "layer\tweight", "layer\x00weight", "layer\x7fweight"):
            with self.subTest(name=name):
                self.assert_invalid(json.dumps({"model_quant_type": "W8A8", name: "FLOAT"}), "field name")

    def test_unrecognized_metadata_is_rejected(self):
        self.assert_invalid('{"model_quant_type":"W8A8","unexpected_option":true}', "unexpected_option")

    def test_ascendv1_metadata_is_not_layer_configuration(self):
        description = {
            "model_quant_type": "W8A8",
            "version": "1.0.0",
            "group_size": 0,
            "metadata": {"quarot": {"max_tp_size": 8, "heads_rotation": {"layers": ["proj"]}}},
            "optional": {"quarot": {"rotation_map": {"global_rotation": "optional/quarot.safetensors"}}},
            "kv_cache_type": "C8",
            "kv_quant_type": "C8",
            "fa_quant_type": "FAQuant",
            "reduce_quant_type": "per_channel",
            "proj.weight": "W8A8",
            "norm.weight": "FLOAT",
        }
        cfg, _ = self.load(json.dumps(description))
        self.assertEqual(set(cfg.quantized_layers), {"proj.weight"})
        self.assertEqual(cfg.exclude_layers, ("norm.weight",))

    def test_invalid_exporter_metadata(self):
        fields = {
            "version": (None, True, 1, [], {}),
            "group_size": (None, True, False, -1, 1.5, "32", [], {}),
            "metadata": (None, True, 1, "W8A8", []),
            "optional": (None, True, 1, "FLOAT", []),
            "kv_cache_type": (None, True, 8, [], {}),
            "kv_quant_type": (None, True, 8, [], {}),
            "fa_quant_type": (None, True, 8, [], {}),
            "reduce_quant_type": (None, True, 8, [], {}),
        }
        for name, values in fields.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    self.assert_invalid(json.dumps({"model_quant_type": "W8A8", name: value}), name)

    def test_unconsumed_metadata_accepts_future_values(self):
        description = {
            "model_quant_type": "W8A8",
            "version": "2",
            "kv_cache_type": "FutureKVType",
            "kv_quant_type": "FutureKVQuantType",
            "fa_quant_type": "FutureFAType",
            "reduce_quant_type": "FutureReduceType",
        }
        cfg, _ = self.load(json.dumps(description))
        self.assertEqual(cfg.quantized_layers, {})

    def test_empty_metadata_and_positive_group_size(self):
        cfg, _ = self.load('{"model_quant_type":"W8A8","version":"1.0.0","group_size":32,"metadata":{},"optional":{}}')
        self.assertEqual(cfg.quantized_layers, {})
        self.assertEqual(cfg.exclude_layers, ())

    def test_invalid_nested_metadata(self):
        for text in (
            '{"model_quant_type":"W8A8","metadata":{"tag":NaN}}',
            '{"model_quant_type":"W8A8","optional":{"tag":1,"tag":2}}',
            '{"model_quant_type":"W8A8","metadata":{"tag":1e999}}',
        ):
            with self.subTest(text=text):
                self.assert_invalid(text, "JSON constant|Duplicate|finite")

    def test_duplicate_keys(self):
        for text, field in (
            ('{"model_quant_type":"W8A8","model_quant_type":"W4A16"}', "model_quant_type"),
            ('{"model_quant_type":"W8A8","layer":"FLOAT","layer":"W8A8"}', "layer"),
            ('{"model_quant_type":"W8A8","layer":"W8A8","layer":"W8A8"}', "layer"),
            ('{"model_quant_type":"W8A8","layer":"FLOAT","\\u006cayer":"W8A8"}', "layer"),
        ):
            with self.subTest(text=text):
                self.assert_invalid(text, "Duplicate.*" + field)

    def test_invalid_json_syntax(self):
        for text in ("", "{", '{"model_quant_type":"W8A8",}', '{"model_quant_type":"W8A8"} trailing'):
            with self.subTest(text=text):
                self.assert_invalid(text, "Invalid quantization description JSON.*line.*column")

    def test_unescaped_control_character(self):
        self.assert_invalid('{"model_quant_type":"W8A8","layer\nweight":"FLOAT"}', "JSON.*line.*column")

    def test_nonstandard_numeric_constants(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                self.assert_invalid('{"model_quant_type":"W8A8","layer":' + value + '}', "JSON constant")

    def test_deeply_nested_input(self):
        self.assert_invalid('[' * 2000 + '0' + ']' * 2000, "JSON.*nesting")

    @unittest.skipUnless(hasattr(sys, "get_int_max_str_digits"), "Python has no integer string length limit")
    def test_integer_parser_limit(self):
        limit = sys.get_int_max_str_digits()
        if not limit:
            self.skipTest("Integer string length limit is disabled")
        self.assert_invalid('{"model_quant_type":"W8A8","group_size":' + '9' * (limit + 1) + '}', "parser limits")

    def test_invalid_utf8_before_weight_access(self):
        self.file_open.return_value.__enter__.return_value.read.side_effect = UnicodeDecodeError(
            "utf-8", b"\xff", 0, 1, "invalid start byte"
        )
        self.assert_invalid("", "UTF-8")

    def test_field_error_is_bounded(self):
        with self.assertRaises(self.parameters_invalid) as context:
            self.load(json.dumps({"model_quant_type": "W8A8", "x" * 10000: "TYPO"}))
        self.assertLess(len(str(context.exception)), 1000)

    def test_ranked_weight_filename(self):
        with mock.patch.object(
            self.quantize_module, "replace_rank_suffix", return_value=("/tmp/quant_2.json", "quant_2.json", 2)
        ):
            cfg, _ = self.load('{"model_quant_type":"W4A4_MXFP4","layer":"W4A4_MXFP4"}')
        self.assertEqual(cfg.quant_algo, self.algorithms.W4A4_MXFP4_DYNAMIC)
        self.weight_open.assert_called_once_with(
            os.path.realpath("/tmp/quant_model_weight_w4a4_mxfp4_2.safetensors"), framework="pytorch"
        )


class TestQuantDescriptionFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mock_torch_npu()
        cls.quantize_module = importlib.import_module("mindiesd.quantization.quantize")
        cls.parameters_invalid = importlib.import_module("mindiesd.utils").ParametersInvalid

    def setUp(self):
        # The directory spans setUp and the test; unittest always runs cleanup.
        temporary_directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(temporary_directory.cleanup)
        self.directory = os.path.realpath(temporary_directory.name)
        self.path = os.path.join(self.directory, "quant_model_description_w8a8.json")

    def write_description(self, data):
        with open(self.path, "wb") as writer:
            writer.write(data)
        os.chmod(self.path, 0o640)

    def test_real_descriptor_and_safetensors(self):
        torch = importlib.import_module("torch")
        save_file = importlib.import_module("safetensors.torch").save_file
        self.write_description(b'{"model_quant_type":"W8A8","proj.weight":"W8A8","norm.weight":"FLOAT"}')
        weight_path = os.path.join(self.directory, "quant_model_weight_w8a8.safetensors")
        expected = torch.arange(8, dtype=torch.int8).reshape(2, 4)
        save_file({"proj.weight": expected}, weight_path)
        os.chmod(weight_path, 0o640)
        cfg, weights = self.quantize_module.get_cfg_and_weights(self.path)
        with weights:
            self.assertTrue(torch.equal(weights.get_tensor("proj.weight"), expected))
        self.assertEqual(cfg.exclude_layers, ("norm.weight",))
        self.assertEqual(set(cfg.quantized_layers), {"proj.weight"})

    def test_real_invalid_utf8(self):
        self.write_description(b"\xff")
        with mock.patch.object(self.quantize_module.safetensors, "safe_open") as weight_open:
            with self.assertRaisesRegex(self.parameters_invalid, "UTF-8"):
                self.quantize_module.get_cfg_and_weights(self.path)
            weight_open.assert_not_called()

    def test_json_permissions_are_still_checked(self):
        self.write_description(b'{"model_quant_type":"W8A8"}')
        os.chmod(self.path, 0o644)
        with mock.patch.object(self.quantize_module.safetensors, "safe_open") as weight_open:
            with self.assertRaises(PermissionError):
                self.quantize_module.get_cfg_and_weights(self.path)
            weight_open.assert_not_called()

    def test_json_symlinks_are_still_rejected(self):
        self.write_description(b'{"model_quant_type":"W8A8"}')
        link_path = os.path.join(self.directory, "linked.json")
        os.symlink(self.path, link_path)
        with self.assertRaisesRegex(self.parameters_invalid, "symbolic link"):
            self.quantize_module.get_cfg_and_weights(link_path)

    def test_weight_permissions_are_still_checked(self):
        self.write_description(b'{"model_quant_type":"W8A8"}')
        weight_path = os.path.join(self.directory, "quant_model_weight_w8a8.safetensors")
        with open(weight_path, "wb") as writer:
            writer.write(b"")
        os.chmod(weight_path, 0o644)
        with mock.patch.object(self.quantize_module.safetensors, "safe_open") as weight_open:
            with self.assertRaises(PermissionError):
                self.quantize_module.get_cfg_and_weights(self.path)
            weight_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
