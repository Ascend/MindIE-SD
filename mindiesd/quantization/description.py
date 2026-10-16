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

"""Validation of the flat MindIE SD quantization description format."""

import json
import math

from ..utils import ParametersInvalid
from .mode import QuantAlgorithm

_MODEL_QUANT_TYPE_FIELD = "model_quant_type"

_METADATA_RULES = {
    "version": "string",
    "group_size": "non_negative_integer",
    "metadata": "object",
    "optional": "object",
    # MindIE SD currently identifies these fields but does not consume them.
    # Keep their validation structural so newer exporter values remain loadable.
    "kv_cache_type": "string",
    "kv_quant_type": "string",
    "fa_quant_type": "string",
    "reduce_quant_type": "string",
}
DESCRIPTION_METADATA_FIELDS = frozenset({_MODEL_QUANT_TYPE_FIELD, *_METADATA_RULES})


def _field_label(name):
    # Field names are untrusted; keep errors bounded and escape control characters.
    return repr(name[:80]) + ("..." if len(name) > 80 else "")


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ParametersInvalid(f"Duplicate quantization description field {_field_label(name)}.")
        result[name] = value
    return result


def _reject_constant(value):
    raise ParametersInvalid(f"Invalid quantization description JSON constant {value}; use standard JSON values.")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ParametersInvalid("Quantization description JSON numbers must be finite.")
    return result


def _validate_metadata(name, value):
    label = _field_label(name)
    rule = _METADATA_RULES.get(name)
    if rule == "string":
        valid = isinstance(value, str)
        expected = "a string"
    elif rule == "non_negative_integer":
        valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        expected = "a non-negative integer (not a boolean)"
    elif rule == "object":
        valid = isinstance(value, dict)
        expected = "a JSON object"
    else:
        raise ParametersInvalid(f"Unsupported quantization description metadata field {_field_label(name)}.")
    if not valid:
        raise ParametersInvalid(f"Quantization description field {label} must be {expected}.")


def parse_quant_description(data):
    """Parse and validate every field before constructing configs or accessing weights.

    The descriptor is a flat object: ``model_quant_type`` is required metadata,
    known exporter metadata has field-specific rules, and remaining entries map
    dynamic layer/tensor names to algorithm strings or ``FLOAT``. Exporter
    metadata is not consumed as inference configuration.
    """
    try:
        description = json.loads(
            data, object_pairs_hook=_unique_fields, parse_constant=_reject_constant, parse_float=_finite_float
        )
    except json.JSONDecodeError as exc:
        raise ParametersInvalid(
            f"Invalid quantization description JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc
    except RecursionError as exc:
        raise ParametersInvalid(
            "Invalid quantization description JSON: excessive nesting; expected a flat object."
        ) from exc
    except ValueError as exc:
        # ParametersInvalid intentionally inherits Exception, so this catches
        # only parser numeric-limit ValueError from the json module.
        raise ParametersInvalid("Invalid quantization description JSON: number exceeds parser limits.") from exc

    if not isinstance(description, dict):
        raise ParametersInvalid("Quantization description must be a JSON object.")
    if _MODEL_QUANT_TYPE_FIELD not in description:
        raise ParametersInvalid("Quantization description field 'model_quant_type' is required.")

    valid_algorithms = {item.value for item in QuantAlgorithm}
    for name, value in description.items():
        label = _field_label(name)
        if not name.strip() or any(ord(char) < 32 or ord(char) == 127 for char in name):
            raise ParametersInvalid(
                f"Invalid quantization description field name {label}: empty or control characters."
            )
        if name in DESCRIPTION_METADATA_FIELDS and name != _MODEL_QUANT_TYPE_FIELD:
            _validate_metadata(name, value)
            continue
        if not isinstance(value, str):
            raise ParametersInvalid(
                f"Quantization description field {label} must be a string, got {type(value).__name__}."
            )
        if name == _MODEL_QUANT_TYPE_FIELD:
            if value not in valid_algorithms:
                raise ParametersInvalid(
                    f"Quantization description field {label} must be an exact QuantAlgorithm value."
                )
        elif value != "FLOAT" and value.upper() not in valid_algorithms:
            raise ParametersInvalid(
                f"Quantization description field {label} must be a QuantAlgorithm value (case-insensitive) or 'FLOAT'."
            )
    return description
