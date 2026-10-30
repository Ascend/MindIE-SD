# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

from unittest.mock import MagicMock

import pytest
import torch

from mindiesd.layers import register_ops as registry
from mindiesd.utils import ParametersInvalid


@pytest.mark.parametrize('version,expected', [('2.9.0+cpu', '2.9'), ('2.10.1.dev', '2.10')])
def test_version_parser(version, expected):
    assert registry._get_torch_major_minor(version) == expected


def test_unparseable_version_is_explicit_error():
    with pytest.raises(RuntimeError, match='Cannot parse'):
        registry._get_torch_major_minor('broken')


@pytest.mark.parametrize('variant', ['torch26', 'torch29', 'torch210'])
def test_forced_plugin_variant_is_selected_without_loading(monkeypatch, tmp_path, variant):
    monkeypatch.setenv(registry.PLUGIN_VARIANT_ENV, ' ' + variant + ' ')
    assert registry._select_mindie_ops_file(str(tmp_path)) == str(tmp_path / variant / registry.PLUGIN_LIBRARY_NAME)


def test_invalid_forced_variant_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(registry.PLUGIN_VARIANT_ENV, 'unknown')
    with pytest.raises(RuntimeError, match='Unsupported'):
        registry._select_mindie_ops_file(str(tmp_path))


def test_auto_variant_and_legacy_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv(registry.PLUGIN_VARIANT_ENV, raising=False)
    monkeypatch.setattr(torch, '__version__', '2.9.0+cpu')
    assert registry._select_mindie_ops_file(str(tmp_path)) == str(tmp_path / registry.PLUGIN_LIBRARY_NAME)
    (tmp_path / 'torch29').mkdir()
    assert registry._has_plugin_variants(str(tmp_path))
    assert registry._select_mindie_ops_file(str(tmp_path)) == str(tmp_path / 'torch29' / registry.PLUGIN_LIBRARY_NAME)
    monkeypatch.setattr(torch, '__version__', '3.0.0')
    with pytest.raises(RuntimeError, match='Unsupported torch version'):
        registry._select_mindie_ops_file(str(tmp_path))


def test_library_path_and_real_file_safety_before_loader_boundary(monkeypatch, tmp_path):
    monkeypatch.delenv(registry.PLUGIN_VARIANT_ENV, raising=False)
    monkeypatch.setattr(torch, '__version__', '2.9.0+cpu')
    source = tmp_path / 'mindiesd/layers/register_ops.py'
    source.parent.mkdir(parents=True)
    source.write_text('# source path for resolver\n')
    library = tmp_path / 'mindiesd/plugin/torch29' / registry.PLUGIN_LIBRARY_NAME
    library.parent.mkdir(parents=True)
    library.write_bytes(b'CPU test loader boundary')
    library.chmod(0o640)
    monkeypatch.setattr(registry, '__file__', str(source))
    load = MagicMock()
    monkeypatch.setattr(torch.ops, 'load_library', load)
    registry._load_mindie_ops_library()
    load.assert_called_once_with(str(library))
    library.unlink()
    load.reset_mock()
    with pytest.raises(ParametersInvalid, match="doesn't exist"):
        registry._load_mindie_ops_library()
    load.assert_not_called()
