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


@pytest.mark.parametrize('variant', ['torch27', 'torch29', 'torch210', 'torch213'])
def test_forced_plugin_variant_is_selected_without_loading(monkeypatch, tmp_path, variant):
    monkeypatch.setenv(registry.PLUGIN_VARIANT_ENV, ' ' + variant + ' ')
    assert registry._select_mindie_ops_file(str(tmp_path)) == str(tmp_path / variant / registry.PLUGIN_LIBRARY_NAME)


@pytest.mark.parametrize('variant', ['unknown', 'torch26'])
def test_invalid_forced_variant_is_rejected(monkeypatch, tmp_path, variant):
    monkeypatch.setenv(registry.PLUGIN_VARIANT_ENV, variant)
    with pytest.raises(RuntimeError, match='Unsupported'):
        registry._select_mindie_ops_file(str(tmp_path))


@pytest.mark.parametrize(
    ('version', 'variant'),
    [
        ('2.7.1+cpu', 'torch27'),
        ('2.8.0', 'torch28'),
        ('2.9.0+cpu', 'torch29'),
        ('2.10.0', 'torch210'),
        ('2.11.0', 'torch211'),
        ('2.12.0', 'torch212'),
        ('2.13.0', 'torch213'),
    ],
)
def test_auto_variant_and_legacy_fallback(monkeypatch, tmp_path, version, variant):
    monkeypatch.delenv(registry.PLUGIN_VARIANT_ENV, raising=False)
    monkeypatch.setattr(torch, '__version__', version)
    assert registry._select_mindie_ops_file(str(tmp_path)) == str(tmp_path / registry.PLUGIN_LIBRARY_NAME)
    (tmp_path / variant).mkdir()
    (tmp_path / variant / registry.PLUGIN_LIBRARY_NAME).write_bytes(b'variant plugin')
    assert registry._has_plugin_variants(str(tmp_path))
    assert registry._select_mindie_ops_file(str(tmp_path)) == str(tmp_path / variant / registry.PLUGIN_LIBRARY_NAME)
    monkeypatch.setattr(torch, '__version__', '3.0.0')
    with pytest.raises(RuntimeError, match='Unsupported torch version'):
        registry._select_mindie_ops_file(str(tmp_path))


@pytest.mark.parametrize('version', ['2.6.0', '2.9.0+cpu', '2.10.0+cpu'])
def test_fa4_directories_do_not_hide_fixed_mindie_plugin(monkeypatch, tmp_path, version):
    monkeypatch.delenv(registry.PLUGIN_VARIANT_ENV, raising=False)
    monkeypatch.setattr(torch, '__version__', version)
    fixed = tmp_path / registry.PLUGIN_LIBRARY_NAME
    fixed.write_bytes(b'fixed MindIE plugin')
    for variant in ('torch29', 'torch210'):
        (tmp_path / variant / 'flash_attn').mkdir(parents=True)
    assert registry._select_mindie_ops_file(str(tmp_path)) == str(fixed)


def test_real_multi_torch_missing_current_variant_does_not_use_fixed_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv(registry.PLUGIN_VARIANT_ENV, raising=False)
    monkeypatch.setattr(torch, '__version__', '2.9.0+cpu')
    (tmp_path / registry.PLUGIN_LIBRARY_NAME).write_bytes(b'wrong fixed fallback')
    other = tmp_path / 'torch210' / registry.PLUGIN_LIBRARY_NAME
    other.parent.mkdir()
    other.write_bytes(b'different Torch native plugin')
    assert registry._select_mindie_ops_file(str(tmp_path)) == str(tmp_path / 'torch29' / registry.PLUGIN_LIBRARY_NAME)


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
