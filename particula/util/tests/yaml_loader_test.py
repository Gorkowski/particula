"""Tests for YAML loader utility."""

import io
import sys
from pathlib import Path

import pytest

from particula.util.yaml_loader import load_yaml


def test_load_from_string():
    """Loading YAML from a string returns parsed mapping."""

    content = "foo: bar\nvalue: 3"
    assert load_yaml(content, from_str=True) == {"foo": "bar", "value": 3}


def test_load_from_file_and_path(tmp_path: Path):
    """Loading YAML from file paths works for str and Path inputs."""

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("alpha: 1\n", encoding="utf-8")

    assert load_yaml(yaml_path) == {"alpha": 1}
    assert load_yaml(str(yaml_path)) == {"alpha": 1}


def test_load_from_text_io():
    """Loading YAML from a text stream succeeds when from_str is True."""

    stream = io.StringIO("nested:\n  key: value\n")
    assert load_yaml(stream, from_str=True) == {"nested": {"key": "value"}}


def test_parse_error_raises_value_error():
    """Malformed YAML surfaces as ValueError with context."""

    with pytest.raises(ValueError) as excinfo:
        load_yaml("- item1\n- item2: [", from_str=True)

    assert "Failed to parse YAML" in str(excinfo.value)


def test_missing_dependency_monkeypatch(monkeypatch: pytest.MonkeyPatch):
    """ImportError message directs users to install dependency."""

    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(ImportError) as excinfo:
        load_yaml("key: value", from_str=True)

    assert "Install pyyaml" in str(excinfo.value)


def test_missing_file_includes_path_context():
    """Missing file errors include the failing path."""

    missing_path = Path("does-not-exist.yaml")

    with pytest.raises(FileNotFoundError) as excinfo:
        load_yaml(missing_path)

    assert str(missing_path) in str(excinfo.value)


def test_invalid_input_type_raises_type_error():
    """Non-path input with from_str=False raises TypeError."""

    with pytest.raises(TypeError) as excinfo:
        load_yaml(io.StringIO("foo"), from_str=False)

    assert "path_or_str must be a path-like object" in str(excinfo.value)
