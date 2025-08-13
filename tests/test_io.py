import pytest
from src.utils.io import load_yaml, ConfigLoadError

def test_load_yaml_valid(tmp_path):
    f = tmp_path / "test.yaml"
    f.write_text("a: 1\nb: 2\n")
    data = load_yaml(f)
    assert data["a"] == 1
    assert data["b"] == 2

def test_load_yaml_missing():
    with pytest.raises(ConfigLoadError):
        load_yaml("notfound.yaml")

def test_load_yaml_invalid(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("a: [1, 2\nb: 2\n")
    with pytest.raises(ConfigLoadError):
        load_yaml(f)
