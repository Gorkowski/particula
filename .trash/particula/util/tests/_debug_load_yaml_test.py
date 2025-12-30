import particula.util as util


def test_debug_load_yaml_presence():
    assert hasattr(util, "yaml_loader"), util.__file__
    assert hasattr(util.yaml_loader, "load_yaml"), util.yaml_loader.__file__
    assert hasattr(util, "load_yaml"), dir(util)
