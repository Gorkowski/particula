"""Basic tests for the Taichi particle-properties package initialiser."""
import taichi as ti
import numpy as np
import numpy.testing as npt

ti.init(arch=ti.cpu)


def test_package_import_and_wrapper_execution():
    """Package must import and expose a registered Taichi wrapper."""
    # importing must not raise
    import importlib
    pkg = importlib.import_module(
        "particula.backend.taichi.particles.properties"
    )

    # one representative wrapper should now be accessible and functional
    kn_func = getattr(pkg, "ti_get_knudsen_number")
    result = kn_func(np.array([1.0]), np.array([0.1]))

    # basic sanity check – result shape and finite value(s)
    assert np.array(result).size == 1
    npt.assert_allclose(np.isfinite(result), True)
