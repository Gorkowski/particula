"""
PyTests for the Taichi-accelerated Knudsen number module.
They verify numerical consistency with the reference NumPy implementation
and exercise both the Python wrapper and the Taichi kernel.
"""

import numpy as np
import pytest
import taichi as ti

ti.init(arch=ti.cpu)  # Initialize Taichi with CPU backend

# --------------------------------------------------------------------- #
# Skip the whole module if Taichi is not available                      #
# --------------------------------------------------------------------- #
try:
    import taichi as ti
except ModuleNotFoundError:
    pytest.skip("Taichi not installed – skipping Taichi backend tests",
                allow_module_level=True)

# Taichi implementation under test
from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    get_knudsen_number_taichi,
    kget_knudsen_number,
)

# Reference (pure-Python/NumPy) version for ground-truth comparison
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number as get_knudsen_number_python,
)


# --------------------------------------------------------------------- #
# Helper: compare with tight tolerance                                  #
# --------------------------------------------------------------------- #
def _assert_close(result, expected):
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------- #
# Tests                                                                 #
# --------------------------------------------------------------------- #


def test_get_knudsen_number_taichi_array():
    """Vector inputs of equal length are handled element-wise."""
    mfp = np.array([6.0e-8, 7.0e-8], dtype=np.float64)
    pr = np.array([1.0e-7, 2.0e-7], dtype=np.float64)

    expected = get_knudsen_number_python(mfp, pr)
    result = get_knudsen_number_taichi(mfp, pr)

    _assert_close(result, expected)


def test_kget_knudsen_number_kernel_direct_call():
    """Directly launch the Taichi kernel and compare its output."""
    mfp_np = np.array([6.0e-8, 7.0e-8], dtype=np.float64)
    pr_np = np.array([1.0e-7, 2.0e-7], dtype=np.float64)
    expected = mfp_np / pr_np

    n = mfp_np.size
    mfp_nd = ti.ndarray(dtype=ti.f64, shape=n)
    pr_nd = ti.ndarray(dtype=ti.f64, shape=n)
    res_nd = ti.ndarray(dtype=ti.f64, shape=n)

    mfp_nd.from_numpy(mfp_np)
    pr_nd.from_numpy(pr_np)

    kget_knudsen_number(mfp_nd, pr_nd, res_nd)

    _assert_close(res_nd.to_numpy(), expected)
