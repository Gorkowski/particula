"""
Unit-tests for the Taichi-accelerated Debye function.
"""
import numpy as np
import numpy.testing as npt
import taichi as ti

# reference Python implementation
from particula.particles.properties.special_functions import get_debye_function
# Taichi wrapper / kernel
from particula.backend.taichi.particles.properties.ti_special_functions import (
    ti_get_debye_function,
    kget_debye_function,
)

# --------------------------------------------------------------------------- #
# initialise Taichi once per test module                                      #
# --------------------------------------------------------------------------- #
ti.init(arch=ti.cpu)


def test_ti_wrapper_matches_numpy_scalar_and_array():
    """Wrapper output (scalar & array) equals NumPy reference."""
    # scalar
    x_scalar = 0.1
    npt.assert_allclose(
        ti_get_debye_function(x_scalar),
        get_debye_function(x_scalar),
        rtol=1e-3,
    )

    # array
    x_array = np.linspace(0.1, 1.0, 10)
    npt.assert_allclose(
        ti_get_debye_function(x_array, n=2),
        get_debye_function(x_array, n=2),
        rtol=1e-4,
    )


def test_kernel_direct_matches_numpy():
    """Kernel result equals NumPy reference (n = 2)."""
    x_array = np.linspace(0.1, 3.0, 8)
    exponent = 2.0

    # allocate / launch kernel
    x_ti = ti.ndarray(dtype=ti.f64, shape=x_array.size)
    res_ti = ti.ndarray(dtype=ti.f64, shape=x_array.size)
    x_ti.from_numpy(x_array)
    kget_debye_function(x_ti, exponent, res_ti)

    npt.assert_allclose(
        res_ti.to_numpy(),
        get_debye_function(x_array, n=int(exponent)),
        rtol=1e-5,
    )
