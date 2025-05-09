"""
PyTests for the Taichi-accelerated Knudsen number module.
They verify numerical consistency with the reference NumPy implementation
and exercise both the Python wrapper and the Taichi kernel.
"""

import numpy as np
import pytest
import taichi as ti
import particula as par

ti.init(arch=ti.cpu)  # Initialize Taichi with CPU backend


# Taichi implementation under test
from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    get_knudsen_number_taichi,
    kget_knudsen_number,
)


def test_get_knudsen_number_taichi_array():
    """Vector inputs of equal length are handled element-wise."""
    mfp = np.array([6.0e-8, 7.0e-8], dtype=np.float64)
    pr = np.array([1.0e-7, 2.0e-7], dtype=np.float64)

    expected = par.particles.get_knudsen_number(mfp, pr)
    result = get_knudsen_number_taichi(mfp, pr)

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0.0)


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

    np.testing.assert_allclose(res_nd.to_numpy(), expected, rtol=1e-12, atol=0.0)
