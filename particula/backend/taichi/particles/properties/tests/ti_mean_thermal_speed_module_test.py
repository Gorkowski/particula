"""Tests for ti_mean_thermal_speed_module."""
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from particula.util.constants import BOLTZMANN_CONSTANT
from particula.backend.taichi.particles.properties.ti_mean_thermal_speed_module import (
    kget_mean_thermal_speed,
    ti_get_mean_thermal_speed,
)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def reference_mean_thermal_speed(particle_mass, temperature):
    """Reference NumPy implementation (same expression as original code)."""
    return np.sqrt((8.0 * BOLTZMANN_CONSTANT * temperature)
                   / (np.pi * particle_mass))


# ------------------------------------------------------------------
# tests
# ------------------------------------------------------------------
def test_taichi_wrapper_matches_numpy():
    """Wrapper should reproduce the NumPy reference values."""
    pm = np.array([1.0e-20, 2.0e-20, 5.0e-21])
    temp = np.array([298.0, 310.0, 250.0])

    np.testing.assert_allclose(
        ti_get_mean_thermal_speed(pm, temp),
        reference_mean_thermal_speed(pm, temp),
    )


def test_kernel_direct_call():
    """Kernel result equals reference when invoked directly."""
    pm = np.array([1.0e-20, 2.0e-20])
    temp = np.array([290.0, 320.0])
    n = pm.size

    # allocate Taichi NDArrays
    pm_ti = ti.ndarray(dtype=ti.f64, shape=n)
    temp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)

    pm_ti.from_numpy(pm)
    temp_ti.from_numpy(temp)

    # launch kernel
    kget_mean_thermal_speed(pm_ti, temp_ti, res_ti)

    np.testing.assert_allclose(
        res_ti.to_numpy(),
        reference_mean_thermal_speed(pm, temp),
    )
