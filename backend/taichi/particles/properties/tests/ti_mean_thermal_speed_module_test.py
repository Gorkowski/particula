import taichi as ti
import numpy as np
import pytest

from particula.backend.taichi.particles.properties.ti_mean_thermal_speed_module import (
    ti_get_mean_thermal_speed,
    kget_mean_thermal_speed,
)
from particula.particles.properties.mean_thermal_speed_module import (
    get_mean_thermal_speed,
)

ti.init(arch=ti.cpu)

def test_ti_get_mean_thermal_speed_matches_numpy():
    particle_mass = np.array([1e-17, 2e-17, 3e-17], dtype=np.float64)
    temperature = np.array([298.0, 300.0, 310.0], dtype=np.float64)
    expected = get_mean_thermal_speed(particle_mass, temperature)
    result = ti_get_mean_thermal_speed(particle_mass, temperature)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)

def test_kget_mean_thermal_speed_kernel_direct():
    particle_mass = np.array([1e-17, 2e-17], dtype=np.float64)
    temperature = np.array([298.0, 310.0], dtype=np.float64)
    result = np.empty_like(particle_mass)
    pm_ti = ti.ndarray(dtype=ti.f64, shape=particle_mass.shape)
    temp_ti = ti.ndarray(dtype=ti.f64, shape=temperature.shape)
    result_ti = ti.ndarray(dtype=ti.f64, shape=particle_mass.shape)
    pm_ti.from_numpy(particle_mass)
    temp_ti.from_numpy(temperature)
    kget_mean_thermal_speed(pm_ti, temp_ti, result_ti)
    result = result_ti.to_numpy()
    expected = get_mean_thermal_speed(particle_mass, temperature)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
