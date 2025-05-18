"""Tests the Taichi backend against the reference Python implementation
of Sutherland’s dynamic-viscosity formula."""

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

from particula.backend.taichi.gas.properties.ti_dynamic_viscosity_module import (
    ti_get_dynamic_viscosity,
    kget_dynamic_viscosity,
)
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
)

def test_wrapper_matches_reference():
    """Taichi wrapper matches NumPy reference within 1 × 10⁻⁹ relative
    tolerance."""
    temperatures = np.array([250.0, 300.0, 350.0])
    np.testing.assert_allclose(
        ti_get_dynamic_viscosity(temperatures),
        get_dynamic_viscosity(temperatures),
        rtol=1e-9,
    )

def test_kernel_direct():
    """Taichi kernel output equals Python backend for a 1-D temperature
    array."""
    temperatures = np.array([280.0, 310.0], dtype=np.float64)
    n_temperatures = temperatures.size
    temperatures_ti = ti.ndarray(dtype=ti.f64, shape=n_temperatures)
    reference_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n_temperatures)
    reference_temperature_ti = ti.ndarray(dtype=ti.f64, shape=n_temperatures)
    dynamic_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=n_temperatures)

    temperatures_ti.from_numpy(temperatures)
    reference_viscosity_ti.from_numpy(
        np.full(n_temperatures, REF_VISCOSITY_AIR_STP, dtype=np.float64)
    )
    reference_temperature_ti.from_numpy(
        np.full(n_temperatures, REF_TEMPERATURE_STP, dtype=np.float64)
    )

    kget_dynamic_viscosity(
        temperatures_ti,
        reference_viscosity_ti,
        reference_temperature_ti,
        dynamic_viscosity_ti,
    )

    np.testing.assert_allclose(
        dynamic_viscosity_ti.to_numpy(),
        get_dynamic_viscosity(temperatures),
        rtol=1e-9,
    )
