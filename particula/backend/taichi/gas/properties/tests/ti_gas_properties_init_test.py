"""
Unit tests for Taichi-based dynamic-viscosity helpers.

Checks that:
    • the Taichi wrapper is callable,
    • the wrapper matches the pure-Python reference,
    • the Taichi kernel matches the reference.
"""

import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)

from particula.backend.taichi.gas.properties import (
    ti_get_dynamic_viscosity,
)

from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.backend.taichi.gas.properties. \
    ti_dynamic_viscosity_module import (
        kget_dynamic_viscosity,
    )

def test_imports_are_callable():
    """Verify that the Taichi wrapper is callable."""
    assert callable(ti_get_dynamic_viscosity)

def test_wrapper_matches_python():
    """Wrapper result must equal the pure-Python reference."""
    temperature = np.array([288.15, 300.0], dtype=np.float64)
    reference_viscosity = np.full_like(temperature, 1.827e-5)
    reference_temperature = np.full_like(temperature, 288.15)

    np.testing.assert_allclose(
        ti_get_dynamic_viscosity(
            temperature,
            reference_viscosity,
            reference_temperature,
        ),
        get_dynamic_viscosity(
            temperature,
            reference_viscosity=reference_viscosity,
            reference_temperature=reference_temperature,
        ),
    )

def test_kernel_direct_call():
    """Kernel output must equal the pure-Python reference."""
    temperature = np.array([288.15, 300.0], dtype=np.float64)
    reference_viscosity = np.full_like(temperature, 1.827e-5)
    reference_temperature = np.full_like(temperature, 288.15)
    n_points = temperature.size

    temperature_field = ti.ndarray(dtype=ti.f64, shape=n_points)
    temperature_field.from_numpy(temperature)
    reference_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_points)
    reference_viscosity_field.from_numpy(reference_viscosity)
    reference_temperature_field = ti.ndarray(dtype=ti.f64, shape=n_points)
    reference_temperature_field.from_numpy(reference_temperature)
    dynamic_viscosity_field = ti.ndarray(dtype=ti.f64, shape=n_points)

    kget_dynamic_viscosity(
        temperature_field,
        reference_viscosity_field,
        reference_temperature_field,
        dynamic_viscosity_field,
    )

    np.testing.assert_allclose(
        dynamic_viscosity_field.to_numpy(),
        get_dynamic_viscosity(
            temperature,
            reference_viscosity=reference_viscosity,
            reference_temperature=reference_temperature,
        ),
    )
