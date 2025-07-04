"""Tests for the aerodynamic_mobility module."""

from contextlib import nullcontext

import numpy as np
import pytest

from particula.particles.properties.aerodynamic_mobility_module import (
    get_aerodynamic_mobility,
)


def test_particle_aerodynamic_mobility_single_value():
    """Verify that the particle_aerodynamic_mobility function calculates the
    correct aerodynamic mobility value for a single particle.
    """
    radius = 0.00005  # 50 micrometers
    slip_correction_factor = 1.1
    dynamic_viscosity = 0.0000181  # Pa.s for air at room temperature
    expected_mobility = slip_correction_factor / (
        6 * np.pi * dynamic_viscosity * radius
    )
    actual_mobility = get_aerodynamic_mobility(
        radius, slip_correction_factor, dynamic_viscosity
    )
    assert np.isclose(
        actual_mobility, expected_mobility
    ), "The value does not match."


def test_particle_aerodynamic_mobility_array_input():
    """Test that the particle_aerodynamic_mobility function handles numpy array
    inputs correctly.
    """
    radius = np.array([0.00005, 0.00007])  # Array of radii
    slip_correction_factor = np.array([1.1, 1.2])
    dynamic_viscosity = 0.0000181
    expected_mobility = slip_correction_factor / (
        6 * np.pi * dynamic_viscosity * radius
    )
    actual_mobility = get_aerodynamic_mobility(
        radius, slip_correction_factor, dynamic_viscosity
    )
    assert np.allclose(actual_mobility, expected_mobility)


def test_particle_aerodynamic_mobility_type_error():
    """Test that the particle_aerodynamic_mobility function raises a TypeError
    with incorrect input types.
    """
    with pytest.raises(TypeError):
        get_aerodynamic_mobility(
            "not a number", "also not a number", "still not a number"
        )


@pytest.mark.parametrize(
    "radius", [0, -1e-6, 1e-12, 1e-3, float("inf"), float("nan")]
)
def test_particle_aerodynamic_mobility_extreme_values(radius: float) -> None:
    """Verify that the particle_aerodynamic_mobility function handles extreme
    values correctly.
    - radius : The radius of the particle.
    """
    slip_correction_factor = 1.1
    dynamic_viscosity = 0.0000181  # Pa.s for air at room temperature
    with (
        pytest.raises(ValueError)
        if (radius <= 0 or np.isnan(radius) or np.isinf(radius))
        else nullcontext()
    ):
        actual_mobility = get_aerodynamic_mobility(
            radius, slip_correction_factor, dynamic_viscosity
        )
        if radius > 0:
            expected_mobility = slip_correction_factor / (
                6 * np.pi * dynamic_viscosity * radius
            )
            assert np.isclose(
                actual_mobility, expected_mobility
            ), f"The value does not match for radius {radius}."
