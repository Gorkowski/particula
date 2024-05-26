"""Test for activity factories. Is the factory function working as expected,
creating the right activity strategy based on the input parameters,
and raising the right exceptions when needed.

The Builder is tested independently."""

import pytest
import numpy as np
from particula.next.particles.activity_factories import (
    activity_factory,
)
from particula.next.particles.activity_strategies import (
    IdealActivityMass,
    IdealActivityMolar,
    KappaParameterActivity,
)


def test_mass_ideal_strategy():
    """Test factory function with default mass_ideal strategy."""
    strategy = activity_factory()
    assert isinstance(
        strategy, IdealActivityMass)


def test_molar_ideal_strategy_with_parameters():
    """Test factory function for molar_ideal strategy with parameters."""
    parameters = {'molar_mass': np.array([100.0, 200.0, 300.0])}
    strategy = activity_factory("molar_ideal", parameters)
    assert isinstance(
        strategy, IdealActivityMolar)
    np.testing.assert_allclose(
        strategy.molar_mass, parameters['molar_mass'], atol=1e-4)


def test_kappa_parameter_strategy_with_parameters():
    """Test factory function for kappa_parameter strategy with full parameters."""
    parameters = {
        'kappa': np.array([0.1, 0.2, 0.3]),
        'density': np.array([1000.0, 2000.0, 3000.0]),
        'molar_mass': np.array([1.0, 2.0, 3.0]),
        'water_index': 0
    }
    strategy = activity_factory(
        "kappa_parameter", parameters)
    assert isinstance(
        strategy, KappaParameterActivity)
    np.testing.assert_allclose(
        strategy.kappa, parameters['kappa'][1:], atol=1e-4)
    np.testing.assert_allclose(
        strategy.density, parameters['density'], atol=1e-4)
    np.testing.assert_allclose(
        strategy.molar_mass, parameters['molar_mass'], atol=1e-4)
    assert strategy.water_index == parameters['water_index']


def test_invalid_strategy_type():
    """Test factory function with an invalid strategy type."""
    with pytest.raises(ValueError) as excinfo:
        activity_factory("invalid_type")
    assert "Unknown strategy type: invalid_type" in str(
        excinfo.value)
