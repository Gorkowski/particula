"""Testing the particle representation factories.

The Strategy is tested independently.
"""

import pytest
import numpy as np
from particula.next.particles.representation_factories import (
    ParticleRepresentationFactory,
)
from particula.next.particles.distribution_strategies import (
    MassBasedMovingBin,
    RadiiBasedMovingBin,
)
from particula.next.particles.representation import ParticleRepresentation
from particula.next.particles.activity_strategies import IdealActivityMass
from particula.next.particles.surface_strategies import SurfaceStrategyVolume


def test_mass_based_build():
    """Test Factory and Build MassParticleRepresentation."""
    parameters = {
        'distribution_strategy': MassBasedMovingBin(),
        'activity_strategy': IdealActivityMass(),
        'surface_strategy': SurfaceStrategyVolume(),
        'mass': np.array([1.0, 2.0, 3.0]),
        'density': np.array([1.0, 2.0, 3.0]),
        'concentration': np.array([10, 20, 30]),
        'charge': 1.0
    }
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "mass", parameters)
    assert isinstance(particle_rep, ParticleRepresentation)


def test_radii_based_build():
    """Test factory and build for RadiusParticleRepresentationBuilder."""
    parameters = {
        'distribution_strategy': RadiiBasedMovingBin(),
        'activity_strategy': IdealActivityMass(),
        'surface_strategy': SurfaceStrategyVolume(),
        'radius': np.array([1.0, 2.0, 3.0]),
        'density': np.array([1.0, 2.0, 3.0]),
        'concentration': np.array([10, 20, 30]),
        'charge': np.array([1.0, 2.0, 3.0])
    }
    strategy = ParticleRepresentationFactory().get_strategy(
        "radius", parameters)
    assert isinstance(strategy, ParticleRepresentation)


def test_limited_radius_build():
    """Test factory and build for LimitedRadiusParticleBuilder."""
    # default values
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "limited_radius")
    assert isinstance(particle_rep, ParticleRepresentation)

    # set values
    parameters = {
        'mode': np.array([100, 2000]),
        'geometric_standard_deviation': np.array([1.4, 1.5]),
        'number_concentration': np.array([1e3, 1e3])
    }
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "limited_radius", parameters)
    assert isinstance(particle_rep, ParticleRepresentation)


def test_invalid_strategy():
    """Test factory function for invalid type."""
    with pytest.raises(ValueError) as excinfo:
        ParticleRepresentationFactory().get_strategy("invalid_type")
    assert "Unknown strategy type: invalid_type" in str(excinfo.value)
