"""Testing the particle representation factories.

The Strategy is tested independently.
"""

import numpy as np
import pytest

from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    ParticleResolvedSpeciatedMass,
    RadiiBasedMovingBin,
)
from particula.particles.representation import ParticleRepresentation
from particula.particles.representation_factories import (
    ParticleRepresentationFactory,
)
from particula.particles.surface_strategies import SurfaceStrategyVolume


def test_mass_based_build():
    """Test Factory and Build MassParticleRepresentation."""
    parameters = {
        "distribution_strategy": MassBasedMovingBin(),
        "activity_strategy": ActivityIdealMass(),
        "surface_strategy": SurfaceStrategyVolume(),
        "mass": np.array([1.0, 2.0, 3.0]),
        "mass_units": "kg",
        "density": np.array([1.0, 2.0, 3.0]),
        "density_units": "kg/m^3",
        "concentration": np.array([10, 20, 30]),
        "concentration_units": "1/m^3",
        "charge": 1.0,
    }
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "mass", parameters
    )
    assert isinstance(particle_rep, ParticleRepresentation)
    assert np.array_equal(particle_rep.get_distribution(), parameters["mass"])
    assert np.array_equal(particle_rep.get_density(), parameters["density"])
    assert np.array_equal(
        particle_rep.get_concentration(), parameters["concentration"]
    )
    assert particle_rep.get_charge() == parameters["charge"]


def test_radii_based_build():
    """Test factory and build for RadiusParticleRepresentationBuilder."""
    parameters = {
        "distribution_strategy": RadiiBasedMovingBin(),
        "activity_strategy": ActivityIdealMass(),
        "surface_strategy": SurfaceStrategyVolume(),
        "radius": np.array([1.0, 2.0, 3.0]),
        "radius_units": "m",
        "density": np.array([1.0, 2.0, 3.0]),
        "density_units": "kg/m^3",
        "concentration": np.array([10, 20, 30]),
        "concentration_units": "1/m^3",
        "charge": np.array([1.0, 2.0, 3.0]),
    }
    strategy = ParticleRepresentationFactory().get_strategy(
        "radius", parameters
    )
    assert isinstance(strategy, ParticleRepresentation)
    assert np.array_equal(strategy.get_distribution(), parameters["radius"])
    assert np.array_equal(strategy.get_density(), parameters["density"])
    assert np.array_equal(
        strategy.get_concentration(), parameters["concentration"]
    )
    assert np.array_equal(strategy.get_charge(), parameters["charge"])


def test_limited_radius_build():
    """Test factory and build for LimitedRadiusParticleBuilder."""
    # default values
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "preset_radius"
    )
    assert isinstance(particle_rep, ParticleRepresentation)

    # set values
    parameters = {
        "mode": np.array([100, 2000]) * 1e-9,
        "mode_units": "m",
        "geometric_standard_deviation": np.array([1.4, 1.5]),
        "number_concentration": np.array([1e3, 1e3]),
        "number_concentration_units": "1/m^3",
    }
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "preset_radius", parameters
    )
    assert isinstance(particle_rep, ParticleRepresentation)


def test_resolved_mass_build():
    """Test factory and build for ResolvedMassParticleRepresentationBuilder."""
    parameters = {
        "distribution_strategy": ParticleResolvedSpeciatedMass(),
        "activity_strategy": ActivityIdealMass(),
        "surface_strategy": SurfaceStrategyVolume(),
        "mass": np.array([1.0, 2.0, 3.0]),
        "mass_units": "kg",
        "density": np.array([1.0, 2.0, 3.0]),
        "density_units": "kg/m^3",
        "charge": 1.0,
        "volume": 1,
        "volume_units": "m^3",
    }
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "resolved_mass", parameters
    )
    assert isinstance(particle_rep, ParticleRepresentation)
    assert np.array_equal(particle_rep.get_distribution(), parameters["mass"])
    assert np.array_equal(particle_rep.get_density(), parameters["density"])
    assert particle_rep.get_charge() == parameters["charge"]
    assert particle_rep.get_volume() == parameters["volume"]


def test_preset_resolved_mass_build():
    """Test factory and build for PresetResolvedMassParticleBuilder."""
    # default values
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "preset_resolved_mass"
    )
    assert isinstance(particle_rep, ParticleRepresentation)

    parameters = {
        "volume": 1,
        "volume_units": "m^3",
        "mode": np.array([100, 2000]) * 1e-9,
        "mode_units": "m",
        "geometric_standard_deviation": np.array([1.4, 1.5]),
        "number_concentration": np.array([1e3, 1e3]),
        "number_concentration_units": "1/m^3",
        "particle_resolved_count": 1000,
    }
    particle_rep = ParticleRepresentationFactory().get_strategy(
        "preset_resolved_mass", parameters
    )
    assert isinstance(particle_rep, ParticleRepresentation)
    assert particle_rep.get_volume() == parameters["volume"]


def test_invalid_strategy():
    """Test factory function for invalid type."""
    with pytest.raises(ValueError) as excinfo:
        ParticleRepresentationFactory().get_strategy("invalid_type")
    assert "Unknown strategy type: invalid_type" in str(excinfo.value)
