"""Tests for representation builders."""

import numpy as np

from particula.next.particles.representation_builders import (
    MassParticleRepresentationBuilder,
    RadiusParticleRepresentationBuilder,
    LimitedRadiusParticleBuilder,
)
from particula.next.particles.representation import ParticleRepresentation
from particula.next.particles.distribution_strategies import (
    RadiiBasedMovingBin,
)
from particula.next.particles.surface_strategies import SurfaceStrategyVolume
from particula.next.particles.activity_strategies import IdealActivityMass


def test_mass_particle_representation_builder():
    """Test MassParticleRepresentationBuilder Builds.
    """
    builder = MassParticleRepresentationBuilder()
    builder.set_distribution_strategy(RadiiBasedMovingBin())
    builder.set_activity_strategy(IdealActivityMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_mass(np.array([1.0, 2.0, 3.0]))
    builder.set_density(np.array([1.0, 2.0, 3.0]))
    builder.set_concentration(np.array([10, 20, 30]))
    builder.set_charge(1.0)
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


# def test_radius_particle_representation_builder():
#     """Test RadiusParticleRepresentationBuilder Builds.
#     """
#     builder = RadiusParticleRepresentationBuilder()
#     builder.set_distribution_strategy(...)
#     builder.set_activity_strategy(...)
#     builder.set_surface_strategy(...)
#     builder.set_radius(...)
#     builder.set_density(...)
#     builder.set_concentration(...)
#     builder.set_charge(...)
#     particle_representation = builder.build()
#     assert isinstance(particle_representation, ParticleRepresentation)


# def test_limited_radius_particle_builder():
#     """Test LimitedRadiusParticleBuilder Builds.
#     """
#     builder = LimitedRadiusParticleBuilder()
#     builder.set_distribution_strategy(...)
#     builder.set_activity_strategy(...)
#     builder.set_surface_strategy(...)
#     builder.set_radius(...)
#     builder.set_density(...)
#     builder.set_concentration(...)
#     builder.set_charge(...)
#     builder.set_mode(...)
#     builder.set_geometric_standard_deviation(...)
#     builder.set_number_concentration(...)
#     builder.set_radius_bins(...)
#     particle_representation = builder.build()
#     assert isinstance(particle_representation, ParticleRepresentation)
