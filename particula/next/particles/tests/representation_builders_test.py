"""Tests for representation builders."""

from particula.next.particles.representation_builders import (
    MassParticleRepresentationBuilder,
    RadiusParticleRepresentationBuilder,
    LimitedRadiusParticleBuilder,
)


def test_mass_particle_representation_builder():
    builder = MassParticleRepresentationBuilder()
    builder.set_distribution_strategy(...)
    builder.set_activity_strategy(...)
    builder.set_surface_strategy(...)
    builder.set_mass(...)
    builder.set_density(...)
    builder.set_concentration(...)
    builder.set_charge(...)
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


def test_radius_particle_representation_builder():
    builder = RadiusParticleRepresentationBuilder()
    builder.set_distribution_strategy(...)
    builder.set_activity_strategy(...)
    builder.set_surface_strategy(...)
    builder.set_radius(...)
    builder.set_density(...)
    builder.set_concentration(...)
    builder.set_charge(...)
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


def test_limited_radius_particle_builder():
    builder = LimitedRadiusParticleBuilder()
    builder.set_distribution_strategy(...)
    builder.set_activity_strategy(...)
    builder.set_surface_strategy(...)
    builder.set_radius(...)
    builder.set_density(...)
    builder.set_concentration(...)
    builder.set_charge(...)
    builder.set_mode(...)
    builder.set_geometric_standard_deviation(...)
    builder.set_number_concentration(...)
    builder.set_radius_bins(...)
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)
