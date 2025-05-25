"""
Test for particle resolved fields.
"""

import numpy as np
import taichi as ti
from particula.backend.taichi.aerosol.ti_particle_resolved_field import (
    ParticleResolvedFieldBuilder,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)

def _assert_shapes(builder, variants, particles, species):
    assert len(builder.fields) == variants
    for fld in builder.fields:
        assert fld.shape == (particles, species)
        for name in ("species_masses",
                     "mass_transport_rate",
                     "transferable_mass"):
            assert getattr(fld, name).shape == (particles, species)

def test_load_single_variant():
    variants, particles, species = 1, 3, 2
    builder = ParticleResolvedFieldBuilder(variants, particles, species)

    _assert_shapes(builder, variants, particles, species)

    arr_mass = np.array([[1.0, 1.1],
                         [2.0, 2.1],
                         [3.0, 3.1]], dtype=np.float32)

    builder.load(0, species_masses=arr_mass)

    np.testing.assert_array_equal(
        builder.variant(0).species_masses.to_numpy(), arr_mass
    )
    zero = np.zeros_like(arr_mass)
    np.testing.assert_array_equal(
        builder.variant(0).mass_transport_rate.to_numpy(), zero
    )
    np.testing.assert_array_equal(
        builder.variant(0).transferable_mass.to_numpy(), zero
    )

def test_load_multiple_variants():
    variants, particles, species = 4, 2, 3
    builder = ParticleResolvedFieldBuilder(variants, particles, species)

    base = np.arange(particles * species, dtype=np.float32).reshape(particles, species)

    for v in range(variants):
        builder.load(v, species_masses=base + v)

    _assert_shapes(builder, variants, particles, species)

    for v in range(variants):
        expected = base + v
        np.testing.assert_array_equal(
            builder.variant(v).species_masses.to_numpy(), expected
        )
        zero = np.zeros_like(expected)
        np.testing.assert_array_equal(
            builder.variant(v).mass_transport_rate.to_numpy(), zero
        )
        np.testing.assert_array_equal(
            builder.variant(v).transferable_mass.to_numpy(), zero
        )
