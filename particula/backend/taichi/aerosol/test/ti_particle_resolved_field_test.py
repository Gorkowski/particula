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
    assert builder.field.shape == (variants, particles, species)
    for name in ("mass", "mtr", "transferable_mass"):
        sub_field = getattr(builder.field, name)
        assert sub_field.shape == (variants, particles, species)

def test_load_single_variant():
    variants, particles, species = 1, 3, 2
    builder = ParticleResolvedFieldBuilder(variants, particles, species)

    _assert_shapes(builder, variants, particles, species)

    arr_mass = np.array([[1.0, 1.1],
                         [2.0, 2.1],
                         [3.0, 3.1]], dtype=np.float32)

    builder.load(0, species_masses=arr_mass)

    np.testing.assert_array_equal(builder.field.mass.to_numpy()[0], arr_mass)
    zero = np.zeros_like(arr_mass)
    np.testing.assert_array_equal(builder.field.mtr.to_numpy()[0], zero)
    np.testing.assert_array_equal(builder.field.t_mass.to_numpy()[0], zero)

def test_load_multiple_variants():
    variants, particles, species = 4, 2, 3
    builder = ParticleResolvedFieldBuilder(variants, particles, species)

    base = np.arange(particles * species, dtype=np.float32).reshape(particles, species)

    for v in range(variants):
        builder.load(v, species_masses=base + v)

    _assert_shapes(builder, variants, particles, species)

    for v in range(variants):
        expected = base + v
        np.testing.assert_array_equal(builder.field.mass.to_numpy()[v], expected)
        zero = np.zeros_like(expected)
        np.testing.assert_array_equal(builder.field.mtr.to_numpy()[v], zero)
        np.testing.assert_array_equal(builder.field.t_mass.to_numpy()[v], zero)
