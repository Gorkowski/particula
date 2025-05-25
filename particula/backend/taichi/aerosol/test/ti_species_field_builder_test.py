""" 
Test for creating and updating species fields in Taichi.
for both single and multiple variants.

"""

import numpy as np
import taichi as ti
import pytest
from particula.backend.taichi.aerosol.ti_species_field_builder import SpeciesFieldBuilder

ti.init(arch=ti.cpu, default_fp=ti.f64)

def _assert_shapes(builder, variants, species):
    assert builder.field.shape == (variants, species)
    for name in (
        "density",
        "molar_mass",
        "pure_vapor_pressure",
        "vapor_concentration",
        "kappa",
        "surface_tension",
        "gas_mass",
    ):
        sub_field = getattr(builder.field, name)
        assert sub_field.shape == (variants, species)

# ─── helpers ────────────────────────────────────────────────────────────
def _make_species_arrays(n):
    rng = np.random.default_rng(seed=42)
    return dict(
        density=rng.random(n, dtype=np.float32),
        molar_mass=rng.random(n, dtype=np.float32),
        pure_vapor_pressure=rng.random(n, dtype=np.float32),
        vapor_concentration=rng.random(n, dtype=np.float32),
        kappa=rng.random(n, dtype=np.float32),
        surface_tension=rng.random(n, dtype=np.float32),
        gas_mass=rng.random(n, dtype=np.float32),
    )

def _assert_values(builder, v, arrays):
    for key, arr in arrays.items():
        np.testing.assert_allclose(
            getattr(builder.field, key).to_numpy()[v],  # [variants, species]
            arr,
            rtol=0, atol=0,
        )

def test_single_variant_three_species():
    variants, species = 1, 3
    builder = SpeciesFieldBuilder(variants, species)
    _assert_shapes(builder, variants, species)


def test_four_variants_three_species():
    variants, species = 4, 3
    builder = SpeciesFieldBuilder(variants, species)
    _assert_shapes(builder, variants, species)

# ─── new tests ──────────────────────────────────────────────────────────
def test_load_single_variant():
    variants, species = 1, 3
    builder = SpeciesFieldBuilder(variants, species)

    arrays = _make_species_arrays(species)
    builder.load(0, **arrays)

    _assert_shapes(builder, variants, species)
    _assert_values(builder, 0, arrays)


def test_load_multiple_variants():
    variants, species = 4, 3
    builder = SpeciesFieldBuilder(variants, species)

    for v in range(variants):
        arrays = _make_species_arrays(species)
        # make each variant unique
        for k in arrays:
            arrays[k] += v
        builder.load(v, **arrays)
        _assert_values(builder, v, arrays)

