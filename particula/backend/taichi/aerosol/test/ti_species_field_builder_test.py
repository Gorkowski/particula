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

def test_single_variant_three_species():
    variants, species = 1, 3
    builder = SpeciesFieldBuilder(variants, species)
    _assert_shapes(builder, variants, species)


def test_four_variants_three_species():
    variants, species = 4, 3
    builder = SpeciesFieldBuilder(variants, species)
    _assert_shapes(builder, variants, species)

