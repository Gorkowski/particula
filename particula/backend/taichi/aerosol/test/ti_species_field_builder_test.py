""" 
Test for creating and updating species fields in Taichi.
for both single and multiple variants.

"""

import numpy as np
import taichi as ti
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

# ─── revised load tests ─────────────────────────────────────────────────
def test_load_single_variant():
    variants, species = 1, 3
    builder = SpeciesFieldBuilder(variants, species)

    # explicit test data (float32 because builder expects that dtype)
    arr_density            = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    arr_molar_mass         = np.array([2.0, 2.1, 2.2], dtype=np.float32)
    arr_pure_vp            = np.array([3.0, 3.1, 3.2], dtype=np.float32)
    arr_vapor_conc         = np.array([4.0, 4.1, 4.2], dtype=np.float32)
    arr_kappa              = np.array([5.0, 5.1, 5.2], dtype=np.float32)
    arr_surface_tension    = np.array([6.0, 6.1, 6.2], dtype=np.float32)
    arr_gas_mass           = np.array([7.0, 7.1, 7.2], dtype=np.float32)

    builder.load(
        0,
        density=arr_density,
        molar_mass=arr_molar_mass,
        pure_vapor_pressure=arr_pure_vp,
        vapor_concentration=arr_vapor_conc,
        kappa=arr_kappa,
        surface_tension=arr_surface_tension,
        gas_mass=arr_gas_mass,
    )

    _assert_shapes(builder, variants, species)
    np.testing.assert_array_equal(builder.field.density[0].to_numpy(), arr_density)
    np.testing.assert_array_equal(builder.field.molar_mass[0].to_numpy(), arr_molar_mass)
    np.testing.assert_array_equal(
        builder.field.pure_vapor_pressure[0].to_numpy(), arr_pure_vp
    )
    np.testing.assert_array_equal(
        builder.field.vapor_concentration[0].to_numpy(), arr_vapor_conc
    )
    np.testing.assert_array_equal(builder.field.kappa[0].to_numpy(), arr_kappa)
    np.testing.assert_array_equal(
        builder.field.surface_tension[0].to_numpy(), arr_surface_tension
    )
    np.testing.assert_array_equal(builder.field.gas_mass[0].to_numpy(), arr_gas_mass)


def test_load_multiple_variants():
    variants, species = 4, 3
    builder = SpeciesFieldBuilder(variants, species)

    # base arrays to clone/offset for each variant so values stay readable
    base = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    for v in range(variants):
        builder.load(
            v,
            density=base + v,
            molar_mass=base + 10 + v,
            pure_vapor_pressure=base + 20 + v,
            vapor_concentration=base + 30 + v,
            kappa=base + 40 + v,
            surface_tension=base + 50 + v,
            gas_mass=base + 60 + v,
        )

    _assert_shapes(builder, variants, species)

    # verify each field/variant retains the expected offset values
    for v in range(variants):
        np.testing.assert_array_equal(builder.field.density[v].to_numpy(), base + v)
        np.testing.assert_array_equal(
            builder.field.molar_mass[v].to_numpy(), base + 10 + v
        )
        np.testing.assert_array_equal(
            builder.field.pure_vapor_pressure[v].to_numpy(), base + 20 + v
        )
        np.testing.assert_array_equal(
            builder.field.vapor_concentration[v].to_numpy(), base + 30 + v
        )
        np.testing.assert_array_equal(builder.field.kappa[v].to_numpy(), base + 40 + v)
        np.testing.assert_array_equal(
            builder.field.surface_tension[v].to_numpy(), base + 50 + v
        )
        np.testing.assert_array_equal(
            builder.field.gas_mass[v].to_numpy(), base + 60 + v
        )

