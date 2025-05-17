import taichi as ti, numpy as np
ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.gas.species import GasSpecies as PySpecies
from particula.backend.taichi.gas.ti_species import GasSpecies as TiSpecies
from particula.gas.vapor_pressure_strategies import ConstantVaporPressureStrategy

def test_scalar_species_partial_pressure():
    vp = ConstantVaporPressureStrategy(2330.0)
    py = PySpecies("H2O", 0.018, vp, True, 1e-3)
    ti_species = TiSpecies("H2O", 0.018, vp, True, 1e-3)

    T = 298.15
    np.testing.assert_allclose(
        py.get_partial_pressure(T),
        ti_species.get_partial_pressure(T),
        rtol=1e-12,
    )

# ‑-- append below the current test_scalar_species_partial_pressure() -----------
import numpy as np

def test_vector_species_partial_pressure():
    """Ensure PySpecies and TiSpecies give identical results for N>1 species."""
    # two arbitrary species
    names          = np.array(["H2O", "CO2"])
    molar_masses   = np.array([0.018, 0.044])
    concentrations = np.array([1e-3, 2e-3])
    vapor_pressure = ConstantVaporPressureStrategy(
        np.array([2_330.0, 101_325.0])
    )

    py_species  = PySpecies(names, molar_masses, vapor_pressure,
                            True, concentrations)
    ti_species  = TiSpecies(names, molar_masses, vapor_pressure,
                            True, concentrations)

    T = 298.15  # K
    np.testing.assert_allclose(
        py_species.get_partial_pressure(T),
        ti_species.get_partial_pressure(T),
        rtol=1e-12,
    )

    # shapes must match the number of species
    assert py_species.get_partial_pressure(T).shape == (2,)
    assert ti_species.get_partial_pressure(T).shape == (2,)

def test_vector_species_list_strategy():
    """Compare Py- and Ti-implementations when each species has its own strategy."""
    names          = np.array(["H2O", "CO2"])
    molar_masses   = np.array([0.018, 0.044])
    concentrations = np.array([1e-3, 2e-3])

    strategies = [
        ConstantVaporPressureStrategy(2_330.0),     # for H2O
        ConstantVaporPressureStrategy(101_325.0),   # for CO2
    ]

    py_species = PySpecies(
        names, molar_masses, strategies, True, concentrations
    )
    ti_species = TiSpecies(
        names, molar_masses, strategies, True, concentrations
    )

    T = 298.15  # K

    # pure vapor pressure must agree
    np.testing.assert_allclose(
        py_species.get_pure_vapor_pressure(T),
        ti_species.get_pure_vapor_pressure(T),
        rtol=1e-12,
    )

    # partial pressure must also agree
    np.testing.assert_allclose(
        py_species.get_partial_pressure(T),
        ti_species.get_partial_pressure(T),
        rtol=1e-12,
    )

    # shapes should equal the number of species
    assert py_species.get_pure_vapor_pressure(T).shape == (2,)
    assert ti_species.get_pure_vapor_pressure(T).shape == (2,)
