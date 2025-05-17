import taichi as ti, numpy as np
ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.gas.species import GasSpecies as PySpecies
from particula.backend.taichi.gas.ti_species import GasSpecies as TiSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy as PyVP,
)
from particula.backend.taichi.gas.ti_vapor_pressure_strategies import (
    ConstantVaporPressureStrategy as TiVP,
)

def test_scalar_species_partial_pressure():
    vp_py  = PyVP(2330.0)
    vp_ti  = TiVP(2330.0)
    py         = PySpecies("H2O", 0.018, vp_py, True, 1e-3)
    ti_species = TiSpecies("H2O", 0.018, vp_ti, True, 1e-3)

    T = 298.15
    np.testing.assert_allclose(
        py.get_partial_pressure(T),
        ti_species.get_partial_pressure(T),
        rtol=1e-12,
    )

def test_vector_species_partial_pressure():
    """Shared-strategy case: same constant vapor-pressure for all species."""
    names          = np.array(["H2O", "CO2"])
    molar_masses   = np.array([0.018, 0.044])
    concentrations = np.array([1e-3, 2e-3])

    shared_strategy_py = PyVP(2_330.0)
    shared_strategy_ti = TiVP(2_330.0)

    py_species = PySpecies(
        names, molar_masses, shared_strategy_py, True, concentrations
    )
    ti_species = TiSpecies(
        names, molar_masses, shared_strategy_ti, True, concentrations
    )

    T = 298.15  # K
    np.testing.assert_allclose(
        py_species.get_partial_pressure(T),
        ti_species.get_partial_pressure(T),
        rtol=1e-12,
    )

    # shapes must still match the number of species
    assert py_species.get_partial_pressure(T).shape == (2,)
    assert ti_species.get_partial_pressure(T).shape == (2,)


def test_vector_species_list_strategy():
    """Compare Py- and Ti-implementations when each species has its own strategy."""
    names          = np.array(["H2O", "CO2"])
    molar_masses   = np.array([0.018, 0.044])
    concentrations = np.array([1e-3, 2e-3])

    strategies_py = [PyVP(2_330.0), PyVP(101_325.0)]
    strategies_ti = [TiVP(2_330.0), TiVP(101_325.0)]

    py_species = PySpecies(
        names, molar_masses, strategies_py, True, concentrations
    )
    ti_species = TiSpecies(
        names, molar_masses, strategies_ti, True, concentrations
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
