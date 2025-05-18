import taichi as ti, numpy as np

ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.gas.species import GasSpecies as PySpecies
from particula.backend.taichi.gas.ti_species import TiGasSpecies as TiSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy as PyVP,
)
from particula.backend.taichi.gas.ti_vapor_pressure_strategies import (
    ConstantVaporPressureStrategy as TiVP,
)


def test_scalar_species_partial_pressure():
    python_vapor_pressure_strategy = PyVP(2330.0)
    taichi_vapor_pressure_strategy = TiVP(2330.0)
    python_species = PySpecies(
        "H2O", 0.018, python_vapor_pressure_strategy, True, 1e-3
    )
    taichi_species = TiSpecies(
        "H2O", 0.018, taichi_vapor_pressure_strategy, True, 1e-3
    )

    temperature_k = 298.15
    np.testing.assert_allclose(
        python_species.get_partial_pressure(temperature_k),
        taichi_species.get_partial_pressure(temperature_k).to_numpy(),
        rtol=1e-8,
    )


def test_vector_species_partial_pressure():
    """Shared-strategy case: same constant vapor-pressure for all species."""
    names = np.array(["H2O", "CO2"])
    molar_masses = np.array([0.018, 0.044])
    concentrations = np.array([1e-3, 2e-3])

    shared_strategy_python = PyVP(2_330.0)
    shared_strategy_taichi = TiVP(2_330.0)

    python_species = PySpecies(
        names, molar_masses, shared_strategy_python, True, concentrations
    )
    taichi_species = TiSpecies(
        names, molar_masses, shared_strategy_taichi, True, concentrations
    )

    temperature_k = 298.15  # K
    np.testing.assert_allclose(
        python_species.get_partial_pressure(temperature_k),
        taichi_species.get_partial_pressure(temperature_k).to_numpy(),
        rtol=1e-8,
    )

    # shapes must still match the number of species
    assert python_species.get_partial_pressure(temperature_k).shape == (2,)
    assert taichi_species.get_partial_pressure(temperature_k).shape == (2,)


def test_vector_species_list_strategy():
    """Compare Py- and Ti-implementations when each species has its own strategy."""
    names = np.array(["H2O", "CO2"])
    molar_masses = np.array([0.018, 0.044])
    concentrations = np.array([1e-3, 2e-3])

    strategies_python = [PyVP(2_330.0), PyVP(101_325.0)]
    strategies_taichi = [TiVP(2_330.0), TiVP(101_325.0)]

    python_species = PySpecies(
        names, molar_masses, strategies_python, True, concentrations
    )
    taichi_species = TiSpecies(
        names, molar_masses, strategies_taichi, True, concentrations
    )

    temperature_k = 298.15  # K

    # pure vapor pressure must agree
    np.testing.assert_allclose(
        python_species.get_pure_vapor_pressure(temperature_k),
        taichi_species.get_pure_vapor_pressure(temperature_k).to_numpy(),
        rtol=1e-8,
    )

    # partial pressure must also agree
    np.testing.assert_allclose(
        python_species.get_partial_pressure(temperature_k),
        taichi_species.get_partial_pressure(temperature_k).to_numpy(),
        rtol=1e-8,
    )

    # shapes should equal the number of species
    assert python_species.get_pure_vapor_pressure(temperature_k).shape == (2,)
    assert taichi_species.get_pure_vapor_pressure(temperature_k).shape == (2,)


def test_concentration_mutators_with_ti_arrays():
    names           = np.array(["H2O", "CO2"])
    molar_masses    = np.array([0.018, 0.044])
    init_conc_np    = np.array([1e-3, 2e-3])

    species = TiSpecies(names, molar_masses, TiVP(2330.0), True, init_conc_np)

    # --- add_concentration ----------------------------------------------
    delta = ti.ndarray(dtype=ti.f64, shape=(2,))
    delta[0] = 1e-4
    delta[1] = -5e-4      # will be clipped to zero for the second entry

    species.add_concentration(delta)

    np.testing.assert_allclose(
        species.get_concentration().to_numpy(),
        np.maximum(0.0, init_conc_np + np.array([1e-4, -5e-4])),
        rtol=1e-12,
    )

    # --- set_concentration ----------------------------------------------
    new_vals = ti.ndarray(dtype=ti.f64, shape=(2,))
    new_vals[0] = 0.0
    new_vals[1] = 3e-3

    species.set_concentration(new_vals)

    np.testing.assert_allclose(
        species.get_concentration().to_numpy(),
        np.array([0.0, 3e-3]),
        rtol=1e-12,
    )
