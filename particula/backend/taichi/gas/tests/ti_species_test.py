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
