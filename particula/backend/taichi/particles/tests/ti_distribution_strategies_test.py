import taichi as ti
import numpy as np
import pytest
from particula.backend.taichi.particles.ti_distribution_strategies import (
    TiMassBasedMovingBin, TiRadiiBasedMovingBin,
    TiSpeciatedMassMovingBin, TiParticleResolvedSpeciatedMass,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)

def _dummy_arrays(ndim=1):
    if ndim == 1:
        d = np.array([1.0, 2.0], dtype=np.float64)
        c = np.array([10.0, 20.0], dtype=np.float64)
    else:
        d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        c = np.array([10.0, 20.0], dtype=np.float64)
    rho = np.ones_like(d, dtype=np.float64)
    return d, c, rho

import pytest

@pytest.mark.parametrize("cls", [
    TiMassBasedMovingBin,
    TiRadiiBasedMovingBin,
    TiSpeciatedMassMovingBin,
    TiParticleResolvedSpeciatedMass,
])
def test_strategy_instantiation(cls):
    d, c, rho = _dummy_arrays()
    strat = cls()                       # constructor must accept no args
    mass = strat.get_species_mass(d, rho)
    assert mass.shape == d.shape
