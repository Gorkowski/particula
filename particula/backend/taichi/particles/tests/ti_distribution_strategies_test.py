import taichi as ti
import pytest
from particula.backend.taichi.particles.ti_distribution_strategies import (
    MassBasedMovingBin, RadiiBasedMovingBin,
    SpeciatedMassMovingBin, ParticleResolvedSpeciatedMass,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)

def _dummy_arrays(ndim: int = 1):
    """
    Create dummy data as Taichi ndarrays (ti.ndarray) for the tests.

    No NumPy objects are returned; everything is ready to be passed
    directly to the Taichi distribution-strategy methods.
    """
    if ndim == 1:
        d = ti.ndarray(dtype=ti.f64, shape=(2,))
        c = ti.ndarray(dtype=ti.f64, shape=(2,))
        for i, val in enumerate((1.0, 2.0)):
            d[i] = val
        for i, val in enumerate((10.0, 20.0)):
            c[i] = val
        rho = ti.ndarray(dtype=ti.f64, shape=(2,))
    else:
        d = ti.ndarray(dtype=ti.f64, shape=(2, 2))
        vals = ((1.0, 2.0), (3.0, 4.0))
        for i in range(2):
            for j in range(2):
                d[i, j] = vals[i][j]
        c = ti.ndarray(dtype=ti.f64, shape=(2,))
        for i, val in enumerate((10.0, 20.0)):
            c[i] = val
        rho = ti.ndarray(dtype=ti.f64, shape=(2, 2))

    rho.fill(1.0)                # set all‐ones density
    return d, c, rho

@pytest.mark.parametrize("cls", [
    MassBasedMovingBin,
    RadiiBasedMovingBin,
    SpeciatedMassMovingBin,
    ParticleResolvedSpeciatedMass,
])
def test_strategy_instantiation(cls):
    d, c, rho = _dummy_arrays()
    strat = cls()                       # constructor must accept no args
    mass = strat.get_species_mass(d, rho)
    assert mass.shape == d.shape
