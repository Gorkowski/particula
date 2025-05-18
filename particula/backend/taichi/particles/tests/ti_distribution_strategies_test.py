import taichi as ti
import pytest
import numpy as np
from particula.backend.taichi.particles.ti_distribution_strategies import (
    TiParticleResolvedSpeciatedMass,
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

# convert a Taichi ndarray to NumPy for comparison
def _np(a):
    return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)

@pytest.mark.parametrize("ndim", [2])
def test_getters(ndim):
    d, c, rho = _dummy_arrays(ndim)
    strat = TiParticleResolvedSpeciatedMass()

    # ─ get_species_mass ─
    sm = strat.get_species_mass(d, rho)
    assert sm.shape == d.shape
    assert np.allclose(_np(sm), _np(d))

    # ─ get_mass ─
    m = strat.get_mass(d, rho)
    if ndim == 1:
        assert np.allclose(_np(m), _np(d))
    else:
        assert np.allclose(_np(m), _np(d).sum(axis=1))

    # ─ get_total_mass ─
    tm = strat.get_total_mass(d, c, rho)
    expected_total = float(np.dot(_np(m), _np(c)))
    assert np.isclose(tm, expected_total)

    # ─ get_radius ─
    r = strat.get_radius(d, rho)
    vol = _np(d) / _np(rho)
    expected_r = (3 * vol / (4 * np.pi)) ** (1 / 3)
    if ndim == 2:
        expected_r = expected_r.sum(axis=1) ** (1 / 3)
    assert np.allclose(_np(r), expected_r)

    # ─ get_name ─
    assert strat.get_name() == "TiParticleResolvedSpeciatedMass"


@pytest.mark.parametrize("ndim", [2])
def test_add_mass(ndim):
    d, c, rho = _dummy_arrays(ndim)
    added = ti.ndarray(dtype=ti.f64, shape=d.shape)
    added.fill(0.1)                         # constant extra mass
    strat = TiParticleResolvedSpeciatedMass()
    new_d, new_c = strat.add_mass(d, c, rho, added)

    expected = _np(d) + _np(added) / _np(c)[..., None] if ndim == 2 else \
               _np(d) + _np(added) / _np(c)
    assert np.allclose(_np(new_d), expected)
    assert np.allclose(_np(new_c), _np(c))      # concentration unchanged


def test_add_concentration():
    d, c, rho = _dummy_arrays(2)
    add_c = ti.ndarray(dtype=ti.f64, shape=c.shape)
    add_c.fill(5.0)
    strat = TiParticleResolvedSpeciatedMass()
    new_d, new_c = strat.add_concentration(d, c, d, add_c)
    assert np.allclose(_np(new_d), _np(d))
    assert np.allclose(_np(new_c), _np(c) + 5.0)


@pytest.mark.parametrize("ndim", [2])
def test_collide_pairs(ndim):
    d, c, rho = _dummy_arrays(ndim)
    idx = ti.ndarray(dtype=ti.f64, shape=(1, 2))
    idx[0, 0], idx[0, 1] = 0, 1             # merge 0 → 1

    strat = TiParticleResolvedSpeciatedMass()
    new_d, new_c = strat.collide_pairs(d, c, rho, idx)

    np_d, np_c = _np(new_d), _np(new_c)
    if ndim == 1:
        assert np_d[0] == 0
        assert np_d[1] == 3                  # 1+2
    else:
        assert np.allclose(np_d[0], 0)
        assert np.allclose(np_d[1],
                           np.array([1, 2]) + np.array([3, 4]))
    assert np_c[0] == 0
