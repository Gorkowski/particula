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
        distribution = ti.ndarray(dtype=ti.f64, shape=(2,))
        concentration = ti.ndarray(dtype=ti.f64, shape=(2,))
        for i, val in enumerate((1.0, 2.0)):
            distribution[i] = val
        for i, val in enumerate((10.0, 20.0)):
            concentration[i] = val
        density = ti.ndarray(dtype=ti.f64, shape=(2,))
    else:
        distribution = ti.ndarray(dtype=ti.f64, shape=(2, 2))
        vals = ((1.0, 2.0), (3.0, 4.0))
        for i in range(2):
            for j in range(2):
                distribution[i, j] = vals[i][j]
        concentration = ti.ndarray(dtype=ti.f64, shape=(2,))
        for i, val in enumerate((10.0, 20.0)):
            concentration[i] = val
        density = ti.ndarray(dtype=ti.f64, shape=(2, 2))

    density.fill(1.0)                # set all‐ones density
    return distribution, concentration, density

# convert a Taichi ndarray to NumPy for comparison
def _np(a):
    return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)

@pytest.mark.parametrize("ndim", [2])
def test_getters(ndim):
    distribution, concentration, density = _dummy_arrays(ndim)
    strat = TiParticleResolvedSpeciatedMass()

    # ─ get_species_mass ─
    species_mass = strat.get_species_mass(distribution, density)
    assert species_mass.shape == distribution.shape
    assert np.allclose(_np(species_mass), _np(distribution))

    # ─ get_mass ─
    mass = strat.get_mass(distribution, density)
    assert np.allclose(_np(mass), _np(distribution).sum(axis=1))

    # ─ get_total_mass ─
    total_mass = strat.get_total_mass(distribution, concentration, density)
    expected_total = float(np.dot(_np(mass), _np(concentration)))
    assert np.isclose(total_mass, expected_total)

    # ─ get_radius ─
    radius = strat.get_radius(distribution, density)
    volume = _np(distribution) / _np(density)
    expected_radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    expected_radius = expected_radius.sum(axis=1) ** (1 / 3)
    assert np.allclose(_np(radius), expected_radius)

    # ─ get_name ─
    assert strat.get_name() == "TiParticleResolvedSpeciatedMass"


@pytest.mark.parametrize("ndim", [2])
def test_add_mass(ndim):
    distribution, concentration, density = _dummy_arrays(ndim)
    added_mass = ti.ndarray(dtype=ti.f64, shape=distribution.shape)
    added_mass.fill(0.1)                         # constant extra mass
    strat = TiParticleResolvedSpeciatedMass()
    new_distribution, new_concentration = strat.add_mass(
        distribution, concentration, density, added_mass
    )

    expected = _np(distribution) + _np(added_mass) / _np(concentration)[..., None]
    assert np.allclose(_np(new_distribution), expected)
    assert np.allclose(_np(new_concentration), _np(concentration))      # concentration unchanged


def test_add_concentration():
    distribution, concentration, density = _dummy_arrays(2)
    added_concentration = ti.ndarray(dtype=ti.f64, shape=concentration.shape)
    added_concentration.fill(5.0)
    strat = TiParticleResolvedSpeciatedMass()
    new_distribution, new_concentration = strat.add_concentration(
        distribution, concentration, distribution, added_concentration
    )
    assert np.allclose(_np(new_distribution), _np(distribution))
    assert np.allclose(_np(new_concentration), _np(concentration) + 5.0)


@pytest.mark.parametrize("ndim", [2])
def test_collide_pairs(ndim):
    distribution, concentration, density = _dummy_arrays(ndim)
    indices = ti.ndarray(dtype=ti.f64, shape=(1, 2))
    indices[0, 0], indices[0, 1] = 0, 1             # merge 0 → 1

    strat = TiParticleResolvedSpeciatedMass()
    new_distribution, new_concentration = strat.collide_pairs(
        distribution, concentration, density, indices
    )

    np_distribution = _np(new_distribution)
    np_concentration = _np(new_concentration)
    assert np.allclose(np_distribution[0], 0)
    assert np.allclose(
        np_distribution[1], np.array([1, 2]) + np.array([3, 4])
    )
    assert np_concentration[0] == 0
