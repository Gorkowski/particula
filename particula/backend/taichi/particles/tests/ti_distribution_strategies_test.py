import taichi as ti
import pytest
import numpy as np
from particula.backend.taichi.particles.ti_distribution_strategies import (
    TiParticleResolvedSpeciatedMass,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)


def _dummy_arrays():
    """
    Create dummy data as Taichi ndarrays (ti.ndarray) for the tests.

    No NumPy objects are returned; everything is ready to be passed
    directly to the Taichi distribution-strategy methods.
    """
    distribution = ti.ndarray(dtype=ti.f64, shape=(10, 3))
    vals = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
        ]
    )
    distribution.from_numpy(vals)
    concentration = ti.ndarray(dtype=ti.f64, shape=(10,))
    concentration.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
    density = ti.ndarray(dtype=ti.f64, shape=(3,))
    density.from_numpy(np.array([1000.0, 1200.0, 1400.0]))

    return distribution, concentration, density

# convert a Taichi ndarray to NumPy for comparison
def _np(a):
    return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)

def test_getters():
    distribution, concentration, density = _dummy_arrays()
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
    radius = _np(radius)
    volume = _np(distribution) / _np(density)[np.newaxis, :]
    expected_radius = (3 * np.sum(volume, axis=1) / (4 * np.pi)) ** (1 / 3)
    assert np.allclose(radius, expected_radius)


def test_add_mass():
    distribution, concentration, density = _dummy_arrays()
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
    distribution, concentration, density = _dummy_arrays()
    added_concentration = ti.ndarray(dtype=ti.f64, shape=concentration.shape)
    added_concentration.fill(5.0)
    strat = TiParticleResolvedSpeciatedMass()
    new_distribution, new_concentration = strat.add_concentration(
        distribution, concentration, distribution, added_concentration
    )
    assert np.allclose(_np(new_distribution), _np(distribution))
    assert np.allclose(_np(new_concentration), _np(concentration) + 5.0)
