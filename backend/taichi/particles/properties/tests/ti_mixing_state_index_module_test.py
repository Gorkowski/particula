import taichi as ti
import numpy as np
import pytest

from particula.backend.taichi.particles.properties.ti_mixing_state_index_module import (
    ti_get_mixing_state_index,
    kget_mixing_state_index,
)

ti.init(arch=ti.cpu)

def reference_mixing_state_index(species_masses):
    """Reference NumPy implementation for testing."""
    species_masses_array = np.asarray(species_masses, dtype=float)
    species_masses_array = species_masses_array[
        species_masses_array.sum(axis=1) > 0
    ]
    if species_masses_array.size == 0:
        return np.nan
    mass_per_particle = species_masses_array.sum(axis=1)
    mass_fraction = species_masses_array / (mass_per_particle[:, None])
    per_particle_diversity = np.exp(
        -(mass_fraction * np.log(mass_fraction)).sum(axis=1)
    )
    total_mass = mass_per_particle.sum()
    if total_mass <= 0:
        return np.nan
    mass_weighted_diversity = (
        np.sum(mass_per_particle * per_particle_diversity) / total_mass
    )
    total_species_mass = species_masses_array.sum(axis=0)
    bulk_mass_fraction = total_species_mass / (total_mass)
    bulk_diversity = np.exp(
        -(bulk_mass_fraction * np.log(bulk_mass_fraction)).sum()
    )
    return (mass_weighted_diversity - 1.0) / (bulk_diversity - 1.0)

def test_ti_get_mixing_state_index_matches_reference():
    masses = np.array([[1.0e-15, 0.0],
                       [5.0e-16, 5.0e-16]])
    chi_ti = ti_get_mixing_state_index(masses)
    chi_ref = reference_mixing_state_index(masses)
    np.testing.assert_allclose(chi_ti, chi_ref, rtol=1e-12, atol=1e-14)

def test_kernel_direct_invocation():
    masses = np.array([[2.0, 0.0, 1.0],
                       [1.0, 1.0, 1.0]])
    n_particles, n_species = masses.shape
    species_masses_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, n_species))
    result_ti = ti.ndarray(dtype=ti.f64, shape=(1,))
    species_masses_ti.from_numpy(masses)
    kget_mixing_state_index(species_masses_ti, n_particles, n_species, result_ti)
    chi_kernel = result_ti.to_numpy().item()
    chi_ref = reference_mixing_state_index(masses)
    np.testing.assert_allclose(chi_kernel, chi_ref, rtol=1e-12, atol=1e-14)
