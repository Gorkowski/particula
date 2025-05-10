"""Taichi implementation of the mixing state index calculation."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_mixing_state_index(
    species_masses: ti.types.ndarray(dtype=ti.f64, ndim=2),
    n_particles: ti.i32,
    n_species: ti.i32
) -> ti.f64:
    """Element-wise calculation of the mixing state index for a set of particles."""
    # Only keep particles with non-zero total mass
    total_mass = 0.0
    for i in range(n_particles):
        mass_sum = 0.0
        for s in range(n_species):
            mass_sum += species_masses[i, s]
        if mass_sum > 0.0:
            total_mass += mass_sum
    if total_mass <= 0.0:
        return float('nan')

    # Per-particle mass fractions and diversity
    mass_weighted_diversity = 0.0
    total_species_mass = ti.Vector([0.0 for _ in range(64)])  # max 64 species
    for i in range(n_particles):
        mass_sum = 0.0
        for s in range(n_species):
            mass_sum += species_masses[i, s]
        if mass_sum > 0.0:
            # Per-particle mass fractions
            per_particle_diversity = 0.0
            for s in range(n_species):
                frac = species_masses[i, s] / mass_sum
                if frac > 0.0:
                    per_particle_diversity += frac * ti.log(frac)
                total_species_mass[s] += species_masses[i, s]
            per_particle_diversity = ti.exp(-per_particle_diversity)
            mass_weighted_diversity += mass_sum * per_particle_diversity

    mass_weighted_diversity = mass_weighted_diversity / total_mass

    # Bulk diversity
    bulk_diversity = 0.0
    for s in range(n_species):
        F_s = total_species_mass[s] / total_mass
        if F_s > 0.0:
            bulk_diversity += F_s * ti.log(F_s)
    bulk_diversity = ti.exp(-bulk_diversity)

    return (mass_weighted_diversity - 1.0) / (bulk_diversity - 1.0)

@ti.kernel
def kget_mixing_state_index(
    species_masses: ti.types.ndarray(dtype=ti.f64, ndim=2),
    n_particles: ti.i32,
    n_species: ti.i32,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1)
):
    """Kernel to compute the mixing state index for a set of particles."""
    result[0] = fget_mixing_state_index(species_masses, n_particles, n_species)

@register("get_mixing_state_index", backend="taichi")
def ti_get_mixing_state_index(species_masses):
    """Taichi-accelerated mixing state index calculation."""
    if not isinstance(species_masses, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy array for species_masses.")

    a = np.atleast_2d(species_masses)
    n_particles, n_species = a.shape
    # Allocate Taichi NDArray buffers
    species_masses_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, n_species))
    result_ti = ti.ndarray(dtype=ti.f64, shape=(1,))
    species_masses_ti.from_numpy(a)
    kget_mixing_state_index(species_masses_ti, n_particles, n_species, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item()
