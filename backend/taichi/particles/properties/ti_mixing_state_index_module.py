"""Taichi implementation of the mixing state index calculation."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

MAX_SPECIES = 64                              # upper limit supported
ts = ti.field(dtype=ti.f64, shape=MAX_SPECIES) # per-species scratch space

@ti.func
def f_shannon_entropy(frac: ti.f64) -> ti.f64:
    return frac * ti.log(frac) if frac > 0.0 else 0.0

@ti.kernel
def kget_mixing_state_index(
    sm: ti.types.ndarray(dtype=ti.f64, ndim=2),   # species masses
    n_particles: ti.i32,
    n_species: ti.i32,
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),  # χ
):
    total_mass = 0.0
    # 1) filter empty particles & accumulate total_mass
    for i in range(n_particles):
        mp = 0.0
        for s in range(n_species):
            mp += sm[i, s]
        if mp > 0.0:
            total_mass += mp

    if total_mass <= 0.0:          # no mass → NaN
        out[0] = ti.nan
        return

    # 2) mass-weighted mean diversity  D̄α
    mw_div = 0.0
    # Step (a) reset per-species accumulator
    for s in range(n_species):
        ts[s] = 0.0

    for i in range(n_particles):
        mp = 0.0
        for s in range(n_species):
            mp += sm[i, s]
        if mp > 0.0:
            H = 0.0
            for s in range(n_species):
                frac = sm[i, s] / mp
                H += f_shannon_entropy(frac)
                ts[s] += sm[i, s]           # accumulate species mass
            D_i = ti.exp(-H)
            mw_div += mp * D_i

    mw_div /= total_mass                     # D̄α

    # 3) bulk diversity Dγ
    H_bulk = 0.0
    for s in range(n_species):
        F_s = ts[s] / total_mass
        H_bulk += f_shannon_entropy(F_s)

    D_bulk = ti.exp(-H_bulk)

    # 4) mixing-state index χ
    out[0] = (mw_div - 1.0) / (D_bulk - 1.0)

@register("get_mixing_state_index", backend="taichi")
def ti_get_mixing_state_index(species_masses):
    """Taichi-accelerated mixing state index calculation."""
    if not isinstance(species_masses, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy array for species_masses.")

    a = np.atleast_2d(species_masses)
    n_particles, n_species = a.shape
    if a.shape[1] > MAX_SPECIES:
        raise ValueError(f"Taichi backend supports up to {MAX_SPECIES} species.")
    # Allocate Taichi NDArray buffers
    species_masses_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, n_species))
    result_ti = ti.ndarray(dtype=ti.f64, shape=(1,))
    species_masses_ti.from_numpy(a)
    kget_mixing_state_index(species_masses_ti, n_particles, n_species, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item()
