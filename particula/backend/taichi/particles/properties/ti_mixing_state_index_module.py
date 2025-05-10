"""Taichi version of get_mixing_state_index."""
import taichi as ti
import numpy as np
from particula.backend import register

# ---------- helpers ---------------------------------------------------
@ti.func
def _safe_log10(x: ti.f64) -> ti.f64:          # avoids log(0)
    eps = 1.0e-300
    return ti.log(ti.max(x, eps)) / ti.log(10.0)

@ti.func
def _safe_exp(x: ti.f64) -> ti.f64:            # avoids overflow
    lim = 709.0                                # ≈ ln(max float64)
    return ti.exp(ti.min(x, lim))

# ---------- scalar χ (works on ONE N×S matrix) ------------------------
@ti.func
def fget_mixing_state_index(
    masses: ti.template(),          # 2-D Taichi NDArray view
    n_particles: ti.i32,
    n_species: ti.i32,
) -> ti.f64:
    bulk_mass = ti.Vector([0.0] * 64)          # compile-time upper bound
    tot_mass = 0.0
    mw_div_sum = 0.0

    for i in range(n_particles):
        m_p = 0.0
        for s in range(n_species):
            m_p += masses[i, s]

        if m_p > 0.0:                          # skip empty particle
            ent = 0.0
            for s in range(n_species):
                m_ns = masses[i, s]
                if m_ns > 0.0:
                    f_ns = m_ns / m_p
                    ent += -f_ns * _safe_log10(f_ns)
                    bulk_mass[s] += m_ns
            D_n = _safe_exp(ent)
            mw_div_sum += m_p * D_n
            tot_mass += m_p

    if tot_mass == 0.0:        # aerosol has no mass
        return float("nan")

    D_bar = mw_div_sum / tot_mass

    bulk_ent = 0.0
    for s in range(n_species):
        if bulk_mass[s] > 0.0:
            F_s = bulk_mass[s] / tot_mass
            bulk_ent += -F_s * _safe_log10(F_s)
    D_gamma = _safe_exp(bulk_ent)

    return (D_bar - 1.0) / (D_gamma - 1.0)

# ---------- kernel ----------------------------------------------------
@ti.kernel
def kget_mixing_state_index(                   # noqa: N802
    species_masses: ti.types.ndarray(dtype=ti.f64, ndim=2),
    n_particles: ti.i32,
    n_species: ti.i32,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),   # shape = (1,)
):
    result[0] = fget_mixing_state_index(species_masses, n_particles, n_species)

# ---------- public wrapper -------------------------------------------
@register("get_mixing_state_index", backend="taichi")
def ti_get_mixing_state_index(species_masses):
    """Taichi backend wrapper for get_mixing_state_index."""
    if not isinstance(species_masses, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy ndarray.")

    sm = np.asarray(species_masses, dtype=float)
    if sm.ndim != 2:
        raise ValueError("Input must be a 2-D (N×S) array.")

    n_particles, n_species = sm.shape
    if n_species > 64:
        raise ValueError("Taichi version supports ≤ 64 species.")

    sm_ti = ti.ndarray(dtype=ti.f64, shape=(n_particles, n_species))
    sm_ti.from_numpy(sm)

    res_ti = ti.ndarray(dtype=ti.f64, shape=1)
    kget_mixing_state_index(sm_ti, n_particles, n_species, res_ti)
    return res_ti.to_numpy()[0]
