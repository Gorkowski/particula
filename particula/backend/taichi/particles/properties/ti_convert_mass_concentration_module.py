"""Taichi backend for mass-to-fraction conversions."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ───── element-wise helpers ─────────────────────────────────
@ti.func
def fget_mole_single(mass: ti.f64, molar_mass: ti.f64,
                     inv_tot: ti.f64) -> ti.f64:            # xᵢ
    return (mass / molar_mass) * inv_tot


@ti.func
def fget_volume_single(mass: ti.f64, density: ti.f64,
                       inv_tot: ti.f64) -> ti.f64:           # ϕᵢ
    return (mass / density) * inv_tot


# ───── kernels ─────────────────────────────────────────────
@ti.kernel
def kget_mole_fraction_from_mass(
    m: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mm: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    total = 0.0
    for i in range(m.shape[0]):
        total += m[i] / mm[i]
    inv_tot = 1.0 / total if total != 0.0 else 0.0
    for i in range(out.shape[0]):
        out[i] = fget_mole_single(m[i], mm[i], inv_tot)


@ti.kernel
def kget_volume_fraction_from_mass(
    m: ti.types.ndarray(dtype=ti.f64, ndim=1),
    rho: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    total = 0.0
    for i in range(m.shape[0]):
        total += m[i] / rho[i]
    inv_tot = 1.0 / total if total != 0.0 else 0.0
    for i in range(out.shape[0]):
        out[i] = fget_volume_single(m[i], rho[i], inv_tot)


# ───── public wrappers + backend registration ──────────────
def _prepare_1d(a, b, name_a, name_b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must have identical shape.")
    return np.atleast_1d(a), np.atleast_1d(b)


@register("get_mole_fraction_from_mass", backend="taichi")
def ti_get_mole_fraction_from_mass(mass_concentrations, molar_masses):
    m, mm = _prepare_1d(mass_concentrations, molar_masses,
                        "mass_concentrations", "molar_masses")
    n = m.size
    m_ti, mm_ti, out_ti = (ti.ndarray(dtype=ti.f64, shape=n) for _ in range(3))
    m_ti.from_numpy(m)
    mm_ti.from_numpy(mm)
    kget_mole_fraction_from_mass(m_ti, mm_ti, out_ti)
    out_np = out_ti.to_numpy()
    return out_np.item() if out_np.size == 1 else out_np


@register("get_volume_fraction_from_mass", backend="taichi")
def ti_get_volume_fraction_from_mass(mass_concentrations, densities):
    m, rho = _prepare_1d(mass_concentrations, densities,
                         "mass_concentrations", "densities")
    n = m.size
    m_ti, rho_ti, out_ti = (ti.ndarray(dtype=ti.f64, shape=n) for _ in range(3))
    m_ti.from_numpy(m)
    rho_ti.from_numpy(rho)
    kget_volume_fraction_from_mass(m_ti, rho_ti, out_ti)
    out_np = out_ti.to_numpy()
    return out_np.item() if out_np.size == 1 else out_np

# ───── mass-fraction ─────────────────────────────────────────────
@ti.kernel
def kget_mass_fraction_from_mass(
    m: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    total = 0.0
    for i in range(m.shape[0]):
        total += m[i]
    inv_tot = 1.0 / total if total != 0.0 else 0.0
    for i in range(out.shape[0]):
        out[i] = m[i] * inv_tot

@register("get_mass_fraction_from_mass", backend="taichi")
def ti_get_mass_fraction_from_mass(mass_concentrations):
    m = np.atleast_1d(np.asarray(mass_concentrations, dtype=np.float64))
    n = m.size
    m_ti, out_ti = (ti.ndarray(dtype=ti.f64, shape=n) for _ in range(2))
    m_ti.from_numpy(m)
    kget_mass_fraction_from_mass(m_ti, out_ti)
    out_np = out_ti.to_numpy()
    return out_np.item() if out_np.size == 1 else out_np
