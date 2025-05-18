"""Taichi backend for mass-to-fraction conversions."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ───── element-wise helpers ─────────────────────────────────
@ti.func
def fget_mole_fraction_single(
    mass: ti.f64, molar_mass: ti.f64, inverse_total: ti.f64
) -> ti.f64:  # xᵢ
    return (mass / molar_mass) * inverse_total


@ti.func
def fget_volume_fraction_single(
    mass: ti.f64, density: ti.f64, inverse_total: ti.f64
) -> ti.f64:  # ϕᵢ
    return (mass / density) * inverse_total


# ───── kernels ─────────────────────────────────────────────
@ti.kernel
def kget_mole_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_masses: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mole_fractions: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    total_moles = 0.0
    for i in range(mass_concentrations.shape[0]):
        total_moles += mass_concentrations[i] / molar_masses[i]
    inverse_total_moles = 1.0 / total_moles if total_moles != 0.0 else 0.0
    for i in range(mole_fractions.shape[0]):
        mole_fractions[i] = fget_mole_fraction_single(
            mass_concentrations[i], molar_masses[i], inverse_total_moles
        )


@ti.kernel
def kget_volume_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    densities: ti.types.ndarray(dtype=ti.f64, ndim=1),
    volume_fractions: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    total_volume = 0.0
    for i in range(mass_concentrations.shape[0]):
        total_volume += mass_concentrations[i] / densities[i]
    inverse_total_volume = 1.0 / total_volume if total_volume != 0.0 else 0.0
    for i in range(volume_fractions.shape[0]):
        volume_fractions[i] = fget_volume_fraction_single(
            mass_concentrations[i], densities[i], inverse_total_volume
        )


# ───── public wrappers + backend registration ──────────────
def _prepare_1d_arrays(array_a, array_b, name_a, name_b):
    array_a = np.asarray(array_a, dtype=np.float64)
    array_b = np.asarray(array_b, dtype=np.float64)
    if array_a.shape != array_b.shape:
        raise ValueError(f"{name_a} and {name_b} must have identical shape.")
    return np.atleast_1d(array_a), np.atleast_1d(array_b)


@register("get_mole_fraction_from_mass", backend="taichi")
def taichi_get_mole_fraction_from_mass(mass_concentrations, molar_masses):
    mass_concentrations_np, molar_masses_np = _prepare_1d_arrays(
        mass_concentrations, molar_masses, "mass_concentrations", "molar_masses"
    )
    n_items = mass_concentrations_np.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    molar_masses_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    mole_fractions_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    mass_concentrations_ti.from_numpy(mass_concentrations_np)
    molar_masses_ti.from_numpy(molar_masses_np)
    kget_mole_fraction_from_mass(
        mass_concentrations_ti, molar_masses_ti, mole_fractions_ti
    )
    mole_fractions_np = mole_fractions_ti.to_numpy()
    return mole_fractions_np.item() if mole_fractions_np.size == 1 else mole_fractions_np


@register("get_volume_fraction_from_mass", backend="taichi")
def taichi_get_volume_fraction_from_mass(mass_concentrations, densities):
    mass_concentrations_np, densities_np = _prepare_1d_arrays(
        mass_concentrations, densities, "mass_concentrations", "densities"
    )
    n_items = mass_concentrations_np.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    densities_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    volume_fractions_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    mass_concentrations_ti.from_numpy(mass_concentrations_np)
    densities_ti.from_numpy(densities_np)
    kget_volume_fraction_from_mass(
        mass_concentrations_ti, densities_ti, volume_fractions_ti
    )
    volume_fractions_np = volume_fractions_ti.to_numpy()
    return volume_fractions_np.item() if volume_fractions_np.size == 1 else volume_fractions_np

# ───── mass-fraction ─────────────────────────────────────────────
@ti.kernel
def kget_mass_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mass_fractions: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    total_mass = 0.0
    for i in range(mass_concentrations.shape[0]):
        total_mass += mass_concentrations[i]
    inverse_total_mass = 1.0 / total_mass if total_mass != 0.0 else 0.0
    for i in range(mass_fractions.shape[0]):
        mass_fractions[i] = mass_concentrations[i] * inverse_total_mass

@register("get_mass_fraction_from_mass", backend="taichi")
def taichi_get_mass_fraction_from_mass(mass_concentrations):
    mass_concentrations_np = np.atleast_1d(np.asarray(mass_concentrations, dtype=np.float64))
    n_items = mass_concentrations_np.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    mass_fractions_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    mass_concentrations_ti.from_numpy(mass_concentrations_np)
    kget_mass_fraction_from_mass(mass_concentrations_ti, mass_fractions_ti)
    mass_fractions_np = mass_fractions_ti.to_numpy()
    return mass_fractions_np.item() if mass_fractions_np.size == 1 else mass_fractions_np
