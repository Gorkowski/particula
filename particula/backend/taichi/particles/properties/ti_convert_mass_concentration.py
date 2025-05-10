"""Taichi-accelerated functions to convert mass concentrations to other concentration units."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_mole_fraction_from_mass(mass_concentration: ti.f64, molar_mass: ti.f64, total_moles: ti.f64) -> ti.f64:
    """Elementwise Taichi function for mole fraction from mass concentration."""
    if total_moles == 0.0:
        return 0.0
    return (mass_concentration / molar_mass) / total_moles

@ti.kernel
def kget_mole_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_masses: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for mole fraction from mass concentration."""
    n = result.shape[0]
    total_moles = 0.0
    for i in range(n):
        total_moles += mass_concentrations[i] / molar_masses[i]
    for i in range(n):
        result[i] = fget_mole_fraction_from_mass(mass_concentrations[i], molar_masses[i], total_moles)

@register("get_mole_fraction_from_mass", backend="taichi")
def ti_get_mole_fraction_from_mass(mass_concentrations, molar_masses):
    """Taichi wrapper for mole fraction from mass concentration."""
    if not (isinstance(mass_concentrations, np.ndarray) and isinstance(molar_masses, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    a1, a2 = np.atleast_1d(mass_concentrations), np.atleast_1d(molar_masses)
    n = a1.size
    a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a1_ti.from_numpy(a1)
    a2_ti.from_numpy(a2)
    kget_mole_fraction_from_mass(a1_ti, a2_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@ti.func
def fget_volume_fraction_from_mass(mass_concentration: ti.f64, density: ti.f64, total_volume: ti.f64) -> ti.f64:
    """Elementwise Taichi function for volume fraction from mass concentration."""
    if total_volume == 0.0:
        return 0.0
    return (mass_concentration / density) / total_volume

@ti.kernel
def kget_volume_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    densities: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for volume fraction from mass concentration."""
    n = result.shape[0]
    total_volume = 0.0
    for i in range(n):
        total_volume += mass_concentrations[i] / densities[i]
    for i in range(n):
        result[i] = fget_volume_fraction_from_mass(mass_concentrations[i], densities[i], total_volume)

@register("get_volume_fraction_from_mass", backend="taichi")
def ti_get_volume_fraction_from_mass(mass_concentrations, densities):
    """Taichi wrapper for volume fraction from mass concentration."""
    if not (isinstance(mass_concentrations, np.ndarray) and isinstance(densities, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    a1, a2 = np.atleast_1d(mass_concentrations), np.atleast_1d(densities)
    n = a1.size
    a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a1_ti.from_numpy(a1)
    a2_ti.from_numpy(a2)
    kget_volume_fraction_from_mass(a1_ti, a2_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@ti.func
def fget_mass_fraction_from_mass(mass_concentration: ti.f64, total_mass: ti.f64) -> ti.f64:
    """Elementwise Taichi function for mass fraction from mass concentration."""
    if total_mass == 0.0:
        return 0.0
    return mass_concentration / total_mass

@ti.kernel
def kget_mass_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taichi kernel for mass fraction from mass concentration."""
    n = result.shape[0]
    total_mass = 0.0
    for i in range(n):
        total_mass += mass_concentrations[i]
    for i in range(n):
        result[i] = fget_mass_fraction_from_mass(mass_concentrations[i], total_mass)

@register("get_mass_fraction_from_mass", backend="taichi")
def ti_get_mass_fraction_from_mass(mass_concentrations):
    """Taichi wrapper for mass fraction from mass concentration."""
    if not isinstance(mass_concentrations, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy array for input.")
    a1 = np.atleast_1d(mass_concentrations)
    n = a1.size
    a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    a1_ti.from_numpy(a1)
    kget_mass_fraction_from_mass(a1_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
