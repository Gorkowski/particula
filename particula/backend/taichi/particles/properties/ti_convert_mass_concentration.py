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
    n_concentrations = result.shape[0]
    total_moles = 0.0
    for i in range(n_concentrations):
        total_moles += mass_concentrations[i] / molar_masses[i]
    for i in range(n_concentrations):
        result[i] = fget_mole_fraction_from_mass(
            mass_concentrations[i], molar_masses[i], total_moles
        )

@register("get_mole_fraction_from_mass", backend="taichi")
def ti_get_mole_fraction_from_mass(mass_concentrations, molar_masses):
    """Taichi wrapper for mole fraction from mass concentration."""
    if not (isinstance(mass_concentrations, np.ndarray) and isinstance(molar_masses, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    mass_concentrations_array = np.atleast_1d(mass_concentrations)
    molar_masses_array = np.atleast_1d(molar_masses)
    n_elements = mass_concentrations_array.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    molar_masses_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    mass_concentrations_ti.from_numpy(mass_concentrations_array)
    molar_masses_ti.from_numpy(molar_masses_array)
    kget_mole_fraction_from_mass(mass_concentrations_ti, molar_masses_ti, result_fraction_ti)
    result_fraction_np = result_fraction_ti.to_numpy()
    return result_fraction_np.item() if result_fraction_np.size == 1 else result_fraction_np

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
    n_concentrations = result.shape[0]
    total_volume = 0.0
    for i in range(n_concentrations):
        total_volume += mass_concentrations[i] / densities[i]
    for i in range(n_concentrations):
        result[i] = fget_volume_fraction_from_mass(
            mass_concentrations[i], densities[i], total_volume
        )

@register("get_volume_fraction_from_mass", backend="taichi")
def ti_get_volume_fraction_from_mass(mass_concentrations, densities):
    """Taichi wrapper for volume fraction from mass concentration."""
    if not (isinstance(mass_concentrations, np.ndarray) and isinstance(densities, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    mass_concentrations_array = np.atleast_1d(mass_concentrations)
    densities_array = np.atleast_1d(densities)
    n_elements = mass_concentrations_array.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    densities_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    mass_concentrations_ti.from_numpy(mass_concentrations_array)
    densities_ti.from_numpy(densities_array)
    kget_volume_fraction_from_mass(mass_concentrations_ti, densities_ti, result_fraction_ti)
    result_fraction_np = result_fraction_ti.to_numpy()
    return result_fraction_np.item() if result_fraction_np.size == 1 else result_fraction_np

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
    n_concentrations = result.shape[0]
    total_mass = 0.0
    for i in range(n_concentrations):
        total_mass += mass_concentrations[i]
    for i in range(n_concentrations):
        result[i] = fget_mass_fraction_from_mass(
            mass_concentrations[i], total_mass
        )

@register("get_mass_fraction_from_mass", backend="taichi")
def ti_get_mass_fraction_from_mass(mass_concentrations):
    """Taichi wrapper for mass fraction from mass concentration."""
    if not isinstance(mass_concentrations, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy array for input.")
    mass_concentrations_array = np.atleast_1d(mass_concentrations)
    n_elements = mass_concentrations_array.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    mass_concentrations_ti.from_numpy(mass_concentrations_array)
    kget_mass_fraction_from_mass(mass_concentrations_ti, result_fraction_ti)
    result_fraction_np = result_fraction_ti.to_numpy()
    return result_fraction_np.item() if result_fraction_np.size == 1 else result_fraction_np
