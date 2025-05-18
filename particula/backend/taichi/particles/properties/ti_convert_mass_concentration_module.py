"""
Taichi backend for mass-to-fraction conversions.

This module provides Taichi-accelerated kernels and helpers for converting
between mass, mole, and volume fractions for particle and gas mixtures.
It registers these conversion routines for the "taichi" backend.

Equations (Unicode format):
    Mole fraction (xᵢ):   xᵢ = (mᵢ / Mᵢ) / Σⱼ (mⱼ / Mⱼ)
    Volume fraction (ϕᵢ): ϕᵢ = (mᵢ / ρᵢ) / Σⱼ (mⱼ / ρⱼ)
    Mass fraction (ωᵢ):   ωᵢ = mᵢ / Σⱼ mⱼ

Examples:
    >>> get_mole_fraction_from_mass([1.0, 2.0], [10.0, 20.0])
    array([0.5, 0.5])
    >>> get_volume_fraction_from_mass([1.0, 2.0], [1.0, 2.0])
    array([0.5, 0.5])

References:
    - "Mole fraction," Wikipedia. https://en.wikipedia.org/wiki/Mole_fraction
    - "Volume fraction," Wikipedia. https://en.wikipedia.org/wiki/Volume_fraction
    - "Mass fraction (chemistry)," Wikipedia. https://en.wikipedia.org/wiki/Mass_fraction_(chemistry)
"""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ───── element-wise helpers ─────────────────────────────────
@ti.func
def fget_mole_fraction_single(
    mass_concentration: ti.f64, molar_mass: ti.f64, inverse_total_moles: ti.f64
) -> ti.f64:
    """
    Compute the mole fraction for a single component.

    Arguments:
        - mass_concentration : Mass of the component [kg or g].
        - molar_mass : Molar mass of the component [kg/mol or g/mol].
        - inverse_total_moles : 1 / (sum of all mⱼ / Mⱼ).

    Returns:
        - Mole fraction (xᵢ) of the component.

    Equation:
        xᵢ = (mᵢ / Mᵢ) × (1 / Σⱼ (mⱼ / Mⱼ))
    """
    return (mass_concentration / molar_mass) * inverse_total_moles


@ti.func
def fget_volume_fraction_single(
    mass_concentration: ti.f64, density: ti.f64, inverse_total_volume: ti.f64
) -> ti.f64:
    """
    Compute the volume fraction for a single component.

    Arguments:
        - mass_concentration : Mass of the component [kg or g].
        - density : Density of the component [kg/m³ or g/cm³].
        - inverse_total_volume : 1 / (sum of all mⱼ / ρⱼ).

    Returns:
        - Volume fraction (ϕᵢ) of the component.

    Equation:
        ϕᵢ = (mᵢ / ρᵢ) × (1 / Σⱼ (mⱼ / ρⱼ))
    """
    return (mass_concentration / density) * inverse_total_volume


# ───── kernels ─────────────────────────────────────────────
@ti.kernel
def kget_mole_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_masses: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mole_fractions: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Compute mole fractions (xᵢ) from mass concentrations and molar masses.

    Args:
        - mass_concentrations : 1D array of mass values.
        - molar_masses : 1D array of molar masses.
        - mole_fractions : 1D output array for mole fractions.

    All arrays must have identical shape.
    """
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
    """
    Compute volume fractions (ϕᵢ) from mass concentrations and densities.

    Args:
        - mass_concentrations : 1D array of mass values.
        - densities : 1D array of densities.
        - volume_fractions : 1D output array for volume fractions.

    All arrays must have identical shape.

    Notes:
        ϕᵢ = (mᵢ / ρᵢ) / Σⱼ (mⱼ / ρⱼ)
    """
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
    """
    Convert two arrays to 1D float64 numpy arrays and check shape match.

    Args:
        - array_a : First input array.
        - array_b : Second input array.
        - name_a : Name of the first array (for error messages).
        - name_b : Name of the second array (for error messages).

    Returns:
        - Tuple of (array_a_1d, array_b_1d), both as 1D float64 arrays.

    Raises:
        - ValueError: If the input arrays do not have identical shape.
    """
    array_a = np.asarray(array_a, dtype=np.float64)
    array_b = np.asarray(array_b, dtype=np.float64)
    if array_a.shape != array_b.shape:
        raise ValueError(f"{name_a} and {name_b} must have identical shape.")
    return np.atleast_1d(array_a), np.atleast_1d(array_b)


@register("get_mole_fraction_from_mass", backend="taichi")
def taichi_get_mole_fraction_from_mass(mass_concentrations, molar_masses):
    """
    Taichi backend: compute mole fractions from mass and molar mass arrays.

    Arguments:
        - mass_concentrations : Array-like of mass values.
        - molar_masses : Array-like of molar masses.

    Returns:
        - Numpy array of mole fractions (or scalar if input is scalar).

    Examples:
        >>> taichi_get_mole_fraction_from_mass([1.0, 2.0], [10.0, 20.0])
        array([0.5, 0.5])
    """
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
    """
    Taichi backend: compute volume fractions from mass and density arrays.

    Arguments:
        - mass_concentrations : Array-like of mass values.
        - densities : Array-like of densities.

    Returns:
        - Numpy array of volume fractions (or scalar if input is scalar).

    Examples:
        >>> taichi_get_volume_fraction_from_mass([1.0, 2.0], [1.0, 2.0])
        array([0.5, 0.5])
    """
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
    """
    Compute mass fractions (ωᵢ) from mass concentrations.

    Args:
        - mass_concentrations : 1D array of mass values.
        - mass_fractions : 1D output array for mass fractions.

    All arrays must have identical shape.

    Equation:
        ωᵢ = mᵢ / Σⱼ mⱼ
    """
    total_mass = 0.0
    for i in range(mass_concentrations.shape[0]):
        total_mass += mass_concentrations[i]
    inverse_total_mass = 1.0 / total_mass if total_mass != 0.0 else 0.0
    for i in range(mass_fractions.shape[0]):
        mass_fractions[i] = mass_concentrations[i] * inverse_total_mass

@register("get_mass_fraction_from_mass", backend="taichi")
def taichi_get_mass_fraction_from_mass(mass_concentrations):
    """
    Taichi backend: compute mass fractions from mass concentration array.

    Arguments:
        - mass_concentrations : Array-like of mass values.

    Returns:
        - Numpy array of mass fractions (or scalar if input is scalar).

    Examples:
        >>> taichi_get_mass_fraction_from_mass([1.0, 1.0])
        array([0.5, 0.5])
    """
    mass_concentrations_np = np.atleast_1d(np.asarray(mass_concentrations, dtype=np.float64))
    n_items = mass_concentrations_np.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    mass_fractions_ti = ti.ndarray(dtype=ti.f64, shape=n_items)
    mass_concentrations_ti.from_numpy(mass_concentrations_np)
    kget_mass_fraction_from_mass(mass_concentrations_ti, mass_fractions_ti)
    mass_fractions_np = mass_fractions_ti.to_numpy()
    return mass_fractions_np.item() if mass_fractions_np.size == 1 else mass_fractions_np
