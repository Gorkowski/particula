"""
Taichi-accelerated conversion of mass concentration to mole, volume, and
mass fractions.

This module provides Taichi-accelerated functions and kernels to convert
mass concentrations to mole fraction, volume fraction, and mass fraction.
All functions are vectorized and compatible with NumPy arrays.

Examples:
    ```py
    import numpy as np
    from particula.backend.taichi.particles.properties import (
        ti_convert_mass_concentration as tcmc
    )

    mass_conc = np.array([1.0, 2.0])
    molar_mass = np.array([18.0, 44.0])
    mole_frac = tcmc.ti_get_mole_fraction_from_mass(mass_conc, molar_mass)
    # Output: array([0.69230769, 0.30769231])
    ```
"""

import taichi as ti
import numpy as np
from typing import Union
from particula.backend.dispatch_register import register

@ti.func
def fget_mole_fraction_from_mass(
    mass_concentration: ti.f64,
    molar_mass: ti.f64,
    total_moles: ti.f64,
) -> ti.f64:
    """
    Compute mole fraction from mass concentration (elementwise).

    χ = (m / M) ÷ Σ(mᵢ / Mᵢ)

    Arguments:
        - mass_concentration : Mass concentration of the component.
        - molar_mass : Molar mass of the component.
        - total_moles : Total moles in the mixture.

    Returns:
        - Mole fraction as a float.
    """
    if total_moles == 0.0:
        return 0.0
    return (mass_concentration / molar_mass) / total_moles

@ti.kernel
def kget_mole_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_masses: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Taichi kernel for mole fraction from mass concentration.

    Arguments:
        - mass_concentrations : Array of mass concentrations.
        - molar_masses : Array of molar masses.
        - out : Output array for mole fractions.

    Returns:
        - None (results written to out).
    """
    n_concentrations = out.shape[0]
    total_moles = 0.0
    for i in range(n_concentrations):
        total_moles += mass_concentrations[i] / molar_masses[i]
    for i in range(n_concentrations):
        out[i] = fget_mole_fraction_from_mass(
            mass_concentrations[i], molar_masses[i], total_moles
        )

@register("get_mole_fraction_from_mass", backend="taichi")
def ti_get_mole_fraction_from_mass(
    mass_concentrations: np.ndarray,
    molar_masses: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Taichi wrapper for mole fraction from mass concentration.

    χ = (m / M) ÷ Σ(mᵢ / Mᵢ)

    Arguments:
        - mass_concentrations : NumPy array of mass concentrations.
        - molar_masses : NumPy array of molar masses.

    Returns:
        - Mole fraction(s) as float or NumPy array.

    Examples:
        ```py
        import numpy as np
        ti_get_mole_fraction_from_mass(
            np.array([1.0, 2.0]), np.array([18.0, 44.0])
        )
        # Output: array([0.69230769, 0.30769231])
        ```
    """
    if not (
        isinstance(mass_concentrations, np.ndarray)
        and isinstance(molar_masses, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    mass_concentrations_array = np.atleast_1d(mass_concentrations)
    molar_masses_array = np.atleast_1d(molar_masses)
    n_elements = mass_concentrations_array.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    molar_masses_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    mass_concentrations_ti.from_numpy(mass_concentrations_array)
    molar_masses_ti.from_numpy(molar_masses_array)
    kget_mole_fraction_from_mass(
        mass_concentrations_ti, molar_masses_ti, result_fraction_ti
    )
    fraction_array = result_fraction_ti.to_numpy()
    return fraction_array.item() if fraction_array.size == 1 else fraction_array

@ti.func
def fget_volume_fraction_from_mass(
    mass_concentration: ti.f64,
    density: ti.f64,
    total_volume: ti.f64,
) -> ti.f64:
    """
    Compute volume fraction from mass concentration (elementwise).

    φ = (m / ρ) ÷ Σ(mᵢ / ρᵢ)

    Arguments:
        - mass_concentration : Mass concentration of the component.
        - density : Density of the component.
        - total_volume : Total volume in the mixture.

    Returns:
        - Volume fraction as a float.
    """
    if total_volume == 0.0:
        return 0.0
    return (mass_concentration / density) / total_volume

@ti.kernel
def kget_volume_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    densities: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Taichi kernel for volume fraction from mass concentration.

    Arguments:
        - mass_concentrations : Array of mass concentrations.
        - densities : Array of densities.
        - out : Output array for volume fractions.

    Returns:
        - None (results written to out).
    """
    n_concentrations = out.shape[0]
    total_volume = 0.0
    for i in range(n_concentrations):
        total_volume += mass_concentrations[i] / densities[i]
    for i in range(n_concentrations):
        out[i] = fget_volume_fraction_from_mass(
            mass_concentrations[i], densities[i], total_volume
        )

@register("get_volume_fraction_from_mass", backend="taichi")
def ti_get_volume_fraction_from_mass(
    mass_concentrations: np.ndarray,
    densities: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Taichi wrapper for volume fraction from mass concentration.

    φ = (m / ρ) ÷ Σ(mᵢ / ρᵢ)

    Arguments:
        - mass_concentrations : NumPy array of mass concentrations.
        - densities : NumPy array of densities.

    Returns:
        - Volume fraction(s) as float or NumPy array.

    Examples:
        ```py
        import numpy as np
        ti_get_volume_fraction_from_mass(
            np.array([1.0, 2.0]), np.array([1.0, 2.0])
        )
        # Output: array([0.33333333, 0.66666667])
        ```
    """
    if not (
        isinstance(mass_concentrations, np.ndarray)
        and isinstance(densities, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    mass_concentrations_array = np.atleast_1d(mass_concentrations)
    densities_array = np.atleast_1d(densities)
    n_elements = mass_concentrations_array.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    densities_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    mass_concentrations_ti.from_numpy(mass_concentrations_array)
    densities_ti.from_numpy(densities_array)
    kget_volume_fraction_from_mass(
        mass_concentrations_ti, densities_ti, result_fraction_ti
    )
    fraction_array = result_fraction_ti.to_numpy()
    return fraction_array.item() if fraction_array.size == 1 else fraction_array

@ti.func
def fget_mass_fraction_from_mass(
    mass_concentration: ti.f64,
    total_mass: ti.f64,
) -> ti.f64:
    """
    Compute mass fraction from mass concentration (elementwise).

    ω = m ÷ Σmᵢ

    Arguments:
        - mass_concentration : Mass concentration of the component.
        - total_mass : Total mass in the mixture.

    Returns:
        - Mass fraction as a float.
    """
    if total_mass == 0.0:
        return 0.0
    return mass_concentration / total_mass

@ti.kernel
def kget_mass_fraction_from_mass(
    mass_concentrations: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorized Taichi kernel for mass fraction from mass concentration.

    Arguments:
        - mass_concentrations : Array of mass concentrations.
        - out : Output array for mass fractions.

    Returns:
        - None (results written to out).
    """
    n_concentrations = out.shape[0]
    total_mass = 0.0
    for i in range(n_concentrations):
        total_mass += mass_concentrations[i]
    for i in range(n_concentrations):
        out[i] = fget_mass_fraction_from_mass(
            mass_concentrations[i], total_mass
        )

@register("get_mass_fraction_from_mass", backend="taichi")
def ti_get_mass_fraction_from_mass(
    mass_concentrations: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Taichi wrapper for mass fraction from mass concentration.

    ω = m ÷ Σmᵢ

    Arguments:
        - mass_concentrations : NumPy array of mass concentrations.

    Returns:
        - Mass fraction(s) as float or NumPy array.

    Examples:
        ```py
        import numpy as np
        ti_get_mass_fraction_from_mass(np.array([1.0, 2.0]))
        # Output: array([0.33333333, 0.66666667])
        ```
    """
    if not isinstance(mass_concentrations, np.ndarray):
        raise TypeError("Taichi backend expects a NumPy array for input.")
    mass_concentrations_array = np.atleast_1d(mass_concentrations)
    n_elements = mass_concentrations_array.size
    mass_concentrations_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    mass_concentrations_ti.from_numpy(mass_concentrations_array)
    kget_mass_fraction_from_mass(
        mass_concentrations_ti, result_fraction_ti
    )
    fraction_array = result_fraction_ti.to_numpy()
    return fraction_array.item() if fraction_array.size == 1 else fraction_array
