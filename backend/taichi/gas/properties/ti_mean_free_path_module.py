"""Taichi implementation of mean free path calculation."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

GAS_CONSTANT = 8.314462618  # J/(mol·K), consistent with particula.util.constants

@ti.func
def fget_molecule_mean_free_path(
    molar_mass: ti.f64, temperature: ti.f64, pressure: ti.f64, dynamic_viscosity: ti.f64
) -> ti.f64:
    """Elementwise mean free path calculation (Taichi)."""
    return (2.0 * dynamic_viscosity / pressure) / ti.sqrt(
        8.0 * molar_mass / (ti.math.pi * GAS_CONSTANT * temperature)
    )

@ti.kernel
def kget_molecule_mean_free_path(
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    pressure: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized mean free path calculation (Taichi)."""
    for i in range(result.shape[0]):
        result[i] = fget_molecule_mean_free_path(
            molar_mass[i], temperature[i], pressure[i], dynamic_viscosity[i]
        )

@register("get_molecule_mean_free_path", backend="taichi")
def get_molecule_mean_free_path_taichi(
    molar_mass, temperature, pressure, dynamic_viscosity=None
):
    """Public Taichi wrapper for mean free path calculation."""
    if not (
        isinstance(molar_mass, np.ndarray)
        and isinstance(temperature, np.ndarray)
        and isinstance(pressure, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    a1 = np.atleast_1d(molar_mass)
    a2 = np.atleast_1d(temperature)
    a3 = np.atleast_1d(pressure)
    n = a1.size

    if dynamic_viscosity is None:
        # Import here to avoid circular import
        from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity

        dynamic_viscosity = get_dynamic_viscosity(a2)
    a4 = np.atleast_1d(dynamic_viscosity)

    mm_ti = ti.ndarray(dtype=ti.f64, shape=n)
    T_ti = ti.ndarray(dtype=ti.f64, shape=n)
    P_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mu_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)

    mm_ti.from_numpy(a1)
    T_ti.from_numpy(a2)
    P_ti.from_numpy(a3)
    mu_ti.from_numpy(a4)

    kget_molecule_mean_free_path(mm_ti, T_ti, P_ti, mu_ti, result_ti)

    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
