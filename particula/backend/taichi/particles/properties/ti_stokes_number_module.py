"""Taichi implementation of Stokes number calculation."""

import taichi as ti
import numpy as np
from particula.backend import register


@ti.func
def fget_stokes_number(
    particle_inertia_time: ti.f64, kolmogorov_time: ti.f64
) -> ti.f64:
    """Elementwise Stokes number calculation."""
    return particle_inertia_time / kolmogorov_time


@ti.kernel
def kget_stokes_number(
    particle_inertia_time: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kolmogorov_time: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Stokes number calculation."""
    for i in range(result.shape[0]):
        result[i] = fget_stokes_number(
            particle_inertia_time[i], kolmogorov_time[i]
        )


@register("get_stokes_number", backend="taichi")
def ti_get_stokes_number(particle_inertia_time, kolmogorov_time):
    """Public Taichi wrapper for Stokes number calculation."""
    if not (
        isinstance(particle_inertia_time, np.ndarray)
        and isinstance(kolmogorov_time, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    pit, kt = np.atleast_1d(particle_inertia_time), np.atleast_1d(
        kolmogorov_time
    )
    n = pit.size

    pit_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pit_ti.from_numpy(pit)
    kt_ti.from_numpy(kt)

    kget_stokes_number(pit_ti, kt_ti, res_ti)
    return res_ti.to_numpy()
