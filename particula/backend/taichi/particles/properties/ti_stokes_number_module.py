"""Taichi implementation of Stokes number calculation."""

import taichi as ti
import numpy as np
from numbers import Number
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
    """
    Taichi backend for get_stokes_number.

    Accepts scalar or array-like inputs, broadcasts them to the same
    shape, launches the Taichi kernel, and returns a NumPy array with
    the broadcast shape (or a scalar if both inputs are scalars).
    """
    # 5 a – coerce to NumPy arrays (scalars become 0-d arrays)
    particle_inertia_time_np = (
        np.asarray(particle_inertia_time, dtype=np.float64)
        if not isinstance(particle_inertia_time, Number)
        else np.array(particle_inertia_time, dtype=np.float64)
    )
    kolmogorov_time_np = (
        np.asarray(kolmogorov_time, dtype=np.float64)
        if not isinstance(kolmogorov_time, Number)
        else np.array(kolmogorov_time, dtype=np.float64)
    )

    # 5 b – broadcast to common shape & flatten
    particle_inertia_time_b, kolmogorov_time_b = np.broadcast_arrays(
        particle_inertia_time_np, kolmogorov_time_np
    )
    flat_particle_inertia_time = particle_inertia_time_b.ravel()
    flat_kolmogorov_time = kolmogorov_time_b.ravel()
    n = flat_particle_inertia_time.size

    # 5 c – allocate Taichi buffers
    particle_inertia_time_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kolmogorov_time_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    particle_inertia_time_ti.from_numpy(flat_particle_inertia_time)
    kolmogorov_time_ti.from_numpy(flat_kolmogorov_time)

    # 5 d – launch kernel
    kget_stokes_number(particle_inertia_time_ti, kolmogorov_time_ti, result_ti)

    # 5 e – reshape back & return scalar if appropriate
    result_array = result_ti.to_numpy().reshape(particle_inertia_time_b.shape)
    return result_array.item() if result_array.size == 1 else result_array
