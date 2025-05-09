"""
Taichi-accelerated implementation of ``get_knudsen_number``.
"""

import taichi as ti
import numpy as np

from particula.backend import register


@ti.func
def fget_knudsen_number(
    mean_free_path: ti.f64, particle_radius: ti.f64
) -> ti.f64:
    """
    Element-wise Knudsen number.
    """
    return mean_free_path / particle_radius


@ti.kernel
def kget_knudsen_number(
    mean_free_path: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Vectorised Taichi kernel computing the Knudsen number.

    All three arguments must be 1-D Taichi NDArrays of identical length.
    The result is written in-place to ``result``.
    """
    for i in range(result.shape[0]):
        result[i] = fget_knudsen_number(mean_free_path[i], particle_radius[i])


@register("get_knudsen_number", backend="taichi")
def get_knudsen_number_taichi(mean_free_path, particle_radius):
    """
    Taichi-accelerated version of ``par.particles.get_knudsen_number``.

    Notes
    -----
    The function normalises the inputs to 1-D NumPy arrays, allocates
    Taichi NDArray buffers, launches :pyfunc:`kget_knudsen_number`,
    and converts the result back to NumPy for the caller.
    """
    if not (
        isinstance(mean_free_path, np.ndarray)
        and isinstance(particle_radius, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    # convert to 1-D numpy arrays if not done
    mean_free_path_np = np.atleast_1d(mean_free_path)
    particle_radius_np = np.atleast_1d(particle_radius)
    # allocate Taichi ndarrays
    n = mean_free_path_np.size
    mean_free_path_ti = ti.ndarray(dtype=ti.f64, shape=n)
    particle_radius_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mean_free_path_ti.from_numpy(mean_free_path_np)
    particle_radius_ti.from_numpy(particle_radius_np)

    # launch kernel
    kget_knudsen_number(mean_free_path_ti, particle_radius_ti, result_ti)
    return result_ti.to_numpy()
