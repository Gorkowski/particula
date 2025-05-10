"""
Taichi-accelerated implementation of ``get_knudsen_number``.
"""

import taichi as ti
import numpy as np

from particula.backend.dispatch_register import register


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
def ti_get_knudsen_number(mean_free_path, particle_radius):
    """
    Taichi-accelerated wrapper for Knudsen number calculation.
    """
    # 5 a – type guard
    if not (isinstance(mean_free_path, np.ndarray) and isinstance(particle_radius, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    # 5 b – ensure 1-D NumPy arrays
    mfp_np = np.atleast_1d(mean_free_path).astype(np.float64)
    pr_np  = np.atleast_1d(particle_radius).astype(np.float64)
    if mfp_np.size != pr_np.size:
        raise ValueError("Input arrays must have identical length.")
    n = mfp_np.size

    # 5 c – allocate Taichi NDArray buffers
    mfp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pr_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mfp_ti.from_numpy(mfp_np)
    pr_ti.from_numpy(pr_np)

    # 5 d – launch the kernel
    kget_knudsen_number(mfp_ti, pr_ti, res_ti)

    # 5 e – convert back & unwrap if scalar
    result_np = res_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
