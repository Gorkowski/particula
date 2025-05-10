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
    if not isinstance(mean_free_path, (int, float, np.ndarray)) \
       or not isinstance(particle_radius, (int, float, np.ndarray)):
        raise TypeError(
            "Taichi backend expects scalar (int/float) or NumPy array inputs.")

    # 5 b – convert to 1-D NumPy arrays and broadcast if a scalar/size-1 vector is given
    mfp_np = np.atleast_1d(np.asarray(mean_free_path,  dtype=np.float64)).ravel()
    pr_np  = np.atleast_1d(np.asarray(particle_radius, dtype=np.float64)).ravel()

    if mfp_np.size == pr_np.size:               # same length -> OK
        n = mfp_np.size
    elif mfp_np.size == 1:                      # scalar mean-free-path
        mfp_np = np.full(pr_np.size, mfp_np.item(), dtype=np.float64)
        n = pr_np.size
    elif pr_np.size == 1:                       # scalar radius
        pr_np = np.full(mfp_np.size, pr_np.item(), dtype=np.float64)
        n = mfp_np.size
    else:
        raise ValueError(
            "Inputs must have identical size or one of them must be scalar.")

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
