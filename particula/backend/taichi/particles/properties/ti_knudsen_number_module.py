"""
Taichi-accelerated implementation of ``get_knudsen_number``.
"""

import taichi as ti
import numpy as np
from numbers import Number

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
    # --- type guard ---------------------------------------------------------
    if not (
        isinstance(mean_free_path, (np.ndarray, Number))
        and isinstance(particle_radius, (np.ndarray, Number))
    ):
        raise TypeError(
            "Taichi backend expects NumPy arrays or scalars for both inputs."
        )

    # --- broadcast ----------------------------------------------------------
    mfp_np = np.asarray(mean_free_path, dtype=np.float64)
    pr_np  = np.asarray(particle_radius, dtype=np.float64)
    mfp_b, pr_b = np.broadcast_arrays(mfp_np, pr_np)

    flat_mfp = mfp_b.ravel()
    flat_pr  = pr_b.ravel()
    n = flat_mfp.size

    # --- Taichi buffers -----------------------------------------------------
    mfp_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pr_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mfp_ti.from_numpy(flat_mfp)
    pr_ti.from_numpy(flat_pr)

    # --- kernel -------------------------------------------------------------
    kget_knudsen_number(mfp_ti, pr_ti, res_ti)

    # --- reshape / unwrap ---------------------------------------------------
    result_np = res_ti.to_numpy().reshape(mfp_b.shape)
    return result_np.item() if result_np.size == 1 else result_np
