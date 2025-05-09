"""
Taichi-accelerated implementation of ``get_knudsen_number``

Imported (and therefore compiled) only when the user activates

    >>> par.use_backend("taichi")

If Taichi is not available, the import silently does nothing; the
dispatch layer will then fall back to the pure-Python version.
"""

import taichi as ti

import numpy as np


# --------------------------------------------------------------------- #
#  Device function (scalar)                                             #
# --------------------------------------------------------------------- #
@ti.func
def _knudsen_element(lambda_mfp: ti.f64, r: ti.f64) -> ti.f64:
    """Return Kn = λ / r for one scalar pair."""
    return lambda_mfp / r


# --------------------------------------------------------------------- #
#  Kernel that works on 1-D arrays                                      #
# --------------------------------------------------------------------- #
@ti.kernel
def kget_knudsen_number(
    mean_free_path: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
) -> None:
    for i in range(result.shape[0]):
        result[i] = _knudsen_element(mean_free_path[i], particle_radius[i])


# # ---------- device function (scalar inputs, no annotations) ----------
# @ti.func
# def knudsen_number(mean_free_path, particle_radius):
#     """Return Kn = λ / r for a single value."""
#     return mean_free_path / particle_radius


# # ---------- kernel (handles ndarrays) ----------
# @ti.kernel
# def kget_knudsen_number(
#     mean_free_path: ti.types.ndarray(),
#     particle_radius: ti.types.ndarray(),
#     result: ti.types.ndarray(),
# ) -> None:
#     n = mean_free_path.shape[0]
#     for i in range(n):
#         result[i] = mean_free_path[i] / particle_radius[i]


def get_knudsen_number_taichi(mean_free_path, particle_radius):
    # --- convert to 1-D numpy arrays -----------------------------------
    mfp_np = np.atleast_1d(mean_free_path).astype(np.float64, copy=False)
    pr_np = np.atleast_1d(particle_radius).astype(np.float64, copy=False)

    # manual broadcasting rules matching reference implementation
    if mfp_np.size == 1 and pr_np.size > 1:
        mfp_np = np.full_like(pr_np, mfp_np.item())
    elif pr_np.size == 1 and mfp_np.size > 1:
        pr_np = np.full_like(mfp_np, pr_np.item())
    elif mfp_np.size != pr_np.size:
        raise ValueError(
            "Input arrays must have the same length or one of them may be length-1."
        )

    n = mfp_np.size
    # --- allocate Taichi ndarrays --------------------------------------
    mfp_nd = ti.ndarray(dtype=ti.f64, shape=n)
    pr_nd  = ti.ndarray(dtype=ti.f64, shape=n)
    res_nd = ti.ndarray(dtype=ti.f64, shape=n)

    mfp_nd.from_numpy(mfp_np)
    pr_nd.from_numpy(pr_np)

    # --- launch kernel -------------------------------------------------
    kget_knudsen_number(mfp_nd, pr_nd, res_nd)

    result_np = res_nd.to_numpy()
    # return scalar if both inputs were scalar
    if np.isscalar(mean_free_path) and np.isscalar(particle_radius):
        return float(result_np[0])
    return result_np
