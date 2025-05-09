"""
Taichi-accelerated implementation of ``get_knudsen_number``

"""

import taichi as ti

import numpy as np


@ti.func
def _knudsen_element(lambda_mfp, r):
    """Return Kn = λ / r for one scalar pair."""
    return lambda_mfp / r

@ti.kernel
def kget_knudsen_number(
    mean_free_path: ti.types.ndarray(),
    particle_radius: ti.types.ndarray(),
    result: ti.types.ndarray(),
):
    for i in range(result.shape[0]):
        result[i] = _knudsen_element(mean_free_path[i], particle_radius[i])


def get_knudsen_number_taichi(mean_free_path, particle_radius):
    # --- convert to 1-D numpy arrays -----------------------------------
    mfp_np = np.atleast_1d(mean_free_path)
    pr_np = np.atleast_1d(particle_radius)

    n = mfp_np.size
    # --- allocate Taichi ndarrays --------------------------------------
    mfp_nd = ti.ndarray(dtype=ti.f64, shape=n)
    pr_nd = ti.ndarray(dtype=ti.f64, shape=n)
    res_nd = ti.ndarray(dtype=ti.f64, shape=n)

    mfp_nd.from_numpy(mfp_np)
    pr_nd.from_numpy(pr_np)

    # --- launch kernel -------------------------------------------------
    kget_knudsen_number(mfp_np, pr_np, res_nd)

    return res_nd.to_numpy()