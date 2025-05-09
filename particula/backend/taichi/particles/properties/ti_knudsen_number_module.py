"""
Taichi-accelerated implementation of ``get_knudsen_number``.

This module is imported only when the Taichi backend is requested.
It provides:

* _knudsen_element  – a reusable ``@ti.func`` that evaluates λ / r
* kget_knudsen_number – a vectorised Taichi kernel operating on 1-D
  NDArray buffers
* get_knudsen_number_taichi – a Python wrapper that conforms to the public
  API of :pyfunc:`particula.particles.properties.knudsen_number_module.get_knudsen_number`

All computations are performed in double precision (ti.f64).
"""

import taichi as ti

import numpy as np


@ti.func
def _knudsen_element(lambda_mfp: ti.f64, r: ti.f64) -> ti.f64:
    """
    Element-wise Knudsen number.

    Parameters
    ----------
    lambda_mfp : ti.f64
        Mean free path of the gas molecules (m).
    r : ti.f64
        Particle radius (m).

    Returns
    -------
    ti.f64
        Knudsen number (dimensionless), computed as λ / r.
    """
    return lambda_mfp / r

@ti.kernel
def kget_knudsen_number(
    mean_free_path: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
) -> None:
    """
    Vectorised Taichi kernel computing the Knudsen number.

    All three arguments must be 1-D Taichi NDArrays of identical length.
    The result is written in-place to ``result``.

    Parameters
    ----------
    mean_free_path : ti.types.ndarray
        Mean free path values (m).
    particle_radius : ti.types.ndarray
        Particle radius values (m).
    result : ti.types.ndarray
        Output buffer that will receive the Knudsen numbers.
    """
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
