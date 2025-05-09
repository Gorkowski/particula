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
    """
    Taichi-accelerated version of ``get_knudsen_number``.

    Parameters
    ----------
    mean_free_path : np.ndarray
        Mean free path(s) in metres.
    particle_radius : np.ndarray
        Particle radius/radii in metres.

    Returns
    -------
    np.ndarray
        Knudsen number(s) as a NumPy array.

    Notes
    -----
    The function normalises the inputs to 1-D NumPy arrays, allocates
    Taichi NDArray buffers, launches :pyfunc:`kget_knudsen_number`,
    and converts the result back to NumPy for the caller.
    """
    if not (isinstance(mean_free_path, np.ndarray) and isinstance(particle_radius, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    # --- convert to 1-D numpy arrays -----------------------------------
    mfp_np = np.atleast_1d(mean_free_path).astype(np.float64, copy=False)
    pr_np = np.atleast_1d(particle_radius).astype(np.float64, copy=False)

    # manual broadcasting (match reference behaviour)
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

    return result_np
