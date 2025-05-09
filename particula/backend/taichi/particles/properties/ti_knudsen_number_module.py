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
def _knudsen_element(mean_free_path: ti.f64, particle_radius: ti.f64) -> ti.f64:
    """
    Element-wise Knudsen number.

    Parameters
    ----------
    mean_free_path : ti.f64
        Mean free path of the gas molecules (m).
    particle_radius : ti.f64
        Particle radius (m).

    Returns
    -------
    ti.f64
        Knudsen number (dimensionless), computed as mean_free_path / particle_radius.
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
        result[i] = _knudsen_element(mean_free_path[i], particle_radius[i])    # arguments unchanged; signature changed


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
    mean_free_path_np = np.atleast_1d(mean_free_path)
    particle_radius_np = np.atleast_1d(particle_radius)

    n = mean_free_path_np.size
    # --- allocate Taichi ndarrays --------------------------------------
    mean_free_path_ti = ti.ndarray(dtype=ti.f64, shape=n)
    particle_radius_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)

    mean_free_path_ti.from_numpy(mean_free_path_np)
    particle_radius_ti.from_numpy(particle_radius_np)

    # --- launch kernel -------------------------------------------------
    kget_knudsen_number(mean_free_path_ti, particle_radius_ti, result_ti)

    result_np = result_ti.to_numpy()

    return result_np
