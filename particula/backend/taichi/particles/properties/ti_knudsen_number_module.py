"""
Taichi-accelerated implementation of ``get_knudsen_number``

Imported (and therefore compiled) only when the user activates

    >>> par.use_backend("taichi")

If Taichi is not available, the import silently does nothing; the
dispatch layer will then fall back to the pure-Python version.
"""

import taichi as ti

import numpy as np


# ---------------------------------------------------------------------
# 1. Device function ─ operates on **scalars only**
# ---------------------------------------------------------------------

ti.init(arch=ti.cpu, default_fp=ti.f64)  # sets scalar precision


@ti.func
def fget_knudsen_number(
    mean_free_path,
    particle_radius,
    result,
):
    """Loop inside the Taichi *function*."""
    n = mean_free_path.shape[0]
    for i in range(n):  # device‑side loop
        result[i] = mean_free_path[i] / particle_radius[i]
    return result


@ti.kernel
def kget_knudsen_number(
    mean_free_path: ti.types.ndarray(),
    particle_radius: ti.types.ndarray(),
    result: ti.types.ndarray(),
):
    # The kernel just delegates all work to the function.
    fget_knudsen_number(mean_free_path, particle_radius, result)
    return result


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
    """
    Vectorised Taichi implementation matching the reference semantics.
    Accepts scalars or 1-D numpy arrays and broadcasts either argument
    when it has length 1.
    """
    # normalise to 1-D float64 numpy arrays
    mfp_np = np.atleast_1d(mean_free_path)
    pr_np = np.atleast_1d(particle_radius)
    res_np = np.zeros_like(mfp_np)

    res_np = kget_knudsen_number(mfp_np, pr_np, res_np)

    return res_np
