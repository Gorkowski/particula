"""
Taichi-accelerated implementation of ``get_knudsen_number``

Imported (and therefore compiled) only when the user activates

    >>> par.use_backend("taichi")

If Taichi is not available, the import silently does nothing; the
dispatch layer will then fall back to the pure-Python version.
"""

from __future__ import annotations

# --------------------------------------------------------------------- #
# Guarded Taichi import – keeps Particula usable when Taichi is absent  #
# --------------------------------------------------------------------- #
try:
    import taichi as ti
except ModuleNotFoundError:      # Taichi not installed → skip registration
    ti = None                    # (dispatchable wrapper will fall back)
    raise SystemExit             # end of file – nothing else to do

import numpy as np

# Particula’s dispatch/registration helpers
from particula.backend import register


# ---------------------------------------------------------------------
# 1. Device function ─ operates on **scalars only**
# ---------------------------------------------------------------------

ti.init(arch=ti.cpu, default_fp=ti.f64)  # sets scalar precision


# ---------- device function (scalar inputs, no annotations) ----------
@ti.func
def knudsen_number(mean_free_path, particle_radius):
    """Return Kn = λ / r for a single value."""
    return mean_free_path / particle_radius


# ---------- kernel (handles ndarrays) ----------
@ti.kernel
def kget_knudsen_number(
    mean_free_path: ti.types.ndarray(),  # 1‑D float64
    particle_radius: ti.types.ndarray(),  # 1‑D float64
    result: ti.types.ndarray(),  # 1‑D float64 (pre‑allocated)
) -> None:
    n = mean_free_path.shape[0]  # number of elements
    for i in range(n):
        result[i] = knudsen_number(mean_free_path[i], particle_radius[i])


# --------------------------------------------------------------------- #
# Python wrapper registered for the “taichi” backend                    #
# --------------------------------------------------------------------- #
# @register("get_knudsen_number", backend="taichi")
# def get_knudsen_number_taichi(
#     mean_free_path,
#     particle_radius,
# ):
#     """
#     Taichi version of ``get_knudsen_number``.
#     Handles scalars as well as 1-D numpy arrays (vectorised).  Broadcasting
#     rules follow the reference implementation: either the two arrays must
#     be the same length, or one of them may be length-1 (scalar broadcast).
#     """
#     # ---- Normalise to 1-D numpy arrays ---------------------------------
#     mean_free_path = np.atleast_1d(mean_free_path)
#     particle_radius = np.atleast_1d(particle_radius)
#     result = np.zeros_like(mean_free_path)

#     kget_knudsen_number(
#         mean_free_path, particle_radius, result
#     )

#     return result
