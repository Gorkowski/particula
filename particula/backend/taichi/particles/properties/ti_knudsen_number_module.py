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
    mean_free_path: ti.types.ndarray(dtype=ti.f64, ndim=1),
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
) -> None:
    n = mean_free_path.shape[0]
    for i in range(n):
        result[i] = knudsen_number(mean_free_path[i], particle_radius[i])


@register("get_knudsen_number", backend="taichi")
def get_knudsen_number_taichi(mean_free_path, particle_radius):
    """
    Vectorised Taichi implementation matching the reference semantics.
    Accepts scalars or 1-D numpy arrays and broadcasts either argument
    when it has length 1.
    """
    # normalise to 1-D float64 numpy arrays
    mfp_np = np.atleast_1d(mean_free_path).astype(np.float64, copy=False)
    pr_np = np.atleast_1d(particle_radius).astype(np.float64, copy=False)

    # manual broadcasting rules
    if mfp_np.size == 1 and pr_np.size > 1:
        mfp_np = np.full_like(pr_np, mfp_np.item())
    elif pr_np.size == 1 and mfp_np.size > 1:
        pr_np = np.full_like(mfp_np, pr_np.item())
    elif mfp_np.size != pr_np.size:
        raise ValueError(
            "Input arrays must have the same length or one of them may be length-1."
        )

    n = mfp_np.size
    mfp_nd = ti.ndarray(dtype=ti.f64, shape=n)
    pr_nd = ti.ndarray(dtype=ti.f64, shape=n)
    res_nd = ti.ndarray(dtype=ti.f64, shape=n)

    mfp_nd.from_numpy(mfp_np)
    pr_nd.from_numpy(pr_np)

    kget_knudsen_number(mfp_nd, pr_nd, res_nd)

    return res_nd.to_numpy()
