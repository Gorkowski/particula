
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


ti.init(arch=ti.cpu, default_fp=ti.f64)        # safe even if called twice


# --------------------------------------------------------------------- #
# 1-D kernel: element-wise Kn = λ / r                                    #
# --------------------------------------------------------------------- #
@ti.kernel
def _knudsen_kernel(
    mfp: ti.types.ndarray(dtype=ti.f64, ndim=1),
    pr: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
) -> None:
    for i in range(res.shape[0]):
        res[i] = mfp[i] / pr[i]


# --------------------------------------------------------------------- #
# Python wrapper registered for the “taichi” backend                    #
# --------------------------------------------------------------------- #
@register("get_knudsen_number", backend="taichi")
def get_knudsen_number_taichi(
    mean_free_path,
    particle_radius,
):
    """
    Taichi version of ``get_knudsen_number``.
    Handles scalars as well as 1-D numpy arrays (vectorised).  Broadcasting
    rules follow the reference implementation: either the two arrays must
    be the same length, or one of them may be length-1 (scalar broadcast).
    """
    # ---- Normalise to 1-D numpy arrays ---------------------------------
    mfp_np = np.atleast_1d(mean_free_path).astype(np.float64, copy=False)
    pr_np = np.atleast_1d(particle_radius).astype(np.float64, copy=False)

    if mfp_np.size == 1 and pr_np.size > 1:
        mfp_np = np.full_like(pr_np, mfp_np.item())
    elif pr_np.size == 1 and mfp_np.size > 1:
        pr_np = np.full_like(mfp_np, pr_np.item())
    elif mfp_np.size != pr_np.size:
        raise ValueError(
            "Input arrays must have the same length "
            "or one of them must be length-1."
        )

    # ---- Allocate Taichi NDArray buffers --------------------------------
    n = mfp_np.size
    mfp_nd = ti.ndarray(dtype=ti.f64, shape=n)
    pr_nd = ti.ndarray(dtype=ti.f64, shape=n)
    res_nd = ti.ndarray(dtype=ti.f64, shape=n)

    mfp_nd.from_numpy(mfp_np)
    pr_nd.from_numpy(pr_np)

    # ---- Launch kernel & fetch result -----------------------------------
    _knudsen_kernel(mfp_nd, pr_nd, res_nd)
    result_np = res_nd.to_numpy()

    # ---- Return scalar or array to match caller semantics ---------------
    if np.isscalar(mean_free_path) and np.isscalar(particle_radius):
        return float(result_np[0])
    return result_np
