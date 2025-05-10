"""Taichi-accelerated Cunningham slip-correction factor."""
import taichi as ti
import numpy as np
from particula.backend import register


# ── 3. element-wise Taichi func ──────────────────────────────────────────────
@ti.func
def fget_cunningham_slip_correction(kn: ti.f64) -> ti.f64:
    """Scalar Cunningham slip-correction."""
    return 1.0 if kn == 0.0 else (
        1.0 + kn * (1.257 + 0.4 * ti.exp(-1.1 / kn))
    )


# ── 4. vectorised kernel ────────────────────────────────────────────────────
@ti.kernel
def kget_cunningham_slip_correction(                       # noqa: N802
    kn: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_cunningham_slip_correction(kn[i])


# ── 5. public wrapper with backend registration ─────────────────────────────
@register("get_cunningham_slip_correction", backend="taichi")  # noqa: D401
def ti_get_cunningham_slip_correction(knudsen_number):          # noqa: N802
    """Taichi wrapper equivalent to NumPy version."""
    kn_np = np.atleast_1d(knudsen_number).astype(np.float64)
    n = kn_np.size

    kn_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kn_ti.from_numpy(kn_np)

    kget_cunningham_slip_correction(kn_ti, res_ti)
    out = res_ti.to_numpy()
    return out[0] if np.isscalar(knudsen_number) else out.reshape(knudsen_number.shape)
