# required imports
import taichi as ti
import numpy as np
from particula.backend import register

# ─── 3. element-wise Taichi function ────────────────────────────────────────
@ti.func
def fget_vapor_transition_correction(kn: ti.f64, alpha: ti.f64) -> ti.f64:
    return (0.75 * alpha * (1.0 + kn)) / (
        kn * kn + kn + 0.283 * alpha * kn + 0.75 * alpha
    )

# ─── 4. vectorised kernel ───────────────────────────────────────────────────
@ti.kernel
def kget_vapor_transition_correction(
    kn: ti.types.ndarray(dtype=ti.f64, ndim=1),
    alpha: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_vapor_transition_correction(kn[i], alpha[i])

# ─── 5. public wrapper (backend registration) ───────────────────────────────
@register("get_vapor_transition_correction", backend="taichi")
def get_vapor_transition_correction_taichi(knudsen_number, mass_accommodation):
    # 5 a – type guard
    if not (
        isinstance(knudsen_number, np.ndarray)
        and isinstance(mass_accommodation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    # 5 b – ensure 1-D NumPy arrays
    kn, alpha = np.atleast_1d(knudsen_number), np.atleast_1d(mass_accommodation)
    n = kn.size

    # 5 c – allocate Taichi buffers
    kn_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    al_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kn_ti.from_numpy(kn)
    al_ti.from_numpy(alpha)

    # 5 d – launch kernel
    kget_vapor_transition_correction(kn_ti, al_ti, res_ti)
    return res_ti.to_numpy()
