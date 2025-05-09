# required imports
import taichi as ti
import numpy as np
from numbers import Number          # new
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
    """
    Taichi backend for get_vapor_transition_correction.

    Accepts scalar or array-like inputs, broadcasts them to the same
    shape, calls the Taichi kernel and returns a NumPy array with the
    broadcast shape (or a scalar if both inputs were scalars).
    """
    # 5 a – coerce to NumPy arrays (scalars become 0-d arrays)
    kn_np = np.asarray(knudsen_number, dtype=np.float64) \
        if not isinstance(knudsen_number, Number) \
        else np.array(knudsen_number, dtype=np.float64)
    alpha_np = np.asarray(mass_accommodation, dtype=np.float64) \
        if not isinstance(mass_accommodation, Number) \
        else np.array(mass_accommodation, dtype=np.float64)

    # 5 b – broadcast to a common shape and flatten
    kn_b, alpha_b = np.broadcast_arrays(kn_np, alpha_np)
    flat_kn, flat_alpha = kn_b.ravel(), alpha_b.ravel()
    n = flat_kn.size

    # 5 c – allocate Taichi buffers
    kn_ti    = ti.ndarray(dtype=ti.f64, shape=n)
    alpha_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti   = ti.ndarray(dtype=ti.f64, shape=n)
    kn_ti.from_numpy(flat_kn)
    alpha_ti.from_numpy(flat_alpha)

    # 5 d – launch the kernel
    kget_vapor_transition_correction(kn_ti, alpha_ti, res_ti)

    # 5 e – reshape back and restore scalar return if needed
    result = res_ti.to_numpy().reshape(kn_b.shape)
    return result.item() if result.size == 1 else result
