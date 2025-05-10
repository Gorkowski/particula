"""Taichi-accelerated Fuchs–Sutugin vapor–transition correction."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# 3 – element-wise Taichi function
@ti.func
def fget_vapor_transition_correction(
    knudsen_number: ti.f64,
    mass_accommodation: ti.f64,
) -> ti.f64:
    return 0.75 * mass_accommodation * (1.0 + knudsen_number) / (
        knudsen_number * knudsen_number
        + knudsen_number
        + 0.283 * mass_accommodation * knudsen_number
        + 0.75 * mass_accommodation
    )

# 4 – vectorised Taichi kernel
@ti.kernel
def kget_vapor_transition_correction(
    knudsen_number: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mass_accommodation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_vapor_transition_correction(
            knudsen_number[i], mass_accommodation[i]
        )

# 5 – public wrapper + backend registration
@register("get_vapor_transition_correction", backend="taichi")
def ti_get_vapor_transition_correction(knudsen_number, mass_accommodation):
    """Wrapper identical to NumPy API but executed with Taichi."""
    if not (
        isinstance(knudsen_number, np.ndarray)
        and isinstance(mass_accommodation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    kn, ma = np.atleast_1d(knudsen_number), np.atleast_1d(mass_accommodation)
    n = kn.size

    kn_ti = ti.ndarray(dtype=ti.f64, shape=n)
    ma_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kn_ti.from_numpy(kn)
    ma_ti.from_numpy(ma)

    kget_vapor_transition_correction(kn_ti, ma_ti, res_ti)

    res_np = res_ti.to_numpy()
    return res_np.item() if res_np.size == 1 else res_np
