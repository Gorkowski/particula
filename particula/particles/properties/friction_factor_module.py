"""Taichi-accelerated friction-factor implementation."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register


PI = 3.141592653589793

@ti.func
def fget_friction_factor(
    particle_radius: ti.f64,
    dynamic_viscosity: ti.f64,
    slip_correction: ti.f64,
) -> ti.f64:
    """Scalar friction factor f = 6πμr / C."""
    return 6.0 * PI * dynamic_viscosity * particle_radius / slip_correction
@ti.kernel
def kget_friction_factor(
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.f64,
    slip_correction: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorised friction factor."""
    for i in range(result.shape[0]):
        result[i] = fget_friction_factor(
            particle_radius[i], dynamic_viscosity, slip_correction[i]
        )

@register("get_friction_factor", backend="taichi")
def ti_get_friction_factor(
    particle_radius,
    dynamic_viscosity,
    slip_correction,
):
    """Taichi wrapper for get_friction_factor."""
    # 5 a – type guard
    if not (
        isinstance(particle_radius, np.ndarray)
        and isinstance(slip_correction, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for radii & slip.")
    if not np.isscalar(dynamic_viscosity):
        raise TypeError("dynamic_viscosity must be a Python / NumPy scalar.")

    # 5 b – ensure 1-D
    pr, sc = np.atleast_1d(particle_radius), np.atleast_1d(slip_correction)
    n = pr.size

    # 5 c – allocate buffers
    pr_ti = ti.ndarray(dtype=ti.f64, shape=n)
    sc_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pr_ti.from_numpy(pr)
    sc_ti.from_numpy(sc)

    # 5 d – launch
    kget_friction_factor(pr_ti, float(dynamic_viscosity), sc_ti, res_ti)

    # 5 e – back to NumPy
    res_np = res_ti.to_numpy()
    return res_np.item() if res_np.size == 1 else res_np
