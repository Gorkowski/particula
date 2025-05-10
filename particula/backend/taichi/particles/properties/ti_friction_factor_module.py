import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_friction_factor(
    particle_radius: ti.f64,
    dynamic_viscosity: ti.f64,
    slip_correction: ti.f64,
) -> ti.f64:
    return 6.0 * 3.141592653589793 * dynamic_viscosity * particle_radius / slip_correction

@ti.kernel
def kget_friction_factor(
    particle_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dynamic_viscosity: ti.f64,
    slip_correction: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result.shape[0]):
        result[i] = fget_friction_factor(
            particle_radius[i], dynamic_viscosity, slip_correction[i]
        )

@register("get_friction_factor", backend="taichi")
def ti_get_friction_factor(particle_radius, dynamic_viscosity, slip_correction):
    # 5 a – type guard
    if not np.isscalar(dynamic_viscosity):
        raise TypeError("dynamic_viscosity must be a scalar (float).")
    if not all(
        np.isscalar(arg) or isinstance(arg, np.ndarray)
        for arg in (particle_radius, slip_correction)
    ):
        raise TypeError(
            "Taichi backend expects scalar(s) or NumPy array(s) for "
            "`particle_radius` and `slip_correction`."
        )

    # 5 b – convert to 1-D NumPy arrays and broadcast shapes
    r_np = np.asarray(particle_radius, dtype=np.float64).ravel()
    c_np = np.asarray(slip_correction, dtype=np.float64).ravel()

    try:
        r_np, c_np = np.broadcast_arrays(r_np, c_np)
    except ValueError as exc:
        raise ValueError(
            "`particle_radius` and `slip_correction` could not be "
            "broadcast to a common shape."
        ) from exc

    r_np = r_np.ravel()
    c_np = c_np.ravel()
    n = r_np.size
    r_ti = ti.ndarray(dtype=ti.f64, shape=n)
    c_ti = ti.ndarray(dtype=ti.f64, shape=n)
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)
    r_ti.from_numpy(r_np)
    c_ti.from_numpy(c_np)

    kget_friction_factor(r_ti, float(dynamic_viscosity), c_ti, out_ti)

    res = out_ti.to_numpy()
    return res.item() if res.size == 1 else res
