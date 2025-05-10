"""Taichi-accelerated Stokes settling velocity (with slip-correction)."""
import taichi as ti
import numpy as np
from particula.backend import register
from particula.util.constants import STANDARD_GRAVITY

# 3 ─ element-wise function
@ti.func
def fget_particle_settling_velocity(          # noqa: N802
    r: ti.f64,
    rho_p: ti.f64,
    c_c: ti.f64,
    mu: ti.f64,
    g: ti.f64,
    rho_f: ti.f64,
) -> ti.f64:
    return (2.0 * r)**2 * (rho_p - rho_f) * c_c * g / (18.0 * mu)

# 4 ─ kernel
@ti.kernel
def kget_particle_settling_velocity(          # noqa: N802
    r: ti.types.ndarray(dtype=ti.f64, ndim=1),
    rho_p: ti.types.ndarray(dtype=ti.f64, ndim=1),
    c_c: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mu: ti.f64,
    g: ti.f64,
    rho_f: ti.f64,
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(out.shape[0]):
        out[i] = fget_particle_settling_velocity(r[i], rho_p[i], c_c[i], mu, g, rho_f)

# 5 ─ wrapper
@register("get_particle_settling_velocity", backend="taichi")   # noqa: D401
def ti_get_particle_settling_velocity(        # noqa: N802
    particle_radius,
    particle_density,
    slip_correction_factor,
    dynamic_viscosity,
    gravitational_acceleration: float = STANDARD_GRAVITY,
    fluid_density: float = 0.0,
):
    r_arr, rho_arr, c_arr = np.broadcast_arrays(
        np.atleast_1d(particle_radius),
        np.atleast_1d(particle_density),
        np.atleast_1d(slip_correction_factor),
    )
    n = r_arr.size
    r_ti   = ti.ndarray(dtype=ti.f64, shape=n); r_ti.from_numpy(r_arr.astype(np.float64))
    rho_ti = ti.ndarray(dtype=ti.f64, shape=n); rho_ti.from_numpy(rho_arr.astype(np.float64))
    c_ti   = ti.ndarray(dtype=ti.f64, shape=n); c_ti.from_numpy(c_arr.astype(np.float64))
    out_ti = ti.ndarray(dtype=ti.f64, shape=n)

    kget_particle_settling_velocity(
        r_ti, rho_ti, c_ti,
        float(dynamic_viscosity),
        float(gravitational_acceleration),
        float(fluid_density),
        out_ti,
    )
    out = out_ti.to_numpy()
    return out[0] if out.size == 1 else out.reshape(r_arr.shape)
