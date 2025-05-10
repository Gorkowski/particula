"""Taichi-accelerated Taylor microscale helpers."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_lagrangian_taylor_microscale_time(
    kolmogorov_time: ti.f64,
    re_lambda: ti.f64,
    accel_variance: ti.f64
) -> ti.f64:
    """Elementwise Lagrangian Taylor microscale time (Taichi)."""
    return kolmogorov_time * ti.sqrt(
        (2.0 * re_lambda) / (ti.pow(15.0, 0.5) * accel_variance)
    )

@ti.func
def fget_taylor_microscale(
    fluid_rms_velocity: ti.f64,
    kinematic_viscosity: ti.f64,
    turbulent_dissipation: ti.f64
) -> ti.f64:
    """Elementwise Taylor microscale (Taichi)."""
    return fluid_rms_velocity * ti.sqrt(
        (15.0 * kinematic_viscosity**2) / turbulent_dissipation
    )

@ti.func
def fget_taylor_microscale_reynolds_number(
    fluid_rms_velocity: ti.f64,
    taylor_microscale: ti.f64,
    kinematic_viscosity: ti.f64
) -> ti.f64:
    """Elementwise Taylor-microscale Reynolds number (Taichi)."""
    return (fluid_rms_velocity * taylor_microscale) / kinematic_viscosity

@ti.kernel
def kget_lagrangian_taylor_microscale_time(
    kolmogorov_time: ti.types.ndarray(dtype=ti.f64, ndim=1),
    re_lambda: ti.types.ndarray(dtype=ti.f64, ndim=1),
    accel_variance: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Lagrangian Taylor microscale time (Taichi)."""
    for i in range(result.shape[0]):
        result[i] = fget_lagrangian_taylor_microscale_time(
            kolmogorov_time[i], re_lambda[i], accel_variance[i]
        )

@ti.kernel
def kget_taylor_microscale(
    fluid_rms_velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taylor microscale (Taichi)."""
    for i in range(result.shape[0]):
        result[i] = fget_taylor_microscale(
            fluid_rms_velocity[i], kinematic_viscosity[i], turbulent_dissipation[i]
        )

@ti.kernel
def kget_taylor_microscale_reynolds_number(
    fluid_rms_velocity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    taylor_microscale: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Taylor-microscale Reynolds number (Taichi)."""
    for i in range(result.shape[0]):
        result[i] = fget_taylor_microscale_reynolds_number(
            fluid_rms_velocity[i], taylor_microscale[i], kinematic_viscosity[i]
        )

@register("get_lagrangian_taylor_microscale_time", backend="taichi")
def ti_get_lagrangian_taylor_microscale_time(
    kolmogorov_time, re_lambda, accel_variance
):
    """Taichi wrapper for Lagrangian Taylor microscale time."""
    if not (
        isinstance(kolmogorov_time, np.ndarray)
        and isinstance(re_lambda, np.ndarray)
        and isinstance(accel_variance, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    kt, rl, av = (
        np.atleast_1d(kolmogorov_time),
        np.atleast_1d(re_lambda),
        np.atleast_1d(accel_variance),
    )
    n = kt.size

    kt_ti = ti.ndarray(dtype=ti.f64, shape=n)
    rl_ti = ti.ndarray(dtype=ti.f64, shape=n)
    av_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kt_ti.from_numpy(kt)
    rl_ti.from_numpy(rl)
    av_ti.from_numpy(av)

    kget_lagrangian_taylor_microscale_time(kt_ti, rl_ti, av_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_taylor_microscale", backend="taichi")
def ti_get_taylor_microscale(
    fluid_rms_velocity, kinematic_viscosity, turbulent_dissipation
):
    """Taichi wrapper for Taylor microscale."""
    if not (
        isinstance(fluid_rms_velocity, np.ndarray)
        and isinstance(kinematic_viscosity, np.ndarray)
        and isinstance(turbulent_dissipation, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    u, nu, eps = (
        np.atleast_1d(fluid_rms_velocity),
        np.atleast_1d(kinematic_viscosity),
        np.atleast_1d(turbulent_dissipation),
    )
    n = u.size

    u_ti = ti.ndarray(dtype=ti.f64, shape=n)
    nu_ti = ti.ndarray(dtype=ti.f64, shape=n)
    eps_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    u_ti.from_numpy(u)
    nu_ti.from_numpy(nu)
    eps_ti.from_numpy(eps)

    kget_taylor_microscale(u_ti, nu_ti, eps_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_taylor_microscale_reynolds_number", backend="taichi")
def ti_get_taylor_microscale_reynolds_number(
    fluid_rms_velocity, taylor_microscale, kinematic_viscosity
):
    """Taichi wrapper for Taylor-microscale Reynolds number."""
    if not (
        isinstance(fluid_rms_velocity, np.ndarray)
        and isinstance(taylor_microscale, np.ndarray)
        and isinstance(kinematic_viscosity, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    u, lam, nu = (
        np.atleast_1d(fluid_rms_velocity),
        np.atleast_1d(taylor_microscale),
        np.atleast_1d(kinematic_viscosity),
    )
    n = u.size

    u_ti = ti.ndarray(dtype=ti.f64, shape=n)
    lam_ti = ti.ndarray(dtype=ti.f64, shape=n)
    nu_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    u_ti.from_numpy(u)
    lam_ti.from_numpy(lam)
    nu_ti.from_numpy(nu)

    kget_taylor_microscale_reynolds_number(u_ti, lam_ti, nu_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
